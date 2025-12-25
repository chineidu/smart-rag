from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import RetryPolicy

from src import create_logger
from src.config import app_config, app_settings
from src.nodes import (
    compress_documents_node,
    final_answer_node,
    generate_plan_node,
    internet_search_node,
    overall_convo_summarization_node,
    reflection_node,
    retrieve_internal_docs_node,
    route_by_tool_condition,
    should_continue_condition,
    should_summarize_overal_convo,
    unrelated_query_node,
    update_lt_memory_node,
    validate_query_node,
)
from src.schemas.types import NextAction, SummarizationConditionType, ToolsType
from src.state import State

logger = create_logger(name="graph_manager")

# Constants
INITIAL_RETRY_INTERVAL: float = 1.0
MAX_ATTEMPTS: int = app_config.custom_config.max_attempts
DB_URI: str = app_settings.database_url.replace("+psycopg2", "")


class GraphManager:
    """Manages LangGraph instances with PostgreSQL checkpointing.

    Methods
    -------
    ainit_graph_memory()
        Initializes both checkpointer and long-term memory.

    acleanup()
        Cleans up resources used by the GraphManager.

    abuild_graph(force_rebuild: bool = False) -> CompiledStateGraph
        Builds and compiles the state graph.
    """

    def __init__(self) -> None:
        # Flags to track initialization status
        self._checkpointer_initialized: bool = False
        self._long_term_memory_initialized: bool = False
        self._graph_initialized: bool = False

        self.checkpointer: AsyncPostgresSaver | None = None
        self.checkpointer_context = None
        self.graph_instance: CompiledStateGraph | None = None
        self.long_term_memory: BaseStore | None = None
        self.long_term_memory_context = None

    async def ainit_graph_memory(self) -> None:
        """Initialize both checkpointer and long-term memory."""
        await self._ainitialize_checkpointer()
        await self._ainitialize_long_term_memory()

    async def acleanup(self) -> None:
        """Clean up resources used by the GraphManager."""
        await self._cleanup_checkpointer()
        await self._cleanup_long_term_memory()

    async def abuild_graph(self, force_rebuild: bool = False) -> CompiledStateGraph:
        """Build and compile the state graph.

        Parameters
        ----------
        force_rebuild : bool, optional
            Force recompilation of the graph even if a compiled instance already exists.

        Returns
        -------
        CompiledStateGraph
            The compiled state graph ready for execution, equipped with checkpointer
            for persistence across sessions.
        """
        # Return cached instance if it exists and rebuild not requested
        if self.graph_instance is not None and not force_rebuild:
            return self.graph_instance

        # Drop previous compiled instance if forcing rebuild
        if force_rebuild:
            self.graph_instance = None

        # Ensure checkpointer is initialized
        if self.checkpointer is None:
            await self._ainitialize_checkpointer()

        # Ensure long-term memory is initialized
        if self.long_term_memory is None:
            await self._ainitialize_long_term_memory()

        # Otherwise, build the graph
        builder: StateGraph = StateGraph(State)
        # =====================================
        # ============= Add nodes =============
        # =====================================

        builder.add_node(  # type: ignore
            "validate_query",
            validate_query_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )
        builder.add_node(  # type: ignore
            "unrelated_query",
            unrelated_query_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )

        builder.add_node(  # type: ignore
            "generate_plan",
            generate_plan_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )
        builder.add_node(  # type: ignore
            "retrieve_internal_docs",
            retrieve_internal_docs_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )
        builder.add_node(  # type: ignore
            "internet_search",
            internet_search_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )
        builder.add_node(  # type: ignore
            "compress_documents",
            compress_documents_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )

        builder.add_node(  # type: ignore
            "reflect",
            reflection_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )
        builder.add_node(  # type: ignore
            "final_answer",
            final_answer_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )
        builder.add_node(  # type: ignore
            "overall_convo_summarization",
            overall_convo_summarization_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )
        builder.add_node(  # type: ignore
            "update_lt_memory",
            update_lt_memory_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )

        # =====================================
        # ============= Add edges =============
        # =====================================
        builder.add_edge(START, "validate_query")
        builder.add_conditional_edges(
            "validate_query",
            should_continue_condition,
            {
                NextAction.CONTINUE: "generate_plan",
                NextAction.FINISH: "unrelated_query",
            },
        )
        builder.add_conditional_edges(
            "generate_plan",
            route_by_tool_condition,  # function to determine which tool to use
            {
                ToolsType.VECTOR_STORE: "retrieve_internal_docs",
                ToolsType.WEB_SEARCH: "internet_search",
            },
        )

        builder.add_edge("retrieve_internal_docs", "compress_documents")
        builder.add_edge("internet_search", "compress_documents")
        builder.add_edge("compress_documents", "reflect")
        builder.add_conditional_edges(
            "reflect",
            should_continue_condition,  # function to determine next action
            {
                NextAction.CONTINUE: "generate_plan",
                NextAction.RE_PLAN: "generate_plan",
                NextAction.FINISH: "final_answer",
            },
        )
        builder.add_edge("final_answer", "update_lt_memory")
        builder.add_conditional_edges(
            "update_lt_memory",
            should_summarize_overal_convo,  # function to determine if summarization is needed,
            {
                SummarizationConditionType.SUMMARIZE: "overall_convo_summarization",
                SummarizationConditionType.END: END,
            },
        )
        builder.add_edge("update_lt_memory", END)
        builder.add_edge("unrelated_query", END)

        # Compile the graph with persistent Postgres checkpointer
        self.graph_instance = builder.compile(
            checkpointer=self.checkpointer, store=self.long_term_memory
        )
        logger.info(
            "Graph instance built and compiled with Postgres checkpointer and long-term memory."
        )
        self._graph_initialized = True

        return self.graph_instance

    async def _ainitialize_checkpointer(self) -> None:
        """Initialize the Postgres checkpointer."""
        if self.checkpointer is None:
            self.checkpointer_context = AsyncPostgresSaver.from_conn_string(DB_URI)
            self.checkpointer = await self.checkpointer_context.__aenter__()  # type: ignore
            self._checkpointer_initialized = True
            await self.checkpointer.setup()

    async def _ainitialize_long_term_memory(self) -> None:
        """Initialize long-term memory store."""
        if self.long_term_memory is None:
            self.long_term_memory_context = AsyncPostgresStore.from_conn_string(DB_URI)
            self.long_term_memory = await self.long_term_memory_context.__aenter__()  # type: ignore
            self._long_term_memory_initialized = True
            await self.long_term_memory.setup()

    async def _cleanup_checkpointer(self) -> None:
        """Clean up the Postgres checkpointer."""
        if self.checkpointer_context is not None and self.checkpointer is not None:
            try:
                await self.checkpointer_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error cleaning up checkpointer: {e}")
            finally:
                self.checkpointer = None
                self.checkpointer_context = None
                self._checkpointer_initialized = False

    async def _cleanup_long_term_memory(self) -> None:
        """Clean up the long-term memory store."""
        if (
            self.long_term_memory_context is not None
            and self.long_term_memory is not None
        ):
            try:
                await self.long_term_memory_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error cleaning up long-term memory: {e}")
            finally:
                self.long_term_memory = None
                self.long_term_memory_context = None
                self._long_term_memory_initialized = False
