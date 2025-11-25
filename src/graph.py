from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import RetryPolicy

from src import create_logger
from src.config import app_config, app_settings
from src.nodes import (
    final_answer_node,
    generate_plan_node,
    internet_search_node,
    rerank_and_compress_node,
    retrieve_internal_docs_node,
    route_by_tool_condition,
    should_continue_condition,
    summarization_node,
    unrelated_query_node,
    validate_query_node,
)
from src.schemas.types import NextAction, ToolsType
from src.state import State

logger = create_logger(name="graph_manager")

# Constants
INITIAL_RETRY_INTERVAL: float = 1.0
MAX_ATTEMPTS: int = app_config.custom_config.max_attempts
DB_URI: str = app_settings.database_url


class GraphManager:
    """Manages LangGraph instances with PostgreSQL checkpointing."""

    def __init__(self) -> None:
        self.checkpointer: AsyncPostgresSaver | None = None
        self.checkpointer_context = None
        self.graph_instance: CompiledStateGraph | None = None
        self.long_term_memory: BaseStore | None = None
        self.long_term_memory_context = None

    async def initialize_checkpointer(self) -> None:
        """Initialize the Postgres checkpointer."""
        if self.checkpointer is None:
            self.checkpointer_context = AsyncPostgresSaver.from_conn_string(DB_URI)
            self.checkpointer = await self.checkpointer_context.__aenter__()  # type: ignore
            await self.checkpointer.setup()

    async def initialize_long_term_memory(self) -> None:
        """Initialize long-term memory store."""
        if self.long_term_memory is None:
            self.long_term_memory_context = AsyncPostgresStore.from_conn_string(DB_URI)
            self.long_term_memory = await self.long_term_memory_context.__aenter__()  # type: ignore
            await self.long_term_memory.setup()

    async def cleanup_checkpointer(self) -> None:
        """Clean up the Postgres checkpointer."""
        if self.checkpointer_context is not None and self.checkpointer is not None:
            try:
                await self.checkpointer_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error cleaning up checkpointer: {e}")
            finally:
                self.checkpointer = None
                self.checkpointer_context = None

    async def cleanup_long_term_memory(self) -> None:
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

    async def build_graph(self) -> CompiledStateGraph:
        """BBuild and compile the state graph. This asynchronous method constructs a
        StateGraph with predefined nodes and edges.

        Returns
        -------
        CompiledStateGraph
            The compiled state graph ready for execution, equipped with checkpointer
            for persistence across sessions.
        """
        # Return cached instance if it exists
        if self.graph_instance is not None:
            return self.graph_instance

        # Ensure checkpointer is initialized
        if self.checkpointer is None:
            await self.initialize_checkpointer()

        # Ensure long-term memory is initialized
        if self.long_term_memory is None:
            await self.initialize_long_term_memory()

        # Otherwise, build the graph
        builder: StateGraph = StateGraph(State)
        # =====================================
        # ============= Add nodes =============
        # =====================================

        builder.add_node(
            "validate_query",
            validate_query_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )
        builder.add_node(
            "unrelated_query",
            unrelated_query_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )

        builder.add_node(
            "generate_plan",
            generate_plan_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )
        builder.add_node(
            "retrieve_internal_docs",
            retrieve_internal_docs_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )
        builder.add_node(
            "internet_search",
            internet_search_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )
        builder.add_node(
            "rerank_and_compress",
            rerank_and_compress_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )

        builder.add_node(
            "summarize",
            summarization_node,
            retry_policy=RetryPolicy(
                max_attempts=MAX_ATTEMPTS, initial_interval=INITIAL_RETRY_INTERVAL
            ),
        )
        builder.add_node(
            "final_answer",
            final_answer_node,
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

        builder.add_edge("retrieve_internal_docs", "rerank_and_compress")
        builder.add_edge("internet_search", "rerank_and_compress")
        builder.add_edge("rerank_and_compress", "summarize")
        builder.add_conditional_edges(
            "summarize",
            should_continue_condition,  # function to determine next action
            {NextAction.CONTINUE: "generate_plan", NextAction.FINISH: "final_answer"},
        )
        builder.add_edge("final_answer", END)
        builder.add_edge("unrelated_query", END)

        # Compile the graph with persistent Postgres checkpointer
        self.graph_instance = builder.compile(
            checkpointer=self.checkpointer, store=self.long_term_memory
        )
        logger.info(
            "Graph instance built and compiled with Postgres checkpointer and long-term memory."
        )

        return self.graph_instance
