from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from schemas.types import NextAction, ToolsType
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
from src.state import State

MAX_RETRIES: int = 3
INITIAL_RETRY_INTERVAL: float = 1.0


def build_graph() -> CompiledStateGraph:
    """Builds and returns the state graph for the application."""
    builder: StateGraph = StateGraph(State)

    # =====================================
    # ============= Add nodes =============
    # =====================================

    builder.add_node(
        "validate_query",
        validate_query_node,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )
    builder.add_node(
        "unrelated_query",
        unrelated_query_node,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )

    builder.add_node(
        "generate_plan",
        generate_plan_node,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )
    builder.add_node(
        "retrieve_internal_docs",
        retrieve_internal_docs_node,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )
    builder.add_node(
        "internet_search",
        internet_search_node,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )
    builder.add_node(
        "rerank_and_compress",
        rerank_and_compress_node,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )

    builder.add_node(
        "summarize",
        summarization_node,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )
    builder.add_node(
        "final_answer",
        final_answer_node,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
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

    # Compile the graph
    return builder.compile()
