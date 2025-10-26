from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from src.nodes import (
    answer_node,
    check_hallucination_node,
    classify_query_node,
    failed_node,
    generate_response,
    generate_web_search_response,
    grade_documents,
    llm_call_node,
    retrieve_documents,
    rewrite_query,
    should_continue_to_final_answer,
    should_continue_to_generate,
    should_continue_to_retrieve,
)
from src.state import State
from src.utilities.tools import search_tool

MAX_RETRIES: int = 3
INITIAL_RETRY_INTERVAL: float = 1.0


def build_graph() -> CompiledStateGraph:
    """Builds and returns the state graph for the application."""
    builder: StateGraph = StateGraph(State)

    # =====================================
    # ============= Add nodes =============
    # =====================================

    tool_node = ToolNode([search_tool])

    builder.add_node(
        "query_analysis",
        classify_query_node,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )
    builder.add_node(
        "tools",
        tool_node,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )
    builder.add_node("llm_call_node", llm_call_node)
    builder.add_node("retrieve", retrieve_documents)
    builder.add_node(
        "grade",
        grade_documents,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )
    builder.add_node(
        "generate",
        generate_response,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )
    builder.add_node(
        "rewrite",
        rewrite_query,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )
    builder.add_node(
        "check_hallucination",
        check_hallucination_node,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )
    builder.add_node(
        "web_search_response",
        generate_web_search_response,
        retry_policy=RetryPolicy(
            max_attempts=MAX_RETRIES, initial_interval=INITIAL_RETRY_INTERVAL
        ),
    )
    builder.add_node("answer", answer_node)
    builder.add_node("failed", failed_node)

    # =====================================
    # ============= Add edges =============
    # =====================================
    builder.add_edge(START, "query_analysis")
    builder.add_conditional_edges(
        "query_analysis",
        should_continue_to_retrieve,
        {"retrieve": "retrieve", "web_search": "llm_call_node"},
    )
    builder.add_conditional_edges(
        "llm_call_node",
        tools_condition,
        {"tools": "tools", END: "failed"},
    )
    builder.add_edge("tools", "web_search_response")
    builder.add_edge("web_search_response", END)
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        should_continue_to_generate,
        {"generate": "generate", "rewrite": "rewrite", "failed": "failed"},
    )
    builder.add_edge("generate", "check_hallucination")
    builder.add_conditional_edges(
        "check_hallucination",
        should_continue_to_final_answer,
        {"answer": "answer", "rewrite": "rewrite", "failed": "failed"},
    )
    builder.add_edge("answer", END)
    builder.add_edge("rewrite", "retrieve")

    # Compile the graph
    return builder.compile()
