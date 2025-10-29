from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from demo.nodes import llm_call_node, should_summarize, summarization_node
from demo.state import State
from demo.tools import date_and_time_tool, search_tool

max_attempts: int = 3

# Shared memory saver instance to persist checkpoints across all graph instances
_memory_saver = MemorySaver()
_graph_instance: CompiledStateGraph | None = None


def build_graph() -> CompiledStateGraph:
    """Builds and returns the state graph with shared memory."""
    global _graph_instance

    # Return cached instance if it exists
    if _graph_instance is not None:
        return _graph_instance

    builder: StateGraph = StateGraph(State)

    # Add nodes
    tool_node = ToolNode([date_and_time_tool, search_tool])

    builder.add_node(
        "llm_call",
        llm_call_node,
        retry_policy=RetryPolicy(max_attempts=max_attempts, initial_interval=1.0),
    )
    builder.add_node(
        "tools",
        tool_node,
        retry_policy=RetryPolicy(max_attempts=max_attempts, initial_interval=1.0),
    )
    builder.add_node(
        "summarize",
        summarization_node,
        retry_policy=RetryPolicy(max_attempts=max_attempts, initial_interval=1.0),
    )

    # Add edges
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call", tools_condition, {"tools": "tools", END: END}
    )
    builder.add_conditional_edges(
        "llm_call", should_summarize, {"summarize": "summarize", END: END}
    )
    builder.add_edge("tools", "llm_call")

    # Compile the graph with shared memory saver
    _graph_instance = builder.compile(checkpointer=_memory_saver)
    return _graph_instance
