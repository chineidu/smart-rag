from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from demo.nodes import llm_call_node
from demo.state import State
from demo.tools import date_tool, search_tool


def build_graph() -> CompiledStateGraph:
    """Builds and returns the state graph for the demo."""
    builder: StateGraph = StateGraph(State)

    # Add nodes
    tool_node = ToolNode([date_tool, search_tool])

    builder.add_node(
        "llm_call",
        llm_call_node,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
    )
    builder.add_node(
        "tools",
        tool_node,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
    )

    # Add edges
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call", tools_condition, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "llm_call")

    # Compile the graph
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
