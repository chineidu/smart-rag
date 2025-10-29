from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class MetaState(TypedDict):
    query: str
    answer: str

    messages: Annotated[list[AnyMessage], add_messages]


class State(MetaState):
    summary: str
