import operator as op
from typing import Annotated, Any, TypedDict


class State(TypedDict):
    messages: Annotated[list[Any], op.add]
