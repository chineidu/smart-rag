import operator as op
from typing import Annotated, Any, TypedDict

from langchain_core.documents.base import Document

from src.utilities.utils import merge_dicts


class OtherInfo(TypedDict):
    source_type: str
    retrieval_relevance: str
    is_hallucinating: str
    rewritten_query: str


class State(TypedDict):
    query: str
    messages: Annotated[list[str], op.add]
    runs: int
    other_info: Annotated[dict[str, Any], merge_dicts]  # Use custom merger
    documents: list[Document]
    response: str
