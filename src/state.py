from typing import Annotated, TypedDict

from langchain_core.documents.base import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.schemas.nodes_schema import Plan
from src.utilities.utils import merge_documents, merge_step_states


class StepState(TypedDict):
    """State of a completed step in the multi-step reasoning process."""

    step_index: int  # Index of the step in the plan
    question: str  # The question asked in this step
    rewritten_queries: list[str]  # Re-written queries for this step
    reranked_documents: list[Document]  # Documents reranked for this step

    # Summary of the step's findings. Used to provide context when deciding whether to
    # continue or end the RAG pipeline
    summary: str


class MetaState(TypedDict):
    """Meta state of the multi-step reasoning process.

    Uses custom reducers for intelligent state merging:
    - merge_step_states: Merges steps by index, combining summaries and docs
    - merge_documents: Deduplicates retrieved/reranked documents
    - concatenate_strings: Combines synthesized context intelligently
    """

    original_question: str  # The original complex question
    is_related_to_context: bool  # Whether the question is related to the context
    plan: Plan  # The multi-step plan

    # List of completed steps (merged by step_index) with custom reducer
    step_state: Annotated[list[StepState], merge_step_states]
    current_step_index: int  # Index of the current step being executed

    # List of documents with custom reducer to deduplicate
    retrieved_documents: Annotated[list[Document], merge_documents]
    # Documents reranked (deduplicated)
    # Synthesized context (stores the summarized context for the last ran step)
    reranked_documents: Annotated[list[Document], merge_documents]
    synthesized_context: str

    num_iterations: int  # Number of iterations completed
    final_answer: str  # The final answer to the original question


class State(MetaState):
    """State of the multi-step reasoning process."""

    messages: Annotated[list[AnyMessage], add_messages]  # Conversation messages
    conversation_summary: str  # Summary of the entire conversation
