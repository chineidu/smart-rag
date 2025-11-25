from typing import TypedDict

from langchain_core.documents.base import Document

from src.schemas.nodes_schema import Plan


class StepState(TypedDict):
    """State of a completed step in the multi-step reasoning process."""

    step_index: int  # Index of the step in the plan
    question: str  # The question asked in this step
    rewritten_queries: list[str]  # Re-written queries for this step
    retrieved_documents: list[Document]  # Documents retrieved for this step
    summary: str  # Summary of the step's findings


class State(TypedDict):
    """State of the multi-step reasoning process."""

    original_question: str  # The original complex question
    is_related_to_context: bool  # Whether the question is related to the context
    plan: Plan  # The multi-step plan
    step_state: list[StepState]  # List of completed steps
    current_step_index: int  # Index of the current step being executed
    retrieved_documents: list[Document]  # Documents retrieved in the current step
    reranked_documents: list[Document]  # Documents reranked based on relevance
    num_iterations: int  # Number of iterations completed
    synthesized_context: str  # Synthesized context from reranked documents
    final_answer: str  # The final answer to the original question
