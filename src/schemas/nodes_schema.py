from pydantic import BaseModel, Field  # type: ignore

from .types import NextAction, RetrieverMethodType, ToolsType


class Step(BaseModel):
    """A single step in the multi-step reasoning process."""

    question: str = Field(description="The question to be answered by the step.")
    rationale: str = Field(description="The brief reasoning behind the question.")
    tool: ToolsType = Field(
        description="The tool to use for this step. For information found ONLY in internal documents, "
        "use 'vector_store'. For the latest information found on the web, use 'web_search'.",
    )
    search_keywords: list[str] = Field(
        description="Critical keywords and phrases to use for web search or vector store "
        "retrieval to ensure quality results are returned.",
    )
    target_section: str | None = Field(
        default=None,
        description="The target section in the document to focus on. This is ONLY required when "
        "the tool is 'vector_store'. e.g., 'ITEM 1A. RISK FACTORS'.",
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="List of step indices (0-based) that this step depends on. "
        "Leave empty if this step can run immediately.",
    )


class Plan(BaseModel):
    """A multi-step plan for answering a complex question."""

    steps: list[Step] = Field(description="A list of steps to execute in the plan.")


class ReWrittenQuery(BaseModel):
    question: str = Field(description="Original query to be re-written.")
    rewritten_queries: list[str] = Field(description="The re-written queries.")
    rationale: str = Field(description="The brief reasoning behind the decision.")


class ValidateQuery(BaseModel):
    is_related_to_context: bool = Field(
        description="Whether the query is related to the context."
    )
    next_action: NextAction = Field(description="The next action to take.")
    rationale: str = Field(description="The brief reasoning behind the decision.")


class RetrieverMethod(BaseModel):
    method: RetrieverMethodType = Field(
        description="The retrieval method to use for retrieving internal documents.",
    )
    rationale: str = Field(description="The brief reasoning behind the decision.")


class Decision(BaseModel):
    next_action: NextAction = Field(..., description="The next action to take.")
    rationale: str = Field(description="The brief reasoning behind the decision.")
