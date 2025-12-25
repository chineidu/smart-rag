from typing import Any

from pydantic import BaseModel, Field, field_validator

from .types import NextAction, RetrieverMethodType, ToolsType


class Step(BaseModel):
    """A single step in the multi-step reasoning process."""

    question: str = Field(description="The question to be answered by the step.")
    rationale: str = Field(description="The brief reasoning behind the question.")
    tool: ToolsType = Field(
        description="The tool to use for this step. For information found ONLY in "
        "internal documents, "
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

    @field_validator("tool", mode="after")
    @classmethod
    def validate_tool(cls, v: Any) -> str:
        """Extract the value form the enum."""
        if isinstance(v, ToolsType):
            return v.value
        return ToolsType(v).value


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

    @field_validator("next_action", mode="after")
    @classmethod
    def validate_next_action(cls, v: Any) -> str:
        """Extract the value form the enum."""
        if isinstance(v, NextAction):
            return v.value
        return NextAction(v).value


class RetrieverMethod(BaseModel):
    method: RetrieverMethodType = Field(
        description="The retrieval method to use for retrieving internal documents.",
    )
    rationale: str = Field(description="The brief reasoning behind the decision.")

    @field_validator("method", mode="after")
    @classmethod
    def validate_method(cls, v: Any) -> str:
        """Extract the value form the enum."""
        if isinstance(v, RetrieverMethodType):
            return v.value
        return RetrieverMethodType(v).value


class Decision(BaseModel):
    next_action: NextAction = Field(..., description="The next action to take.")
    rationale: str = Field(description="The brief reasoning behind the decision.")

    @field_validator("next_action", mode="after")
    @classmethod
    def validate_next_action(cls, v: Any) -> str:
        """Extract the value form the enum."""
        if isinstance(v, NextAction):
            return v.value
        return NextAction(v).value


class StructuredMemoryResponse(BaseModel):
    """Schema for structured memory response."""

    # Long-term (durable) information
    technical_preferences: list[str] = Field(
        default_factory=list,
        description="User's technical preferences and expertise level",
    )
    communication_preferences: list[str] = Field(
        default_factory=list,
        description="User's communication preferences and style",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Standing limitations or constraints. (e.g., avoid using things specified by the user)",
    )
    interests: list[str] = Field(
        default_factory=list,
        description="User's hobbies, interests, and activities they enjoy",
    )
    #
    pain_points: list[str] = Field(
        default_factory=list,
        description="Challenges, pain points, or recurring issues the user faces",
    )
    other_preferences: list[str] = Field(
        default_factory=list,
        description="Any other relevant user details that don't fit other categories",
    )
