from datetime import datetime
from typing import Annotated

from pydantic import BeforeValidator, ConfigDict, Field  # type: ignore

from src.schemas.base import BaseSchema
from src.schemas.types import FeedbackType


def normalize_feedback(value: str | None) -> str | None:
    """Normalize feedback value, converting string 'None' to None."""
    if value == "None":
        return None
    return value


class FeedbackRequestSchema(BaseSchema):
    """Feedback request model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "abc123",
                "message_index": 1,
                "user_message": "What is LangGraph?",
                "assistant_message": "LangGraph is a framework...",
                "sources": ["https://example.com"],
                "feedback": "positive",
            }
        }
    )

    session_id: str = Field(..., description="Session/checkpoint ID")
    user_id: int | None = Field(
        default=None,
        description="ID of the user providing feedback (auto-populated from auth)",
    )
    username: str | None = Field(
        default=None,
        description="Username of the user providing feedback (auto-populated from auth)",
    )
    message_index: int = Field(
        ..., ge=0, description="Index of the message in conversation"
    )
    user_message: str = Field(default="", description="User's question/prompt")
    assistant_message: str = Field(..., description="Assistant's response")
    sources: list[str] = Field(default_factory=list, description="List of source URLs")
    feedback: Annotated[FeedbackType, BeforeValidator(normalize_feedback)] = Field(
        default=FeedbackType.NEUTRAL,
        description="Feedback type: 'positive', 'negative', or null",
    )
    timestamp: str | None = Field(
        default_factory=lambda _: datetime.now().isoformat(timespec="seconds"),
        description="Timestamp (auto-generated if not provided)",
    )
