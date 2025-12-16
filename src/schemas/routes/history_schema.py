from datetime import datetime
from uuid import uuid4

from pydantic import Field

from src.schemas.base import BaseSchema


class ChatHistorySchema(BaseSchema):
    """Chat history model."""

    session_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Session ID"
    )
    messages: list[dict[str, str]] = Field(
        default_factory=list, description="List of chat messages"
    )
    message_count: int = Field(
        0, description="Total number of messages in the chat history"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat() + "Z",
        description="Timestamp when the chat history was created",
    )
