from dataclasses import dataclass, field
from typing import Any

from pydantic import Field  # type: ignore

from src.schemas.base import BaseSchema
from src.schemas.types import TaskStatusType


@dataclass(slots=True, kw_only=True)
class Result:
    """Schema for the truncated result."""

    status: TaskStatusType = field(
        default=TaskStatusType.ERROR, metadata={"description": "The response status"}
    )
    session_id: str = field(default="", metadata={"description": "The session ID"})
    task_id: str = field(default="", metadata={"description": "The task ID"})
    event_count: int = field(
        default=0, metadata={"description": "The total number of events"}
    )
    sample_records: list[dict[str, Any]] = field(
        default_factory=list, metadata={"description": "The truncated result"}
    )


class TaskStatusResponse(BaseSchema):
    """Schema for the task status response."""

    task_id: str = Field(default="", description="The task ID")
    state: TaskStatusType = Field(
        default=TaskStatusType.PENDING, description="The task state"
    )
    ready: bool = Field(default=False, description="Whether the task is ready")
    result: Result | dict = Field(default_factory=dict, description="The task result")
