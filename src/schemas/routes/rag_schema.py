from pydantic import Field

from src.schemas.base import BaseSchema


class QuerySchema(BaseSchema):
    """Schema for user query input."""

    query: str = Field(description="User query")
