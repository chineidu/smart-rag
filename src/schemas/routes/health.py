from dataclasses import dataclass, field

from pydantic import Field

from src.schemas.base import BaseSchema


class HealthStatusSchema(BaseSchema):
    """Health status model."""

    name: str = Field(description="Name of the API")
    status: str = Field(description="Current status of the API")
    version: str = Field(description="API version")


@dataclass(slots=True, kw_only=True)
class HealthStatus:
    """Health status model."""

    name: str = field(metadata={"description": "Name of the API"})
    status: str = field(metadata={"description": "Current status of the API"})
    version: str = field(metadata={"description": "API version"})
