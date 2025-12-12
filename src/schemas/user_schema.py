from datetime import datetime

from pydantic import ConfigDict, Field  # type: ignore

from src.schemas.base import BaseSchema
from src.schemas.types import RoleType


class UserSchema(BaseSchema):
    """User schema."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "firstname": "John",
                "lastname": "Doe",
                "username": "johndoe",
                "email": "john.doe@example.com",
                "is_active": True,
            }
        }
    )

    id: int | None = Field(default=None)
    firstname: str | None = Field(default=None)
    lastname: str | None = Field(default=None)
    username: str
    email: str
    is_active: bool = True
    roles: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda _: datetime.now().isoformat(timespec="seconds")
    )
    updated_at: str | None = Field(default=None)


class UserWithHashSchema(UserSchema):
    """User schema with password hash."""

    hashed_password: str


class RoleSchema(BaseSchema):
    """Role schema."""

    id: int | None = None
    name: str | RoleType
    description: str | None = None
    created_at: str = Field(
        default_factory=lambda _: datetime.now().isoformat(timespec="seconds")
    )
    updated_at: str | None = Field(default=None)
