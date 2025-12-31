"""Admin-only endpoints for role and user management."""

from typing import Any

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from src import create_logger
from src.api.core.auth import get_current_admin_user
from src.api.core.cache import cached
from src.api.core.exceptions import HTTPError
from src.api.core.ratelimit import limiter
from src.api.core.responses import MsgSpecJSONResponse
from src.config import app_config
from src.db.crud import CRUDFactory
from src.db.models import get_db
from src.schemas.types import RoleType
from src.schemas.user_schema import RoleSchema, UserWithHashSchema

logger = create_logger(name="admin")
LIMIT_VALUE: int = app_config.api_config.ratelimit.default_rate
router = APIRouter(tags=["admin"], default_response_class=MsgSpecJSONResponse)


@router.post("/admin/roles", status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_new_role(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    role: RoleSchema,
    current_admin: UserWithHashSchema = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
) -> RoleSchema:
    """
    Create a new role. Admin access required.

    Parameters
    ----------
        role: RoleSchema
            Role information to create
        current_admin: UserWithHashSchema
            Current authenticated admin user
        db: Session
            Database session

    Returns
    -------
        Created role information
    """
    try:
        crud_factory = CRUDFactory(db=db)
        # Validate role name is a valid RoleType
        if role.name not in [r.value for r in RoleType]:
            raise HTTPError(
                status_code=status.HTTP_400_BAD_REQUEST,
                details=f"Invalid role name. Must be one of: {', '.join([r.value for r in RoleType])}",
            )
        # Check if role already exists
        db_role = crud_factory.create_role(role=role)

        logger.info(f"Admin {current_admin.username} created role: {db_role.name}")

        return RoleSchema(
            id=db_role.id,
            name=db_role.name,
            description=db_role.description,
            created_at=db_role.created_at.isoformat(timespec="seconds"),
            updated_at=db_role.updated_at.isoformat(timespec="seconds")
            if db_role.updated_at
            else None,
        )

    except Exception as e:
        logger.error(f"Error creating role: {str(e)}")
        raise HTTPError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=f"Failed to create role: {str(e)}",
        ) from e


@router.post(
    "/admin/users/{username}/roles/{role_name}", status_code=status.HTTP_200_OK
)
@limiter.limit("10/minute")
async def assign_role_to_user_endpoint(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    username: str,
    role_name: RoleType | str,
    current_admin: UserWithHashSchema = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """
    Assign a role to a user. Admin access required.

    Parameters
    ----------
        username: str
            Username to assign role to
        role_name: RoleType | str
            Role name to assign
        current_admin: UserWithHashSchema
            Current authenticated admin user
        db: Session
            Database session

    Returns
    -------
        Success message
    """
    try:
        crud_factory = CRUDFactory(db=db)
        # Validate role name
        if role_name not in [r.value for r in RoleType]:
            raise HTTPError(
                status_code=status.HTTP_400_BAD_REQUEST,
                details=f"Invalid role name. Must be one of: {', '.join([r.value for r in RoleType])}",
            )

        # Check if user exists
        db_user = crud_factory.get_user_by_username(username=username)
        if not db_user:
            raise HTTPError(
                status_code=status.HTTP_404_NOT_FOUND,
                details=f"User {username!r} not found",
            )

        # Check if role exists
        db_role = crud_factory.get_role_by_name(name=role_name)
        if not db_role:
            raise HTTPError(
                status_code=status.HTTP_404_NOT_FOUND,
                details=f"Role {role_name!r} not found",
            )

        # Assign role
        crud_factory.assign_role_to_user(username=username, role=RoleType(role_name))
        logger.info(
            f"Admin {current_admin.username} assigned role {role_name} to user {username!r}"
        )

        return {
            "message": f"Successfully assigned role {role_name!r} to user {username!r}"
        }

    except Exception as e:
        logger.error(f"Error assigning role: {str(e)}")
        raise HTTPError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=f"Failed to assign role: {str(e)}",
        ) from e


@router.delete(
    "/admin/users/{username}/roles/{role_name}", status_code=status.HTTP_200_OK
)
@limiter.limit("10/minute")
async def remove_role_from_user(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    username: str,
    role_name: str,
    current_admin: UserWithHashSchema = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """
    Remove a role from a user. Admin access required.

    Parameters
    ----------
        username: str
            Username to remove role from
        role_name: str
            Role name to remove
        current_admin: UserWithHashSchema
            Current authenticated admin user
        db: Session
            Database session

    Returns
    -------
        Success message
    """
    try:
        crud_factory = CRUDFactory(db=db)
        # Validate role name
        if role_name not in [r.value for r in RoleType]:
            raise HTTPError(
                status_code=status.HTTP_400_BAD_REQUEST,
                details=f"Invalid role name. Must be one of: {', '.join([r.value for r in RoleType])}",
            )

        # Check if user exists
        db_user = crud_factory.get_user_by_username(username=username)
        if not db_user:
            raise HTTPError(
                status_code=status.HTTP_404_NOT_FOUND,
                details=f"User '{username}' not found",
            )

        # Check if role exists
        db_role = crud_factory.get_role_by_name(name=role_name)
        if not db_role:
            raise HTTPError(
                status_code=status.HTTP_404_NOT_FOUND,
                details=f"Role '{role_name}' not found",
            )

        # Check if user has the role
        if db_user not in db_role.users:
            raise HTTPError(
                status_code=status.HTTP_400_BAD_REQUEST,
                details=f"User '{username}' does not have role '{role_name}'",
            )

        # Remove role
        db_role.users.remove(db_user)
        db.commit()

        logger.info(
            f"Admin {current_admin.username} removed role '{role_name}' from user '{username}'"
        )

        return {
            "message": f"Successfully removed role '{role_name}' from user '{username}'"
        }
    except Exception as e:
        logger.error(f"Error removing role: {str(e)}")
        raise HTTPError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=f"Failed to remove role: {str(e)}",
        ) from e


@router.get("/admin/roles", status_code=status.HTTP_200_OK)
@limiter.limit("10/minute")
@cached(ttl=600, key_prefix="roles")  # type: ignore
async def list_roles(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    current_admin: UserWithHashSchema = Depends(get_current_admin_user),  # noqa: ARG001
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """List all roles in the system. Admin access required."""
    crud_factory = CRUDFactory(db=db)
    all_roles = crud_factory.get_all_roles()
    if all_roles:
        roles_list: list[RoleSchema] = [
            RoleSchema(
                id=role.id,
                name=role.name,
                description=role.description,
                created_at=role.created_at.isoformat(timespec="seconds"),
                updated_at=role.updated_at.isoformat(timespec="seconds")
                if role.updated_at
                else None,
            )
            for role in all_roles
        ]
        return {"roles": roles_list}

    return {"roles": []}
