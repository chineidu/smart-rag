"""
Crud operations for the database.

(Using SQLAlchemy ORM v2.x)
"""

import json
from typing import Any, cast

from sqlalchemy import insert, select
from sqlalchemy.orm import Session, selectinload

from src import create_logger
from src.db.models import DBRole, DBUser, DBUserFeedback
from src.schemas.routes.streamer_schema import FeedbackRequestSchema
from src.schemas.types import RoleType
from src.schemas.user_schema import RoleSchema, UserWithHashSchema

logger = create_logger("crud")


def get_user_by_email(db: Session, email: str) -> DBUser | None:
    """Get a user by their email address."""
    stmt = select(DBUser).where(DBUser.email == email)
    return db.scalar(stmt)


def get_user_by_username(db: Session, username: str) -> DBUser | None:
    """Get a user by their username."""
    stmt = select(DBUser).where(DBUser.username == username)
    return db.scalar(stmt)


def get_user_by_id(db: Session, user_id: int) -> DBUser | None:
    """Get a user by their ID."""
    stmt = select(DBUser).where(DBUser.id == user_id)
    return db.scalar(stmt)


def get_feedback_by_username(
    db: Session, session_id: str, message_index: int, username: str
) -> DBUserFeedback | None:
    """Get feedback by session ID, message index, and user name."""
    stmt = select(DBUserFeedback).where(
        DBUserFeedback.session_id == session_id,
        DBUserFeedback.message_index == message_index,
        DBUserFeedback.username == username,
    )
    return db.scalar(stmt)


def get_role_by_name(db: Session, name: RoleType | str) -> DBRole | None:
    """Get a role by its name."""
    stmt = (
        select(DBRole)
        # Select the list of users associated with the role
        .options(selectinload(DBRole.users))
        .where(DBRole.name == name)
    )
    return db.scalar(stmt)


def get_all_roles(db: Session) -> list[DBRole] | None:
    """Get all roles from the database."""
    result = db.scalars(select(DBRole)).all()
    return cast(list[DBRole], result)


def convert_userdb_to_schema(db_user: DBUser) -> UserWithHashSchema | None:
    """Convert a DBUser object to a UserWithHashSchema object."""
    try:
        return UserWithHashSchema(
            id=db_user.id,
            firstname=db_user.firstname,
            lastname=db_user.lastname,
            username=db_user.username,
            email=db_user.email,
            is_active=db_user.is_active,
            created_at=db_user.created_at.isoformat(timespec="seconds"),
            updated_at=db_user.updated_at.isoformat(timespec="seconds")
            if db_user.updated_at
            else None,
            hashed_password=db_user.hashed_password,
            roles=[role.name for role in db_user.roles],
        )
    except Exception as e:
        logger.error(f"Error converting DBUser to UserWithHashSchema: {e}")
        return None


def create_user(db: Session, user: UserWithHashSchema) -> DBUser:
    """Create a new user in the database."""
    try:
        # Check if email or username already exists
        existing_user = get_user_by_email(db, user.email)
        if existing_user:
            raise ValueError(f"Email {user.email!r} is already registered.")

        values: dict[str, Any] = user.model_dump(
            exclude={"id", "roles", "updated_at", "created_at"}
        )
        stmt = insert(DBUser).values(**values).returning(DBUser)
        db_user = db.scalar(stmt)
        db.commit()
        logger.info(f"Logged new user with ID: {db_user.id!r} to the database.")
        return db_user

    except Exception as e:
        logger.error(f"Error creating user: {e}")
        db.rollback()
        raise e


def create_feedback(db: Session, feedback: FeedbackRequestSchema) -> DBUserFeedback:
    """Create a new user feedback entry in the database."""
    try:
        values: dict[str, Any] = feedback.model_dump(exclude={"timestamp"})
        # Serialize sources list to JSON string for database storage
        if "sources" in values and isinstance(values["sources"], list):
            values["sources"] = json.dumps(values["sources"])
        stmt = insert(DBUserFeedback).values(**values).returning(DBUserFeedback)
        db_feedback = db.scalar(stmt)
        db.commit()
        logger.info(f"Logged new feedback with ID: {db_feedback.id!r} to the database.")
        return db_feedback

    except Exception as e:
        logger.error(f"Error creating feedback: {e}")
        db.rollback()
        raise e


def create_role(db: Session, role: RoleSchema) -> DBRole:
    """Create a new role in the database."""
    try:
        # Check if role exists
        db_role = get_role_by_name(db=db, name=role.name)
        if db_role is not None:
            logger.info(f"Role '{role.name}' already exists with ID: {db_role.id}")
            return db_role

        stmt = (
            insert(DBRole)
            .values(role.model_dump(exclude={"id", "created_at", "updated_at"}))
            .returning(DBRole)
        )
        db_role = db.scalar(stmt)
        db.commit()
        logger.info(f"Logged new role with ID: {db_role.id!r} to the database.")
        return db_role

    except Exception as e:
        logger.error(f"Error creating role: {e}")
        db.rollback()
        raise e


def assign_role_to_user(db: Session, username: str, role: RoleType) -> None:
    """Assign a role to a user."""
    try:
        # Check that role and user exist
        if not (db_role := get_role_by_name(db=db, name=role)):
            raise ValueError(f"Role {role} does not exist!")

        if not (db_user := get_user_by_username(db=db, username=username)):
            raise ValueError(f"User {username} does not exist!")

        if db_user not in db_role.users:
            db_role.users.append(db_user)
            db.commit()
            logger.info(f"Assigned role {role!r} to user {username!r}.")
        else:
            logger.info(f"User {username!r} already has role {role!r}.")

        return

    except Exception as e:
        logger.error(f"Error assigning role to user: {e}")
        raise
