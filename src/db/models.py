from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Table,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from src.config import app_settings
from src.db import DatabasePool


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


# =========================================================
# ==================== Database Models ====================
# =========================================================
# Association table for many-to-many relationship between users and roles
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", ForeignKey("users.id"), primary_key=True),
    Column("role_id", ForeignKey("roles.id"), primary_key=True),
)


class DBUser(Base):
    """Data model for storing user information."""

    __tablename__: str = "users"

    id: Mapped[int] = mapped_column("id", primary_key=True)
    firstname: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    lastname: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

    # =========================================================
    # ==================== Relationships ======================
    # =========================================================

    # Many-to-many relationship with roles (through user_roles association table)
    roles = relationship("DBRole", secondary=user_roles, back_populates="users")

    # One-to-many relationship: a user can have many feedback entries
    # `all, delete-orphan` cascade to remove feedback when user is deleted
    user_feedback = relationship(
        "DBUserFeedback", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        """
        Returns a string representation of the Users object.

        Returns
        -------
        str
        """
        return f"{self.__class__.__name__}(id={self.id!r}, username={self.username!r}, email={self.email!r})"


class DBUserFeedback(Base):
    """Data model for storing user feedback."""

    __tablename__: str = "user_feedback"

    id: Mapped[int] = mapped_column("id", primary_key=True)
    username: Mapped[str] = mapped_column(String(50), nullable=False)
    session_id: Mapped[str] = mapped_column(String(50), nullable=False)
    message_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # It can vary based on user input
    user_message: Mapped[str] = mapped_column(String(), nullable=False)
    assistant_message: Mapped[str] = mapped_column(String(), nullable=False)
    sources: Mapped[str] = mapped_column(String(), nullable=True)
    # e.g., 'positive', 'negative', 'neutral'
    feedback: Mapped[str] = mapped_column(String(20), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )

    # =========================================================
    # ==================== Relationships ======================
    # =========================================================
    # Many-to-many relationship with users
    # Link to a single user (one-to-many): each feedback row belongs to ONLY one user (Requires a FK!)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    user = relationship("DBUser", back_populates="user_feedback")

    def __repr__(self) -> str:
        """
        Returns a string representation of the Users object.

        Returns
        -------
        str
        """
        return (
            f"{self.__class__.__name__}(id={self.id!r}, session_id={self.session_id!r}, "
            f"message_index={self.message_index!r}, "
            f"timestamp={self.timestamp!r})"
        )


class DBRole(Base):
    """Data model for storing user roles."""

    __tablename__: str = "roles"

    id: Mapped[int] = mapped_column("id", primary_key=True)
    name: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False
    )  # e.g., 'admin', 'user', 'guest'
    description: Mapped[str] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

    # =========================================================
    # ==================== Relationships ======================
    # =========================================================
    # Many-to-many relationship with users
    users = relationship("DBUser", secondary=user_roles, back_populates="roles")

    def __repr__(self) -> str:
        """
        Returns a string representation of the DBRole object.

        Returns
        -------
        str
        """
        return f"{self.__class__.__name__}(id={self.id!r}, name={self.name!r})"


class CeleryTaskCleanup(Base):
    """Read-only model for Celery task cleanup operations.

    This model reflects the actual Celery table structure with an 'id' column
    that Celery expects, separate from your application models.
    """

    __tablename__: str = "celery_taskmeta"
    __table_args__: dict[str, Any] = {"extend_existing": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(155), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    result: Mapped[bytes] = mapped_column(LargeBinary, nullable=True)
    date_done: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    traceback: Mapped[str] = mapped_column(Text, nullable=True)
    name: Mapped[str] = mapped_column(String(155), nullable=True)
    args: Mapped[str] = mapped_column(Text, nullable=True)
    kwargs: Mapped[str] = mapped_column(Text, nullable=True)
    worker: Mapped[str] = mapped_column(String(155), nullable=True)
    retries: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    queue: Mapped[str] = mapped_column(String(155), nullable=True)

    def __repr__(self) -> str:
        """
        Returns a string representation of the CeleryTaskCleanup object.

        Returns
        -------
        str
        """
        return (
            f"{self.__class__.__name__}(id={self.id!r}, task_id={self.task_id!r}, status={self.status!r}, "
            f"date_done={self.date_done!r})"
        )


# =========================================================
# ==================== Utilities ==========================
# =========================================================
# Global pool instance
_db_pool: DatabasePool | None = None


def get_db_pool() -> DatabasePool:
    """Get or create the global database pool."""
    global _db_pool
    if _db_pool is None:
        _db_pool = DatabasePool(app_settings.database_url_2)
    return _db_pool


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session context manager.

    Use this for manual session management with 'with' statements.

    Yields
    ------
    Session
        A database session

    Example
    -------
        with get_db_session() as session:
            # use session here
    """
    db_pool = get_db_pool()
    with db_pool.get_session() as session:
        yield session


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions.

    This is a generator function that FastAPI will handle automatically.
    Use this with Depends() in your route handlers.

    Yields
    ------
    Session
        A database session that will be automatically closed after the request
    """
    db_pool = get_db_pool()
    with db_pool.get_session() as session:
        try:
            yield session
        finally:
            # Session cleanup is handled by the context manager
            pass
