import asyncio
from typing import TYPE_CHECKING

from celery import Task

from src import create_logger
from src.graph import GraphManager

# from src.celery_app.app import celery_app
from src.stream_manager import StreamSessionManager

if TYPE_CHECKING:
    pass

logger = create_logger(name="celery_app")

__all__ = ["BaseTask", "celery_app"]
# Import signals to register them (this ensures signal handlers are connected)
from src.celery_app import signals  # noqa: E402, F401


class BaseTask(Task):
    """
    A custom base task class for Celery tasks with automatic retry configuration.

    Attributes
    ----------
    autoretry_for : tuple
        A tuple of exception types for which the task should automatically retry.
    throws : tuple
        A tuple of exception types for which full traceback should be logged on retry.
    default_retry_delay : int
        The default delay between retries in seconds.
    max_retries : int
        The maximum number of retries allowed for the task.
    retry_backoff : bool
        Enables exponential backoff for retry delays.
    retry_backoff_max : int
        The maximum delay in seconds for exponential backoff.
    """

    autoretry_for = (Exception,)
    throws = (Exception,)  # Log full traceback on retry
    default_retry_delay = 30  # 30 seconds
    max_retries = 5


class CallbackTask(BaseTask):
    """A Celery task with callback capabilities for streaming responses."""

    # Class-level singletons (per worker process)
    _stream_session_manager: "StreamSessionManager | None" = None
    _graph_builder: "GraphManager | None" = None

    def on_failure(self, exc, task_id, args, kwargs, einfo) -> None:  # noqa: ANN001, ARG002
        """Log task failure details."""
        session_id: str = kwargs.get("session_id", "unknown")
        logger.error(
            f"Task {self.name} [{task_id}] failed for session {session_id}: {exc}"
        )

        if session_id:
            asyncio.run(
                self.asend_error_event(
                    session_id=session_id,
                    error_message=str(exc),
                )
            )

    def on_retry(self, exc, task_id, args, kwargs, einfo) -> None:  # noqa: ANN001, ARG002
        """Log task retry details."""
        session_id: str = kwargs.get("session_id", "unknown")
        logger.warning(
            f"Task {self.name} [{task_id}] retrying for session {session_id}: {exc}"
        )

    def on_success(self, retval, task_id, args, kwargs) -> None:  # noqa: ANN001, ARG002
        """Log task success details."""
        session_id: str = kwargs.get("session_id", "unknown")
        logger.info(f"Task {self.name} [{task_id}] succeeded for session {session_id}")

    async def asend_error_event(self, session_id: str, error_message: str) -> None:
        """Send an error event to the client via the StreamSessionManager."""
        manager = await self.aget_stream_session_manager()
        await manager.asend_event(
            session_id=session_id,
            event_type="error",
            data={"error": error_message},
        )

    @classmethod
    async def aget_stream_session_manager(cls) -> StreamSessionManager:
        """Get or create the StreamSessionManager instance (per-process singleton).

        Uses an async lock to avoid race conditions during concurrent creation.
        """
        if cls._stream_session_manager is None:
            cls._stream_session_manager = StreamSessionManager()
        return cls._stream_session_manager

    @classmethod
    async def aget_graph_builder(cls) -> StreamSessionManager:
        """Get or create the GraphManager instance (per-process singleton)."""
        if cls._graph_builder is None:
            cls._graph_builder = GraphManager()
        return cls._graph_builder

    @classmethod
    async def aclose_stream_session_manager(cls) -> None:
        """Close and clear the shared StreamSessionManager instance if present."""
        if cls._stream_session_manager is not None:
            try:
                await cls._stream_session_manager.aclose()
            finally:
                cls._stream_session_manager = None

    @classmethod
    async def aclose_graph_builder(cls) -> None:
        """Close and clear the shared GraphManager instance if present."""
        if cls._graph_builder is not None:
            try:
                await cls._graph_builder.aclose()
            finally:
                cls._graph_builder = None
