import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from aiocache import Cache
from fastapi import Request

from src.api.core.exceptions import ResourcesNotFoundError
from src.graph import GraphManager
from src.schemas.types import ResourcesType

if TYPE_CHECKING:
    from src.stream_manager import StreamSessionManager


_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def get_graph_manager(request: Request) -> GraphManager:
    """Get the GraphManager from the app state."""
    if (
        not hasattr(request.app.state, "graph_manager")
        or request.app.state.graph_manager is None
    ):
        raise ResourcesNotFoundError(resource_type=ResourcesType.GRAPH)
    return request.app.state.graph_manager


async def get_cache(request: Request) -> Cache:
    """Dependency to inject cache into endpoints."""
    if not hasattr(request.app.state, "cache") or request.app.state.cache is None:
        raise ResourcesNotFoundError(resource_type=ResourcesType.CACHE)
    return request.app.state.cache


async def get_stream_session_manager(request: Request) -> "StreamSessionManager":
    """Dependency to inject StreamSessionManager into endpoints."""
    if (
        not hasattr(request.app.state, "stream_session_manager")
        or request.app.state.stream_session_manager is None
    ):
        raise ResourcesNotFoundError(resource_type=ResourcesType.STREAM_SESSION)
    return request.app.state.stream_session_manager


def get_executor(max_workers: int | None = None) -> ThreadPoolExecutor:
    """Return a shared global ThreadPoolExecutor.

    Sharing a single executor across all instances ensures:
    1. Efficient use of system resources (avoids creating a new executor per request).
    2. Proper timeout and cancellation handling with asyncio.
    3. Thread reuse for better performance under concurrent predictions.
    """
    global _executor
    global _executor_lock

    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="global_threadpool_",
                )

    #  _executor is guaranteed to be initialized
    assert _executor is not None
    return _executor
