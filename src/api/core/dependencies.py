from typing import TYPE_CHECKING

from aiocache import Cache
from fastapi import Request

from src.api.core.exceptions import ResourcesNotFoundError
from src.schemas.types import ResourcesType

if TYPE_CHECKING:
    from src.graph import GraphManager


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
