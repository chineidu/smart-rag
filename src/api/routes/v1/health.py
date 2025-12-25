from aiocache import Cache
from fastapi import APIRouter, Depends, Request, status

from src import create_logger
from src.api.core.cache import cached
from src.api.core.dependencies import get_cache
from src.api.core.exceptions import BaseAPIError, UnexpectedError
from src.api.core.ratelimit import limiter
from src.api.core.reponses import MsgSpecJSONResponse
from src.config import app_config
from src.schemas.routes.health import HealthStatus

logger = create_logger(name="health")
LIMIT_VALUE: int = app_config.api_config.ratelimit.default_rate
router = APIRouter(tags=["health"], default_response_class=MsgSpecJSONResponse)


@router.get("/health", status_code=status.HTTP_200_OK)
@cached(ttl=300, key_prefix="health")  # type: ignore
@limiter.limit(f"{LIMIT_VALUE}/minute")
async def health_check(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    cache: Cache = Depends(get_cache),  # Required by caching decorator  # noqa: ARG001
) -> HealthStatus:
    """Route for health checks"""
    try:
        response = HealthStatus(
            name=app_config.api_config.title,
            status=app_config.api_config.status,
            version=app_config.api_config.version,
        )

        if not response:
            BaseAPIError(
                message="Health check failed",
            )

        return response

    except BaseAPIError as e:
        logger.error(f"Health check error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during health check: {e}")
        raise UnexpectedError(
            details="Unexpected error occurred while performing hybrid search."
        ) from e
