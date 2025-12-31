""" "API routes for document retrieval operations"""

from typing import Any

from aiocache import Cache
from fastapi import APIRouter, Depends, Query, Request, status

from src import create_logger
from src.api.core.cache import cached
from src.api.core.dependencies import get_cache
from src.api.core.exceptions import BaseAPIError, UnexpectedError
from src.api.core.ratelimit import limiter
from src.api.core.responses import MsgSpecJSONResponse
from src.config import app_config
from src.schemas.types import SectionNamesType
from src.utilities.tools.tools import (
    ahybrid_search_tool,
    akeyword_search_tool,
    arerank_documents,
    avector_search_tool,
)

logger = create_logger(name="retrieval")
LIMIT_VALUE: int = app_config.api_config.ratelimit.burst_rate
RERANK_K: int = app_config.custom_config.rerank_k
router = APIRouter(
    tags=["retrieval"],
    default_response_class=MsgSpecJSONResponse,
)


@router.get("/retrievals/keyword", status_code=status.HTTP_200_OK)
@cached(ttl=300, key_prefix="keyword")  # type: ignore
@limiter.limit(f"{LIMIT_VALUE}/minute")
async def keyword_search(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    query: str = Query(description="The search query string"),
    k: int = Query(8, description="Number of top documents to retrieve"),
    # Required by caching decorator
    cache: Cache = Depends(get_cache),  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Route for keyword-based document retrieval"""
    try:
        documents = await akeyword_search_tool(query, filter=None, k=k)
        documents = await arerank_documents(query, documents, k=RERANK_K)
        if not documents:
            BaseAPIError(
                message="Keyword search failed",
            )

        return [doc.model_dump() for doc in documents]

    except BaseAPIError as e:
        raise e
    except Exception:
        raise UnexpectedError(
            details="Unexpected error occurred while performing keyword search."
        ) from None


@router.get("/retrievals/semantic", status_code=status.HTTP_200_OK)
@cached(ttl=300, key_prefix="semantic")  # type: ignore
@limiter.limit(f"{LIMIT_VALUE}/minute")
async def semantic_search(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    query: str = Query(description="The search query string"),
    filter: SectionNamesType | None = Query(
        None, description="Optional metadata filter for 'metadata.section'"
    ),
    k: int = Query(8, description="Number of top documents to retrieve"),
    # Required by caching decorator
    cache: Cache = Depends(get_cache),  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Route for semantic (vector) document retrieval"""
    try:
        documents = await avector_search_tool(query, filter=filter, k=k)
        documents = await arerank_documents(query, documents, k=RERANK_K)

        if not documents:
            BaseAPIError(
                message="Vector search failed",
            )

        return [doc.model_dump() for doc in documents]

    except BaseAPIError as e:
        raise e
    except Exception:
        raise UnexpectedError(
            details="Unexpected error occurred while performing semantic search."
        ) from None


@router.get("/retrievals/hybrid", status_code=status.HTTP_200_OK)
@cached(ttl=300, key_prefix="hybrid")  # type: ignore
@limiter.limit(f"{LIMIT_VALUE}/minute")
async def hybrid_search(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    query: str = Query(description="The search query string"),
    filter: SectionNamesType | None = Query(
        None, description="Optional metadata filter for 'metadata.section'"
    ),
    k: int = Query(8, description="Number of top documents to retrieve"),
    # Required by caching decorator
    cache: Cache = Depends(get_cache),  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Route for hybrid (vector + keywords) document retrieval"""
    try:
        documents = await ahybrid_search_tool(query, filter=filter, k=k)
        documents = await arerank_documents(query, documents, k=RERANK_K)

        if not documents:
            BaseAPIError(
                message="Vector search failed",
            )

        return [doc.model_dump() for doc in documents]

    except BaseAPIError as e:
        raise e
    except Exception:
        raise UnexpectedError(
            details="Unexpected error occurred while performing hybrid search."
        ) from None
