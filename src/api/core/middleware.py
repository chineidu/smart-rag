"""Custom middleware for request ID assignment, logging, and error handling."""

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src import create_logger
from src.api.core.exceptions import (
    BaseAPIError,
    InvalidInputError,
    PredictionError,
    ResourcesNotFoundError,
)
from src.schemas.types import ErrorCodeEnum

logger = create_logger(name="middleware")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add a unique request ID to each incoming request."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Add a unique request ID to the request and response headers."""
        request_id = str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log incoming requests and outgoing responses."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Log request and response details."""
        start_time: float = time.perf_counter()

        request_id = getattr(request.state, "request_id", "N/A")
        response: Response = await call_next(request)
        process_time: float = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        log: dict[str, Any] = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": round(process_time, 4),
            "request_id": request_id,
        }

        logger.info(f"{json.dumps(log)}")

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware to handle exceptions and return standardized error responses."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Catch exceptions and return standardized error responses."""
        try:
            response: Response = await call_next(request)
            return response

        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "status": "error",
                    "error": {"message": exc.detail, "code": ErrorCodeEnum.HTTP_ERROR},
                    "request_id": getattr(request.state, "request_id", "N/A"),
                    "path": str(request.url.path),
                },
            )

        except InvalidInputError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "status": "error",
                    "error": {"message": exc.message, "code": exc.error_code},
                    "request_id": getattr(request.state, "request_id", "N/A"),
                    "path": str(request.url.path),
                },
            )

        except ResourcesNotFoundError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "status": "error",
                    "error": {"message": exc.message, "code": exc.error_code},
                    "request_id": getattr(request.state, "request_id", "N/A"),
                    "path": str(request.url.path),
                },
            )

        except PredictionError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "status": "error",
                    "error": {"message": exc.message, "code": exc.error_code},
                    "request_id": getattr(request.state, "request_id", "N/A"),
                    "path": str(request.url.path),
                },
            )
        except BaseAPIError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "status": "error",
                    "error": {
                        "message": "An unexpected error occurred.",
                        "code": exc.error_code,
                    },
                    "request_id": getattr(request.state, "request_id", "N/A"),
                    "path": str(request.url.path),
                },
            )
