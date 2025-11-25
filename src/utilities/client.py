import json
from typing import Any

import httpx
import instructor
from openai import AsyncOpenAI

from src.config import app_settings


class HTTPXClient:
    def __init__(
        self,
        base_url: str = "",
        timeout: int = 30,
        http2: bool = True,
        max_connections: int = 20,
        max_keepalive_connections: int = 5,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.http2 = http2
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            http2=self.http2,
            limits=httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
            ),
        )

    async def __aenter__(self) -> "HTTPXClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.client.aclose()

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform an asynchronous GET request."""
        try:
            response = await self.client.get(url, params=params, headers=headers)
            return self._parse_response(response)
        except Exception as e:
            return self._handle_exception(e)

    async def post(
        self,
        url: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform an asynchronous POST request."""
        try:
            response = await self.client.post(
                url, data=data, params=params, headers=headers
            )
            return self._parse_response(response)
        except Exception as e:
            return self._handle_exception(e)

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        """Parse the HTTPX response and return a standardized dictionary."""
        try:
            data = response.json()
        except json.JSONDecodeError:
            data = response.text

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "data": data,
            "headers": dict(response.headers),
            "error": (
                None
                if response.status_code < 400
                else f"HTTP {response.status_code} Error"
            ),
        }

    def _handle_exception(self, e: Exception) -> dict[str, Any]:
        """Handle exceptions and return a standardized error response."""
        if isinstance(e, httpx.ConnectError):
            error_msg = f"Connection Error: {str(e)}"
        elif isinstance(e, httpx.TimeoutException):
            error_msg = f"Request Timeout: {str(e)}"
        else:
            error_msg = f"Unexpected Error: {str(e)}"

        return {
            "success": False,
            "status_code": None,
            "data": None,
            "headers": None,
            "error": error_msg,
        }


async def get_instructor_openrouter_client() -> instructor.AsyncClient:  # type: ignore
    """Create an Instructor AsyncOpenRouter client."""
    _async_client = AsyncOpenAI(
        api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(),
        base_url=app_settings.OPENROUTER_URL,
    )

    return instructor.from_openai(
        _async_client, mode=instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS
    )
