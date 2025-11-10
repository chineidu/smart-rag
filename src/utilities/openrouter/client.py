from typing import Any

import httpx

from src.utilities.model_config import RemoteModel
from src.utilities.openrouter.exceptions import (
    OpenRouterError,
)
from src.utilities.openrouter.types import OpenRouterClientPaths, RequestMethods
from src.utilities.openrouter.utils import _validate_response

CONTENT_TYPE: str = "application/json"
USER_AGENT: str = "openrouter_sdk/0.0.1"


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = OpenRouterClientPaths.BASE_URL.value,
        default_model: str = RemoteModel.GEMINI_2_0_FLASH_001.value,
        timeout: int = 20,
    ) -> None:
        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": CONTENT_TYPE,
                "user-agent": USER_AGENT,
            },
        )

        # Import and initialize resources here to avoid circular imports
        from src.utilities.openrouter.resources.chat import (
            ChatResource,
            CompletionsResource,
        )
        from src.utilities.openrouter.resources.embeddings import EmbeddingsResource
        from src.utilities.openrouter.resources.models import ModelsResource

        self.chat = ChatResource(client=self)
        self._completions = CompletionsResource(client=self)
        self.embeddings = EmbeddingsResource(client=self)
        self.models = ModelsResource(client=self)

    def close(self) -> None:
        """Close the HTTP client session."""
        self._client.close()

    # Context manager
    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _request(self, method: RequestMethods, path: str, **kwargs: Any) -> Any:
        """Internal method to make HTTP requests."""
        response = self._client.request(method, path, **kwargs)
        return _validate_response(self, response)

    def completions(
        self,
        prompt: str | list[str],
        model: str | None = None,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Get completions from OpenRouter API."""
        return self._completions.completions(prompt=prompt, model=model, **kwargs)


class AsyncOpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = OpenRouterClientPaths.BASE_URL.value,
        default_model: str = RemoteModel.GEMINI_2_0_FLASH_001.value,
        timeout: int = 20,
    ) -> None:
        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": CONTENT_TYPE,
                "user-agent": USER_AGENT,
            },
        )

        # Import and initialize resources here to avoid circular imports
        from src.utilities.openrouter.resources.chat import (
            ChatResource,
            CompletionsResource,
        )
        from src.utilities.openrouter.resources.embeddings import EmbeddingsResource
        from src.utilities.openrouter.resources.models import ModelsResource

        self.chat = ChatResource(client=self)
        self._completions = CompletionsResource(client=self)
        self.embeddings = EmbeddingsResource(client=self)
        self.models = ModelsResource(client=self)

    async def aclose(self) -> None:
        """Close the HTTP client session."""
        await self._client.aclose()

    # Context manager
    async def __aenter__(self) -> "AsyncOpenRouterClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await self.aclose()

    async def _arequest(self, method: RequestMethods, path: str, **kwargs: Any) -> Any:
        """Internal method to make asynchronous HTTP requests."""
        async with self._client as client:
            response = await client.request(method, path, **kwargs)

        return _validate_response(self, response)


from src.config import app_settings  # noqa: E402

# client = OpenRouterClient(api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(), timeout=10)


async def main() -> None:
    """Main async function for testing."""
    client = AsyncOpenRouterClient(
        api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(), timeout=10
    )

    try:
        # res = client.chat.completions(
        #     messages=[{"role": "user", "content": "Hello, how are you?"}]
        # )
        res = await client.embeddings.acreate(
            input="Hello, how are you?", model="openai/text-embedding-3-small"
        )
        print(res)
    except OpenRouterError as e:
        print(f"An error occurred: {e}")
    finally:
        await client.aclose()


if __name__ == "__main__":
    # Example usage

    import asyncio

    asyncio.run(main())
