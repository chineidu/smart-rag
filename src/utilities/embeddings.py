import os
from typing import Any

import together
from langchain_core.embeddings import Embeddings
from langchain_core.utils import convert_to_secret_str
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)

from src.config.settings import setup_env_once
from src.utilities.openrouter.client import AsyncOpenRouterClient, OpenRouterClient

# Ensure .env values are loaded into os.environ before any client construction
setup_env_once()


def set_together_api(value: str | None = None) -> SecretStr:
    """Set the Together API key"""
    if value is None:
        return convert_to_secret_str(os.getenv("TOGETHER_API_KEY", ""))
    return convert_to_secret_str(value)


def set_openrouter_api(value: str | None = None) -> SecretStr:
    """Set the OpenRouter API key"""
    if value is None:
        return convert_to_secret_str(os.getenv("OPENROUTER_API_KEY", ""))
    return convert_to_secret_str(value)


class TogetherEmbeddings(BaseModel, Embeddings):
    """Together embeddings helper that wires the client once the key is present."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: together.Together | None = Field(default=None)
    together_api_key: SecretStr = Field(default_factory=set_together_api)
    model: str = Field(default="togethercomputer/m2-bert-80M-32k-retrieval")

    @model_validator(mode="after")
    def validate_environment(self) -> "TogetherEmbeddings":
        """Validate the environment and set up the Together client."""
        _api_key: SecretStr | str = self.together_api_key or os.getenv(
            "TOGETHER_API_KEY", ""
        )
        if not _api_key:
            raise ValueError(
                "Together API key not found. Please set the TOGETHER_API_KEY environment variable."
            )

        if isinstance(_api_key, str):
            _api_key = convert_to_secret_str(_api_key)

        # Lazily create the Together client to avoid constructing it without credentials
        self.client = together.Together(
            api_key=_api_key.get_secret_value(),  # type: ignore
        )
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        if self.client is None:
            msg = "Together client is not initialized. Did you call TogetherEmbeddings() with a valid API key?"
            raise RuntimeError(msg)

        return [
            i.embedding
            for i in self.client.embeddings.create(input=texts, model=self.model).data  # type: ignore
        ]  # type: ignore

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]


class OpenRouterEmbeddings(BaseModel, Embeddings):
    """Using Field with default_factory for automatic client creation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: OpenRouterClient = Field(default_factory=OpenRouterClient)
    aclient: AsyncOpenRouterClient = Field(default_factory=AsyncOpenRouterClient)

    openrouter_api_key: SecretStr = Field(default_factory=set_openrouter_api)
    model: str = Field(default="openai/text-embedding-3-small")

    @model_validator(mode="after")
    def validate_environment(self) -> "OpenRouterEmbeddings":
        """Validate the environment and set up the OpenRouter client."""
        _api_key: SecretStr | str = self.openrouter_api_key or os.getenv(
            "OPENROUTER_API_KEY", ""
        )
        if not _api_key:
            raise ValueError(
                "OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable."
            )

        if isinstance(_api_key, str):
            _api_key = convert_to_secret_str(_api_key)

        # Set up the OpenRouter client if not already set
        self.client = OpenRouterClient(
            api_key=_api_key.get_secret_value(),  # type: ignore
            default_model=self.model,
        )
        self.aclient = AsyncOpenRouterClient(
            api_key=_api_key.get_secret_value(),  # type: ignore
            default_model=self.model,
        )
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        response: dict[str, Any] = self.client.embeddings.create(
            input=texts, model=self.model
        )

        # Check for errors in the response
        if "error" in response:
            error_msg = response["error"].get("message", "Unknown error")
            error_code = response["error"].get("code", "Unknown code")
            raise ValueError(
                f"OpenRouter embeddings API error ({error_code}): {error_msg}. "
                f"Model: {self.model}. Please check if the model is available and properly configured."
            )

        if "data" not in response:
            raise KeyError(
                f"'data' key not found in OpenRouter response. Available keys: {list(response.keys())}. Response: {response}"
            )

        return [emb["embedding"] for emb in response["data"]]

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        response: dict[str, Any] = await self.aclient.aembeddings.create(
            input=texts, model=self.model
        )

        # Check for errors in the response
        if "error" in response:
            error_msg = response["error"].get("message", "Unknown error")
            error_code = response["error"].get("code", "Unknown code")
            raise ValueError(
                f"OpenRouter embeddings API error ({error_code}): {error_msg}. "
                f"Model: {self.model}. Please check if the model is available and properly configured."
            )

        if "data" not in response:
            raise KeyError(
                f"'data' key not found in OpenRouter response. Available keys: {list(response.keys())}. Response: {response}"
            )

        return [emb["embedding"] for emb in response["data"]]

    async def aembed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return (await self.aembed_documents([text]))[0]
