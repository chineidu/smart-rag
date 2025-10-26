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


def set_together_api(value: str | None = None) -> SecretStr:
    """Set the Together API key"""
    if value is None:
        return convert_to_secret_str(os.getenv("TOGETHER_API_KEY", ""))
    return convert_to_secret_str(value)


class TogetherEmbeddings(BaseModel, Embeddings):
    """Using Field with default_factory for automatic client creation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: together.Together = Field(default_factory=together.Together)
    together_api_key: SecretStr = Field(default_factory=lambda: set_together_api)  # type: ignore
    model: str = Field(default="togethercomputer/m2-bert-80M-32k-retrieval")

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Set up the Together API key and client before model instantiation."""
        # Handle API key setup
        api_key = values.get("together_api_key") or os.getenv("TOGETHER_API_KEY", "")
        if isinstance(api_key, str):
            api_key = set_together_api(api_key)
        values["together_api_key"] = api_key
        values["client"] = together.Together()

        # Set global API key
        together.api_key = api_key.get_secret_value()  # type: ignore

        return values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        return [
            i.embedding
            for i in self.client.embeddings.create(input=texts, model=self.model).data  # type: ignore
        ]  # type: ignore

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
