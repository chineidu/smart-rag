from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import Field

from src import PACKAGE_PATH
from src.schemas import BaseSchema
from src.schemas.types import FileFormatsType


class VectorStoreConfig(BaseSchema):
    """Configuration for vector store."""

    filepaths: str = Field(description="Path to the football news data directory.")
    jq_schema: str | None = Field(
        default=None, description="JQ schema to apply when loading documents"
    )
    format: FileFormatsType | str = Field(
        description="Format of the files to be loaded."
    )
    collection: str = Field(
        description="Name of the Qdrant collection to use or create"
    )
    filepaths_is_glob: bool = Field(
        default=False,
        description="If True, treat `filepaths` as a glob pattern to match multiple files, by default False",
    )
    force_recreate: bool = Field(
        default=False,
        description="If True, delete and recreate existing collection to match embedding dimension.",
    )


class CustomConfig(BaseSchema):
    """Configuration for other fields."""

    topics: list[str] = Field(description="List of acceptable topics for retrieval.")
    fetch_full_page: bool = Field(
        description="If True, fetch and parse full HTML content for each result."
    )
    k: int = Field(
        5,
        description="Number of top documents to retrieve per query before deduplication.",
    )
    max_chars: int | None = Field(
        description="Maximum characters to return per result. If None, no truncation."
    )
    max_attempts: int = Field(
        3, description="The maximum number of times to retry an operation."
    )


class CreativeModelConfig(BaseSchema):
    """Configuration for creative model."""

    model_name: str = Field(..., description="The name of the creative LLM to use.")
    temperature: float = Field(
        0.7, description="The temperature setting for the creative LLM."
    )
    seed: int = Field(42, description="The random seed for the creative LLM.")


class StructuredOutputModelConfig(BaseSchema):
    """Configuration for structured output model."""

    model_name: str = Field(
        ..., description="The name of the structured output LLM to use."
    )
    temperature: float = Field(
        0.0, description="The temperature setting for the structured output LLM."
    )
    seed: int = Field(42, description="The random seed for the structured output LLM.")


class EmbeddingModelConfig(BaseSchema):
    """Configuration for embedding model."""

    model_name: str = Field(..., description="The name of the embedding model to use.")


class CrossEncoderConfig(BaseSchema):
    """Configuration for cross encoder model."""

    model_name: str = Field(
        ..., description="The name of the cross encoder model to use."
    )
    num_labels: int | None = Field(
        default=None, description="Number of labels for the cross encoder model."
    )
    max_length: int | None = Field(
        default=None, description="Maximum sequence length for the cross encoder model."
    )
    device: str | None = Field(
        default=None, description="Device to load the cross encoder model onto."
    )


class CORS(BaseSchema):
    """CORS configuration class."""

    allow_origins: list[str]
    allow_credentials: bool
    allow_methods: list[str]
    allow_headers: list[str]


class Middleware(BaseSchema):
    """Middleware configuration class."""

    cors: CORS


class LLMModelConfig(BaseSchema):
    """Configuration for models."""

    creative_model: CreativeModelConfig = Field(
        description="Creative model configuration."
    )
    structured_output_model: StructuredOutputModelConfig = Field(
        description="Structured output model configuration."
    )
    embedding_model: EmbeddingModelConfig = Field(
        description="Embedding model configuration."
    )
    cross_encoder_model: CrossEncoderConfig = Field(
        description="Cross Encoder model configuration"
    )


class APIConfig(BaseSchema):
    """API-level configuration."""

    title: str = Field(..., description="The title of the API.")
    name: str = Field(..., description="The name of the API.")
    description: str = Field(..., description="The description of the API.")
    version: str = Field(..., description="The version of the API.")
    status: str = Field(..., description="The current status of the API.")
    prefix: str = Field(..., description="The prefix for the API routes.")
    middleware: Middleware = Field(description="Middleware configuration.")


class AppConfig(BaseSchema):
    """Application-level configuration."""

    vectorstore_config: VectorStoreConfig = Field(
        description="Vector store configurations."
    )
    custom_config: CustomConfig = Field(description="Custom configurations")
    llm_model_config: LLMModelConfig = Field(description="LLM model configurations.")
    api_config: APIConfig = Field(description="API configurations.")


config_path: Path = PACKAGE_PATH / "src/config/config.yaml"
config: DictConfig = OmegaConf.load(config_path).config
# # Resolve all the variables
resolved_cfg = OmegaConf.to_container(config, resolve=True)
# Validate the config
app_config: AppConfig = AppConfig(**dict(resolved_cfg))  # type: ignore
