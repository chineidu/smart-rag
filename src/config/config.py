from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import Field

from src import PACKAGE_PATH
from src.schemas import BaseSchema
from src.schemas.types import BrokerOrBackendType, FileFormatsType


@dataclass(slots=True, kw_only=True)
class VectorStoreConfig(BaseSchema):
    """Configuration for vector store."""

    filepaths: str = field(
        metadata={"description": "Path to the football news data directory."}
    )
    jq_schema: str | None = field(
        default=None,
        metadata={"description": "JQ schema to apply when loading documents"},
    )
    format: FileFormatsType | str = field(
        metadata={"description": "Format of the files to be loaded."}
    )
    collection: str = field(
        metadata={"description": "Name of the Qdrant collection to use or create"}
    )
    filepaths_is_glob: bool = field(
        default=False,
        metadata={
            "description": "If True, treat `filepaths` as a glob pattern to match multiple files, "
            "by default False"
        },
    )
    force_recreate: bool = field(
        default=False,
        metadata={
            "description": "If True, delete and recreate existing collection to match embedding dimension."
        },
    )


@dataclass(slots=True, kw_only=True)
class CustomConfig:
    """Configuration for other fields."""

    topics: list[str] = field(
        default_factory=list,
        metadata={"description": "List of acceptable topics for retrieval."},
    )
    fetch_full_page: bool = field(
        metadata={
            "description": "If True, fetch and parse full HTML content for each result."
        }
    )
    k: int = field(
        default=5,
        metadata={
            "description": "Number of top documents to retrieve per query before deduplication."
        },
    )
    rerank_k: int = field(
        default=3,
        metadata={
            "description": "Number of top documents to rerank after initial retrieval."
        },
    )
    max_chars: int | None = field(
        default=None,
        metadata={
            "description": "Maximum characters to return per result. If None, no truncation."
        },
    )
    max_attempts: int = field(
        default=3,
        metadata={"description": "The maximum number of times to retry an operation."},
    )
    max_messages: int = field(
        default=20,
        metadata={
            "description": "Maximum number of messages to keep in conversation history."
        },
    )
    recursion_limit: int = field(
        default=50,
        metadata={"description": "Maximum recursion depth for multi-step plans."},
    )


@dataclass(slots=True, kw_only=True)
class CreativeModelConfig(BaseSchema):
    """Configuration for creative model."""

    model_name: str = field(
        metadata={"description": "The name of the creative LLM to use."}
    )
    temperature: float = field(
        default=0.7,
        metadata={"description": "The temperature setting for the creative LLM."},
    )
    seed: int = field(
        default=42, metadata={"description": "The random seed for the creative LLM."}
    )


@dataclass(slots=True, kw_only=True)
class StructuredOutputModelConfig(BaseSchema):
    """Configuration for structured output model."""

    model_name: str = field(
        metadata={"description": "The name of the structured output LLM to use."}
    )
    temperature: float = field(
        default=0.0,
        metadata={
            "description": "The temperature setting for the structured output LLM."
        },
    )
    seed: int = field(
        default=42,
        metadata={"description": "The random seed for the structured output LLM."},
    )


@dataclass(slots=True, kw_only=True)
class EmbeddingModelConfig(BaseSchema):
    """Configuration for embedding model."""

    model_name: str = field(
        metadata={"description": "The name of the embedding model to use."}
    )


@dataclass(slots=True, kw_only=True)
class CrossEncoderConfig(BaseSchema):
    """Configuration for cross encoder model."""

    model_name: str = field(
        metadata={"description": "The name of the cross encoder model to use."}
    )
    num_labels: int | None = field(
        default=None,
        metadata={"description": "Number of labels for the cross encoder model."},
    )
    max_length: int | None = field(
        default=None,
        metadata={
            "description": "Maximum sequence length for the cross encoder model."
        },
    )
    device: str | None = field(
        default=None,
        metadata={"description": "Device to load the cross encoder model onto."},
    )


@dataclass(slots=True, kw_only=True)
class CeleryConfig(BaseSchema):
    """Configuration for Celery."""

    celery_broker: BrokerOrBackendType = field(
        metadata={"description": "The broker URL for Celery."}
    )
    celery_backend: BrokerOrBackendType = field(
        metadata={"description": "The backend URL for Celery."}
    )
    celery_task_track_started: bool = field(
        metadata={"description": "If True, track task start events."}
    )
    celery_task_time_limit: int = field(
        metadata={"description": "Hard time limit for tasks in seconds."}
    )
    celery_task_soft_time_limit: int = field(
        metadata={"description": "Soft time limit for tasks in seconds."}
    )


@dataclass(slots=True, kw_only=True)
class StreamConfig(BaseSchema):
    """Configuration for streams."""

    stream_prefix: str = field(
        metadata={"description": "Prefix for the Redis stream keys."}
    )
    stream_ttl: int = field(
        metadata={"description": "Time-to-live for the streams in seconds."}
    )
    max_stream_length: int = field(
        metadata={"description": "Maximum length of the streams."}
    )


@dataclass(slots=True, kw_only=True)
class SSEConfig(BaseSchema):
    """Configuration for Server-Sent Events (SSE)."""

    sse_retry_timeout: int = field(
        metadata={"description": "Retry timeout for SSE connections in seconds."}
    )
    sse_keepalive_interval: int = field(
        metadata={"description": "Interval for sending keepalive messages in seconds."}
    )


@dataclass(slots=True, kw_only=True)
class SessionConfig(BaseSchema):
    """Configuration for session management."""

    session_cleanup_interval: int = field(
        metadata={"description": "Interval for cleaning up idle sessions in seconds."}
    )
    session_idle_timeout: int = field(
        metadata={"description": "Idle timeout for sessions in seconds."}
    )


@dataclass(slots=True, kw_only=True)
class CORS(BaseSchema):
    """CORS configuration class."""

    allow_origins: list[str] = field(
        default_factory=list, metadata={"description": "Allowed origins for CORS."}
    )
    allow_credentials: bool = field(
        metadata={"description": "Allow credentials for CORS."}
    )
    allow_methods: list[str] = field(
        default_factory=list, metadata={"description": "Allowed methods for CORS."}
    )
    allow_headers: list[str] = field(
        default_factory=list, metadata={"description": "Allowed headers for CORS."}
    )


@dataclass(slots=True, kw_only=True)
class Middleware(BaseSchema):
    """Middleware configuration class."""

    cors: CORS = field(metadata={"description": "CORS configuration."})


@dataclass(slots=True, kw_only=True)
class LLMModelConfig(BaseSchema):
    """Configuration for models."""

    creative_model: CreativeModelConfig = field(
        metadata={"description": "Creative model configuration."}
    )
    structured_output_model: StructuredOutputModelConfig = field(
        metadata={"description": "Structured output model configuration."}
    )
    embedding_model: EmbeddingModelConfig = field(
        metadata={"description": "Embedding model configuration."}
    )
    cross_encoder_model: CrossEncoderConfig = field(
        metadata={"description": "Cross Encoder model configuration"}
    )


@dataclass(slots=True, kw_only=True)
class APIConfig(BaseSchema):
    """API-level configuration."""

    title: str = field(metadata={"description": "The title of the API."})
    name: str = field(metadata={"description": "The name of the API."})
    description: str = field(metadata={"description": "The description of the API."})
    version: str = field(metadata={"description": "The version of the API."})
    status: str = field(metadata={"description": "The current status of the API."})
    prefix: str = field(metadata={"description": "The prefix for the API routes."})
    middleware: Middleware = field(
        metadata={"description": "Middleware configuration."}
    )


class AppConfig(BaseSchema):
    """Application-level configuration (Pydantic Model)."""

    vectorstore_config: VectorStoreConfig = Field(
        description="Vector store configurations."
    )
    custom_config: CustomConfig = Field(description="Custom configurations")
    llm_model_config: LLMModelConfig = Field(description="LLM model configurations.")
    celery_config: CeleryConfig = Field(description="Celery configurations.")
    stream_config: StreamConfig = Field(description="Stream configurations.")
    sse_config: SSEConfig = Field(description="SSE configurations.")
    session_config: SessionConfig = Field(
        description="Session management configurations."
    )
    api_config: APIConfig = Field(description="API configurations.")


config_path: Path = PACKAGE_PATH / "src/config/config.yaml"
config: DictConfig = OmegaConf.load(config_path).config
# # Resolve all the variables
resolved_cfg = OmegaConf.to_container(config, resolve=True)
# Validate the config
app_config: AppConfig = AppConfig(**dict(resolved_cfg))  # type: ignore
