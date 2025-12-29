from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import Field

from src import PACKAGE_PATH
from src.schemas import BaseSchema
from src.schemas.types import BrokerOrBackendType, FileFormatsType


@dataclass(slots=True, kw_only=True)
class VectorStoreConfig:
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
    split_by_sections: bool = field(
        default=False,
        metadata={
            "description": "If True, split documents based on extracted sections; otherwise, "
            "use character-based splitting."
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
class CreativeModelConfig:
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
class StructuredOutputModelConfig:
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
class EmbeddingModelConfig:
    """Configuration for embedding model."""

    model_name: str = field(
        metadata={"description": "The name of the embedding model to use."}
    )


@dataclass(slots=True, kw_only=True)
class CrossEncoderConfig:
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
class QueueName:
    queue: str


@dataclass(slots=True, kw_only=True)
class QueuesConfig:
    """Configuration for the queues."""

    high_priority_ml: str = field(
        metadata={"description": "Queue for high priority ML tasks"}
    )
    normal_priority_ml: str = field(
        metadata={"description": "Queue for normal priority ML tasks"}
    )
    low_priority_ml: str = field(
        metadata={"description": "Queue for low priority ML tasks"}
    )
    cleanups: str = field(metadata={"description": "Queue for cleanup tasks"})
    notifications: str = field(metadata={"description": "Queue for user notifications"})
    success_queue: str = field(
        metadata={"description": "Queue for publishing successful NER tasks"}
    )
    failed_queue: str = field(
        metadata={"description": "Queue for publishing failed NER tasks"}
    )


@dataclass(slots=True, kw_only=True)
class TaskConfig:
    """Configuration for Celery tasks."""

    task_serializer: str = field(
        metadata={"description": "The serializer to use for Celery tasks."}
    )
    result_serializer: str = field(
        metadata={"description": "The serializer to use for Celery task results."}
    )
    accept_content: list[str] = field(
        default_factory=list,
        metadata={"description": "List of accepted content types for Celery tasks."},
    )
    timezone: str = field(metadata={"description": "The timezone for Celery tasks."})
    enable_utc: bool = field(
        metadata={"description": "If True, enable UTC for Celery tasks."}
    )
    task_time_limit: int = field(
        metadata={"description": "Hard time limit for tasks in seconds."}
    )
    task_soft_time_limit: int = field(
        metadata={"description": "Soft time limit for tasks in seconds."}
    )


@dataclass(slots=True, kw_only=True)
class WorkerConfig:
    """Configuration for Celery workers."""

    worker_prefetch_multiplier: int = field(
        metadata={"description": "The prefetch multiplier for Celery workers."}
    )
    task_reject_on_worker_lost: bool = field(
        metadata={
            "description": "If True, tasks are re-queued if a worker crashes. (Redis broker only)"
        }
    )
    worker_max_tasks_per_child: int = field(
        metadata={
            "description": "Maximum number of tasks a worker can execute before being replaced."
        }
    )
    worker_max_memory_per_child: int = field(
        metadata={
            "description": "Maximum memory (in bytes) a worker can use before being replaced."
        }
    )
    task_acks_late: bool = field(
        metadata={"description": "If True, tasks are acknowledged after execution."}
    )


@dataclass(slots=True, kw_only=True)
class TaskAndSchedule:
    """Configuration for a Celery task and its schedule."""

    task: str = field(metadata={"description": "The name of the Celery task."})
    schedule: int = field(metadata={"description": "The schedule interval in seconds."})


@dataclass(slots=True, kw_only=True)
class BeatSchedule:
    """Configuration for a Celery Beat scheduled task."""

    cleanup_old_records: TaskAndSchedule = field(
        metadata={"description": "Scheduled task for cleaning up old records."}
    )


@dataclass(slots=True, kw_only=True)
class BeatConfig:
    """Configuration for Celery Beat."""

    beat_schedule: BeatSchedule = field(
        metadata={"description": "The schedule for periodic tasks."}
    )
    health_check: TaskAndSchedule = field(
        metadata={"description": "Health check configuration."}
    )


@dataclass(slots=True, kw_only=True)
class RedisConfig:
    """Configuration for Redis-Celery."""

    master_name: str = field(
        metadata={"description": "The name of the Redis master for Sentinel."}
    )
    socket_timeout: float = field(
        metadata={"description": "Socket timeout in seconds."}
    )
    socket_connect_timeout: float = field(
        metadata={"description": "Socket connect timeout in seconds."}
    )
    socket_keepalive: bool = field(
        metadata={"description": "If True, enable socket keepalive."}
    )
    socket_keepalive_options: dict[str, int] = field(
        default_factory=dict,
        metadata={"description": "Socket keepalive options."},
    )
    health_check_interval: int = field(
        metadata={"description": "Interval for health checks in seconds."}
    )


@dataclass(slots=True, kw_only=True)
class OtherConfig:
    """Other Celery configurations."""

    celery_broker: BrokerOrBackendType = field(
        metadata={"description": "The type of message broker used by Celery."}
    )
    result_backend_always_retry: bool = field(
        metadata={
            "description": "If True, always retry connecting to the result backend."
        }
    )
    result_persistent: bool = field(
        metadata={"description": "If True, persist task results in the backend."}
    )
    result_backend_max_retries: int = field(
        metadata={
            "description": "Maximum number of retries for connecting to the result backend."
        }
    )
    result_expires: int = field(
        metadata={"description": "Time in seconds before a task result expires."}
    )
    num_processes: int = field(
        metadata={"description": "Number of worker processes to spawn."}
    )
    redis_config: RedisConfig = field(
        metadata={"description": "Redis-specific configurations."}
    )


@dataclass(slots=True, kw_only=True)
class CeleryConfig:
    """Configuration for Celery."""

    task_config: TaskConfig = field(
        metadata={"description": "Task-related configurations."}
    )
    task_routes: dict[str, QueueName] = field(
        metadata={"description": "Routing configuration for Celery tasks."}
    )
    worker_config: WorkerConfig = field(
        metadata={"description": "Worker-related configurations."}
    )
    beat_config: BeatConfig = field(
        metadata={"description": "Beat-related configurations."}
    )
    other_config: OtherConfig = field(
        metadata={"description": "Other Celery configurations."}
    )


@dataclass(slots=True, kw_only=True)
class StreamConfig:
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
class SSEConfig:
    """Configuration for Server-Sent Events (SSE)."""

    sse_retry_timeout: int = field(
        metadata={"description": "Retry timeout for SSE connections in seconds."}
    )
    sse_keepalive_interval: int = field(
        metadata={"description": "Interval for sending keepalive messages in seconds."}
    )


@dataclass(slots=True, kw_only=True)
class SessionConfig:
    """Configuration for session management."""

    session_cleanup_interval: int = field(
        metadata={"description": "Interval for cleaning up idle sessions in seconds."}
    )
    session_idle_timeout: int = field(
        metadata={"description": "Idle timeout for sessions in seconds."}
    )


@dataclass(slots=True, kw_only=True)
class CORS:
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
class Middleware:
    """Middleware configuration class."""

    cors: CORS = field(metadata={"description": "CORS configuration."})


@dataclass(slots=True, kw_only=True)
class Ratelimit:
    """Ratelimit configuration class."""

    default_rate: int = field(
        metadata={"description": "Default rate limit (e.g., 50)."}
    )
    burst_rate: int = field(metadata={"description": "Burst rate limit (e.g., 100)."})
    login_rate: int = field(metadata={"description": "Login rate limit (e.g., 10)."})


@dataclass(slots=True, kw_only=True)
class LLMModelConfig:
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
class APIConfig:
    """API-level configuration."""

    title: str = field(metadata={"description": "The title of the API."})
    name: str = field(metadata={"description": "The name of the API."})
    description: str = field(metadata={"description": "The description of the API."})
    version: str = field(metadata={"description": "The version of the API."})
    status: str = field(metadata={"description": "The current status of the API."})
    prefix: str = field(metadata={"description": "The prefix for the API routes."})
    auth_prefix: str = field(
        metadata={"description": "The prefix for the authentication routes."}
    )
    middleware: Middleware = field(
        metadata={"description": "Middleware configuration."}
    )
    ratelimit: Ratelimit = field(metadata={"description": "Ratelimit configuration."})


class AppConfig(BaseSchema):
    """Application-level configuration (Pydantic Model)."""

    vectorstore_config: VectorStoreConfig = Field(
        description="Vector store configurations."
    )
    custom_config: CustomConfig = Field(description="Custom configurations")
    llm_model_config: LLMModelConfig = Field(description="LLM model configurations.")
    queues_config: QueuesConfig = Field(description="Queues configurations.")
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
