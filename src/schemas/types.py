from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

if TYPE_CHECKING:
    from pydantic import BaseModel

    from src.schemas.base import ModelList

# TypeVar bound to BaseModel - accepts any Pydantic model
T = TypeVar("T", bound="BaseModel")

type MemoryData = "ModelList[T] | list[dict[str, Any]] | dict[str, Any] | TypedDict"  # type: ignore


class RetrieverMethodType(StrEnum):
    """The type of retrieval method to use for internal document search."""

    VECTOR_SEARCH = "vector_search"
    KEYWORD_SEARCH = "keyword_search"
    HYBRID_SEARCH = "hybrid_search"


class ToolsType(StrEnum):
    """The type of tool to use for each step."""

    VECTOR_STORE = "vector_store"
    WEB_SEARCH = "web_search"


class NextAction(StrEnum):
    """Tells the executor what to do after the current planning step."""

    CONTINUE = "continue"
    FINISH = "finish"
    RE_PLAN = "re_plan"


class SummarizationConditionType(StrEnum):
    """Condition to determine if overall conversation summarization is needed."""

    SUMMARIZE = "summarize"
    END = "END"


class FileFormatsType(StrEnum):
    """The type of file to process."""

    CSV = "csv"
    JSON = "json"
    TXT = "txt"
    PDF = "pdf"


class ErrorCodeEnum(StrEnum):
    HTTP_ERROR = "http_error"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    INVALID_INPUT = "invalid_input"
    RESOURCES_NOT_FOUND = "resources_not_found"
    STREAMING_ERROR = "streaming_error"
    UNAUTHORIZED = "unauthorized"
    UNEXPECTED_ERROR = "unexpected_error"


class ResourcesType(StrEnum):
    """The type of resource to use."""

    CACHE = "cache"
    DATABASE = "database"
    GRAPH = "graph"
    RATE_LIMITER = "rate_limiter"
    VECTOR_STORE = "vector_store"
    STREAM_SESSION = "stream_session"
    UNKNOWN = "unknown"  # Used when the resource type is not recognized


class MemoryKeys(StrEnum):
    """The keys used for storing and retrieving memory content."""

    NAMESPACE_KEY = "memory"
    USER_PREFERENCES_KEY = "user_preferences"


class RoleType(StrEnum):
    """Enumeration of possible user roles."""

    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class FeedbackType(str, Enum):
    """Enumeration of possible feedback types.

    Note
    ----
    NEUTRAL is represented as None
    Can't use `StrEnum` here because we want None as a value
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = None


class SectionNamesType(StrEnum):
    """The type of filters available for retrieval searches."""

    ITEM_1 = "ITEM 1. BUSINESS"
    ITEM_1A = "ITEM 1A. RISK FACTORS"
    ITEM_1B = "ITEM 1B. UNRESOLVED STAFF COMMENTS"
    ITEM_2 = "ITEM 2. PROPERTIES"
    ITEM_3 = "ITEM 3. LEGAL PROCEEDINGS"
    ITEM_4 = "ITEM 4. MINE SAFETY DISCLOSURES"
    ITEM_5 = (
        "ITEM 5. MARKET FOR REGISTRANTS' COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER "
        "PURCHASES OF EQUITY SECURITIES"
    )
    ITEM_6 = "ITEM 6. [RESERVED]"
    ITEM_7 = "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS"
    ITEM_7A = "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK"
    ITEM_8 = "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA"
    ITEM_9 = "ITEM 9. CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AND FINANCIAL DISCLOSURE"
    ITEM_9A = "ITEM 9A. CONTROLS AND PROCEDURES"
    ITEM_9C = (
        "ITEM 9C. DISCLOSURE REGARDING FOREIGN JURISDICTIONS THAT PREVENT INSPECTIONS"
    )
    ITEM_10 = "ITEM 10. DIRECTORS, EXECUTIVE OFFICERS AND CORPORATE GOVERNANCE"
    ITEM_11 = "ITEM 11. EXECUTIVE COMPENSATION"
    ITEM_12 = (
        "ITEM 12. SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT "
        "AND RELATED STOCKHOLDER MATTERS"
    )
    ITEM_13 = "ITEM 13. CERTAIN RELATIONSHIPS AND RELATED TRANSACTIONS, AND DIRECTOR INDEPENDENCE"
    ITEM_14 = "ITEM 14. PRINCIPAL ACCOUNTANT FEES AND SERVICES"
    ITEM_15 = "ITEM 15. EXHIBIT AND FINANCIAL STATEMENT SCHEDULES"
    ITEM_16 = "ITEM 16. FORM 10-K SUMMARY"


class BrokerOrBackendType(StrEnum):
    """The type of broker or backend to use for task queuing and execution."""

    REDIS = "redis"
    RABBITMQ = "rabbitmq"


class EventsType(StrEnum):
    """The type of events for SSE."""

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    ERROR = "error"
    KEEPALIVE = "keepalive"

    # Custom event types
    VALIDATE_QUERY = "validate_query"
    UNRELATED_QUERY = "unrelated_query"
    GENERATE_PLAN = "generate_plan"
    RETRIEVE_INTERNAL_DOCS = "retrieve_internal_docs"
    INTERNET_SEARCH = "internet_search"
    COMPRESS_DOCUMENTS = "compress_documents"
    REFLECT = "reflect"
    FINAL_ANSWER = "final_answer"
    OVERALL_CONVO_SUMMARIZATION = "overall_convo_summarization"
    UPDATE_LT_MEMORY = "update_lt_memory"


class TaskStatusType(StrEnum):
    """The status of a Celery task."""

    ERROR = "ERROR"
    FAILURE = "FAILURE"
    PENDING = "PENDING"
    RETRY = "RETRY"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"


class PoolType(StrEnum):
    """Celery worker pool strategies.

    Notes
    -----
    `PREFORK`: Process-based workers (separate Python processes). Provides strong
    isolation and true parallelism for CPU-bound or non-thread-safe tasks, but
    increases memory usage (model loaded per process).

    `THREADS`: Thread-based workers (single process). Lower memory usage since
    models can be shared; required for GPU/CUDA workloads to avoid context
    conflicts. The Python GIL may limit pure-Python parallelism, though many ML
    runtimes release the GIL during inference.

    Selection (short):
        - Prefer `THREADS` for GPUs, large models, ONNX/optimized ML inference,
          or fast startup.
        - Prefer `PREFORK` for isolation, CPU-bound work, or non-thread-safe code.
    """

    PREFORK = "prefork"
    THREADS = "threads"


class LoggingLevelEnum(StrEnum):
    """Enum for logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
