import os
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv  # type: ignore
from pydantic import SecretStr  # type: ignore
from pydantic.functional_validators import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore

from src.schemas.types import LoggingLevelEnum, PoolType


class BaseSettingsConfig(BaseSettings):
    """Base configuration class for settings.

    This class extends BaseSettings to provide common configuration options
    for environment variable loading and processing.

    Attributes
    ----------
    model_config : SettingsConfigDict
        Configuration dictionary for the settings model specifying env file location,
        encoding and other processing options.
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(".env").absolute()),
        env_file_encoding="utf-8",
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class Settings(BaseSettingsConfig):
    """Application settings class containing database and other credentials."""

    # ===== DATABASE =====
    POSTGRES_USER: str = "langgraph"
    POSTGRES_PASSWORD: SecretStr = SecretStr("your_postgres_password")
    POSTGRES_DB: str = "langgraph"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    API_DB_NAME: str = "user_feedback_db"

    # ===== REDIS CACHE =====
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: SecretStr = SecretStr("your_redis_password")
    REDIS_DB: int = 0

    # ===== CELERY =====
    C_FORCE_ROOT: int = 1  # Suppress root user warning
    CELERY_OPTIMIZATION: str = "fair"  # Optimize Celery performance
    CELERY_CONCURRENCY: str = "1"  # Number of concurrent worker processes/threads
    CELERY_POOL: PoolType = PoolType.THREADS  # Worker pool type
    CELERY_QUEUES_LOW: str = "low_priority_ml,user_notifications,periodic"
    CELERY_LOGLEVEL: LoggingLevelEnum = LoggingLevelEnum.WARNING
    CELERY_HOSTNAME_SUFFIX: str = ""

    # ===== REMOTE INFERENCE =====
    # TOGETHER AI
    TOGETHER_API_KEY: SecretStr = SecretStr("your_api_key")
    TOGETHER_API_URL: str = "https://api.together.xyz/v1"

    # OPENROUTER
    OPENROUTER_API_KEY: SecretStr = SecretStr("your_api_key")
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1"

    # TAVILY
    TAVILY_API_KEY: SecretStr = SecretStr("your_api_key")

    # BRAVE SEARCH
    BRAVE_SEARCH_API_KEY: SecretStr = SecretStr("your_api_key")

    # ===== OBSERVABILITY =====
    # LANGFUSE
    LANGCHAIN_API_KEY: SecretStr = SecretStr("your_api_key")
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT: str = "Smart-RAG"

    # ===== VECTOR STORE =====
    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: SecretStr = SecretStr("your_api_key")

    # ===== APPLICATION SETTINGS =====
    ENVIRONMENT: str = "development"  # development | production
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 2
    RELOAD: bool = False
    DEBUG: bool = False

    # ===== API AUTHENTICATION =====
    SECRET_KEY: SecretStr = SecretStr("your_secret_key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    @field_validator(
        "PORT", "POSTGRES_PORT", "REDIS_PORT", "QDRANT_PORT", mode="before"
    )
    @classmethod
    def parse_port_fields(cls, v: str | int) -> int:
        """Parses port fields to ensure they are integers."""
        if isinstance(v, str):
            try:
                return int(v.strip())
            except ValueError:
                raise ValueError(f"Invalid port value: {v}") from None

        if isinstance(v, int) and not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}")

        return v

    @field_validator("REDIS_DB", mode="before")
    @classmethod
    def parse_int_fields(cls, v: str | int) -> int:
        """Parses int fields to ensure they are integers."""
        if isinstance(v, str):
            try:
                return int(v.strip())
            except ValueError:
                raise ValueError(f"Invalid integer value: {v}") from None

        return v

    @property
    def database_url(self) -> str:
        """
        Constructs the API database connection URL.

        This is the database used for user authentication and API-specific tables.
        It's separate from MLflow's database to avoid conflicts.

        Returns
        -------
        str
            Complete database connection URL in the format:
            postgresql+psycopg2://user:password@host:port/dbname
        """
        password: str = quote(self.POSTGRES_PASSWORD.get_secret_value(), safe="")
        url: str = (
            f"postgresql+psycopg2://{self.POSTGRES_USER}"
            f":{password}"
            f"@{self.POSTGRES_HOST}"
            f":{self.POSTGRES_PORT}"
            f"/{self.POSTGRES_DB}"
        )
        return url

    @property
    def database_url_2(self) -> str:
        """
        Constructs the API database connection URL.

        This is the database used for user authentication and API-specific tables.
        It's separate from MLflow's database to avoid conflicts.

        Returns
        -------
        str
            Complete database connection URL in the format:
            postgresql+psycopg2://user:password@host:port/dbname
        """
        password: str = quote(self.POSTGRES_PASSWORD.get_secret_value(), safe="")
        url: str = (
            f"postgresql+psycopg2://{self.POSTGRES_USER}"
            f":{password}"
            f"@{self.POSTGRES_HOST}"
            f":{self.POSTGRES_PORT}"
            f"/{self.API_DB_NAME}"
        )
        return url

    @property
    def redis_url(self) -> str:
        """
        Constructs the Redis connection URL.

        Returns
        -------
        str
            Complete Redis connection URL in the format:
            redis://[:password@]host:port/db
        """
        raw_password = self.REDIS_PASSWORD.get_secret_value()
        if raw_password:
            password = quote(raw_password, safe="")
            url: str = f"redis://:{password}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        else:
            url = f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return url

    @property
    def qdrant_url(self) -> str:
        """
        Constructs the Qdrant connection URL.

        Returns
        -------
        str
            Complete Qdrant connection URL in the format:
            http://host:port
        """
        url: str = f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
        return url


def refresh_settings() -> Settings:
    """Refresh environment variables and return new Settings instance.

    This function reloads environment variables from .env file and creates
    a new Settings instance with the updated values.

    Returns
    -------
    Settings
        A new Settings instance with refreshed environment variables
    """
    load_dotenv(override=True)
    return Settings()


def setup_env() -> None:
    """Sets environment variables for Together AI and OpenRouter clients."""
    os.environ["TOGETHER_API_KEY"] = app_settings.TOGETHER_API_KEY.get_secret_value()
    os.environ["TOGETHER_API_URL"] = app_settings.TOGETHER_API_URL
    os.environ["OPENROUTER_API_KEY"] = (
        app_settings.OPENROUTER_API_KEY.get_secret_value()
    )
    os.environ["OPENROUTER_URL"] = app_settings.OPENROUTER_URL


app_settings: Settings = refresh_settings()

# Call setup_env only once at startup
_setup_env_called: bool = False


def setup_env_once() -> None:
    """Sets environment variables for Together AI and OpenRouter clients. Called only once."""
    global _setup_env_called
    if not _setup_env_called:
        setup_env()
        _setup_env_called = True


# Automatically call setup_env when the module is imported
setup_env_once()
