from pydantic.functional_validators import field_validator
from pathlib import Path

from dotenv import load_dotenv  # type: ignore
from pydantic import SecretStr  # type: ignore
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


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
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: SecretStr = SecretStr("your_api_key")

    @field_validator("PORT", mode="before")
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


app_settings: Settings = refresh_settings()
