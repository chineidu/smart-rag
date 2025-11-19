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

    # ===== APP CONFIGURATION =====
    SEARCH_API: str = "tavily"  # Options: 'tavily', 'serper'
    FETCH_FULL_PAGE: bool = True

    # ===== REMOTE INFERENCE =====

    # TOGETHER AI
    TOGETHER_API_KEY: SecretStr = SecretStr("")
    TOGETHER_API_URL: str = "https://api.together.xyz/v1"

    # OPENROUTER
    OPENROUTER_API_KEY: SecretStr = SecretStr("")
    OPENROUTER_URL: str = "https://openrouter.ai/api/v1"

    # TAVILY
    TAVILY_API_KEY: SecretStr = SecretStr("")

    # ===== OBSERVABILITY =====
    # LANGFUSE
    LANGCHAIN_API_KEY: SecretStr = SecretStr("")
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT: str = "Smart-RAG"

    # ===== VECTOR STORE =====
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: SecretStr = SecretStr("")


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
