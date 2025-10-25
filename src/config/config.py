from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import Field

from src import PACKAGE_PATH
from src.schemas import BaseSchema


class FootballConfig(BaseSchema):
    """Configuration  for football news data source."""

    filepaths: str = Field(..., description="Path to the football news data directory.")


class AIConfig(BaseSchema):
    """Configuration  for AI news data source."""

    filepaths: str = Field(..., description="Path to the AI news data directory.")


class VectorStoreConfig(BaseSchema):
    """Configuration for vector store."""

    football_config: FootballConfig = Field(
        description="Football news vector store configuration."
    )
    ai_config: AIConfig = Field(description="AI news vector store configuration.")


class CreativeModelConfig(BaseSchema):
    """Configuration for creative model."""

    model_name: str = Field(..., description="The name of the creative LLM to use.")
    temperature: float = Field(
        0.7, description="The temperature setting for the creative LLM."
    )
    max_tokens: int = Field(
        1024, description="The maximum number of tokens for the creative LLM response."
    )


class ClassifierModelConfig(BaseSchema):
    """Configuration for classifier model."""

    model_name: str = Field(..., description="The name of the classifier LLM to use.")
    temperature: float = Field(
        0.0, description="The temperature setting for the classifier LLM."
    )
    max_tokens: int = Field(
        512, description="The maximum number of tokens for the classifier LLM response."
    )


class EmbeddingModelConfig(BaseSchema):
    """Configuration for embedding model."""

    model_name: str = Field(..., description="The name of the embedding model to use.")


class LLMModelConfig(BaseSchema):
    """Configuration for models."""

    creative_model: CreativeModelConfig = Field(
        description="Creative model configuration."
    )
    classifier_model: ClassifierModelConfig = Field(
        description="Classifier model configuration."
    )
    embedding_model: EmbeddingModelConfig = Field(
        description="Embedding model configuration."
    )


class AppConfig(BaseSchema):
    """Application-level configuration."""

    vectorstore_config: VectorStoreConfig = Field(
        description="Vector store configurations."
    )
    llm_model_config: LLMModelConfig = Field(description="LLM model configurations.")


config_path: Path = PACKAGE_PATH / "src/config/config.yaml"
config: DictConfig = OmegaConf.load(config_path).config
# # Resolve all the variables
resolved_cfg = OmegaConf.to_container(config, resolve=True)
# Validate the config
app_config: AppConfig = AppConfig(**dict(resolved_cfg))  # type: ignore
