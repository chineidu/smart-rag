from typing import Any

from sentence_transformers import CrossEncoder
from torch import nn

from src import create_logger

logger = create_logger("reranker")


class CrossEncoderSetup:
    """Class to manage reranker model setup and lifecycle."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int | None = None,
        max_length: int | None = None,
        device: Any | None = None,
    ) -> None:
        self._initialized: bool = False
        self._model: nn.Module | None = None

        # others
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device

    def load_model(self) -> nn.Module | None:
        """Load and return the reranker model.

        Returns
        -------
        nn.Module | None
            The loaded reranker model instance.
        """
        if self._initialized and self._model:
            return self._model

        self._model = CrossEncoder(
            model_name_or_path=self.model_name_or_path,
            num_labels=self.num_labels,
            max_length=self.max_length,
            device=self.device,
            local_files_only=True,  # Use locally cached models only
        )
        self._initialized = True
        logger.info(
            f"Reranker model '{self.model_name_or_path}' has been loaded into memory."
        )

        return self._model

    def get_model(self) -> nn.Module | None:
        """Return the cached reranker model instance if initialized.

        Returns
        -------
        nn.Module | None
            The cached reranker model instance or None if not yet set up.
        """
        return self._model

    def is_ready(self) -> bool:
        """Check if the reranker setup is ready for use."""
        return self._initialized and self._model is not None

    def close(self) -> None:
        """Unload the reranker model from memory."""
        if self._initialized and self._model:
            self._model = None
            self._initialized = False
            logger.info("📝 Reranker model has been unloaded from memory.")
