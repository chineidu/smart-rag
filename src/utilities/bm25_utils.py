from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src import create_logger

logger = create_logger("bm25")


class BM25Setup:
    """Class to manage reranker model setup and lifecycle."""

    def __init__(self, documents: list[Document]) -> None:
        self._initialized: bool = False
        self._model_dict: dict[str, Any] | None = None

        # others
        self.documents = documents
        from src.utilities.tools.helpers import CustomTokenizer

        self.custom_tokenizer = CustomTokenizer(to_lower=True)

    def load_model(self) -> dict[str, Any] | None:
        """Load and return the bm25 model dictionary.
        Returns
        -------
        dict[str, Any] | None
            The loaded bm25 model dictionary or None if not initialized.
        """
        if self._initialized and self._model_dict:
            return self._model_dict

        # Create a list where each element is a list of words from a document
        tokenized_corpus: list[list[str]] = [
            self.custom_tokenizer.format_data(doc.page_content).split(" ")
            for doc in self.documents
        ]
        # Create a list of all unique document IDs
        doc_ids: list[str] = [doc.metadata["chunk_id"] for doc in self.documents]

        # Create a mapping from a document's ID back to the full Document object for easy lookup
        doc_dict: dict[str, Document] = {
            doc.metadata["chunk_id"]: doc for doc in self.documents
        }

        # Initialize the BM25Okapi index with our tokenized corpus
        bm25 = BM25Okapi(tokenized_corpus)
        self._initialized = True
        self._model_dict = {"bm25": bm25, "doc_ids": doc_ids, "doc_dict": doc_dict}
        logger.info("BM25 index built and loaded successfully.")

        return self._model_dict

    def get_model_dict(self) -> dict[str, Any] | None:
        """Return the cached bm25 model dictionary if initialized."""
        return self._model_dict

    def is_ready(self) -> bool:
        """Check if the bm25 setup is ready for use."""
        return self._initialized and self._model_dict is not None

    def close(self) -> None:
        """Unload the bm25 resources from memory."""
        if self._initialized and self._model_dict:
            self._model_dict = None
            self._initialized = False
            logger.info("📝 BM25 resources have been unloaded from memory.")
