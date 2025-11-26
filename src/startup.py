from typing import TYPE_CHECKING

from src import create_logger

if TYPE_CHECKING:
    from src.utilities.bm25_utils import BM25Setup
    from src.utilities.reranker import CrossEncoderSetup
    from src.utilities.vectorstores import VectorStoreSetup

logger = create_logger("startup")

# Module-level cached VectorStore setup (should be configured during app startup)
_bm25_setup: "BM25Setup | None" = None
_reranker_setup: "CrossEncoderSetup | None" = None
_vs_setup: "VectorStoreSetup | None" = None


def set_vectorstore_setup(vs_setup: "VectorStoreSetup") -> None:
    """Provide an existing VectorStoreSetup instance for helpers to use.

    Call this during application startup.
    """
    global _vs_setup
    _vs_setup = vs_setup
    logger.info("Global VectorStoreSetup has been set.")


def get_vectorstore_setup() -> "VectorStoreSetup | None":
    """Return the currently configured VectorStoreSetup instance (or None)."""
    return _vs_setup


def set_reranker_setup(reranker_setup: "CrossEncoderSetup") -> None:
    """Provide an existing CrossEncoderSetup instance for helpers to use.

    Call this during application startup.
    """
    global _reranker_setup
    _reranker_setup = reranker_setup
    logger.info("Global CrossEncoderSetup has been set.")


def get_reranker_setup() -> "CrossEncoderSetup | None":
    """Return the currently configured CrossEncoderSetup instance (or None)."""
    return _reranker_setup


def set_bm25_setup(bm25_setup: "BM25Setup") -> None:
    """Provide an existing BM25Setup instance for helpers to use.

    Call this during application startup.
    """
    global _bm25_setup
    _bm25_setup = bm25_setup
    logger.info("Global BM25Setup has been set.")


def get_bm25_setup() -> "BM25Setup | None":
    """Return the currently configured BM25Setup instance (or None)."""
    return _bm25_setup
