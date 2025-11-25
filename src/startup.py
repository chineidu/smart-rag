from src import create_logger
from src.utilities.vectorstores import VectorStoreSetup

logger = create_logger("startup")

# Module-level cached VectorStore setup (should be configured during app startup)
_vs_setup: VectorStoreSetup | None = None


def set_vectorstore_setup(vs_setup: VectorStoreSetup) -> None:
    """Provide an existing VectorStoreSetup instance for helpers to use.

    Call this during application startup to allow avector_search_tool to use the
    cached vectorstore.
    """
    global _vs_setup
    _vs_setup = vs_setup
    logger.info("Global VectorStoreSetup has been set.")


def get_vectorstore_setup() -> VectorStoreSetup | None:
    """Return the currently configured VectorStoreSetup instance (or None)."""
    return _vs_setup
