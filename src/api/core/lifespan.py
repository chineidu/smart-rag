import os
import time
import warnings
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import FastAPI
from langchain_core.embeddings import Embeddings
from qdrant_client.qdrant_client import QdrantClient

from src import create_logger
from src.api.core.cache import CacheSetup
from src.api.core.ratelimit import limiter
from src.config import app_config, app_settings
from src.graph import GraphManager
from src.utilities.embeddings import OpenRouterEmbeddings
from src.utilities.vectorstores import VectorStoreSetup

if TYPE_CHECKING:
    pass
warnings.filterwarnings("ignore")

logger = create_logger(name="api_lifespan")

# Constants
MAX_WORKERS: int = os.cpu_count() - 1  # type: ignore
FILEPATHS: str | list[str] = app_config.vectorstore_config.filepaths
JQ_SCHEMA: str | None = app_config.vectorstore_config.jq_schema
FORMAT: str = app_config.vectorstore_config.format
COLLECTION: str = app_config.vectorstore_config.collection
FILEPATHS_IS_GLOB: bool = app_config.vectorstore_config.filepaths_is_glob
EMBEDDING_MODEL: Embeddings = OpenRouterEmbeddings(
    model=app_config.llm_model_config.embedding_model.model_name
)
QDRANT_CLIENT: QdrantClient = QdrantClient(url=app_settings.qdrant_url)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Initialize and cleanup FastAPI application lifecycle.

    This context manager handles the initialization of required resources
    during startup and cleanup during shutdown.
    """
    try:
        start_time: float = time.perf_counter()
        logger.info(
            f"ENVIRONMENT: {app_settings.ENVIRONMENT} | DEBUG: {app_settings.DEBUG} "
        )
        logger.info("Starting up application and loading model...")

        # ====================================================
        # ================= Load Dependencies ================
        # ====================================================

        # ==== Setup graph manager ====
        graph_manager = GraphManager()
        app.state.graph_manager = graph_manager

        # ==== Setup vectorstore ====
        vs_setup = VectorStoreSetup()
        vectorstore = await vs_setup.asetup_vectorstore(
            FILEPATHS,
            JQ_SCHEMA,
            FORMAT,
            EMBEDDING_MODEL,
            QDRANT_CLIENT,
            COLLECTION,
            FILEPATHS_IS_GLOB,
        )
        if not vs_setup.is_ready() or vectorstore is None:
            raise RuntimeError("Failed to initialize vectorstore during startup")
        app.state.vs_setup = vs_setup

        # Make vectorstore available to helper functions
        from src.startup import set_vectorstore_setup

        set_vectorstore_setup(vs_setup)

        # ==== Setup cache ====
        cache_setup = CacheSetup()
        app.state.cache_setup = cache_setup
        app.state.cache = cache_setup.setup_cache()
        logger.info("Cache initialized")

        # ==== Setup rate limiter ====
        app.state.limiter = limiter

        logger.info(
            f"Application startup completed in {time.perf_counter() - start_time:.2f} seconds"
        )

        # Yield control to the application
        yield

    # ====================================================
    # =============== Cleanup Dependencies ===============
    # ====================================================
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        raise

    finally:
        logger.info("Shutting down application...")

        # ==== Cleanup graph manager ====
        if hasattr(app.state, "graph_manager"):
            try:
                await app.state.graph_manager.cleanup_checkpointer()
                await app.state.graph_manager.cleanup_long_term_memory()
                logger.info("🚨 Graph manager shutdown completed.")
            except Exception as e:
                logger.error(f"Error cleaning up graph manager: {e}")

        # ==== Cleanup vectorstore setup ====
        if hasattr(app.state, "vs_setup") and app.state.vs_setup.is_ready():
            try:
                app.state.vs_setup.close()
            except Exception as e:
                logger.error(f"Error closing vectorstore setup: {e}")

        # ==== Cleanup rate limiter ====
        if hasattr(app.state, "limiter"):
            try:
                app.state.limiter = None
                logger.info("🚨 Rate limiter shutdown.")

            except Exception as e:
                logger.error(f"Error shutting down the rate limiter: {e}")

        # ==== Cleanup cache ====
        if hasattr(app.state, "cache_setup"):
            # Cache will be automatically garbage collected
            try:
                app.state.cache_setup.close()

            except Exception as e:
                logger.warning(f"Error clearing cache during shutdown: {e}")
            logger.info("Cache shutdown initiated")
