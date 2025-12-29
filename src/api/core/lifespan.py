import time
import warnings
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import FastAPI
from langchain_core.embeddings import Embeddings
from qdrant_client.qdrant_client import QdrantClient

from src import create_logger
from src.api.core.cache import setup_cache
from src.api.core.ratelimit import limiter
from src.config import app_config, app_settings
from src.graph import GraphManager
from src.utilities.bm25_utils import BM25Setup
from src.utilities.embeddings import OpenRouterEmbeddings
from src.utilities.reranker import CrossEncoderSetup
from src.utilities.vectorstores import VectorStoreSetup

if TYPE_CHECKING:
    pass
warnings.filterwarnings("ignore")

logger = create_logger(name="api_lifespan")

# Constants
FILEPATHS: str | list[str] = app_config.vectorstore_config.filepaths
JQ_SCHEMA: str | None = app_config.vectorstore_config.jq_schema
FORMAT: str = app_config.vectorstore_config.format
COLLECTION: str = app_config.vectorstore_config.collection
FILEPATHS_IS_GLOB: bool = app_config.vectorstore_config.filepaths_is_glob
EMBEDDING_MODEL: Embeddings = OpenRouterEmbeddings(
    model=app_config.llm_model_config.embedding_model.model_name
)
QDRANT_CLIENT: QdrantClient = QdrantClient(url=app_settings.qdrant_url)
SPLIT_BY_SECTIONS: bool = app_config.vectorstore_config.split_by_sections


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

        # ---------- Setup rate limiter ----------
        app.state.limiter = limiter

        # ---------- Setup GraphManager ----------
        graph_manager = GraphManager()
        app.state.graph_manager = graph_manager

        # ==== Setup vectorstore ====
        vs_setup = VectorStoreSetup()
        vectorstore = await vs_setup.asetup_vectorstore(
            filepaths=FILEPATHS,
            jq_schema=JQ_SCHEMA,
            format=FORMAT,
            embedding_model=EMBEDDING_MODEL,
            client=QDRANT_CLIENT,
            collection=COLLECTION,
            filepaths_is_glob=FILEPATHS_IS_GLOB,
            split_by_sections=SPLIT_BY_SECTIONS,
        )
        if not vs_setup.is_ready() or vectorstore is None:
            raise RuntimeError("Failed to initialize vectorstore during startup")
        app.state.vs_setup = vs_setup

        # Make vectorstore available to helper functions
        from src.startup import set_vectorstore_setup

        set_vectorstore_setup(vs_setup)

        # ---------- Setup bm25 ----------
        bm25_setup = BM25Setup(documents=vs_setup.get_documents() or [])
        bm25_model_dict = bm25_setup.load_model()
        if not bm25_setup.is_ready() or bm25_model_dict is None:
            raise RuntimeError("Failed to initialize BM25 during startup")
        app.state.bm25_setup = bm25_setup
        # Make bm25 available to helper functions
        from src.startup import set_bm25_setup

        set_bm25_setup(bm25_setup)
        # ---------- Setup reranker ----------
        reranker_setup = CrossEncoderSetup(
            model_name_or_path=app_config.llm_model_config.cross_encoder_model.model_name,
            num_labels=app_config.llm_model_config.cross_encoder_model.num_labels,
            max_length=app_config.llm_model_config.cross_encoder_model.max_length,
            device=app_config.llm_model_config.cross_encoder_model.device,
        )
        reranker_model = reranker_setup.load_model()
        if not reranker_setup.is_ready() or reranker_model is None:
            raise RuntimeError("Failed to initialize reranker during startup")
        app.state.reranker_setup = reranker_setup
        # Make reranker available to helper functions
        from src.startup import set_reranker_setup

        set_reranker_setup(reranker_setup)

        # ---------- Setup cache ----------
        app.state.cache = setup_cache()
        logger.info("Cache initialized")

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

        # ---------- Cleanup GraphManager ----------
        if hasattr(app.state, "graph_manager"):
            try:
                app.state.graph_manager = None
                logger.info("GraphManager shutdown.")
            except Exception as e:
                logger.error(f"Error shutting down GraphManager: {e}")

        # ---------- Cleanup vectorstore setup ----------
        if hasattr(app.state, "vs_setup") and app.state.vs_setup.is_ready():
            try:
                app.state.vs_setup.close()
            except Exception as e:
                logger.error(f"❌ Error closing vectorstore setup: {e}")

        # ---------- Cleanup bm25 setup ----------
        if hasattr(app.state, "bm25_setup") and app.state.bm25_setup.is_ready():
            try:
                app.state.bm25_setup.close()
            except Exception as e:
                logger.error(f"❌ Error closing bm25 setup: {e}")

        # ---------- Cleanup reranker setup ----------
        if hasattr(app.state, "reranker_setup") and app.state.reranker_setup.is_ready():
            try:
                app.state.reranker_setup.close()
            except Exception as e:
                logger.error(f"❌ Error closing reranker setup: {e}")
        # ---------- Cleanup rate limiter ----------
        if hasattr(app.state, "limiter"):
            try:
                app.state.limiter = None
                logger.info("🚨 Rate limiter shutdown.")

            except Exception as e:
                logger.error(f"❌ Error shutting down the rate limiter: {e}")
