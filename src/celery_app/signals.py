import asyncio
import os
import time
from typing import TYPE_CHECKING, Any

import torch
from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
    worker_init,
    worker_ready,
    worker_shutdown,
)
from qdrant_client.qdrant_client import QdrantClient

from src import create_logger
from src.config import app_config, app_settings
from src.db.init import init_db
from src.graph import GraphManager
from src.startup import (
    set_bm25_setup,
    set_reranker_setup,
    set_vectorstore_setup,
)
from src.utilities.bm25_utils import BM25Setup
from src.utilities.embeddings import OpenRouterEmbeddings
from src.utilities.reranker import CrossEncoderSetup
from src.utilities.vectorstores import VectorStoreSetup

if TYPE_CHECKING:
    pass

logger = create_logger(name="celery_app")


# ===========================================
# ======= Worker Lifecycle Management =======
# ===========================================
@worker_init.connect
def worker_init_handler(sender: Any | None = None, **kwargs: Any) -> None:  # noqa: ANN003, ARG001
    """Callback function triggered when a Celery worker process is initialized.

    Similar to Lifespan in FastAPI, this class ensures that the model is loaded
    and ready when the Celery worker starts, and provides access to the model
    loader for task instances.
    """
    # Import here to avoid circular imports
    from src.celery_app import CallbackTask

    logger.info(f"Celery worker {os.getpid()} is initializing (worker_init).")

    start_time: float = time.perf_counter()
    # Initialize database
    init_db()

    try:
        # ================= Load Dependencies =================
        # ---- Graph manager ----
        # Create the manager, but defer async initialization to per-task loops
        graph_manager = GraphManager()
        CallbackTask._graph_builder = graph_manager

        # ---- Vectorstore ----
        embedding_model = OpenRouterEmbeddings(
            model=app_config.llm_model_config.embedding_model.model_name
        )
        qdrant_client: QdrantClient = QdrantClient(url=app_settings.qdrant_url)

        vs_setup = VectorStoreSetup()
        asyncio.run(
            vs_setup.asetup_vectorstore(
                app_config.vectorstore_config.filepaths,
                app_config.vectorstore_config.jq_schema,
                app_config.vectorstore_config.format,
                embedding_model,
                qdrant_client,
                app_config.vectorstore_config.collection,
                app_config.vectorstore_config.filepaths_is_glob,
            )
        )
        if not vs_setup.is_ready():
            raise RuntimeError(
                "Vectorstore failed to initialize in Celery worker startup"
            )
        set_vectorstore_setup(vs_setup)

        # ---- BM25 ----
        bm25_setup = BM25Setup(documents=vs_setup.get_documents() or [])
        if bm25_setup.load_model() is None or not bm25_setup.is_ready():
            raise RuntimeError("BM25 failed to initialize in Celery worker startup")
        set_bm25_setup(bm25_setup)

        # ---- Reranker ----
        reranker_setup = CrossEncoderSetup(
            model_name_or_path=app_config.llm_model_config.cross_encoder_model.model_name,
            num_labels=app_config.llm_model_config.cross_encoder_model.num_labels,
            max_length=app_config.llm_model_config.cross_encoder_model.max_length,
            device=app_config.llm_model_config.cross_encoder_model.device,
        )
        if reranker_setup.load_model() is None or not reranker_setup.is_ready():
            raise RuntimeError("Reranker failed to initialize in Celery worker startup")
        set_reranker_setup(reranker_setup)

        logger.info(
            f"Celery worker {os.getpid()} startup completed in {time.perf_counter() - start_time:.2f}s"
        )
    except Exception as e:
        logger.error(f"Celery worker {os.getpid()} failed to initialize: {e}")
        raise


@worker_ready.connect
def on_worker_ready(sender: Any | None = None, **kwargs: Any) -> None:  # noqa: ARG001
    """Callback function triggered when a Celery worker is ready to accept tasks."""
    logger.info(f"Celery worker {os.getpid()} is ready to accept tasks.")


@worker_shutdown.connect
def on_worker_shutdown(sender: Any | None = None, **kwargs: Any) -> None:  # noqa: ARG001
    """Event handler for the worker shutdown signal."""
    logger.info(f"Celery worker {os.getpid()} is shutting down.")

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared on worker shutdown.")


# ===========================================
# ======== Task Lifecycle Management ========
# ===========================================
@task_prerun.connect
def task_prerun_handler(
    task_id: str, task: Any, args: Any, kwargs: Any, **extra: Any
) -> None:  # noqa: ANN001, ARG001
    """Log when task starts"""
    session_id = kwargs.get("session_id", "unknown")
    logger.info(f"Task {task.name} started: {task_id} (session: {session_id})")


@task_postrun.connect
def task_postrun_handler(
    task_id: str, task: Any, args: Any, kwargs: Any, retval: Any, **extra: Any
) -> None:  # noqa: ARG001
    """Log when task completes"""
    session_id = kwargs.get("session_id", "unknown")
    logger.info(f"Task {task.name} finished: {task_id} (session: {session_id})")


@task_failure.connect
def task_failure_handler(
    task_id: str,
    exception: Any,
    args: Any,  # noqa: ARG001
    kwargs: Any,
    traceback: Any,  # noqa: ARG001
    einfo: Any,  # noqa: ARG001
    **extra: Any,  # noqa: ARG001, E501
) -> None:
    """Log when task fails"""
    session_id = kwargs.get("session_id", "unknown")
    logger.error(
        f"Task failed: {task_id} (session: {session_id}), exception: {exception}"
    )
