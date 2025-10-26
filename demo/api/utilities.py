import asyncio
import os
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status

from demo import create_logger

warnings.filterwarnings("ignore")
logger = create_logger(name="app_utils")


MAX_WORKERS: int = os.cpu_count() - 1  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Initialize and cleanup FastAPI application lifecycle.

    This context manager handles the initialization of model components during startup
    and cleanup during shutdown.
    """
    try:
        start_time: float = time.perf_counter()
        logger.info("Starting up application and loading model...")

        # Load model here
        await asyncio.sleep(1)  # Simulate async model loading
        app.state.my_model = {"dummy": "model"}  # Placeholder for model instance

        logger.info(
            f"Application startup completed in {time.perf_counter() - start_time:.2f} seconds"
        )

        # Yield control to the application
        yield

    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        raise

    finally:
        # Cleanup on shutdown
        if hasattr(app.state, "my_model"):
            try:
                app.state.my_model = None
                logger.info("Model unloaded during shutdown")

            except Exception as e:
                logger.error(f"Error during shutdown cleanup: {e}")


def get_model_manager(request: Request) -> dict[str, Any]:
    """Get the prediction service from the app state."""
    if not hasattr(request.app.state, "my_model") or request.app.state.my_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not loaded. Please try again later.",
        )
    return request.app.state.my_model
