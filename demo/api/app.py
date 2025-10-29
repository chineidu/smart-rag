import sys
import warnings

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from demo.api.routes import feedback, get_history, get_stream
from demo.api.utilities.utilities import lifespan

warnings.filterwarnings("ignore")


def create_application() -> FastAPI:
    """Create and configure a FastAPI application instance.

    This function initializes a FastAPI application with custom configuration settings,
    adds CORS middleware, and includes API route handlers.

    Returns
    -------
    FastAPI
        A configured FastAPI application instance.
    """
    app = FastAPI(
        title="My Demo API",
        description="API for my demo application",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Type"],
    )

    # Include routers
    app.include_router(feedback.router, prefix="/api/v1")
    app.include_router(get_stream.router, prefix="/api/v1")
    app.include_router(get_history.router, prefix="/api/v1")

    return app


app: FastAPI = create_application()

if __name__ == "__main__":
    try:
        uvicorn.run(
            "demo.api.app:app",
            host="0.0.0.0",
            port=8080,
            reload=False,
        )
    except (Exception, KeyboardInterrupt) as e:
        print(f"Error creating application: {e}")
        print("Exiting gracefully...")
        sys.exit(1)
