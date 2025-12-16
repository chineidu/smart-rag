import os
import sys

import numpy as np
import torch

from src import create_logger
from src.celery_app.app import celery_app
from src.config import app_config
from src.utilities.utils import log_system_info

logger = create_logger(name="worker")
rng = np.random.default_rng()
name: str = str(rng.integers(0, 10_000))


def _default_pool_and_concurrency() -> tuple[str, int]:
    """Return sensible defaults for pool and concurrency based on hardware.

    - GPU: prefer `threads` to avoid CUDA fork issues
    - CPU: prefer `prefork` for process-level parallelism
    """
    if torch.cuda.is_available():
        num_processes: int = min(os.cpu_count() or 1, 4)
        return ("threads", num_processes)
    # CPU-only
    configured = app_config.celery_config.other_config.num_processes
    cpu_count = (os.cpu_count() or 1) // 2
    # Use the smaller of configured and available CPUs, but at least 1
    num_processes = max(1, min(configured, cpu_count))
    return ("prefork", num_processes)


def run_worker() -> None:
    """Run the Celery worker with optimized pool configuration."""

    try:
        logger.info("Database initialized successfully.")

        log_system_info()

        # Resolve dynamic settings from environment (with safe defaults)
        env_queues: str | None = os.getenv("CELERY_QUEUES")
        env_concurrency: str | None = os.getenv("CELERY_CONCURRENCY")
        env_pool: str | None = os.getenv("CELERY_POOL")
        env_loglevel: str | None = os.getenv("CELERY_LOGLEVEL")
        env_hostname_suffix: str | None = os.getenv("CELERY_HOSTNAME_SUFFIX")

        # Defaults
        default_pool, default_concurrency = _default_pool_and_concurrency()
        pool_type: str = (env_pool or default_pool).strip()
        try:
            num_processes: int = (
                int(env_concurrency) if env_concurrency else default_concurrency
            )
        except ValueError:
            num_processes = default_concurrency

        # Queues: fall back to configured set if not provided
        default_queues: str = ",".join(
            [
                app_config.queues_config.low_priority_ml,
                app_config.queues_config.normal_priority_ml,
                app_config.queues_config.high_priority_ml,
                app_config.queues_config.notifications,
                "celery",
            ]
        )
        queues_arg: str = (env_queues or default_queues).strip()

        # Log level
        loglevel: str = (env_loglevel or "warning").strip()

        # Hostname suffix
        hostname_suffix: str = (env_hostname_suffix or f"worker_{name}").strip()

        # Configure worker arguments
        argv: list[str] = [
            "worker",  # Command to start the Celery worker
            f"--loglevel={loglevel}",
            f"--concurrency={num_processes}",
            f"--queues={queues_arg}",
            f"--hostname={hostname_suffix}@%h",
            f"--pool={pool_type}",
            "--without-gossip",  # Disable gossip to reduce network overhead
            "--without-mingle",  # Disable mingle to speed up worker startup
            "--without-heartbeat",  # Disable heartbeat to reduce network traffic
        ]

        # Set thread limits for CPU efficiency if using threads (defensive; prefork doesn't use these)
        num_threads = os.cpu_count() or 1
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)

        logger.info(
            f"Starting Celery worker with pool={pool_type}, "
            f"concurrency={num_processes}, threads={num_threads}, queues={queues_arg}, loglevel={loglevel}"
        )
        logger.info(f"Worker args: {' '.join(argv)}")

        celery_app.worker_main(argv=argv)

    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Failed to start Celery worker: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    run_worker()
