from dataclasses import asdict
from typing import Any

from celery import Celery
from kombu import Queue

from src.config import app_config, app_settings
from src.schemas.types import BrokerOrBackendType

# Redis Config
master_name: str = app_config.celery_config.other_config.redis_config.master_name
socket_timeout: float = (
    app_config.celery_config.other_config.redis_config.socket_timeout
)
socket_connect_timeout: float = (
    app_config.celery_config.other_config.redis_config.socket_connect_timeout
)
socket_keepalive: bool = (
    app_config.celery_config.other_config.redis_config.socket_keepalive
)
socket_keepalive_options: dict[str, int] = (
    app_config.celery_config.other_config.redis_config.socket_keepalive_options
)
health_check_interval: int = (
    app_config.celery_config.other_config.redis_config.health_check_interval
)


def create_celery_app() -> Celery:
    """Create and configure a Celery application instance."""
    celery = Celery("ner_tasks")

    # Define priority queues for RabbitMQ
    task_queues = (
        # Priority queues
        Queue(
            app_config.queues_config.high_priority_ml,
            routing_key=app_config.queues_config.high_priority_ml,
        ),
        Queue(
            app_config.queues_config.normal_priority_ml,
            routing_key=app_config.queues_config.normal_priority_ml,
        ),
        Queue(
            app_config.queues_config.low_priority_ml,
            routing_key=app_config.queues_config.low_priority_ml,
        ),
        # Default queues_config
        Queue("celery", routing_key="celery"),  # default celery queue
        Queue(
            app_config.queues_config.cleanups,
            routing_key=app_config.queues_config.cleanups,
        ),
        Queue(
            app_config.queues_config.notifications,
            routing_key=app_config.queues_config.notifications,
        ),
    )

    # Convert the beat_schedule to a dictionary
    beat_config_dict: dict[str, dict[str, Any]] = dict(
        asdict(app_config.celery_config.beat_config.beat_schedule).items()
    )

    # Add the health_check
    beat_config_dict["health_check"] = asdict(
        app_config.celery_config.beat_config.health_check
    )

    # Configuration
    celery.conf.update(
        # DB result backend config
        # result_backend=app_settings.celery_database_url, # Using a SQL DB backend
        result_backend_always_retry=app_config.celery_config.other_config.result_backend_always_retry,
        result_persistent=app_config.celery_config.other_config.result_persistent,
        result_backend_max_retries=app_config.celery_config.other_config.result_backend_max_retries,
        result_expires=app_config.celery_config.other_config.result_expires,
        # Broker config
        broker_url=app_settings.redis_url,
        broker_connection_retry=True,
        broker_connection_max_retries=3,
        # Priority queue definitions
        task_queues=task_queues,
        # Task config
        task_serializer=app_config.celery_config.task_config.task_serializer,
        task_time_limit=app_config.celery_config.task_config.task_time_limit,
        task_soft_time_limit=app_config.celery_config.task_config.task_soft_time_limit,
        result_serializer=app_config.celery_config.task_config.result_serializer,
        accept_content=app_config.celery_config.task_config.accept_content,
        timezone=app_config.celery_config.task_config.timezone,
        enable_utc=app_config.celery_config.task_config.enable_utc,
        # Task routing
        task_routes=app_config.celery_config.task_routes,
        # Beat schedule
        beat_schedule=beat_config_dict,  # dict is required!
        # Worker config
        worker_prefetch_multiplier=app_config.celery_config.worker_config.worker_prefetch_multiplier,
        worker_max_tasks_per_child=app_config.celery_config.worker_config.worker_max_tasks_per_child,
        worker_max_memory_per_child=app_config.celery_config.worker_config.worker_max_memory_per_child,
        task_acks_late=app_config.celery_config.worker_config.task_acks_late,
        # (Redis broker only) Re-queue tasks if a worker crashes
        task_reject_on_worker_lost=app_config.celery_config.worker_config.task_reject_on_worker_lost,
    )
    if app_config.celery_config.other_config.celery_broker == BrokerOrBackendType.REDIS:
        celery.conf.update(
            result_backend=app_settings.redis_url,
            broker_transport="redis",
            broker_transport_options={
                # For Redis Sentinel
                "master_name": master_name,
                "socket_timeout": socket_timeout,
                "socket_connect_timeout": socket_connect_timeout,
                "socket_keepalive": socket_keepalive,
                "socket_keepalive_options": socket_keepalive_options,
                "health_check_interval": health_check_interval,
            },
        )
    # Task discovery
    celery.autodiscover_tasks(
        [
            "src.celery_app.tasks.periodic",
            "src.celery_app.tasks.prediction",
            "src.celery_app.tasks.notifications",
        ]
    )

    return celery


celery_app = create_celery_app()
