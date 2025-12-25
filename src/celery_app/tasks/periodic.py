from datetime import datetime, timedelta

from celery import shared_task
from sqlalchemy import delete, func, select, text

from src import create_logger
from src.celery_app import BaseTask
from src.db.models import CeleryTaskCleanup, get_db_session

logger = create_logger(name="prediction_tasks")
logger.propagate = False  # This prevents double logging to the root logger


@shared_task(bind=True, base=BaseTask)
def cleanup_old_records(self) -> None:  # noqa: ANN001
    """Clean up old Celery task records."""
    try:
        cutoff_date = datetime.now() - timedelta(days=30)

        with get_db_session() as session:
            count_stmt = (
                select(func.count())
                .select_from(CeleryTaskCleanup)
                .where(CeleryTaskCleanup.date_done < cutoff_date)
            )
            old_tasks: int | None = session.scalar(count_stmt)
            if old_tasks is None:
                old_tasks = 0

            # Delete old records
            delete_stmt = delete(CeleryTaskCleanup).where(
                CeleryTaskCleanup.date_done < cutoff_date
            )
            session.execute(delete_stmt)
            session.commit()
            logger.info(f"Cleaned up {old_tasks:,} old Celery task records.")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise self.retry(exc=e) from e


@shared_task
def health_check() -> None:
    """
    Perform system health check
    """
    try:
        with get_db_session() as session:
            session.execute(text("SELECT 1"))

            # Get some statistics
            total_tasks_stmt = select(func.count()).select_from(CeleryTaskCleanup)
            total_tasks: int | None = session.scalar(total_tasks_stmt)
            if total_tasks is None:
                total_tasks = 0

            failed_tasks_stmt = (
                select(func.count())
                .select_from(CeleryTaskCleanup)
                .where(CeleryTaskCleanup.status == "FAILURE")
            )
            failed_tasks = session.scalar(failed_tasks_stmt)

            logger.info(
                f"Health check completed successfully. {total_tasks} total tasks, "
                f"{failed_tasks} failed tasks."
            )

    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        logger.exception("Full traceback:")
