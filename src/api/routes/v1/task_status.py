import json
from typing import Any

from celery.result import AsyncResult
from fastapi import APIRouter, Request, status

from src import create_logger
from src.api.core.exceptions import BaseAPIError
from src.api.core.ratelimit import limiter
from src.api.core.responses import MsgSpecJSONResponse
from src.celery_app.app import celery_app
from src.config import app_config
from src.schemas.routes.task_status import TaskStatusResponse
from src.schemas.types import TaskStatusType

logger = create_logger(name="task-status")
LIMIT_VALUE: int = app_config.api_config.ratelimit.burst_rate

router = APIRouter(tags=["task-status"], default_response_class=MsgSpecJSONResponse)


@router.get("/task-status/{task_id}", status_code=status.HTTP_200_OK)
@limiter.limit(f"{LIMIT_VALUE}/minute")
async def get_task_status(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    task_id: str,
) -> dict[str, Any]:
    """
    Get the status of a task (works for both individual tasks and chord callbacks)
    """
    try:
        result = AsyncResult(task_id, app=celery_app)

        state: str = result.state or ""

        response: dict[str, Any] = {
            "task_id": task_id,
            "state": state,
            "ready": result.ready(),
        }

        if state == TaskStatusType.PENDING.value:
            response.update(
                {
                    "status": "Task is waiting to be processed or does not exist",
                    "current": 0,
                    "total": 1,
                }
            )

        elif state == TaskStatusType.SUCCESS.value:
            response.update(
                {
                    "status": "Task completed successfully",
                    "result": result.result,
                    "successful": True,
                }
            )

        elif state == TaskStatusType.FAILURE.value:
            response.update(
                {
                    "status": "Task failed",
                    "error": str(result.info),
                    "traceback": result.traceback,
                    "successful": False,
                }
            )

        elif state == TaskStatusType.RETRY.value:
            # result.info for retries often contains exception info
            info = result.info
            response.update(
                {
                    "status": "Task is being retried",
                    "error": str(info),
                    "percentage": 0,
                }
            )

        elif state == TaskStatusType.ERROR.value:
            response.update(
                {"status": "Task encountered an error", "error": str(result.info)}
            )

        return TaskStatusResponse(**response).model_dump(by_alias=True)  # type: ignore

    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {e}")
        response = {
            "task_id": task_id,
            "state": TaskStatusType.ERROR.value,
            "status": f"Error retrieving task status: {str(e)}",
            "error": str(e),
        }
        raise BaseAPIError(f"{json.dumps(response, indent=2)}") from e
