import asyncio
import json
from typing import Any

from langchain_core.runnables.config import RunnableConfig

from src import create_logger
from src.celery_app import CallbackTask
from src.celery_app.app import celery_app
from src.celery_app.globals import _get_worker_event_loop
from src.config import app_config
from src.graph import GraphManager
from src.schemas.types import EventsType, TaskStatusType
from src.stream_manager import StreamSessionManager
from src.utilities.utils import format_generated_content

logger = create_logger(name="predict_task")
logger.propagate = False  # This prevents double logging to the root logger

RECURSION_LIMIT = app_config.custom_config.recursion_limit


# Note: When `bind=True`, celery automatically passes the task instance as the first argument
# meaning that we need to use `self` and this provides additional functionality like retries, etc
@celery_app.task(bind=True, base=CallbackTask)
def generate_streaming_response_task(
    self,  # noqa: ANN001
    session_id: str,
    message: str,
    user_id: str,  # noqa: ANN001
) -> dict[str, Any]:  # noqa: ANN001
    """Task for generating and streaming response for a given query.

    Parameters
    ----------
    session_id : str | None
        The unique identifier for the session.
    message : str
        The input query string to generate a response for.
    user_id : str
        The identifier for the user making the request.

    Returns
    -------
    dict[str, Any]
        _description_

    Raises
    ------
    self.retry
        _description_
    """
    task_id = self.request.id  # type: ignore
    queue = self.request.delivery_info.get("routing_key", "unknown")

    async def _run_generation() -> dict[str, Any]:
        """This uses a single event loop to run the async generation and streaming.

        It prevents "Event loop is closed" errors that can occur when trying to create
        multiple event loops in the same thread/process.
        """
        stream_manager: StreamSessionManager = await self.aget_stream_session_manager()
        graph_builder: GraphManager = await self.aget_graph_builder()
        return await _agenerate_and_stream(
            session_id=session_id,
            message=message,
            stream_manager=stream_manager,
            graph_builder=graph_builder,
            user_id=user_id,
            task_id=task_id,
            queue=queue,
        )

    try:
        # Reuse a single event loop per worker process; recreate if it was closed
        loop = _get_worker_event_loop()
        return loop.run_until_complete(_run_generation())
    except Exception as e:
        logger.error(f"Error in task {task_id}: {e}")
        raise self.retry(exc=e) from e


async def _agenerate_and_stream(
    session_id: str,
    message: str,
    stream_manager: StreamSessionManager,
    graph_builder: GraphManager,
    user_id: str,
    task_id: str,
    queue: str,
) -> dict[str, Any]:
    """Helper function to generate and stream response asynchronously.
    Returns
    -------
    dict[str, Any]
        A dictionary containing the results of the generation and streaming process.
    """
    graph = await graph_builder.abuild_graph()
    user_id = "dummy_user"  # Placeholder for user ID extraction
    input_: dict[str, Any] = {"original_question": message}
    event_count: int = 0

    # Ensure session_id is always a string
    if session_id is None:
        raise ValueError("session_id must be provided and cannot be None.")

    _session_id: str = session_id

    try:
        session_exists: bool = await stream_manager.asession_exists(
            session_id=_session_id
        )
        if not session_exists:
            _session_id = await stream_manager.acreate_session(
                session_id=_session_id, user_id=user_id
            )

        config: RunnableConfig = {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {
                "thread_id": _session_id,
                "user_id": user_id,
            },
        }
        events = graph.astream(
            input_,
            config=config,
            subgraphs=True,
            stream_mode="updates",
        )

        # ------------------ Send session started ------------------
        await stream_manager.asend_event(
            _session_id,
            event_type=EventsType.SESSION_STARTED,
            data={
                "queue": queue,
                "session_id": _session_id,
                "task_id": task_id,
                "query": message,
            },
        )
        event_count += 1

        # Process graph events
        async for event in events:
            event_data: dict[str, Any] = event[1]  # type: ignore
            event_type: str = list(event_data.keys())[0]

            logger.info(f"📤 Streaming event: {event_type}")

            result = json.dumps(
                {
                    "data": format_generated_content(event_type, event_data),
                    "task_id": task_id,
                    "is_streaming": True,
                }
            )
            success = await stream_manager.asend_event(
                session_id=_session_id,
                event_type=EventsType(event_type),
                data=result,
            )
            if success:
                event_count += 1

        # ------------------ Send completion event ------------------
        result = json.dumps(
            {
                "data": {"data": "Generation complete.", "session_id": _session_id},
                "task_id": task_id,
                "is_streaming": False,
                "status": TaskStatusType.SUCCESS,
            }
        )
        await stream_manager.asend_event(
            session_id=_session_id,
            event_type=EventsType.SESSION_ENDED,
            data=result,
        )
        event_count += 1

        return {
            "session_id": _session_id,
            "task_id": task_id,
            "event_count": event_count,
            "status": TaskStatusType.SUCCESS,
        }
    except asyncio.CancelledError:
        logger.warning(
            f"Streaming cancelled for session {_session_id} and task {task_id}"
        )
        await stream_manager.asend_event(
            session_id=_session_id,
            event_type=EventsType.ERROR,
            data={"data": "Streaming was cancelled."},
        )
        raise

    except Exception as e:
        logger.error(
            f"Error during streaming for session {_session_id} and task {task_id}: {e}"
        )
        await stream_manager.asend_event(
            session_id=_session_id,
            event_type=EventsType.ERROR,
            data={"data": f"An error occurred: {str(e)}"},
        )
        raise

    finally:
        pass
