"""API routes for streaming chat responses. Runs in the background via Celery tasks."""

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.param_functions import Depends
from fastapi.responses import StreamingResponse
from kombu.exceptions import OperationalError

from src import create_logger
from src.api.core.auth import get_current_active_user
from src.api.core.exceptions import (
    HTTPError,
    StreamingError,
    UnauthorizedError,
    UnexpectedError,
)
from src.api.core.ratelimit import limiter
from src.api.core.reponses import MsgSpecJSONResponse
from src.celery_app.tasks.prediction import generate_streaming_response_task
from src.config import app_config
from src.schemas.routes.streamer_schema import SessionResponse, UserSessions
from src.schemas.types import EventsType, RoleType
from src.schemas.user_schema import UserWithHashSchema
from src.stream_manager import StreamSessionManager

logger = create_logger("streaming_response")
router = APIRouter(tags=["streaming"], default_response_class=MsgSpecJSONResponse)
RECURSION_LIMIT = app_config.custom_config.recursion_limit
LIMIT_VALUE: int = app_config.api_config.ratelimit.burst_rate

session_manager = StreamSessionManager()


def format_sse(data: str, event: str | None = None, retry: int | None = None) -> str:
    """Format data as Server-Sent Event"""
    message = ""
    if event:
        message += f"event: {event}\n"
    if retry:
        message += f"retry: {retry}\n"

    # Handle multiline data
    for line in data.split("\n"):
        message += f"data: {line}\n"

    message += "\n"
    return message


@router.get("/sessions")
@limiter.limit(f"{LIMIT_VALUE}/minute")
async def create_session(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    user_id: str | None = None,
    current_user: UserWithHashSchema = Depends(get_current_active_user),
) -> SessionResponse:
    """Create a new streaming session.

    Parameters
    ----------
        user_id : str | None, optional
            The user ID to associate with this session for conversation history tracking.
    """
    try:
        if not current_user:
            raise UnauthorizedError("User must be authenticated to create a session.")

        session_id: str = await session_manager.acreate_session(user_id=user_id)
        logger.info(
            f"Created new session with ID: {session_id}"
            + (f" for user {user_id}" if user_id else "")
        )
        return SessionResponse(
            task_id=None, session_id=session_id, message="Session created successfully"
        )

    except Exception as e:
        logger.error(f"Error creating session: {e}", exc_info=True)
        raise UnexpectedError("Error occurred while creating the session.") from e


@router.get("/users/{user_id}/sessions")
@limiter.limit(f"{LIMIT_VALUE}/minute")
async def list_user_sessions(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    user_id: str,
    current_user: UserWithHashSchema = Depends(get_current_active_user),
) -> UserSessions:
    """List all sessions for a user with metadata.

    Parameters
    ----------
        user_id : str
            The unique identifier for the user.

    Returns
    -------
        UserSessions
            List of session objects with metadata sorted by creation time (newest first).
    """
    try:
        if not current_user:
            raise UnauthorizedError("User must be authenticated to create a session.")

        sessions = await session_manager.aget_user_sessions(user_id=user_id)
        logger.info(f"Retrieved {len(sessions)} sessions for user {user_id}")
        return UserSessions(user_id=user_id, sessions=sessions, count=len(sessions))

    except Exception as e:
        logger.error(f"Error listing user sessions: {e}", exc_info=True)
        raise UnexpectedError("Error occurred while listing user the sessions.") from e


@router.post("/sessions/{session_id}/query")
@limiter.limit(f"{LIMIT_VALUE}/minute")
async def submit_query(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    message: str,
    user_id: str,
    session_id: str,
    role: RoleType | None = RoleType.GUEST,
    current_user: UserWithHashSchema = Depends(get_current_active_user),
) -> SessionResponse:
    """Trigger a Celery task to generate a streaming response."""
    try:
        if not current_user:
            raise UnauthorizedError("User must be authenticated to create a session.")

        # Route task to queue based on role
        # Default to GUEST if not provided
        selected_role = role or RoleType.GUEST
        if selected_role in (RoleType.ADMIN, RoleType.USER):
            target_queue = app_config.queues_config.high_priority_ml
        else:
            target_queue = app_config.queues_config.low_priority_ml

        try:
            task = generate_streaming_response_task.apply_async(  # type: ignore
                kwargs={
                    "message": message,
                    "session_id": session_id,
                    "user_id": user_id,
                },
                # Trigger immediately (no countdown)
                countdown=0,
                queue=target_queue,
            )
            return SessionResponse(
                task_id=task.id,
                session_id=session_id,
                message=f"Published to {target_queue} queue successfully.",
            )
        except OperationalError:
            logger.error(f"Failed to publish task to queue {target_queue}")
            raise HTTPError(
                details="Task queue service is unavailable. Please ensure the message broker is running.",
            ) from None

    except OperationalError as e:
        logger.error(f"Celery broker connection failed: {e}")
        raise HTTPError(
            details="Task queue service is unavailable. Please ensure the message broker is running.",
        ) from e


async def events_generator(
    request: Request, session_id: str, last_id: str
) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events for streaming responses."""
    stop_on_session_end: bool = last_id == "$"

    # ---- Send initial connection event ----
    try:
        yield format_sse(
            data=f"{{'event_type': 'connected', 'session_id':{session_id} ,"
            f"'message': 'Connection established successfully.'}}",
            event="connected",
            retry=3000,
        )

        async for event in session_manager.aread_events(
            session_id=session_id,
            last_id=last_id,
            # For replays (last_id != "$"), keep reading past SESSION_ENDED to return full history.
            stop_on_session_end=(last_id == "$"),
        ):
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info(f"Client disconnected from session {session_id}")
                break

            # Format and yield event
            event_type = event.get("event_type")
            message_id = event.get("id")  # Redis message ID

            # Format data based on event type
            if event_type == EventsType.KEEPALIVE:
                yield format_sse(
                    data='{"type": "keepalive"}', event=EventsType.KEEPALIVE
                )
            else:
                yield format_sse(
                    data=json.dumps({**event, "message_id": message_id}),
                    event=event_type,
                )

            # If done, close stream
            if stop_on_session_end and event_type == EventsType.SESSION_ENDED:
                logger.info(f"Stream completed for session {session_id}")
                break

    except Exception as e:
        logger.error(f"Error streaming to session {session_id}: {e}", exc_info=True)
        yield format_sse(data=f'{{"error": "{str(e)}"}}', event="error")

    finally:
        # Clean up connection
        session_manager.remove_connection(session_id)
        logger.info(f"Connection closed for session {session_id}")


@router.get("/sessions/{session_id}/stream")
@limiter.limit(f"{LIMIT_VALUE}/minute")
async def chat_stream(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    session_id: str,
    last_id: str = "$",  # Default to $ for new events only (continue mode)
    current_user: UserWithHashSchema = Depends(get_current_active_user),
) -> StreamingResponse:
    """Endpoint to stream chat responses for a given session.

    Parameters
    ----------
        session_id : str
            The unique identifier for the session.
        last_id : str, optional
            The Redis stream ID to start reading from:
            - "0": Replay all events from the beginning
            - "$": Stream only new events (continue conversation)
            - "{message_id}": Start from a specific message ID
    """
    try:
        if not current_user:
            raise UnauthorizedError("User must be authenticated to create a session.")

        # Validate session exists (only stream existing sessions)
        if not await session_manager.asession_exists(session_id=session_id):
            logger.warning(f"Stream requested for unknown session {session_id!r}")
            raise HTTPError(details="Session not found")

        # Register connection and stream
        session_manager.add_connection(session_id)

        return StreamingResponse(
            content=events_generator(
                request,
                session_id=session_id,
                last_id=last_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    except Exception as e:
        logger.error(f"Error streaming to session {session_id}: {e}", exc_info=True)
        raise StreamingError(details=f"Error streaming to session {session_id}") from e
