import asyncio
import json
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, Request, status
from fastapi.responses import StreamingResponse

from src import create_logger
from src.api.core.exceptions import BaseAPIError
from src.api.core.ratelimit import limiter
from src.config.config import app_config
from src.schemas.routes import QuerySchema

logger = create_logger(name="events_streaming")

router = APIRouter(tags=["streaming"])
END_OF_STREAM: str = "<END_OF_STREAM>"


def format_sse(
    data: list[Any] | dict[str, Any], event: str | None = None, retry: int | None = None
) -> str:
    """Format a string as a Server-Sent Event (SSE)

    Parameters
    ----------
    data : list[Any] | dict[str, Any]
        The data to be sent in the SSE
    event : str | None, optional
        The event type, by default None
    retry : int | None, optional
        The retry interval in milliseconds, by default None

    Returns
    -------
    str
        The formatted SSE string
    """
    msg: str = ""
    if event:
        msg += f"event: {event}\n"

    if retry:
        msg += f"retry: {retry}\n"

    # Support lists/dicts (JSON) and primitive/string payloads
    if isinstance(data, (list, dict)):
        data_str: str = json.dumps(data)
    else:
        data_str = str(data)

    # Per SSE spec, data lines should be prefixed with `data:`. Multi-line
    # payloads are supported but we'll keep it simple and send as a single
    # JSON/string line.
    msg += f"data: {data_str}\n\n"

    return msg


class StreamSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, asyncio.Queue] = {}
        self._maxsize: int = 200
        self._active_connections: dict[
            str, int
        ] = {}  # Track connection count per session

    async def create_session(self) -> tuple[str, asyncio.Queue]:
        session_id: str = str(uuid4())
        queue = asyncio.Queue(self._maxsize)
        self._sessions[session_id] = queue
        self._active_connections[session_id] = 0
        logger.info(
            f"Session {session_id} created. Total sessions: {self.active_sessions}"
        )

        return (session_id, queue)

    def get_queue(self, session_id: str) -> asyncio.Queue | None:
        """Retrieve an existing session's queue by session ID"""
        return self._sessions.get(session_id)

    def add_connection(self, session_id: str) -> None:
        """Increment the connection count for a session"""
        if session_id in self._active_connections:
            self._active_connections[session_id] += 1
            logger.info(
                f"Connection added to session {session_id}. Active connections: {self._active_connections[session_id]}"
            )

    def remove_connection(self, session_id: str) -> None:
        """Decrement connection count"""
        if session_id in self._active_connections:
            self._active_connections[session_id] -= 1
            logger.info(
                f"Connection removed from session {session_id}. Active connections: {self._active_connections[session_id]}"
            )

    def close_session(self, session_id: str) -> None:
        """Close and remove a session by session ID"""
        self._sessions.pop(session_id, None)
        self._active_connections.pop(session_id, None)
        logger.info(
            f"Session {session_id} closed. Total sessions: {self.active_sessions}"
        )

    async def send_to_session(
        self, session_id: str, data: Any, event: str | None = None
    ) -> bool:
        """Send data to a specific session's queue"""
        queue = self.get_queue(session_id)

        if queue:
            message = format_sse(data, event=event)
            await queue.put(message)
            return True

        return False

    @property
    def active_sessions(self) -> int:
        """Get the number of active SSE connections"""
        return len(self._sessions)


# Global instance
session_manager = StreamSessionManager()


# ============================================================================
# SSE Stream Generator
# ============================================================================
async def event_stream_generator(
    request: Request, session_id: str
) -> AsyncGenerator[str, None]:
    """Async generator that yields Server-Sent Events (SSE) from a queue"""
    queue = session_manager.get_queue(session_id)

    if not queue:
        yield format_sse(data={"error": "Invalid session ID"}, event="error")
        return

    # Register this connection
    session_manager.add_connection(session_id)

    try:
        # Send initial connection confirmation
        yield format_sse(
            data={"status": "connected", "session_id": session_id},
            event="connection_established",
            retry=5000,
        )

        while True:
            # If client disconnects, exit the loop
            if await request.is_disconnected():
                logger.info(f"Client disconnected from SSE session {session_id}")
                break

            try:
                # This is where messages are received from the queue
                # Wait for the next message with a timeout so we can send keep-alives
                message: str = await asyncio.wait_for(queue.get(), timeout=15.0)

                # Check for end-of-stream signal
                if message == END_OF_STREAM:
                    yield format_sse({"status": "complete"}, event="end")
                    # Put it back for other connections
                    try:
                        queue.put_nowait(END_OF_STREAM)
                    except asyncio.QueueFull:
                        pass
                    break

                yield message

            except asyncio.TimeoutError:
                # Send a comment to keep the connection alive
                yield ": keep-alive\n\n"

    finally:
        # Decrement connection count
        session_manager.remove_connection(session_id)


@router.post("/run", status_code=status.HTTP_200_OK)
async def run_graph(request: Request, payload: QuerySchema) -> dict[str, Any]:
    """Create a new streaming session and start graph execution."""
    try:
        # Normalize payload to extract the query string
        query: str = payload.query

        if not query:
            raise BaseAPIError(message="Missing 'query' in request body")

        # Create a new session
        session_id, queue = await session_manager.create_session()

        # Trigger graph execution in background
        asyncio.create_task(execute_graph(session_id, query))

        try:
            stream_url = str(request.url_for("sse_stream", session_id=session_id))
        except Exception:
            stream_url = f"{app_config.api_config.prefix}/stream/{session_id}"

        return {"session_id": session_id, "stream_url": stream_url}

    except BaseAPIError:
        raise
    except Exception as e:
        logger.error(f"Failed to create run session: {e}")
        raise BaseAPIError(message="Failed to create run session") from e


@router.get("/stream/{session_id}", status_code=status.HTTP_200_OK)
# @cached(ttl=300, key_prefix="sse")  # type: ignore
@limiter.limit("60/minute")
async def sse_stream(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    session_id: str,
    # cache: Cache = Depends(get_cache),  # Required by caching decorator  # noqa: ARG001
) -> StreamingResponse:
    """Route for Server-Sent Events (SSE) streaming endpoint"""
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Disable proxy buffering
    }
    try:
        return StreamingResponse(
            event_stream_generator(request, session_id),
            media_type="text/event-stream",
            headers=headers,
        )

    except BaseAPIError as e:
        logger.error(f"Error: {e}")

        raise

    except Exception as e:
        logger.error(f"Unexpected error during sse: {e}")
        raise BaseAPIError(
            message="An unexpected error occurred during sse",
        ) from e


# ============================================================================
# LangGraph Integration
# ============================================================================


async def execute_graph(session_id: str, query: str):
    try:
        # Initialize your graph here
        # from langgraph import ...
        # graph = create_your_graph()

        # Stream graph execution
        # Replace this with your actual LangGraph streaming code
        async for event in simulate_graph_stream(query):
            await session_manager.send_to_session(
                session_id,
                event,
                event="node_output",  # Or use event.get("type") for dynamic events
            )

        # Send completion signal to all listening clients
        queue = session_manager.get_queue(session_id)
        if queue:
            await queue.put(END_OF_STREAM)
            logger.info(f"Graph execution completed for session {session_id}")

    except Exception as e:
        # Send error to client
        logger.error(f"Error executing graph for session {session_id}: {e}")
        await session_manager.send_to_session(
            session_id, {"error": str(e)}, event="error"
        )
        # Send end signal so clients know the session is finished
        queue = session_manager.get_queue(session_id)
        if queue:
            await queue.put(END_OF_STREAM)


async def simulate_graph_stream(query: str):
    """
    Simulated graph execution - REPLACE THIS with your actual LangGraph code

    Example with real LangGraph:

        async for event in graph.astream({"input": query}):
            # event will contain node outputs
            yield event
    """
    # Simulate different node outputs
    nodes = ["retriever", "reasoning", "generation"]

    for node in nodes:
        await asyncio.sleep(1)  # Simulate processing time

        yield {
            "node": node,
            "output": f"Output from {node} node processing: {query}",
            "status": "running",
        }

    # Final result
    yield {
        "node": "final",
        "output": f"Final answer for: {query}",
        "status": "complete",
    }
