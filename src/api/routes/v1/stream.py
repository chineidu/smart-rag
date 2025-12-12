import json
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables.config import RunnableConfig

from src import create_logger
from src.api.core.dependencies import get_graph_manager
from src.api.core.ratelimit import limiter
from src.api.core.reponses import MsgSpecJSONResponse
from src.config import app_config
from src.graph import GraphManager
from src.utilities.utils import format_generated_content

logger = create_logger("streaming_response")
router = APIRouter(tags=["streaming"], default_response_class=MsgSpecJSONResponse)
RECURSION_LIMIT = app_config.custom_config.recursion_limit


async def generate_streaming_response(
    message: str,
    session_id: str | None,
    get_current_user: Any,
    graph_manager: GraphManager,  # type: ignore
) -> AsyncGenerator[str, None]:
    """Generate a streaming response for chat messages."""

    graph = await graph_manager.abuild_graph()
    is_new_session = session_id is None
    # Implement user authentication extraction here
    # user_id = get_current_user()
    user_id = "dummy_user"  # Placeholder for user ID extraction
    input_: dict[str, Any] = {"original_question": message}

    if is_new_session:
        session_id = str(uuid4())
        config: RunnableConfig = {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {"thread_id": session_id, "user_id": user_id},
        }
        events = graph.astream(
            input_,
            config=config,
            subgraphs=True,
            stream_mode="updates",
        )

        # Send the session ID first
        yield (
            json.dumps({"event_type": "new_session", "session_id": session_id}) + "\n\n"
        )

    else:
        config: RunnableConfig = {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {"thread_id": session_id, "user_id": user_id},
        }
        events: Any = graph.astream(
            input_,
            config=config,
            subgraphs=True,
            stream_mode="updates",
        )

    # Process graph events
    async for event in events:
        event_data: dict[str, Any] = event[1]  # type: ignore
        event_type: str = list(event_data.keys())[0]
        node_output = event_data.get(event_type, {})

        logger.info(f"📤 Streaming event: {event_type}")

        # For final_answer node, stream the complete answer
        if event_type == "final_answer" and "final_answer" in node_output:
            final_answer = node_output["final_answer"]
            result = json.dumps(
                {
                    "event_type": "final_answer",
                    "data": final_answer,
                    "is_streaming": True,
                }
            )
            yield result + "\n\n"
        else:
            # For other nodes, emit their completion events
            result = json.dumps(
                {
                    "event_type": event_type,
                    "data": format_generated_content(event_type, event_data),
                    "is_streaming": False,
                }
            )
            yield result + "\n\n"

    # ==========================================================
    # =================== Send completion event ====================
    # ==========================================================
    result = json.dumps({"event_type": "done", "data": "Generation complete."})
    yield result + "\n\n"


@router.get("/chat_stream")
@limiter.limit("5/minute")
async def chat_stream(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    message: str,
    graph_manager: GraphManager = Depends(get_graph_manager),
    # current_user: UserWithHashSchema = Depends(get_current_user),
    # langfuse_handler: CallbackHandler = Depends(get_langfuse_handler),
    session_id: str | None = None,
) -> StreamingResponse:
    """Endpoint to stream chat responses for a given message."""

    # Extract user_id from authenticated user
    # user_id = str(current_user.id)

    return StreamingResponse(
        content=generate_streaming_response(
            message,
            session_id=session_id,
            get_current_user="dummy_user",
            graph_manager=graph_manager,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
