from aiocache import Cache
from fastapi import APIRouter, Depends, HTTPException, Request, status
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig

from src import create_logger
from src.api.core.cache import cached
from src.api.core.dependencies import get_cache, get_graph_manager
from src.api.core.ratelimit import limiter
from src.api.core.reponses import MsgSpecJSONResponse
from src.config import app_settings
from src.graph import GraphManager
from src.schemas.routes.history_schema import ChatHistorySchema

logger = create_logger(name="status_route")
router = APIRouter(tags=["history"], default_response_class=MsgSpecJSONResponse)
LIMIT_VALUE: int = app_settings.LIMIT_VALUE


@router.get("/chat_history", status_code=status.HTTP_200_OK)
@limiter.limit(f"{LIMIT_VALUE}/minute")
@cached(ttl=60, key_prefix="chat_history")  # type: ignore
async def get_chat_history(
    request: Request,  # Required by SlowAPI  # noqa: ARG001
    session_id: str,
    graph_manager: GraphManager = Depends(get_graph_manager),
    cache: Cache = Depends(get_cache),  # noqa: ARG001
) -> ChatHistorySchema:
    """
    Retrieve the conversation history for a given checkpoint ID.

    Parameters
    ----------
    session_id:
        The checkpoint ID to retrieve history for

    Returns
    -------
    ChatHistorySchema
        The chat history including messages and message count
    """
    try:
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        graph = await graph_manager.abuild_graph()

        # Get the state from the checkpoint
        state = await graph.aget_state(config)

        if not state or not state.values or not state.values.get("messages"):
            logger.error(f"Session '{session_id}' not found or has no messages")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session '{session_id}' not found or has no messages",
            )

        # Extract messages from state
        messages: list[BaseMessage] = state.values.get("messages", [])

        # Convert messages to a serializable format
        formatted_messages: list[dict[str, str]] = []
        for msg in messages:
            if hasattr(msg, "type"):
                msg_type = msg.type
            else:
                msg_type = msg.__class__.__name__.replace("Message", "").lower()

            formatted_messages.append({"role": msg_type, "content": msg.content})  # type: ignore

        logger.info(
            f"Retrieved {len(formatted_messages)} messages from checkpoint '{session_id}'"
        )

        return ChatHistorySchema(
            session_id=session_id,
            messages=formatted_messages,
            message_count=len(formatted_messages),
        ).model_dump(by_alias=True)

    except HTTPException:
        logger.error("HTTP error occurred")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving session_id: {str(e)}"
        ) from e
