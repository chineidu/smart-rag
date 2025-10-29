"""API route for retrieving conversation history from a checkpoint."""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from langchain_core.messages import BaseMessage

from demo import create_logger
from demo.graph import build_graph

router = APIRouter(tags=["history"])
graph = build_graph()
logger = create_logger(name="get_history_api")


@router.get("/chat_history")
async def get_chat_history(checkpoint_id: str) -> dict[str, Any]:
    """
    Retrieve the conversation history for a given checkpoint ID.

    Parameters
    -----------
    checkpoint_id:
        The checkpoint ID to retrieve history for

    Returns
    --------
    dict[str, Any]:
        Dictionary containing the conversation history and metadata
    """
    try:
        config: dict[str, Any] = {"configurable": {"thread_id": checkpoint_id}}

        # Get the state from the checkpoint
        state = graph.get_state(config)  # type: ignore

        if not state or not state.values or not state.values.get("messages"):
            logger.error(f"Checkpoint '{checkpoint_id}' not found or has no messages")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint '{checkpoint_id}' not found or has no messages",
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
            f"Retrieved {len(formatted_messages)} messages from checkpoint '{checkpoint_id}'"
        )

        return {
            "checkpoint_id": checkpoint_id,
            "messages": formatted_messages,
            "message_count": len(formatted_messages),
        }

    except HTTPException:
        logger.error("HTTP error occurred")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving checkpoint: {str(e)}"
        ) from e
