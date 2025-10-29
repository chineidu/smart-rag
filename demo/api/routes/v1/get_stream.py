import json
from typing import Any, AsyncGenerator, LiteralString
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage

from demo.api.utilities.utilities import Events
from demo.graph import build_graph

router = APIRouter()
graph = build_graph()


def serialise_ai_message_chunk(
    chunk: AIMessageChunk,
) -> str | list[str | dict[Any, Any]]:
    """Serialise an AIMessageChunk for streaming."""
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    raise TypeError(
        f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
    )


async def generate_chat_responses(
    message: str, checkpoint_id: str | None = None
) -> AsyncGenerator[str | LiteralString, Any]:
    """Generate chat responses as a stream of Server-Sent Events (SSE)."""
    # ==========================================================
    # ==================== New Conversation ====================
    # ==========================================================

    is_new_conversation: bool = checkpoint_id is None

    if is_new_conversation:
        # Generate new checkpoint ID for first message in conversation
        new_checkpoint_id: str = str(uuid4())

        config: dict[str, dict[str, str]] = {
            "configurable": {"thread_id": new_checkpoint_id}
        }

        # Initialize with first message
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config,  # type: ignore
        )

        # First send the checkpoint ID
        yield f"data: {json.dumps({'type': Events.CHECKPOINT, 'checkpoint_id': new_checkpoint_id})}\n\n"

    else:
        config = {
            "configurable": {"thread_id": checkpoint_id}  # type: ignore
        }
        # Continue existing conversation
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config,  # type: ignore
        )

    async for event in events:
        event_type: str = event.get("event", "")

        # Handle different event types
        # ==========================================================
        # ================ Stream AI message chunks ================
        # ==========================================================
        if (
            event_type == "on_chat_model_stream"
            and event.get("metadata", {}).get("langgraph_node") == "llm_call"
        ):
            # `llm_call` node ensures we only stream LLM responses and not other node outputs
            # like tool calls, summarization, etc.
            chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])  # type: ignore
            payload = {"type": Events.CONTENT, "content": chunk_content}
            yield f"data: {json.dumps(payload)}\n\n"

        # ==========================================================
        # =========== Check if model made any tool calls ===========
        # ==========================================================
        # Get the input arguments for the tool calls
        elif event_type == "on_chat_model_end":
            # Check if there are tool calls for search
            tool_calls = (
                event["data"]["output"].tool_calls  # type: ignore
                if hasattr(event["data"]["output"], "tool_calls")  # type: ignore
                else []
            )
            search_calls = [
                call for call in tool_calls if call["name"] == "search_tool"
            ]

            if search_calls:
                # Signal that a search is starting
                search_query: str = search_calls[0]["args"].get("query", "")
                payload = {"type": Events.SEARCH_START, "query": search_query}
                yield f"data: {json.dumps(payload)}\n\n"
        # date_and_time_tool has NO input arguments to extract, so we skip that step

        # ===========================================================
        # ============== Handle tool call completions ===============
        # ===========================================================
        # Get tool completion events
        # Handle search_tool completions
        elif event_type == "on_tool_end" and event["name"] == "tavily_search":
            output = event["data"]["output"]  # type: ignore

            # Extract URLs directly from the Tavily response
            urls = []
            if isinstance(output, dict) and "results" in output:
                # Extract URLs from the results array
                for result in output["results"]:
                    if isinstance(result, dict) and "url" in result:
                        urls.append(result["url"])  # noqa: PERF401
            # Send the URLs if we found any
            if urls:
                urls_json = json.dumps(urls)
                yield f'data: {{"type": {Events.SEARCH_RESULT}, "urls": {urls_json}}}\n\n'

        # Handle date_and_time_tool completions
        elif event_type == "on_tool_end" and event["name"] == "date_and_time_tool":
            output = event["data"]["output"]  # type: ignore
            date_str: str = output.content
            _date: str = date_str.split("T")[0]
            _time: str = date_str.split("T")[1]
            formatted_date = f"{_date} {_time}"
            payload = {"type": Events.DATE_RESULT, "result": formatted_date}
            yield f"data: {json.dumps(payload)}\n\n"

    # ==========================================================
    # =================== Send an end event ====================
    # ==========================================================
    yield f'data: {{"type": {Events.COMPLETION_END}}}\n\n'


@router.get("/chat_stream")
async def chat_stream(
    message: str, checkpoint_id: str | None = None
) -> StreamingResponse:
    """Endpoint to stream chat responses for a given message."""
    return StreamingResponse(
        content=generate_chat_responses(message, checkpoint_id),
        media_type="text/event-stream",
    )
