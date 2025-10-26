"""Streamlit app for Smart RAG chat interface with streaming support."""

import json
from typing import Any

import httpx
import streamlit as st

# Configuration
API_BASE_URL: str = "http://localhost:8080"  # FastAPI server URL
CHAT_STREAM_ENDPOINT: str = f"{API_BASE_URL}/api/v1/chat_stream"


def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "checkpoint_id" not in st.session_state:
        st.session_state.checkpoint_id = None


def parse_sse_event(line: str) -> dict[str, Any] | None:
    """Parse a Server-Sent Event line.

    Parameters:
    -----------
    line: str
        The SSE line to parse.

    Returns:
    --------
    dict | None
        Parsed event data or None if invalid.
    """
    if line.startswith("data: "):
        try:
            data = line[6:]  # Remove "data: " prefix
            return json.loads(data)
        except json.JSONDecodeError:
            return None
    return None


async def stream_chat_response(message: str, checkpoint_id: str | None = None) -> None:
    """Stream chat response from the API.

    Parameters:
    -----------
    message: str
        The user message to send.
    checkpoint_id: str | None
        Optional checkpoint ID for conversation continuity.
    """
    # Prepare request parameters
    params: dict[str, str] = {"checkpoint_id": checkpoint_id} if checkpoint_id else {}

    # Display user message
    with st.chat_message("user"):
        st.markdown(message)
    st.session_state.messages.append({"role": "user", "content": message})

    # Create placeholder for assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        sources_placeholder = st.empty()
        full_response: str = ""
        sources: list[str] = []

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "GET",
                    f"{CHAT_STREAM_ENDPOINT}/{message}",
                    params=params,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        event = parse_sse_event(line)
                        if not event:
                            continue

                        event_type = event.get("type")

                        if event_type == "checkpoint":
                            # Store checkpoint ID for conversation continuity
                            st.session_state.checkpoint_id = event.get("checkpoint_id")

                        elif event_type == "search_start":
                            # Indicate search is in progress
                            search_query = event.get("query", "")
                            with sources_placeholder.container():
                                st.info(f"🔍 Searching: {search_query}")

                        elif event_type == "search_results":
                            # Display search sources
                            sources = event.get("urls", [])
                            if sources:
                                with sources_placeholder.container():
                                    st.success(f"✅ Found {len(sources)} sources")
                                    with st.expander("📚 View Sources"):
                                        for idx, url in enumerate(sources, 1):
                                            st.markdown(f"{idx}. [{url}]({url})")

                        elif event_type == "date_result":
                            # Display date result
                            date_result = event.get("result", "")
                            with sources_placeholder.container():
                                st.info(f"📅 Current Date and Time: {date_result}")

                        elif event_type == "content":
                            # Stream content token by token
                            content = event.get("content", "")
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")

                        elif event_type == "end":
                            # Finalize response
                            message_placeholder.markdown(full_response)
                            break

        except httpx.HTTPError as e:
            st.error(f"❌ Error connecting to API: {str(e)}")
            st.error(
                "Make sure the FastAPI server is running at `http://localhost:8000`"
            )
            return
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            return

    # Save assistant response to session state
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": sources if sources else None,
        }
    )


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Smart RAG Chat",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Smart RAG Chat")
    st.caption("Chat with AI powered by agentic RAG")

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.checkpoint_id = None
            st.rerun()

        st.divider()

        st.subheader("About")
        st.markdown(
            """
            This chat interface uses:
            - **LangGraph** for agentic workflows
            - **Tavily Search** for web research
            - **FastAPI** for streaming backend
            - **Streamlit** for the UI
            """
        )

        # Display conversation info
        if st.session_state.checkpoint_id:
            st.divider()
            st.caption(f"Session ID: `{st.session_state.checkpoint_id}`")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for idx, url in enumerate(msg["sources"], 1):
                        st.markdown(f"{idx}. [{url}]({url})")

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        import asyncio

        asyncio.run(
            stream_chat_response(
                prompt,
                checkpoint_id=st.session_state.checkpoint_id,
            )
        )
        st.rerun()


if __name__ == "__main__":
    main()
