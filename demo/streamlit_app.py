"""Streamlit app for Smart RAG chat interface with streaming support and enhanced UI."""

import asyncio
import json
import re
from typing import Any

import httpx
import streamlit as st

from demo.api.utilities.utilities import Events, Feedback

# Configuration
API_BASE_URL: str = "http://localhost:8080"
CHAT_STREAM_ENDPOINT: str = f"{API_BASE_URL}/api/v1/chat_stream"
FEEDBACK_ENDPOINT: str = f"{API_BASE_URL}/api/v1/feedback"
CHAT_HISTORY_ENDPOINT: str = f"{API_BASE_URL}/api/v1/chat_history"


def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "checkpoint_id" not in st.session_state:
        st.session_state.checkpoint_id = None
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}


def parse_sse_event(line: str) -> dict[str, Any] | None:
    """Parse a Server-Sent Event line."""
    if line.startswith("data: "):
        try:
            data: str = line[6:]
            return json.loads(data)

        except json.JSONDecodeError:
            return None

    return None


def markdown_to_html(text: str) -> str:
    """Convert markdown links to HTML links."""
    # Convert markdown links [text](url) to HTML <a> tags
    pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
    html_text = re.sub(pattern, r'<a href="\2" target="_blank">\1</a>', text)

    # Convert newlines to <br> tags
    return html_text.replace("\n", "<br>")


async def send_feedback_to_api(message_index: int, feedback_type: str | None) -> None:
    """Send feedback data to FastAPI endpoint."""
    try:
        # Get the message data
        if message_index >= len(st.session_state.messages):
            st.toast("⚠️ Invalid message index", icon="⚠️")
            return

        message_data = st.session_state.messages[message_index]

        # Ensure this is an assistant message
        if message_data["role"] != "assistant":
            st.toast("⚠️ Can only provide feedback on assistant messages", icon="⚠️")
            return

        # Get the corresponding user message (should be the one before)
        user_message = ""
        if (
            message_index > 0
            and st.session_state.messages[message_index - 1]["role"] == "user"
        ):
            user_message = st.session_state.messages[message_index - 1]["content"]

        payload: dict[str, Any] = {
            "session_id": st.session_state.checkpoint_id
            if st.session_state.checkpoint_id
            else "no_session",
            "message_index": message_index,
            "user_message": user_message,
            "assistant_message": message_data["content"],
            "sources": message_data.get("sources")
            if message_data.get("sources")
            else [],
            "feedback": feedback_type,
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(FEEDBACK_ENDPOINT, json=payload)
            response.raise_for_status()
            st.toast("✅ Feedback saved!", icon="✅")

    except httpx.HTTPStatusError as e:
        error_detail = e.response.text
        print(f"HTTP Error {e.response.status_code}: {error_detail}")
        st.toast(f"⚠️ Server error: {e.response.status_code}", icon="⚠️")
    except httpx.HTTPError as e:
        print(f"HTTP Error: {str(e)}")
        st.toast(f"⚠️ Failed to save feedback: {str(e)}", icon="⚠️")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        st.toast(f"⚠️ Error: {str(e)}", icon="⚠️")


async def load_chat_history(checkpoint_id: str) -> bool:
    """Load chat history from a checkpoint ID.

    Parameters
    -----------
    checkpoint_id:
        The checkpoint ID to load history from

    Returns
    --------
    bool
        True if successful, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                CHAT_HISTORY_ENDPOINT, params={"checkpoint_id": checkpoint_id}
            )
            response.raise_for_status()
            data = response.json()

            # Convert the loaded messages to the format used in the app
            loaded_messages = []
            for msg in data.get("messages", []):
                # Map message types (human -> user, ai -> assistant)
                role = msg["role"]
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"

                loaded_messages.append(
                    {"role": role, "content": msg["content"], "sources": None}
                )

            # Update session state
            st.session_state.messages = loaded_messages
            st.session_state.checkpoint_id = checkpoint_id
            st.session_state.message_count = len(
                [m for m in loaded_messages if m["role"] == "assistant"]
            )
            st.session_state.feedback = {}

            return True

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            st.toast(f"⚠️ Checkpoint '{checkpoint_id}' not found", icon="⚠️")
        else:
            st.toast(f"⚠️ Server error: {e.response.status_code}", icon="⚠️")
        return False
    except httpx.HTTPError as e:
        st.toast(f"⚠️ Connection error: {str(e)}", icon="⚠️")
        return False
    except Exception as e:
        st.toast(f"⚠️ Error loading checkpoint: {str(e)}", icon="⚠️")
        return False


def render_sources_section(sources: list[str]) -> None:
    """Render a collapsible sources section with enhanced styling."""
    if not sources:
        return

    with st.expander(
        f"📚 **{len(sources)} Source{'' if len(sources) == 1 else 's'} Referenced**",
        expanded=False,
    ):
        st.markdown(
            """
            <style>
            .source-item {
                padding: 0.75rem 1rem;
                margin: 0.5rem 0;
                background: white;
                border-left: 4px solid #667eea;
                border-radius: 6px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                transition: all 0.2s ease;
            }
            .source-item:hover {
                box-shadow: 0 3px 8px rgba(102, 126, 234, 0.2);
                transform: translateX(4px);
            }
            .source-number {
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                width: 28px;
                height: 28px;
                line-height: 28px;
                text-align: center;
                border-radius: 50%;
                font-weight: bold;
                font-size: 0.85rem;
                margin-right: 0.75rem;
            }
            .source-link {
                text-decoration: none;
                color: #667eea;
                font-weight: 500;
                transition: color 0.2s ease;
            }
            .source-link:hover {
                color: #764ba2;
                text-decoration: underline;
            }
            .source-domain {
                color: #6c757d;
                font-size: 0.85rem;
                margin-left: 2.5rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        for idx, url in enumerate(sources, 1):
            # Extract domain and path for display
            try:
                parts = url.split("/")
                domain = parts[2] if len(parts) > 2 else url
                path = "/" + "/".join(parts[3:]) if len(parts) > 3 else ""
                display_path = (path[:50] + "...") if len(path) > 50 else path
            except Exception:
                domain = url
                display_path = ""

            st.markdown(
                f"""
                <div class="source-item">
                    <span class="source-number">{idx}</span>
                    <a href="{url}" target="_blank" class="source-link">
                        {domain}
                    </a>
                    {f'<div class="source-domain">{display_path}</div>' if display_path else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_feedback_buttons(message_index: int) -> None:
    """Render feedback buttons for a message."""
    feedback_key = f"msg_{message_index}"
    current_feedback = st.session_state.feedback.get(feedback_key)

    col1, col2, col3 = st.columns([0.1, 0.1, 0.8])

    with col1:
        if st.button(
            "👍" if current_feedback != Feedback.POSITIVE else "✅",
            key=f"thumbs_up_{message_index}",
            help="Helpful response",
            use_container_width=True,
        ):
            # Toggle positive feedback: if already positive, remove it; otherwise, set to positive
            if current_feedback == Feedback.POSITIVE:
                new_feedback = Feedback.NEUTRAL.value  # Remove/clear feedback
            else:
                new_feedback = Feedback.POSITIVE  # Set positive feedback

            st.session_state.feedback[feedback_key] = new_feedback

            # Send feedback to API
            asyncio.run(send_feedback_to_api(message_index, new_feedback))
            st.rerun()

    with col2:
        if st.button(
            "👎" if current_feedback != Feedback.NEGATIVE else "❌",
            key=f"thumbs_down_{message_index}",
            help="Not helpful",
            use_container_width=True,
        ):
            # Toggle negative feedback: if already negative, remove it; otherwise, set to negative
            if current_feedback == Feedback.NEGATIVE:
                new_feedback = Feedback.NEUTRAL.value  # Remove/clear feedback
            else:
                new_feedback = Feedback.NEGATIVE  # Set negative feedback
            st.session_state.feedback[feedback_key] = new_feedback

            # Send feedback to API
            asyncio.run(send_feedback_to_api(message_index, new_feedback))
            st.rerun()


def render_message_with_avatar(
    role: str,
    content: str,
    sources: list[str] | None = None,
    message_index: int | None = None,
) -> None:
    """Render a message with custom styling and avatar."""
    if role == "user":
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 1rem 1.5rem;
                            border-radius: 18px 18px 4px 18px;
                            margin: 0.5rem 0;
                            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);">
                    {content}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown("### 👤")
    else:
        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            st.markdown("### 🤖")
        with col2:
            st.markdown(
                f"""
                <div style="background: #f8f9fa;
                            color: #212529;
                            padding: 1rem 1.5rem;
                            border-radius: 18px 18px 18px 4px;
                            margin: 0.5rem 0;
                            border: 1px solid #e9ecef;
                            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                    {markdown_to_html(content)}
                </div>
                <style>
                div[style*="background: #f8f9fa"] a {{
                    color: #667eea;
                    text-decoration: none;
                    font-weight: 500;
                    transition: color 0.2s ease;
                }}
                div[style*="background: #f8f9fa"] a:hover {{
                    color: #764ba2;
                    text-decoration: underline;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

            if sources:
                render_sources_section(sources)

            # Add feedback buttons for assistant messages
            if message_index is not None:
                render_feedback_buttons(message_index)


async def stream_chat_response(message: str, checkpoint_id: str | None = None) -> None:
    """Stream chat response from the API with enhanced UI feedback."""

    # Build query parameters including the message
    params: dict[str, str] = {"message": message}
    if checkpoint_id:
        params["checkpoint_id"] = checkpoint_id

    # Add user message with custom rendering
    st.session_state.messages.append({"role": "user", "content": message})
    render_message_with_avatar("user", message)

    # Create assistant response container
    col1, col2 = st.columns([0.15, 0.85])
    with col1:
        st.markdown("### 🤖")

    with col2:
        status_container = st.empty()
        message_placeholder = st.empty()
        sources_container = st.empty()

        full_response: str = ""
        sources: list[str] = []
        search_query: str = ""

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                async with client.stream(
                    "GET", CHAT_STREAM_ENDPOINT, params=params
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        event = parse_sse_event(line)
                        if not event:
                            continue

                        event_type = event.get("type")

                        if event_type == Events.CHECKPOINT:
                            st.session_state.checkpoint_id = event.get("checkpoint_id")

                        elif event_type == Events.SEARCH_START:
                            search_query = event.get("query", "")
                            with status_container:
                                st.markdown(
                                    f"""
                                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                                color: white;
                                                padding: 0.75rem 1rem;
                                                border-radius: 8px;
                                                margin-bottom: 1rem;
                                                display: flex;
                                                align-items: center;
                                                animation: pulse 2s infinite;">
                                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">🔍</span>
                                        <span>Searching: <strong>{search_query}</strong></span>
                                    </div>
                                    <style>
                                    @keyframes pulse {{
                                        0%, 100% {{ opacity: 1; }}
                                        50% {{ opacity: 0.8; }}
                                    }}
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        elif event_type == Events.SEARCH_RESULT:
                            sources = event.get("urls", [])
                            if sources:
                                with status_container:
                                    st.markdown(
                                        f"""
                                        <div style="background: #d4edda;
                                                    color: #155724;
                                                    padding: 0.75rem 1rem;
                                                    border-radius: 8px;
                                                    margin-bottom: 1rem;
                                                    border-left: 4px solid #28a745;">
                                            <span style="font-size: 1.2rem; margin-right: 0.5rem;">✅</span>
                                            <span>Found <strong>{len(sources)}</strong> relevant sources</span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                        elif event_type == Events.DATE_RESULT:
                            date_result = event.get("result", "")
                            with status_container:
                                st.markdown(
                                    f"""
                                    <div style="background: #cce5ff;
                                                color: #004085;
                                                padding: 0.75rem 1rem;
                                                border-radius: 8px;
                                                margin-bottom: 1rem;
                                                border-left: 4px solid #007bff;">
                                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">📅</span>
                                        <span><strong>{date_result}</strong></span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        elif event_type == Events.CONTENT:
                            content = event.get("content", "")
                            full_response += content
                            with message_placeholder:
                                st.markdown(
                                    f"""
                                    <div style="background: #f8f9fa;
                                                color: #212529;
                                                padding: 1rem 1.5rem;
                                                border-radius: 18px 18px 18px 4px;
                                                border: 1px solid #e9ecef;
                                                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                                        {markdown_to_html(full_response)}<span style="animation: blink 1s infinite;">▌</span>
                                    </div>
                                    <style>
                                    @keyframes blink {{
                                        0%, 50% {{ opacity: 1; }}
                                        51%, 100% {{ opacity: 0; }}
                                    }}
                                    div[style*="background: #f8f9fa"] a {{
                                        color: #667eea;
                                        text-decoration: none;
                                        font-weight: 500;
                                    }}
                                    div[style*="background: #f8f9fa"] a:hover {{
                                        color: #764ba2;
                                        text-decoration: underline;
                                    }}
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        elif event_type == Events.COMPLETION_END:
                            status_container.empty()
                            with message_placeholder:
                                st.markdown(
                                    f"""
                                    <div style="background: #f8f9fa;
                                                padding: 1rem 1.5rem;
                                                border-radius: 18px 18px 18px 4px;
                                                border: 1px solid #e9ecef;
                                                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                                        {markdown_to_html(full_response)}
                                    </div>
                                    <style>
                                    div[style*="background: #f8f9fa"] a {{
                                        color: #667eea;
                                        text-decoration: none;
                                        font-weight: 500;
                                    }}
                                    div[style*="background: #f8f9fa"] a:hover {{
                                        color: #764ba2;
                                        text-decoration: underline;
                                    }}
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            if sources:
                                with sources_container:
                                    render_sources_section(sources)
                            break

        except httpx.HTTPError as e:
            st.error(f"❌ Connection Error: {str(e)}")
            st.info(
                "💡 Make sure the FastAPI server is running at `http://localhost:8080`"
            )
            return
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")
            return

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": sources if sources else None,
        }
    )
    st.session_state.message_count += 1


def main() -> None:
    """Main Streamlit app with enhanced UI."""
    st.set_page_config(
        page_title="AI Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(to bottom, #ffffff, #f8f9fa);
        }
        .stChatMessage {
            background-color: transparent !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(
            """
            <h1 style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 3rem; margin-bottom: 0;">
                🤖 AI Chat Assistant
            </h1>
            <p style="text-align: center; color: #6c757d; margin-top: 0;">
                Powered by Agentic RAG • Real-time Web Search • Smart Responses
            </p>
            """,
            unsafe_allow_html=True,
        )

    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.checkpoint_id = None
            st.session_state.message_count = 0
            st.session_state.feedback = {}
            st.rerun()

        st.markdown("---")

        # Continue from previous chat
        st.markdown("## 🔄 Continue Previous Chat")
        checkpoint_input = st.text_input(
            "Enter Checkpoint ID",
            placeholder="Paste checkpoint ID here...",
            help="Enter a previous checkpoint ID to continue that conversation",
        )
        if st.button(
            "📥 Load Checkpoint",
            use_container_width=True,
            disabled=not checkpoint_input,
        ):
            if checkpoint_input:
                with st.spinner("Loading checkpoint..."):
                    success = asyncio.run(load_chat_history(checkpoint_input))
                    if success:
                        st.success(
                            f"✅ Loaded checkpoint with {st.session_state.message_count} messages"
                        )
                        st.rerun()

        st.markdown("---")

        # Stats
        st.markdown("## 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", st.session_state.message_count)
        with col2:
            status = "🟢 Active" if st.session_state.checkpoint_id else "⚪ New"
            st.metric("Status", status)

        # Feedback stats
        if st.session_state.feedback:
            positive = sum(
                1 for v in st.session_state.feedback.values() if v == "positive"
            )
            negative = sum(
                1 for v in st.session_state.feedback.values() if v == "negative"
            )
            total = positive + negative

            st.markdown("### 💭 Feedback")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("👍 Helpful", positive)
            with col2:
                st.metric("👎 Not Helpful", negative)

            if total > 0:
                satisfaction = (positive / total) * 100
                st.progress(satisfaction / 100)
                st.caption(f"Satisfaction: {satisfaction:.0f}%")

        st.markdown("---")

        # About
        st.markdown("## 💡 About")
        st.markdown(
            """
            This AI assistant uses:

            **🧠 LangGraph**
            Agentic workflow orchestration

            **🔍 Tavily Search**
            Real-time web research

            **🗓️ Date Tool**
            Date and time manipulation

            **⚡ FastAPI**
            High-performance streaming

            **🎨 Streamlit**
            Beautiful UI interface
            """
        )

        if st.session_state.checkpoint_id:
            st.markdown("---")
            st.markdown("## 🔑 Checkpoint ID")

            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.code(st.session_state.checkpoint_id, language=None)
            with col2:
                if st.button(
                    "📋",
                    key="copy_checkpoint",
                    help="Copy to clipboard",
                    use_container_width=True,
                ):
                    st.toast("✅ Copied to clipboard!", icon="✅")

            st.caption(
                "💡 Copy this ID to resume your conversation later"
            )  # Main chat area
    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            # Welcome message
            st.markdown(
                """
                <div style="text-align: center; padding: 3rem 1rem; color: #6c757d;">
                    <h2 style="color: #495057;">👋 Welcome! How can I help you today?</h2>
                    <p style="font-size: 1.1rem; margin-top: 1rem;">
                        Ask me anything! I can search the web, provide detailed answers, and more.
                    </p>
                    <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 2rem; flex-wrap: wrap;">
                        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 200px;">
                            💡 Explain concepts
                        </div>
                        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 200px;">
                            🔍 Research topics
                        </div>
                        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 200px;">
                            📊 Analyze data
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Display message history
            assistant_message_count = 0
            for idx, msg in enumerate(st.session_state.messages):
                # Track assistant messages separately for feedback
                if msg["role"] == "assistant":
                    message_index = idx
                    assistant_message_count += 1
                else:
                    message_index = None

                render_message_with_avatar(
                    msg["role"], msg["content"], msg.get("sources"), message_index
                )

    # Chat input
    if prompt := st.chat_input("💬 Type your message here...", key="chat_input"):
        asyncio.run(
            stream_chat_response(prompt, checkpoint_id=st.session_state.checkpoint_id)
        )
        st.rerun()


if __name__ == "__main__":
    main()
