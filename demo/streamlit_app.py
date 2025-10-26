"""Streamlit app for Smart RAG chat interface with streaming support and enhanced UI."""

import json
from typing import Any

import httpx
import streamlit as st

# Configuration
API_BASE_URL: str = "http://localhost:8080"
CHAT_STREAM_ENDPOINT: str = f"{API_BASE_URL}/api/v1/chat_stream"


def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "checkpoint_id" not in st.session_state:
        st.session_state.checkpoint_id = None
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0


def parse_sse_event(line: str) -> dict[str, Any] | None:
    """Parse a Server-Sent Event line."""
    if line.startswith("data: "):
        try:
            data = line[6:]
            return json.loads(data)
        except json.JSONDecodeError:
            return None
    return None


def render_message_with_avatar(
    role: str, content: str, sources: list[str] | None = None
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
                    {content}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if sources:
                with st.expander(
                    f"📚 {len(sources)} Sources Referenced", expanded=False
                ):
                    for idx, url in enumerate(sources, 1):
                        domain = url.split("/")[2] if len(url.split("/")) > 2 else url
                        st.markdown(
                            f"""
                            <div style="padding: 0.5rem; margin: 0.25rem 0; background: white;
                                        border-left: 3px solid #667eea; border-radius: 4px;">
                                <strong>{idx}.</strong> <a href="{url}" target="_blank"
                                style="text-decoration: none; color: #667eea;">
                                {domain}</a>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


async def stream_chat_response(message: str, checkpoint_id: str | None = None) -> None:
    """Stream chat response from the API with enhanced UI feedback."""
    params: dict[str, str] = {"checkpoint_id": checkpoint_id} if checkpoint_id else {}

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
                            st.session_state.checkpoint_id = event.get("checkpoint_id")

                        elif event_type == "search_start":
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

                        elif event_type == "search_results":
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

                        elif event_type == "date_result":
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

                        elif event_type == "content":
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
                                        {full_response}<span style="animation: blink 1s infinite;">▌</span>
                                    </div>
                                    <style>
                                    @keyframes blink {{
                                        0%, 50% {{ opacity: 1; }}
                                        51%, 100% {{ opacity: 0; }}
                                    }}
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        elif event_type == "end":
                            status_container.empty()
                            with message_placeholder:
                                st.markdown(
                                    f"""
                                    <div style="background: #f8f9fa;
                                                padding: 1rem 1.5rem;
                                                border-radius: 18px 18px 18px 4px;
                                                border: 1px solid #e9ecef;
                                                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                                        {full_response}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            if sources:
                                with sources_container:
                                    with st.expander(
                                        f"📚 {len(sources)} Sources Referenced",
                                        expanded=False,
                                    ):
                                        for idx, url in enumerate(sources, 1):
                                            domain = (
                                                url.split("/")[2]
                                                if len(url.split("/")) > 2
                                                else url
                                            )
                                            st.markdown(
                                                f"""
                                                <div style="padding: 0.5rem; margin: 0.25rem 0; background: white;
                                                            border-left: 3px solid #667eea; border-radius: 4px;">
                                                    <strong>{idx}.</strong> <a href="{url}" target="_blank"
                                                    style="text-decoration: none; color: #667eea;">
                                                    {domain}</a>
                                                </div>
                                                """,
                                                unsafe_allow_html=True,
                                            )
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

            **⚡ FastAPI**
            High-performance streaming

            **🎨 Streamlit**
            Beautiful UI interface
            """
        )

        if st.session_state.checkpoint_id:
            st.markdown("---")
            st.markdown("## 🔑 Session Info")
            st.code(st.session_state.checkpoint_id[:16] + "...", language=None)

    # Main chat area
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
            for msg in st.session_state.messages:
                render_message_with_avatar(
                    msg["role"], msg["content"], msg.get("sources")
                )

    # Chat input
    if prompt := st.chat_input("💬 Type your message here...", key="chat_input"):
        import asyncio

        asyncio.run(
            stream_chat_response(prompt, checkpoint_id=st.session_state.checkpoint_id)
        )
        st.rerun()


if __name__ == "__main__":
    main()
