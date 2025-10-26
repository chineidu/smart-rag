# Smart RAG Demo

This directory contains the demo application with both FastAPI backend and Streamlit frontend.

## Running the Application

### 1. Start the FastAPI Server

```bash
# From the project root
uv run -m demo.api.app
```

The API will be available at `http://localhost:8000`

### 2. Start the Streamlit App

In a new terminal:

```bash
# From the project root
uv run streamlit run demo/streamlit_app.py
```

The Streamlit app will open in your browser at `http://localhost:8501`

## Features

- 💬 **Real-time streaming**: See responses as they're generated
- 🔍 **Search integration**: Watch as the AI searches the web for information
- 📚 **Source citations**: View the sources used for each response
- 💾 **Conversation memory**: Continue conversations across messages
- 🎨 **Clean UI**: Modern chat interface with Streamlit

## Architecture

```
User → Streamlit UI → FastAPI Backend → LangGraph Agent → Tavily Search
                                              ↓
                                         LLM (OpenRouter)
```

## API Endpoints

- `GET /chat_stream/{message}?checkpoint_id={id}`: Stream chat responses

## Environment Variables

Make sure your `.env` file contains:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_URL=https://openrouter.ai/api/v1
TAVILY_API_KEY=your_key_here
```
