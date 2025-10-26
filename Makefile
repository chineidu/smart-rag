.PHONY: help install api-run streamlit demo clean test lint format

help:
	@echo "Smart RAG - Available commands:"
	@echo ""
	@echo "  make install    - Install all dependencies"
	@echo "  make api-run    - Run FastAPI server"
	@echo "  make streamlit  - Run Streamlit app"
	@echo "  make demo       - Run both API and Streamlit (requires tmux)"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter (ruff)"
	@echo "  make format     - Format code (ruff)"
	@echo "  make clean      - Clean up cache and temporary files"
	@echo ""

install:
	@echo "📦 Installing dependencies..."
	uv sync

api-run:
	@echo "🚀 Starting FastAPI server on http://localhost:8080"
	uv run -m demo.api.app

streamlit:
	@echo "🎨 Starting Streamlit app on http://localhost:8501"
	uv run streamlit run demo/streamlit_app.py

demo:
	@echo "🚀 Starting both FastAPI and Streamlit..."
	@echo "📡 FastAPI will run on http://localhost:8080"
	@echo "🎨 Streamlit will run on http://localhost:8501"
	@echo ""
	@echo "Press Ctrl+C in each pane to stop"
	@if command -v tmux > /dev/null; then \
		tmux new-session -d -s smart-rag 'uv run -m demo.api.app' \; \
		split-window -v 'sleep 3 && uv run streamlit run demo/streamlit_app.py' \; \
		attach-session -t smart-rag; \
	else \
		echo "❌ tmux is not installed. Please install it or run servers separately:"; \
		echo "   Terminal 1: make api-run"; \
		echo "   Terminal 2: make streamlit"; \
	fi

test:
	@echo "🧪 Running tests..."
	uv run pytest

lint:
	@echo "🔍 Running linter..."
	uv run ruff check .

format:
	@echo "✨ Formatting code..."
	uv run ruff check --fix .
	uv run ruff format .

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"
