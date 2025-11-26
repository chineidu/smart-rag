
.PHONY: help install api-run frontend src clean test test-verbose lint format clean-cache up down restart logs clean-all setup status

help:
	@echo "Tool Chat - Available commands:"
	@echo ""
	@echo "📦 Development:"
	@echo "  make install        - Install all dependencies"
	@echo "  make test           - Run tests"
	@echo "  make test-verbose   - Run tests with verbose output"
	@echo "  make lint           - Run linter (ruff)"
	@echo "  make format         - Format code (ruff)"
	@echo "  make clean-cache    - Clean up cache and temporary files"
	@echo ""
	@echo "🚀 Running the Application:"
	@echo "  make api-run        - Run FastAPI server on http://localhost:8000"
	@echo "  make frontend       - Run Streamlit app on http://localhost:8501"
	@echo "  make src            - Run both API and Streamlit (requires tmux)"
	@echo ""
	@echo "🐳 Docker Services:"
	@echo "  make up             - Start all services"
	@echo "  make down           - Stop all services"
	@echo "  make restart        - Restart services"
	@echo "  make logs           - View logs"
	@echo "  make status         - Check status"
	@echo "  make setup          - Setup from scratch"
	@echo "  make clean-all      - Clean everything (including volumes)"
	@echo ""

install:
	@echo "📦 Installing dependencies..."
	uv sync

api-run:
	@echo "🚀 Starting FastAPI server on http://localhost:8000"
	uv run fastapi dev src/api/app.py
# 	uv run -m src.api.app

frontend:
	@echo "🎨 Starting Streamlit app on http://localhost:8501"
	uv run -m streamlit run src/frontend/app.py

src:
	@echo "🚀 Starting both FastAPI and Streamlit..."
	@echo "📡 FastAPI will run on http://localhost:8000"
	@echo "🎨 Streamlit will run on http://localhost:8501"
	@echo ""
	@echo "Press Ctrl+C in each pane to stop"
	@if command -v tmux > /dev/null; then \
		tmux new-session -d -s chat-bot 'uv run fastapi dev src/api/app.py' \; \
		split-window -v 'sleep 3 && uv run -m streamlit run src/frontend/app.py' \; \
		attach-session -t chat-bot; \
	else \
		echo "❌ tmux is not installed. Please install it or run servers separately:"; \
		echo "   Terminal 1: make api-run"; \
		echo "   Terminal 2: make frontend"; \
	fi

test:
	@echo "🧪 Running tests..."
	uv run -m pytest

test-verbose:
	@echo "🧪 Running tests..."
	uv run -m pytest -v

lint:
	@echo "🔍 Running linter..."
	uv run ruff check .

format:
	@echo "✨ Formatting code..."
	uv run ruff check --fix .
	uv run ruff format .

clean-cache:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"

.PHONY: up down restart logs clean setup status clean-all

# Start all services
up:
	@chmod +x docker/init_databases.sh
	docker-compose up -d

# Stop all services
down:
	docker-compose down

# Restart services
restart: down up

# Clean everything (including volumes)
clean-all:
	docker-compose down -v --remove-orphans

# View logs
logs:
	docker-compose logs -f

# Setup from scratch
setup: clean up
	@echo "Setup complete! Services are running."

# Check status
status:
	docker-compose ps
