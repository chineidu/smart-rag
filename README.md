# SMART RAG AGENT

This agentic RAG app answers multi-subject questions using RAG Fusion, precise ranking (RRF), tool use, and safety guardrails for accurate, evidence-based results.

## Table of Contents
<!-- TOC -->

- [SMART RAG AGENT](#smart-rag-agent)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Obtain Bearer Token](#obtain-bearer-token)
  - [How To Stream Requests](#how-to-stream-requests)
  - [API Endpoints](#api-endpoints)
    - [Supervisord Commands](#supervisord-commands)
  - [Database Migrations](#database-migrations)
    - [Alembic Setup](#alembic-setup)
    - [Create a new Migration](#create-a-new-migration)
    - [Apply Migrations](#apply-migrations)
    - [Rollback Migrations](#rollback-migrations)
    - [Current Revision](#current-revision)
    - [Check Migration History](#check-migration-history)
  - [Technologies Used](#technologies-used)

<!-- /TOC -->

## Features

- Hybrid retrieval: dense embeddings + BM25 with RRF merging for grounded answers.
- Cross-encoder reranking and safety filters to keep responses concise and low-risk.
- Streaming chat endpoint with source snippets and latency metadata.
- Task pipeline backed by Celery with priority queues (high/normal/low) plus scheduled health checks and cleanups.
- Vector storage in Qdrant, Redis caching, and Postgres-backed LangGraph checkpoints for resilience.
- Built-in migrations (Alembic) and supervisor scripts for easy worker management.

## Obtain Bearer Token

```bash
curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=<your_username>&password=<your_password>"

# E.g. Use a GUEST account
curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=guest&password=guest123"
```

Sample response:

```json
{"access_token":"eyJhbGciO...","token_type":"bearer"}
```

## How To Stream Requests

- Base URL: <http://localhost:8000/api/v1>
- Use `-N` for streaming endpoints to keep the connection open.
- Example:

```bash
TOKEN="<paste_access_token_here>"

curl -N --get \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: text/event-stream" \
  --data-urlencode "last_id=$" \
  "http://localhost:8000/api/v1/sessions/<your-session-id>/stream"

# e.g.
curl -N --get \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: text/event-stream" \
  --data-urlencode "last_id=$" \
  "http://localhost:8000/api/v1/sessions/9ba93a69-18e4-4416-b186-ea76f7ac8906/stream"
```

Typical streamed chunk:

```txt
data: {"event_type": "final_answer", "data": {"answer": "Nvidia reported ~$27B revenue and ~$4.4B net income in FY2023...", "sources": [{"title": "2023 10-K", "url": "https://example.com/10k"}]}}
```

## API Endpoints

- **Health** – quick service check

```bash
curl -s http://localhost:8000/api/v1/health | jq
```

Sample response:

```json
{"name": "SMART RAG AGENT", "status": "ok", "version": "0.1.0"}
```

- **Chat (direct streaming, no Celery)** – SSE stream of graph output

```bash
curl -N --get \
  --data-urlencode "message=Summarize the latest Nvidia filing" \
  "http://localhost:8000/api/v1/chat_stream"
```

First event for new sessions includes `session_id`, followed by node events and a final `done` event.

- **Streaming sessions (Celery-backed)**
  - Create session

  ```bash
  curl -s http://localhost:8000/api/v1/sessions | jq
  ```

  - Submit a prompt to a session (routes to high/low priority queue by role)

  ```bash
  curl -s -X POST \
    "http://localhost:8000/api/v1/sessions/{session_id}/query" \
    -d "message=Explain RAG fusion" -d "user_id=user-123"
  ```

  - Stream the response via SSE

  ```bash
  curl -N "http://localhost:8000/api/v1/sessions/{session_id}/stream?last_id=$"
  ```

  - **Chat history** – fetch stored messages for a session

  ```bash
  curl -s "http://localhost:8000/api/v1/chat_history?session_id={session_id}" | jq
  ```

  - **Retrieval** – keyword, semantic, or hybrid search

  ```bash
  # Keyword
  curl -s "http://localhost:8000/api/v1/retrievals/keyword?query=AI%20chips&k=5" | jq

  # Semantic with optional section filter
  curl -s "http://localhost:8000/api/v1/retrievals/semantic?query=GPU%20forecast&filter=exec" | jq

  # Hybrid (vector + keyword)
  curl -s "http://localhost:8000/api/v1/retrievals/hybrid?query=datacenter%20revenue&k=5" | jq
  ```

Example item:

```json
{"id": "doc-123", "content": "Nvidia reported record data center revenue...", "score": 0.82, "metadata": {"section": "financials", "source": "10-K"}}
```

- **Task status** – poll Celery tasks

```bash
curl -s "http://localhost:8000/api/v1/task-status/{task_id}" | jq
```

### Supervisord Commands

```bash
# Install supervisor
sudo apt-get update

# Start supervisord with the config
supervisord -c docker/supervisord.conf

# Check worker status
supervisorctl -c docker/supervisord.conf status

# Start/stop/restart workers
supervisorctl -c docker/supervisord.conf start celery:*
supervisorctl -c docker/supervisord.conf stop celery:*
supervisorctl -c docker/supervisord.conf restart celery:*

# View logs
tail -f /var/log/celery-*.log
```

## Database Migrations

### Alembic Setup

- Initialize Alembic (if not already done):

```bash
alembic init alembic
```

- Configure `alembic/env.py` with your database URL.

```py
# =============================================================
# ==================== Add DB Config ==========================
# =============================================================
config.set_main_option("sqlalchemy.url", app_settings.database_url)

...

# ============ Fiter out unneeded metadata ============
excluded_tables: set[str] = {"celery_taskmeta"}
# Only include tables not in excluded_tables for Alembic migrations
filtered_metadata = MetaData()
for table_name, table in Base.metadata.tables.items():
    if table_name not in excluded_tables:
        table.tometadata(filtered_metadata)
target_metadata = filtered_metadata
```

### Create a new Migration

- `Autogenerate`: Compares your database schema to the SQLAlchemy models and automatically creates a migration script that reflects any differences.

```bash
alembic revision --autogenerate -m "Your migration message"

# E.g.
alembic revision --autogenerate -m "Add users table"

# View the SQL statements that will be executed
alembic revision --autogenerate -m "Add users table" --sql
```

### Apply Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Apply migrations to a specific revision
alembic upgrade <revision_id>
```

### Rollback Migrations

```bash
# Downgrade one revision
alembic downgrade -1

# Downgrade to a specific revision
alembic downgrade <revision_id>
```

### Current Revision

```bash
alembic current
```

### Check Migration History

```bash
alembic history
```

## Technologies Used

- FastAPI, Uvicorn, Gunicorn for the API layer
- Celery + Redis for task queuing, scheduling, and streaming responses
- LangChain, LangGraph, Sentence-Transformers, Instructor for orchestration and embeddings
- Qdrant for vector storage; Rank-BM25 for lexical retrieval
- Tavily/DDGS/Wikipedia for web enrichment; httpx for IO
- Alembic + psycopg for migrations and Postgres connectivity
- Streamlit for a simple UI; aiocache for caching
