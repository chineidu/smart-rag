# SMART RAG AGENT

This agentic RAG app answers multi-subject questions using RAG Fusion, precise ranking (RRF), tool use, and safety guardrails for accurate, evidence-based results.

## How To Make Requests

- Using CURL:

```bash
curl -N --get \
  --data-urlencode "query=your_query" \
  "http://localhost:8000/api/v1/your_endpoint"

# E.g. an endpoint with `message` parameter:
curl -N --get \
  --data-urlencode "message=What was Nvidia's yearly income in 2023? Include revenue, expenses, and net income." \
  "http://localhost:8000/api/v1/chat_stream"
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
