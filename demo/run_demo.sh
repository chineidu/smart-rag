#!/bin/bash

# Script to run both FastAPI and Streamlit servers
# Usage: ./demo/run_demo.sh

echo "🚀 Starting Smart RAG Demo..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null
    exit 0
}

# Set up trap to catch Ctrl+C
trap cleanup SIGINT SIGTERM

# Start FastAPI server
echo -e "${BLUE}📡 Starting FastAPI server...${NC}"
uv run -m demo.api.app &
FASTAPI_PID=$!

# Wait a bit for FastAPI to start
sleep 3

# Start Streamlit app
echo -e "${GREEN}🎨 Starting Streamlit app...${NC}"
uv run streamlit run demo/streamlit_app.py &
STREAMLIT_PID=$!

echo ""
echo -e "${GREEN}✅ Both servers are running!${NC}"
echo ""
echo "📡 FastAPI:   http://localhost:8000"
echo "🎨 Streamlit: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for both processes
wait $FASTAPI_PID $STREAMLIT_PID
