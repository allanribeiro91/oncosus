#!/bin/bash
# Start OncoSUS FastAPI backend
# Run from project root: ./deploy/start-api.sh

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(pwd)"

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

cd backend/rag
uvicorn app:app --host 0.0.0.0 --port 8000
