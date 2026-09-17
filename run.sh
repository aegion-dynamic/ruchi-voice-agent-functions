#!/usr/bin/env bash
# Start the standalone voice_agent_functions server.
#
# Usage:
#   ./run.sh                  # serves on 127.0.0.1:8000
#   PORT=8001 ./run.sh        # custom port
#
# Then open the demo at http://127.0.0.1:8000/demo/  (NOT the file://
# path — browsers only allow mic access on localhost/HTTPS).
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"

# Use the local venv if it exists, otherwise fall back to system python.
if [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi

echo "──────────────────────────────────────────────────────────"
echo "  Ruchi voice_agent_functions"
echo "  Demo:   http://127.0.0.1:${PORT}/demo/"
echo "  Docs:   http://127.0.0.1:${PORT}/docs"
echo "  Health: http://127.0.0.1:${PORT}/api/health"
echo "──────────────────────────────────────────────────────────"

exec "$PYTHON" -m uvicorn app:app --host 127.0.0.1 --port "$PORT" "$@"
