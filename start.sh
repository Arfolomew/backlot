#!/bin/bash
# Start Backlot against a ComfyUI that is already running.
#
#   ./start.sh                      ComfyUI on http://127.0.0.1:8188 (the default)
#   COMFYUI_URL=http://127.0.0.1:8189 ./start.sh
#
# What this does:
#   1. starts comfy-api-proxy on :8199 in front of your ComfyUI (the Comfy Developer
#      Platform API, served locally), unless something already answers there;
#   2. starts the app on :8801 and opens it.
set -euo pipefail
cd "$(dirname "$0")"

COMFYUI_URL="${COMFYUI_URL:-http://127.0.0.1:8188}"
PROXY_PORT="${PROXY_PORT:-8199}"
PORT="${PORT:-8801}"
VENV=".venv"

if [ ! -x "$VENV/bin/python" ]; then
  echo "No .venv yet, creating one and installing requirements..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -r requirements.txt
fi

if ! curl -s -m 2 "$COMFYUI_URL/system_stats" >/dev/null; then
  echo "ComfyUI is not answering at $COMFYUI_URL. Start ComfyUI first, or set COMFYUI_URL." >&2
  exit 1
fi

if ! curl -s -m 2 "http://127.0.0.1:$PROXY_PORT/api/v2/health" >/dev/null; then
  mkdir -p .proxy-state
  nohup "$VENV/bin/comfy-api-proxy" run --comfyui "$COMFYUI_URL" --port "$PROXY_PORT" \
    --state-dir .proxy-state > proxy.log 2>&1 &
  for i in $(seq 1 20); do
    curl -s -m 1 "http://127.0.0.1:$PROXY_PORT/api/v2/health" >/dev/null && break
    sleep 0.5
  done
fi

export BACKLOT_COMFY_URL="http://127.0.0.1:$PROXY_PORT"
export BACKLOT_COMFYUI_URL="$COMFYUI_URL"
export PORT
if ! curl -s -m 1 "http://127.0.0.1:$PORT/api/status" >/dev/null; then
  nohup "$VENV/bin/python" app/server.py > app.log 2>&1 &
  for i in $(seq 1 20); do
    curl -s -m 1 "http://127.0.0.1:$PORT/api/status" >/dev/null && break
    sleep 0.5
  done
fi

echo "Backlot -> http://localhost:$PORT   (Comfy API: $BACKLOT_COMFY_URL -> $COMFYUI_URL)"
if [ -n "${BACKLOT_NO_OPEN:-}" ]; then :    # a wrapper will open its own window
elif command -v xdg-open >/dev/null; then xdg-open "http://localhost:$PORT" >/dev/null 2>&1 || true
elif command -v open >/dev/null; then open "http://localhost:$PORT" || true; fi
