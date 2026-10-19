#!/bin/bash
# Stop the app and the local Comfy API proxy. Leaves ComfyUI itself running.
PORT="${PORT:-8801}"
PROXY_PORT="${PROXY_PORT:-8199}"
fuser -k "$PORT/tcp" 2>/dev/null && echo "stopped app on :$PORT"
fuser -k "$PROXY_PORT/tcp" 2>/dev/null && echo "stopped proxy on :$PROXY_PORT"
exit 0
