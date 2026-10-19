#!/bin/bash
# Desktop-launcher entry point. Runs start.sh and, since a desktop icon has no
# terminal to print to, surfaces a failure (usually "ComfyUI is not running")
# as a desktop notification instead of failing silently.
cd "$(dirname "$0")/.." || exit 1
LOG="launch.log"
if ./start.sh >"$LOG" 2>&1; then
  exit 0
fi
MSG="Backlot could not start. Is ComfyUI running? Details: $(pwd)/$LOG"
if command -v notify-send >/dev/null 2>&1; then
  notify-send -i backlot "Backlot" "$MSG"
elif command -v osascript >/dev/null 2>&1; then
  osascript -e "display notification \"$MSG\" with title \"Backlot\""
fi
echo "$MSG" >&2
exit 1
