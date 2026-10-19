#!/bin/bash
# Latent chaining (ref clip 1 -> clip 2) and Qwen 2.1 text + edit, through the SDK path.
# usage: tools/sdk_smoke2.sh <port> <source-image.png>
set -u
PORT="$1"; IMG="$2"; D="$(dirname "$0")"; LOG="$D/sdk_smoke2.log"; : > "$LOG"
TMP=$(mktemp -d); B64=$(base64 -w0 "$IMG")
run() {
  local t0=$(date +%s)
  local out; out=$(curl -s -m 1500 -X POST "http://127.0.0.1:$PORT$2" -H 'Content-Type: application/json' -d @"$3")
  python3 - "$1" "$(( $(date +%s) - t0 ))" "$LOG" <<PY
import sys, json
print(json.dumps({"test": sys.argv[1], "wall": int(sys.argv[2]), "resp": """$(echo "$out" | head -c 600 | sed 's/"""//g')"""}), file=open(sys.argv[3], "a"))
PY
}
printf '%s' "{\"prompt\":\"The red paper boat drifts slowly to the right across the pond, gentle ripples, soft wind\",\"mode\":\"ref\",\"width\":1024,\"height\":576,\"seconds\":3,\"seed\":21,\"chain_index\":1,\"chain_project\":\"backlot test\",\"images\":[\"data:image/png;base64,$B64\"]}" > "$TMP/c1.json"
printf '%s' "{\"prompt\":\"The red paper boat keeps drifting right and a paper duck paddles into view, soft wind\",\"mode\":\"ref\",\"width\":1024,\"height\":576,\"seconds\":3,\"seed\":22,\"chain_index\":2,\"chain_project\":\"backlot test\",\"images\":[\"data:image/png;base64,$B64\"]}" > "$TMP/c2.json"
printf '%s' "{\"prompt\":\"A tiny paper lighthouse on a paper cliff at dusk, cut-paper layers, warm window light\",\"model\":\"qwen21\",\"mode\":\"text\",\"width\":1152,\"height\":768,\"seed\":31}" > "$TMP/q1.json"
printf '%s' "{\"prompt\":\"Make it night time with a paper crescent moon and stars, keep everything else the same\",\"model\":\"qwen21\",\"mode\":\"edit\",\"seed\":32,\"image\":\"data:image/png;base64,$B64\"}" > "$TMP/q2.json"
run chain1    /api/video "$TMP/c1.json"
run chain2    /api/video "$TMP/c2.json"
run qwen-text /api/image "$TMP/q1.json"
run qwen-edit /api/image "$TMP/q2.json"
echo '{"test":"done"}' >> "$LOG"; rm -rf "$TMP"
