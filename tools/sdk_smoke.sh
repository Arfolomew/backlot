#!/bin/bash
# Sequential H3 smoke test through the SDK path. One JSON line per test in tools/sdk_smoke.log.
# usage: tools/sdk_smoke.sh <port> <source-image.png>
set -u
PORT="$1"; IMG="$2"; D="$(dirname "$0")"; LOG="$D/sdk_smoke.log"; : > "$LOG"
TMP=$(mktemp -d); B64=$(base64 -w0 "$IMG")
run() {  # run <label> <path> <payload-file>
  local t0=$(date +%s)
  local out; out=$(curl -s -m 1500 -X POST "http://127.0.0.1:$PORT$2" -H 'Content-Type: application/json' -d @"$3")
  python3 - "$1" "$(( $(date +%s) - t0 ))" "$LOG" <<PY
import sys, json
print(json.dumps({"test": sys.argv[1], "wall": int(sys.argv[2]), "resp": """$(echo "$out" | head -c 600 | sed 's/"""//g')"""}), file=open(sys.argv[3], "a"))
PY
}
printf '%s' "{\"prompt\":\"A red paper boat drifts across a calm pond at golden hour, paper cut-out world, gentle ripples, soft wind sound\",\"mode\":\"t2v\",\"width\":1024,\"height\":576,\"seconds\":3,\"seed\":11}" > "$TMP/t2v.json"
printf '%s' "{\"prompt\":\"The scene slowly comes alive, subtle camera push in, soft ambient sound\",\"mode\":\"i2v\",\"width\":1024,\"height\":576,\"seconds\":3,\"seed\":12,\"image\":\"data:image/png;base64,$B64\"}" > "$TMP/i2v.json"
printf '%s' "{\"prompt\":\"The same scene at night, slow pan left, crickets\",\"mode\":\"ref\",\"width\":1024,\"height\":576,\"seconds\":3,\"seed\":13,\"images\":[\"data:image/png;base64,$B64\"]}" > "$TMP/ref.json"
printf '%s' "{\"mode\":\"orbit\",\"width\":1152,\"height\":768,\"seed\":14,\"images\":[\"data:image/png;base64,$B64\"]}" > "$TMP/orbit.json"
run t2v   /api/video "$TMP/t2v.json"
run i2v   /api/video "$TMP/i2v.json"
run ref   /api/video "$TMP/ref.json"
run orbit /api/video "$TMP/orbit.json"
echo '{"test":"done"}' >> "$LOG"; rm -rf "$TMP"
