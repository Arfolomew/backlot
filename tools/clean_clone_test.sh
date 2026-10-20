#!/bin/bash
# Clean-clone install test: what a judge gets from `git clone`. Clones the public repo (or a local
# branch), builds a fresh venv, starts the app on a spare port against the SAME running ComfyUI +
# proxy this machine already has, and exercises the new surfaces without touching the live
# instance. Prints PASS/FAIL per check and exits non-zero on any failure. Norm, Oct 20 2026.
#
#   tools/clean_clone_test.sh                      # clone github.com/Arfolomew/backlot main
#   tools/clean_clone_test.sh /path/to/repo branch # clone a local repo + branch instead
set -u
SRC="${1:-https://github.com/Arfolomew/backlot}"; BRANCH="${2:-main}"
PORT=8807; PROXY="${PROXY:-http://127.0.0.1:8199}"; COMFY="${COMFY:-http://127.0.0.1:8189}"
QPROXY="${QPROXY:-http://127.0.0.1:8198}"; QCOMFY="${QCOMFY:-http://127.0.0.1:8191}"
DIR=$(mktemp -d /tmp/backlot-clone-XXXX); fails=0
ok(){ echo "PASS  $1"; }; bad(){ echo "FAIL  $1"; fails=$((fails+1)); }

echo "clone -> $DIR"
git clone -q --depth 1 --branch "$BRANCH" "$SRC" "$DIR" || { echo "FAIL  clone"; exit 1; }
cd "$DIR"
python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt && ok "venv + requirements" || bad "venv + requirements"
[ -f app/moves.json ] && [ "$(python3 -c "import json;print(len(json.load(open('app/moves.json'))['moves']))")" = 30 ] && ok "30 moves in catalog" || bad "moves catalog"
n=$(ls app/previews/moves/*.mp4 2>/dev/null | wc -l); [ "$n" = 30 ] && ok "30 preview clips shipped" || bad "preview clips ($n)"
[ -f app/previews/cinema/calm-lighthouse-1960s.mp4 ] && ok "cinema sample shipped" || bad "cinema sample"
bash -n setup/download-models.sh && ok "download-models.sh parses" || bad "download-models.sh"
grep -q h3_character_swap_v1.5 setup/download-models.sh docs/models.md && ok "swap LoRA documented" || bad "swap LoRA docs"

fuser -k $PORT/tcp >/dev/null 2>&1
PORT=$PORT BACKLOT_COMFY_URL=$PROXY BACKLOT_COMFYUI_URL=$COMFY BACKLOT_QWEN_URL=$QPROXY BACKLOT_QWEN_COMFYUI_URL=$QCOMFY \
  nohup .venv/bin/python app/server.py > server.log 2>&1 &
for i in $(seq 1 30); do curl -sf localhost:$PORT/api/status >/dev/null && break; sleep 0.5; done
curl -sf localhost:$PORT/api/status | grep -q '"comfy": true' && ok "server up, backend reachable" || bad "server/backend"
[ "$(curl -s localhost:$PORT/api/moves | python3 -c 'import json,sys;d=json.load(sys.stdin)["moves"];print(sum(1 for m in d if m["preview"]))')" = 30 ] && ok "/api/moves serves 30 previews" || bad "/api/moves"
curl -s localhost:$PORT/api/cinema | grep -q '"pacing"' && ok "/api/cinema" || bad "/api/cinema"
curl -s -X POST localhost:$PORT/api/cinema/compose -H 'Content-Type: application/json' -d '{"scene":"a cat on a windowsill","pacing":"calm","seconds":6}' | grep -q '\[Shot 3\]' && ok "cinema compose -> 3 shots" || bad "cinema compose"
curl -s -o /dev/null -w '%{http_code}' localhost:$PORT/previews/moves/push-in.mp4 | grep -q 200 && ok "preview served" || bad "preview route"
curl -s -o /dev/null -w '%{http_code}' localhost:$PORT/ | grep -q 200 && ok "index served" || bad "index"
# one real GPU job through the SDK path: SeedVR2 2x on a shipped poster (~10 s)
python3 - $PORT <<'EOF' && ok "image upscale job through the SDK" || bad "image upscale job"
import base64,json,sys,time,urllib.request
port=sys.argv[1]
img=base64.b64encode(open('app/previews/moves/push-in.jpg','rb').read()).decode()
body=json.dumps({"image":"data:image/jpeg;base64,"+img,"size":"2x","filename":"push-in.jpg"}).encode()
req=urllib.request.Request(f"http://127.0.0.1:{port}/api/image-upscale",data=body,headers={"Content-Type":"application/json","X-Backlot-Async":"1"})
job=json.load(urllib.request.urlopen(req))["job"]; t0=time.time()
while time.time()-t0<600:
    time.sleep(3)
    try:
        r=urllib.request.urlopen(f"http://127.0.0.1:{port}/api/jobs/{job}")
        if r.status==200: d=json.load(r); print("   ", d["url"], d.get("took"), "s"); sys.exit(0)
    except urllib.error.HTTPError as e: print("   ", e.code, e.read().decode()[:200]); sys.exit(1)
sys.exit(1)
EOF
fuser -k $PORT/tcp >/dev/null 2>&1
echo; echo "clone dir kept at $DIR (delete when done)"; [ $fails = 0 ] && echo "ALL PASS" || echo "$fails FAILED"; exit $fails
