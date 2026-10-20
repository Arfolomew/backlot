#!/usr/bin/env python3
"""Character Swap end-to-end test through the dev server (Oct 20 2026).
Source clip = the Side Tracking preview (a runner), character = a frame from the Push In preview (the keeper).
Both are app-rendered, invented people. ~20 min on the 16 GB card."""
import base64, json, subprocess, sys, time, urllib.request
SERVER = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8805"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "4.5", "-i", "app/previews/moves/push-in.mp4", "-frames:v", "1", "/tmp/swap-char.png"], check=True)
vid = base64.b64encode(open("app/previews/moves/tracking-side.mp4", "rb").read()).decode()
img = base64.b64encode(open("/tmp/swap-char.png", "rb").read()).decode()
body = json.dumps({"mode": "swap", "video": "data:video/mp4;base64," + vid, "image": "data:image/png;base64," + img,
                   "subject": "the runner", "width": 864, "height": 480, "label": "App · Character Swap"}).encode()
req = urllib.request.Request(SERVER + "/api/video", data=body, headers={"Content-Type": "application/json", "X-Backlot-Async": "1"})
job = json.load(urllib.request.urlopen(req))["job"]; print("job", job, flush=True); t0 = time.time()
while True:
    time.sleep(10)
    try:
        r = urllib.request.urlopen(f"{SERVER}/api/jobs/{job}")
        if r.status == 200:
            d = json.load(r); print("OK", d["url"], d.get("took"), "s", d.get("seconds"), "s long"); break
    except urllib.error.HTTPError as e:
        print("FAILED", e.code, e.read().decode()[:400]); sys.exit(1)
    if time.time() - t0 > 3600: print("timeout"); sys.exit(1)
