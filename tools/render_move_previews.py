#!/usr/bin/env python3
"""Render the preview clip for every Move in app/moves.json through a running Backlot.

    .venv/bin/python tools/render_move_previews.py --server http://127.0.0.1:8805 [--only push-in,orbit-left] [--force]

Each preview is a 5 s text-to-video render of the move's `preview_subject` at 864x480
(about 3 to 4 minutes each on a 16 GB card). The finished clip is copied to
app/previews/moves/<id>.mp4 and a poster frame to <id>.jpg. Already-rendered moves are
skipped unless --force. Uses the server's job API so a dropped connection does not lose a render.
Norm, Oct 20 2026.
"""
import argparse, json, os, shutil, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CATALOG = os.path.join(ROOT, "app", "moves.json")
OUT = os.path.join(ROOT, "app", "previews", "moves")


def call(server, path, body=None, headers=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(server + path, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode() or "{}")


def render(server, move, width, height, seconds, password):
    hdr = {"X-Backlot-Async": "1"}
    if password:
        import base64
        hdr["Authorization"] = "Basic " + base64.b64encode(f"backlot:{password}".encode()).decode()
    st, d = call(server, "/api/video", {
        "mode": "t2v", "move": move["id"], "subject": move["preview_subject"],
        "width": width, "height": height, "seconds": seconds, "seed": 7}, hdr)
    if st != 202 or "job" not in d:
        raise RuntimeError(f"submit failed: {st} {d}")
    job = d["job"]
    t0 = time.time()
    while True:
        time.sleep(5)
        try:
            st, d = call(server, f"/api/jobs/{job}", headers=hdr)
        except urllib.error.HTTPError as e:   # 4xx/5xx carries the real error body
            raise RuntimeError(f"job {job}: {e.code} {e.read().decode()[:300]}")
        if st == 202:
            continue
        if st != 200:
            raise RuntimeError(f"job {job} failed: {st} {d}")
        return d, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8805")
    ap.add_argument("--only", default="", help="comma-separated move ids")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--password", default=os.environ.get("BACKLOT_PASSWORD", ""))
    a = ap.parse_args()
    moves = json.load(open(CATALOG))["moves"]
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    os.makedirs(OUT, exist_ok=True)
    done = failed = 0
    for m in moves:
        if only and m["id"] not in only:
            continue
        dst = os.path.join(OUT, m["id"] + ".mp4")
        if os.path.isfile(dst) and not a.force:
            print(f"skip  {m['id']} (exists)", flush=True)
            continue
        print(f"render {m['id']:18s} ...", end=" ", flush=True)
        try:
            d, took = render(a.server, m, a.width, a.height, a.seconds, a.password)
            shutil.copyfile(d["file"], dst)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1.0", "-i", dst,
                            "-frames:v", "1", "-q:v", "4", dst[:-4] + ".jpg"], check=True)
            print(f"ok  {took:5.0f}s  {d.get('frames')}f  seed {d.get('seed')}", flush=True)
            done += 1
        except Exception as e:
            print(f"FAILED  {e}", flush=True)
            failed += 1
    print(f"\n{done} rendered, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
