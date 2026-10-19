#!/usr/bin/env python3
"""One-shot migration of the private MiniMax Studio server.py into this fork.
Run from the repo root after copying the studio's server.py to app/server.py."""
import re, sys
p = 'app/server.py'; s = open(p).read()
def rep(a, b, count=1):
    global s
    n = s.count(a); assert n == count, (n, a[:80]); s = s.replace(a, b)

head_end = s.index('import base64')
s = '''#!/usr/bin/env python3
"""
Backlot: a local, open-weights video + image studio on ComfyUI.

Video on MiniMax H3 (text, image, reference images, 360 orbit, latent chaining), stills on
Krea 2 Turbo and Qwen Image 2.1 (text, img2img, style reference, identity edit, instruction
edit, camera angles), SeedVR2 / RealESRGAN upscaling and RIFE interpolation.

Every render goes through the Comfy Developer Platform API via the Comfy SDK. Locally that is
`comfy-api-proxy` in front of a plain ComfyUI (see README); point BACKLOT_COMFY_URL at Comfy Cloud
and the same code runs there.

Workflow graphs live in workflows/ (one builder per model family); this file is the HTTP app.
"""
''' + s[head_end:]

rep('''HERE = os.path.dirname(os.path.abspath(__file__))
H3_DIR = os.path.expanduser("~/ComfyUI-h3")
sys.path.insert(0, H3_DIR)
from build_h3_workflow import build as build_h3            # noqa: E402
from build_h3_workflow import CAMERA_LORA, CAMERA_TRIGGER   # noqa: E402
from build_ref2va_workflow import build as build_ref2va      # noqa: E402
from build_postfx_workflow import build as build_postfx    # noqa: E402
from build_krea_workflow import build as build_krea        # noqa: E402
from build_seedvr2_video_workflow import build as build_seedvr2_video   # noqa: E402  (Oct 5 2026)
''', '''HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from workflows.build_h3_workflow import build as build_h3            # noqa: E402
from workflows.build_h3_workflow import CAMERA_LORA, CAMERA_TRIGGER   # noqa: E402
from workflows.build_ref2va_workflow import build as build_ref2va      # noqa: E402
from workflows.build_postfx_workflow import build as build_postfx    # noqa: E402
from workflows.build_krea_workflow import build as build_krea        # noqa: E402
from workflows.build_seedvr2_video_workflow import build as build_seedvr2_video   # noqa: E402
from comfy_backend import Backend, gpu_used_mib   # noqa: E402
''')

a = s.index('COMFY = "http://127.0.0.1:8189"')
b = s.index('QWEN21_UNET = ')
s = s[:a] + '''# ---- configuration (environment) ----
#   BACKLOT_COMFY_URL          Comfy API endpoint: the local comfy-api-proxy (default) or Comfy Cloud.
#   BACKLOT_COMFYUI_URL        the plain ComfyUI behind the proxy, used ONLY for the "free VRAM" button.
#   BACKLOT_QWEN_URL           optional second Comfy API endpoint for Qwen Image 2.1 (default: same as above).
#   BACKLOT_QWEN_COMFYUI_URL   its plain ComfyUI, same purpose as BACKLOT_COMFYUI_URL.
#   BACKLOT_OUTPUT_DIR         where finished videos and images are filed (default: ./output).
#   BACKLOT_COMFY_OUTPUT_DIR   optional: ComfyUI's output folder, so latent chaining can list saved clips.
#   PORT                  app port (default 8801).
COMFY = os.environ.get("BACKLOT_COMFY_URL", "http://127.0.0.1:8199").rstrip("/")
COMFY_DIRECT = os.environ.get("BACKLOT_COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
VLO = os.environ.get("BACKLOT_QWEN_URL", COMFY).rstrip("/")
VLO_DIRECT = os.environ.get("BACKLOT_QWEN_COMFYUI_URL", COMFY_DIRECT if VLO == COMFY else "").rstrip("/")
BACKEND = Backend(COMFY, COMFY_DIRECT, label="ComfyUI")
QWEN = BACKEND if VLO == COMFY else Backend(VLO, VLO_DIRECT, label="Qwen ComfyUI")
# ComfyUI's own output folder. Optional; only latent chaining (Motion Context) reads it.
COMFY_STAGING = os.path.expanduser(os.environ.get("BACKLOT_COMFY_OUTPUT_DIR", ""))

# Qwen Image 2.1 + Kijai's turbo LoRA: base INT8, 8 steps, cfg 1. Needs the native Qwen 2.1
# nodes (ComfyUI 0.35+). By default it runs on the same backend as everything else.
''' + s[b:]

rep('''# 🔴 Every output this app produces lands under Studio Output, nowhere else.
OUTPUT_ROOT = os.path.expanduser("~/Desktop/Studio Output")
VIDEO_DIR = os.path.join(OUTPUT_ROOT, "Minimax_Videos")
IMAGE_DIR = os.path.join(OUTPUT_ROOT, "Krea_2_Turbo")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
''', '''# Every output this app produces lands under OUTPUT_ROOT, nowhere else.
OUTPUT_ROOT = os.path.expanduser(os.environ.get("BACKLOT_OUTPUT_DIR", os.path.join(HERE, "..", "output")))
VIDEO_DIR = os.path.join(OUTPUT_ROOT, "videos")
IMAGE_DIR = os.path.join(OUTPUT_ROOT, "images")
TMP_DIR = os.path.join(OUTPUT_ROOT, ".tmp")
for _d in (VIDEO_DIR, IMAGE_DIR, TMP_DIR):
    os.makedirs(_d, exist_ok=True)
''')

rep('''KREA_NSFW_LORA = "krea2_nsfw_plus_0x7.safetensors"   # lorian0x7 "NSFW+ Krea2" (Civitai 2742640), Oct 14 2026, Jake asked
KREA_NSFW_STRENGTH = 1.0
''', '')
rep('''                nsfw = body.get("model") == "krea2nsfw" and mode != "edit"
                if nsfw:
                    # Krea NSFW button (Oct 14 2026): same Krea graph, NSFW LoRA spliced in right
                    # after the UNET so every mode (and the style/identity LoRAs) runs on top of it.
                    for node in wf.values():
                        for k, v in node["inputs"].items():
                            if k == "model" and v == ["1", 0]:
                                node["inputs"][k] = ["40", 0]
                    wf["40"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                        "model": ["1", 0], "lora_name": KREA_NSFW_LORA,
                        "strength_model": float(body.get("nsfw_strength", KREA_NSFW_STRENGTH))}}
''', '')
rep('''                if nsfw:
                    kind = kind.replace("krea", "krea-nsfw", 1)
''', '')

a = s.index('HEARTBEAT = {"last": time.time(), "seen": False}')
b = s.index('def save_output(')
s = s[:a] + '''HEARTBEAT = {"last": time.time(), "seen": False}


# ---------- backend helpers (all rendering goes through the Comfy SDK) ----------
def comfy_alive():
    return BACKEND.alive()


def vlo_alive():
    return QWEN.alive()


def comfy_up(wait=0):
    """True if the main backend answers. The app never starts ComfyUI itself."""
    if QWEN is not BACKEND:
        QWEN.free()   # two ComfyUIs cannot both hold models on one card
    return BACKEND.alive()


def vlo_up(wait=0):
    return QWEN.alive()


def comfy_free():
    """Drop resident model weights from the card (needs BACKLOT_COMFYUI_URL). Never starts anything."""
    return BACKEND.free()


def vlo_free():
    if QWEN is not BACKEND:
        QWEN.free()


BACKEND_DOWN = (f"ComfyUI is not reachable through the Comfy API at {COMFY}. "
                "Start ComfyUI and comfy-api-proxy (see README).")


def submit_and_wait(wf, timeout=1800, exts=(".mp4", ".webm", ".gif"), base=None, staging=None, backend=None):
    """Submit a graph through the Comfy SDK and wait for it.
    Returns (bytes, ext, None); the third slot is kept for call-site compatibility."""
    be = backend or BACKEND
    blob, ext = be.run(wf, timeout=timeout, exts=exts)
    return blob, ext, None


def upload_to_comfy(raw, filename, backend=None):
    """Stage an input (still, clip, audio) as a Comfy asset. Returns the name to reference."""
    return (backend or BACKEND).upload(raw, filename)


def discard_upload(name, backend=None):
    (backend or BACKEND).discard(name)
    if backend is None and QWEN is not BACKEND:
        QWEN.discard(name)


def stage_local(raw, name):
    """A local scratch copy of an input for ffprobe/ffmpeg. Lives under OUTPUT_ROOT/.tmp."""
    path = os.path.join(TMP_DIR, os.path.basename(name))
    with open(path, "wb") as f:
        f.write(raw)
    return path


def cleanup_local(path):
    if path and os.path.dirname(os.path.abspath(path)) == os.path.abspath(TMP_DIR):
        try:
            os.remove(path)
        except OSError:
            pass


''' + s[b:]

s, n = re.subn(r'os\.remove\(os\.path\.join\(COMFY_INPUT, ([A-Za-z_]+)\)\)', r'discard_upload(\1)', s)
assert n == 5, n
s, n = re.subn(r'os\.remove\(os\.path\.join\(VLO_INPUT, ([A-Za-z_]+)\)\)', r'discard_upload(\1, QWEN)', s)
assert n == 1, n

rep('''                    name = f"mmstudio_qwen21_{stamp}_{i}.png"
                    with open(os.path.join(VLO_INPUT, name), "wb") as f:
                        f.write(buf.getvalue())
''', '''                    name = upload_to_comfy(buf.getvalue(), f"mmstudio_qwen21_{stamp}_{i}.png", backend=QWEN)
''')
rep('''                blob, ext, staged = submit_and_wait(wf, timeout=1800, exts=(".png",),
                                                    base=VLO, staging=VLO_STAGING)''',
    '''                blob, ext, staged = submit_and_wait(wf, timeout=1800, exts=(".png",), backend=QWEN)''')
rep('''                    if chain_index >= 2:
                        ctx_dir = os.path.join(COMFY_STAGING, proj)''',
    '''                    if chain_index >= 2 and COMFY_STAGING:
                        ctx_dir = os.path.join(COMFY_STAGING, proj)''')
rep('''                    src_name = upload_to_comfy(raw, f"mmstudio_up_{fname}")
                    src_path = os.path.join(COMFY_INPUT, src_name)
                elif body.get("last"):
                    cand = os.path.join(VIDEO_DIR, os.path.basename(body["last"]))
                    if not os.path.isfile(cand):
                        return self._send(400, json.dumps({"error": "last output not found"}))
                    with open(cand, "rb") as f:
                        src_name = upload_to_comfy(f.read(), f"mmstudio_up_{os.path.basename(cand)}")
                    src_path = os.path.join(COMFY_INPUT, src_name)
''', '''                    src_name = upload_to_comfy(raw, f"mmstudio_up_{fname}")
                    src_path = stage_local(raw, src_name)
                elif body.get("last"):
                    cand = os.path.join(VIDEO_DIR, os.path.basename(body["last"]))
                    if not os.path.isfile(cand):
                        return self._send(400, json.dumps({"error": "last output not found"}))
                    with open(cand, "rb") as f:
                        src_name = upload_to_comfy(f.read(), f"mmstudio_up_{os.path.basename(cand)}")
                    src_path = cand
''')
rep('''                name, path = save_output(blob, kind="up" if fast else "up-seedvr2", ext=ext, staged=staged)
                try:
                    os.remove(src_path)
                except OSError:
                    pass
''', '''                name, path = save_output(blob, kind="up" if fast else "up-seedvr2", ext=ext, staged=staged)
                discard_upload(src_name)
                cleanup_local(src_path)
''')
rep('''                src_name = upload_to_comfy(raw, f"mmstudio_slow_{int(time.time())}_{fname}")
                src_path = os.path.join(COMFY_INPUT, src_name)
''', '''                src_name = upload_to_comfy(raw, f"mmstudio_slow_{int(time.time())}_{fname}")
                src_path = stage_local(raw, src_name)
''')
rep('''            finally:
                if src_path:
                    try:
                        os.remove(src_path)
                    except OSError:
                        pass
''', '''            finally:
                if src_path:
                    discard_upload(os.path.basename(src_path))
                    cleanup_local(src_path)
''')
s, n = re.subn(r'\{"error": "MiniMax backend \(:8189\) would not start"\}', '{"error": BACKEND_DOWN}', s)
assert n >= 5, n
rep('''                        {"error": "Qwen 2.1 backend (Vlo ComfyUI :8191) would not start"}''',
    '''                        {"error": f"Qwen backend is not reachable at {VLO}"}''')
rep('''    port = int(os.environ.get("PORT", "8801"))
    print(f"MiniMax Studio on http://localhost:{port}")
    print(f"Backend  -> {COMFY}")
    print(f"Videos   -> {VIDEO_DIR}")
    print(f"Images   -> {IMAGE_DIR}")
    threading.Thread(target=_watchdog, daemon=True).start()
''', '''    port = int(os.environ.get("PORT", "8801"))
    print(f"Backlot on http://localhost:{port}")
    print(f"Comfy API -> {COMFY}" + (f"   (Qwen -> {VLO})" if VLO != COMFY else ""))
    print(f"Videos    -> {VIDEO_DIR}")
    print(f"Images    -> {IMAGE_DIR}")
''')
open(p, 'w').write(s)
print("migrated ok")
