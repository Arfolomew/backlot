#!/usr/bin/env python3
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
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from workflows.build_h3_workflow import build as build_h3            # noqa: E402
from workflows.build_h3_workflow import CAMERA_LORA, CAMERA_TRIGGER   # noqa: E402
from workflows.build_ref2va_workflow import build as build_ref2va      # noqa: E402
from workflows.build_postfx_workflow import build as build_postfx    # noqa: E402
from workflows.build_krea_workflow import build as build_krea        # noqa: E402
from workflows.build_ltx25_workflow import build as build_ltx25      # noqa: E402
from workflows.build_seedvr2_video_workflow import build as build_seedvr2_video   # noqa: E402
from comfy_backend import Backend, gpu_used_mib   # noqa: E402

# ---- configuration (environment) ----
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
QWEN21_UNET = "qwen_image_2.1_int8_convrot.safetensors"
QWEN21_LORA = "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"
QWEN21_TE = "qwen3vl_8b_int8_convrot.safetensors"
QWEN21_VAE = "qwen_image_2.1_vae_bf16.safetensors"

# MOSS-Image (ComfyUI 0.42+). The text-and-layout model: every prompted word rendered correctly in
# our tests, including a five-line price list and a URL. Official template graph: skip-layer guidance
# on single layer 22 for the first half, 25 steps, CFG 2.5, er_sde/simple, fixed negative. 16 s per
# image at about 1 MP on a 16 GB card. Layout mode sends the node's own JSON (pixel boxes centered on
# the canvas) so each text block lands where the form put it. Photoreal stays on Krea.
MOSS_UNET = "moss_image_sft_int8_convrot.safetensors"
MOSS_TE = "qwen3.5_9b_w4a8_gptq.safetensors"
MOSS_VAE = "flux2-vae_small.safetensors"
MOSS_NEG = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
MOSS_STEPS = 40                  # Jake, Oct 20: 40 steps looked much better than 25
MOSS_CFG = 2.5
MOSS_MAX_PIXELS = 2_000_000      # raised Oct 20 (Jake): 1152x1728 posters render at full size on 16 GB
QWEN21_STEPS = 8
QWEN21_MAX_REFS = 10
# ORB360 360-orbit LoRA (MatlowAI, Ref2VA). Prompts = the author's captions, in prompts/.
ORB360_LORA = "h3-tier2/minimax_h3_orb360_step1500.safetensors"
ORB360_PROMPTS = {1: "orb360-1photo.txt", 3: "orb360-3photo.txt"}
ORB360_FRAMES = 124
ORB360_SIZES = {"land": (1152, 768), "port": (832, 1024), "square": (800, 800)}

# Every output this app produces lands under OUTPUT_ROOT, nowhere else.
OUTPUT_ROOT = os.path.expanduser(os.environ.get("BACKLOT_OUTPUT_DIR", os.path.join(HERE, "..", "output")))
VIDEO_DIR = os.path.join(OUTPUT_ROOT, "videos")
IMAGE_DIR = os.path.join(OUTPUT_ROOT, "images")
AUDIO_DIR = os.path.join(OUTPUT_ROOT, "audio")
META_DIR = os.path.join(OUTPUT_ROOT, "meta")      # one small JSON per output: what made it (Oct 20 2026)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Audio tab (Oct 20 2026). Two engines, both core ComfyUI nodes:
#   song = YuE2 3B (lyrics + style -> a full song with vocals, 48 kHz). ComfyUI's own "Text to Music
#          (YuE2)" blueprint: checkpoint -> [LoRA chain] -> (ABC plan) -> music tokens -> KSampler
#          32 steps dpm_2/sgm_uniform -> VAE decode. plan "off" sings the lyrics straight (clearest
#          delivery, best for rap); "full" writes a section plan and melody first; "melody" = plan only.
#   sfx  = Stable Audio 3 Medium (any sound or music bed up to ~45 s). T5Gemma encoder, 8 steps lcm,
#          CFG 1, prompt ends "Length: N seconds" (the model listens to that).
# Both decode through SaveAudioFloatWAV (one 40-line custom node in setup/) because the stock 16-bit
# SaveAudio clips YuE2's overs; the WAV is then loudness-normalized to MP3 here.
YUE2_CKPT = "yue2_3b_bf16.safetensors"
SA3_CKPT = "stable_audio_3_medium.safetensors"
SA3_TE = "t5gemma_b_b_ul2.safetensors"
SONG_MAX_SECONDS = 240
SFX_MAX_SECONDS = 45
# YuE2 LoRAs: slot says which half the weights train (clip = planner, model = decoder, both).
YUE2_LORAS = {
    "realaudio-nar-v4.safetensors": {"label": "Real production sound", "slot": "model", "default": 1.0},
    "instrumental-ar-v3.safetensors": {"label": "Instrumental (no vocals)", "slot": "clip", "default": 1.0},
}
for _s, _l in [("male", "Male vocal"), ("female", "Female vocal"), ("hiphop", "Hip-hop"), ("rnb", "R&B"),
               ("pop", "Pop"), ("pop-punk", "Pop punk"), ("indie-rock", "Indie rock"), ("metal", "Metal"),
               ("country", "Country"), ("acoustic-folk", "Acoustic folk"), ("house", "House"),
               ("disco-funk", "Disco funk"), ("lofi", "Lo-fi"), ("kpop", "K-pop"), ("reggaeton", "Reggaeton"),
               ("afrobeats", "Afrobeats")]:
    YUE2_LORAS[f"slider-{_s}.safetensors"] = {"label": _l, "slot": "clip", "default": 0.6}
TMP_DIR = os.path.join(OUTPUT_ROOT, ".tmp")
for _d in (VIDEO_DIR, IMAGE_DIR, TMP_DIR):
    os.makedirs(_d, exist_ok=True)

# The PROVEN H3 recipe, measured Sep 11 2026. Do not "tune" these without a render:
#   clip_cpu       the Qwen3-VL text encoder is 13.58GiB and ComfyUI-GGUF
#                  dequantizes ON-GPU -> OOM in dequantize_blocks_Q4_K if on CUDA.
#   head_chunks 16 MiniMaxLowVRAMAttention; higher = less VRAM, slower (max 56).
#   steps 8        with the 4-step turbo LoRA at strength 1.0.
# Moves: one-click camera moves and effects (Oct 20 2026). Catalog in moves.json,
# previews rendered by tools/render_move_previews.py into previews/moves/<id>.mp4.
with open(os.path.join(HERE, "moves.json")) as _f:
    MOVES = {m["id"]: m for m in json.load(_f)["moves"]}
PREVIEW_DIR = os.path.join(HERE, "previews")
# Character Swap (akatz-ai H3 Character Swap LoRA V1.5, Oct 17 2026 A/B winner over Viggle):
# Ref2VA graph + swap LoRA 1.0 + the ref2v turbo, 8 steps, er_sde / normal. Source clip = <Video 1>
# (its soundtrack rides along), character = <Picture 1>. ~22 min per 5 s clip on a 16 GB card, most
# of it the 32B text encoder reading the source video.
SWAP_LORA = "h3_character_swap_v1.5.safetensors"
SWAP_MAX_SECONDS = 10.0

# Cinema Studio (Oct 20 2026): pickers -> one H3 prompt in the model's own three-field,
# multi-shot format. Catalog of fragments in cinema.json.
with open(os.path.join(HERE, "cinema.json")) as _f:
    CINEMA = json.load(_f)

def _pick(group, key):
    items = CINEMA[group]
    for it in items:
        if it["key"] == key:
            return it
    return items[0]

def _mmss(t):
    return "%02d:%06.3f" % (int(t // 60), t % 60)

def compose_cinema_prompt(o):
    """o: {scene, shots?, genre, era, light, camera, lens, aperture, pacing, seconds}.
    Returns (prompt, cuts). 'shots' = optional list of per-shot lines (custom mode);
    otherwise every shot restates the same scene from the ladder's framing."""
    scene = " ".join(str(o.get("scene") or "").split()).rstrip(".")
    seconds = max(2.0, min(10.0, float(o.get("seconds") or 5)))
    genre = _pick("genre", o.get("genre") or "general")
    era = _pick("era", o.get("era") or "auto")
    light = _pick("light", o.get("light") or "auto")
    cam = _pick("camera", o.get("camera") or "modern")
    lens = _pick("lens", o.get("lens") or "clean")
    ap = _pick("aperture", o.get("aperture") or "f4")
    pacing = _pick("pacing", o.get("pacing") or "single")
    custom = [" ".join(x.split()).rstrip(".") for x in (o.get("shots") or []) if str(x).strip()]
    # cut list in seconds, dropping cuts that would leave a shot shorter than 0.6 s
    cuts = [round(f * seconds, 3) for f in pacing["cuts"]]
    keep = [cuts[0]]
    for c in cuts[1:]:
        if c - keep[-1] >= 0.6 and seconds - c >= 0.6:
            keep.append(c)
    cuts = keep
    ladder = pacing["ladder"]
    style = ", ".join(x for x in [
        "Live-action, cinematic", era["frag"], cam["frag"], lens["frag"], ap["frag"], light["frag"], genre["look"]] if x)
    parts = []
    for i, t in enumerate(cuts):
        shot = CINEMA["shots"][ladder[i % len(ladder)]]
        content = custom[i] if i < len(custom) else (scene if i == 0 else f"the same scene, {scene}")
        if i == 0:
            parts.append(f"[Shot 1] {style}, {shot['frame']} frames {content}. {shot['motion']}")
        else:
            parts.append(f"[Shot {i + 1}] At {_mmss(t)}, the camera cuts to {shot['frame']} of {content}. {shot['motion']}")
    prompt = ("integrated_multimodal_description: " + " ".join(parts) + "\n\n"
              f"overall_soundscape: {genre['sound']}, continuous across every cut.\n\n"
              f"non_diegetic_music: {genre['music']}")
    return prompt, cuts


def compose_move_prompt(move, subject):
    """Turn a Move + the user's subject line into the prompt H3 gets.
    Camera moves ride the Camera Motion LoRA, which was trained on comma lists
    ('camera motion, subject, scene, move, look, quality'), so that is what we send.
    Effects carry no LoRA and use the model's own three-field format."""
    subject = " ".join(subject.split()).rstrip(".,;")
    if move.get("lora"):
        return (f"{CAMERA_TRIGGER}, {subject}, {move['move']}, "
                "cinematic lighting, sharp detail, 4k, high quality")
    return (f"integrated_multimodal_description: [Shot 1] {subject}. {move['move']}.\n\n"
            "overall_soundscape: natural ambience matching the scene, subtle and continuous.\n\n"
            "non_diegetic_music: none")

LTX_FPS = 24
LTX_MAX_SECONDS = 10.0
H3_STEPS = 8
H3_HEAD_CHUNKS = 16
H3_FPS = 24

# Krea 2 Turbo (Flux2-family still image) on the SAME :8189 backend.
# 🔴 EmptyFlux2LatentImage builds a //16 latent but qwen_image_vae decodes at 8x,
#    so the decoded image is HALF the requested size. Ask for 2x, get what you meant.
# 🔴 But Flux2Scheduler must get the REAL output size: it sets the noise shift from the token
#    count, and handing it the 2x size gave 4x the shift -> waxy, soft skin. Fixed Oct 4 2026
#    (bake-off ~/Desktop/image-model-bakeoff/, the user's pick), with ~2MP as the new default size.
# Flux2Scheduler accepts steps 1..4096. Measured at 1216x832: 8 steps 15s (waxy),
# 20 steps 27s (sharp skin + wood grain), 32 steps 42s (label/pit-level detail).
KREA_LATENT_SCALE = 2
KREA_STEP_MIN = 1
KREA_STEP_MAX = 4096
KREA_TARGET_PIXELS = 1728 * 1152   # ~2 MP, Oct 4 bake-off winner
KREA_REF_METHOD = "index_timestep_zero"   # what the official style-reference template uses
KREA_ID_LORA = "krea2_identity_edit_v1_2.safetensors"   # conradlocke, Oct 4 2026
KREA_ID_PIXELS = 1024 * 1024        # Identity Edit author: 1 MP sweet spot
KREA_STYLE_LORA = "krea2_style_reference.safetensors"   # ostris, ~/ComfyUI-h3/.../models/loras
QWEN_EDIT_GGUF = "qwen-image-edit-2511-Q4_K_M.gguf"            # unsloth, ~/ComfyUI-h3/.../models/unet (Oct 6 2026)
QWEN_EDIT_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"   # lightx2v, 4 steps
QWEN_EDIT_TE = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
# Oct 17 2026 (the user: "a really cool lora"). fal Multiple Angles LoRA for Qwen-Image-Edit-2511:
# re-shoots the photo from one of 96 camera positions. Prompt is "<sks> <azimuth> <elevation> <distance>".
# Stacked after Lightning on the Edit graph, ~22 s per angle. Test: Shelved-Projects/AI-Persona/angle-test/.
QWEN_ANGLES_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
ANGLE_AZIMUTHS = ("front view", "front-right quarter view", "right side view", "back-right quarter view",
                  "back view", "back-left quarter view", "left side view", "front-left quarter view")
ANGLE_ELEVATIONS = ("low-angle shot", "eye-level shot", "elevated shot", "high-angle shot")
ANGLE_DISTANCES = ("close-up", "medium shot", "wide shot")
# Oct 5 2026: SeedVR2 replaces RealESRGAN for stills (the user: ESRGAN looks smooth and plastic;
# SeedVR2 keeps real pores, lashes and lettering). numz/ComfyUI-SeedVR2_VideoUpscaler in
# ~/ComfyUI-h3, models in ComfyUI/models/SEEDVR2/. 1152x1664 -> 2160x3120 in ~20 s on the 4070 Ti S.
SEEDVR2_DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
SEEDVR2_VAE = "ema_vae_fp16.safetensors"
SEEDVR2_MAX_LONG = 4096   # long-edge cap so a panorama can't blow past 16 GB


def seedvr2_nodes(image_ref, short_side, first_id=50):
    """SeedVR2 upscale nodes. `image_ref` is a node output like ["1", 0]; `short_side` is the
    target length of the image's SHORT edge. Returns (nodes, output_ref)."""
    a, b, c = str(first_id), str(first_id + 1), str(first_id + 2)
    return {
        a: {"class_type": "SeedVR2LoadDiTModel", "inputs": {"model": SEEDVR2_DIT, "device": "cuda:0"}},
        b: {"class_type": "SeedVR2LoadVAEModel", "inputs": {
            "model": SEEDVR2_VAE, "device": "cuda:0", "encode_tiled": True, "decode_tiled": True}},
        c: {"class_type": "SeedVR2VideoUpscaler", "inputs": {
            "image": image_ref, "dit": [a, 0], "vae": [b, 0], "seed": 42,
            "resolution": int(short_side), "max_resolution": 0, "batch_size": 1,
            "uniform_batch_size": False, "color_correction": "lab"}},
    }, [c, 0]

HEARTBEAT = {"last": time.time(), "seen": False}
ACTIVE_JOBS = {"n": 0}          # renders in flight; the watchdog never frees under one
IDLE_FREE_AFTER = 20.0          # seconds without a heartbeat before the card is handed back
CLOSE_GRACE = 8.0               # a pagehide beacon shortens the wait to this


def _idle_watchdog():
    """Hand the GPU back when every Backlot window is gone.
    Each open window heartbeats every 5 s. When the beats stop (window closed, or a
    pagehide beacon pulled the clock forward) and nothing is rendering, drop the
    resident models once. A reload re-arms it within a second, so refreshing never
    unloads anything. The app and proxy keep running; ComfyUI is never stopped."""
    while True:
        time.sleep(2.0)
        try:
            if not HEARTBEAT["seen"] or ACTIVE_JOBS["n"]:
                continue
            if time.time() - HEARTBEAT["last"] < IDLE_FREE_AFTER:
                continue
            HEARTBEAT["seen"] = False
            vlo_free()
            freed = comfy_free()
            if freed is not None:
                print(f"[idle] no Backlot window for {IDLE_FREE_AFTER:.0f}s, "
                      f"freed {freed} MiB of VRAM (now {gpu_used_mib()} MiB used)", flush=True)
        except Exception as e:
            print(f"[idle] free failed: {e}", flush=True)


# ---------- background jobs ----------
# Any render endpoint can run as a job: the client sends X-Backlot-Async: 1, gets a job id
# back at once, and polls /api/jobs/<id>. The handler code is unchanged; in job mode
# _send() records the response instead of writing it to the socket. This is what lets a
# 7-minute H3 render survive a reverse proxy's request timeout or a page reload.
JOBS = {}                      # id -> {"state", "path", "t0", "code", "body", "ctype", "t1"}
JOBS_LOCK = __import__("threading").Lock()
JOB_PATHS = {"/api/video", "/api/image", "/api/audio", "/api/vertical", "/api/image-upscale",
             "/api/upscale", "/api/slowmo"}
JOB_KEEP_SEC = 3600            # finished jobs stay fetchable this long
JOB_CTX = __import__("threading").local()   # .jid on a job worker thread, so submit_and_wait can report progress


def _new_job(path):
    jid = base64.urlsafe_b64encode(os.urandom(9)).decode().rstrip("=")
    with JOBS_LOCK:
        now = time.time()
        for k in [k for k, j in JOBS.items() if j.get("t1") and now - j["t1"] > JOB_KEEP_SEC]:
            del JOBS[k]
        JOBS[jid] = {"state": "running", "path": path, "t0": now}
    return jid


# Optional HTTP Basic auth, for an instance exposed beyond localhost (BACKLOT_PASSWORD).
# Any username works; only the password is checked.
PASSWORD = os.environ.get("BACKLOT_PASSWORD", "")


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


def build_moss(prompt, width, height, steps=MOSS_STEPS, seed=7, prefix="moss/Backlot"):
    """The MOSS-Image text-to-image graph (ComfyUI's own template), unchanged."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MOSS_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": MOSS_TE, "type": "moss_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MOSS_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": MOSS_NEG, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "15": {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {
            "model": ["1", 0], "double_layers": "", "single_layers": "22", "start_percent": 0.0, "end_percent": 0.5}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["15", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
            "seed": seed, "steps": steps, "cfg": MOSS_CFG, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


# Layout mode: the poster form -> MOSS layout JSON. Each field becomes one box, stacked top to
# bottom; empty fields drop out and the rest spread to fill the canvas. Heights are weights
# (share of the usable canvas), captions tell MOSS how to set each block.
MOSS_LAYOUT_SLOTS = (
    # key, weight, is_text, caption
    ("headline", 0.18, True,  "Large bold display capitals, centered, the dominant element."),
    ("subhead",  0.08, True,  "Smaller script or italic banner line, centered."),
    ("lines",    0.30, True,  "Left-aligned list, one item per line, clean sans-serif, even spacing, every line legible."),
    ("art",      0.28, False, None),
    ("footer",   0.05, True,  "Small spaced capitals, centered."),
    ("url",      0.04, True,  "Small lowercase web address, centered."),
)


def moss_layout_prompt(summary, fields, width, height):
    """Return the MOSS layout JSON string, or None when no field is filled."""
    active = []
    for key, weight, is_text, caption in MOSS_LAYOUT_SLOTS:
        val = (fields.get(key) or "").strip()
        if not val:
            continue
        if key == "lines":
            lines = [ln.strip() for ln in val.splitlines() if ln.strip()]
            val = "\n".join(lines)
            weight = min(0.45, 0.05 * len(lines) + 0.02)
        active.append((key, weight, is_text, caption, val))
    if not active:
        return None
    margin_x, top, bottom, gap = 0.08, 0.06, 0.05, 0.025
    usable = 1.0 - top - bottom - gap * (len(active) - 1)
    total = sum(a[1] for a in active)
    y = top
    items = []
    for key, weight, is_text, caption, val in active:
        h = weight / total * usable
        x0, x1 = round(width * margin_x), round(width * (1 - margin_x))
        y0, y1 = round(height * y), round(height * (y + h))
        if key == "art":
            # art sits in the middle 70% of the width so it reads as an illustration, not a band
            x0, x1 = round(width * 0.15), round(width * 0.85)
            item = {"label": "object", "caption": val}
        else:
            item = {"label": "text", "caption": caption, "text": val}
        item["bbox"] = [y0 - height // 2, x0 - width // 2, y1 - height // 2, x1 - width // 2]
        items.append(item)
        y += h + gap
    summary = (summary or "").strip() or ("Clean printed poster on a flat background, sharp evenly kerned "
                                          "typography, every word spelled exactly as written.")
    layout = {"summary": {"caption": summary}, "items": items,
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def build_yue2(style, lyrics, seconds, seed, plan="off", loras=(), prefix="audio/Backlot_song"):
    wf = {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": YUE2_CKPT}}}
    model, clip = ["ckpt", 0], ["ckpt", 1]
    for i, (name, strength) in enumerate(loras):
        meta = YUE2_LORAS[name]
        nid = f"lora{i}"
        wf[nid] = {"class_type": "LoraLoader", "inputs": {
            "model": model, "clip": clip, "lora_name": name,
            "strength_model": strength if meta["slot"] in ("model", "both") else 0.0,
            "strength_clip": strength if meta["slot"] in ("clip", "both") else 0.0}}
        model, clip = [nid, 0], [nid, 1]
    abc = ""
    if plan in ("full", "melody"):
        wf["abc"] = {"class_type": "YuE2GenerateABC", "inputs": {
            "clip": clip, "style": style, "lyrics": lyrics, "seed": seed, "mode": plan,
            "max_abc_tokens": 8192, "temperature": 0.7, "top_p": 0.9, "top_k": 30,
            "repetition_penalty": 1.005, "penalty_window": 100}}
        abc = ["abc", 0]
    wf["music"] = {"class_type": "YuE2GenerateMusic", "inputs": {
        "clip": clip, "style": style, "lyrics": lyrics, "abc": abc, "seed": seed,
        "mode": "full" if plan == "off" else plan, "max_duration": float(seconds),
        "temperature": 1.0, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.2}}
    wf["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["music", 0]}}
    wf["latent"] = {"class_type": "EmptyYuE2LatentAudio", "inputs": {"seconds": ["music", 1], "batch_size": 1}}
    wf["sample"] = {"class_type": "KSampler", "inputs": {
        "model": model, "positive": ["music", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
        "seed": seed, "steps": 32, "cfg": 1.0, "sampler_name": "dpm_2", "scheduler": "sgm_uniform", "denoise": 1.0}}
    wf["decode"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["ckpt", 2]}}
    wf["save"] = {"class_type": "SaveAudioFloatWAV", "inputs": {"audio": ["decode", 0], "filename_prefix": prefix}}
    return wf


def build_sa3(prompt, seconds, seed, steps=8, prefix="audio/Backlot_sfx"):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SA3_CKPT}},
        "9": {"class_type": "CLIPLoader", "inputs": {"clip_name": SA3_TE, "type": "stable_audio", "device": "default"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": f"{prompt} Length: {seconds} seconds"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": ""}},
        "4": {"class_type": "ConditioningStableAudio", "inputs": {
            "positive": ["2", 0], "negative": ["3", 0], "seconds_start": 0.0, "seconds_total": float(seconds)}},
        "5": {"class_type": "EmptyLatentAudio", "inputs": {"seconds": float(seconds), "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0],
            "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "lcm", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveAudioFloatWAV", "inputs": {"audio": ["7", 0], "filename_prefix": prefix}},
    }


def wav_to_mp3(wav_bytes, out_path, title=None):
    """Loudness-normalize (-14 LUFS, -1.5 dBTP, the streaming target) and write 320k MP3.
    ``title`` goes into the ID3 tag so players show it."""
    tmp = out_path + ".wav"
    with open(tmp, "wb") as f:
        f.write(wav_bytes)
    tags = ["-metadata", f"title={title}", "-metadata", "artist=Backlot"] if title else []
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                        "-codec:a", "libmp3lame", "-b:a", "320k", "-id3v2_version", "3", *tags, out_path],
                       check=True, timeout=300)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def submit_and_wait(wf, timeout=1800, exts=(".mp4", ".webm", ".gif"), base=None, staging=None, backend=None):
    """Submit a graph through the Comfy SDK and wait for it.
    Returns (bytes, ext, None); the third slot is kept for call-site compatibility."""
    be = backend or BACKEND
    jid = getattr(JOB_CTX, "jid", None)

    def on_progress(pr):
        with JOBS_LOCK:
            j = JOBS.get(jid)
            if j and j["state"] == "running":
                j["progress"] = pr
    ACTIVE_JOBS["n"] += 1
    try:
        blob, ext = be.run(wf, timeout=timeout, exts=exts, on_progress=on_progress if jid else None)
    finally:
        ACTIVE_JOBS["n"] -= 1
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


def save_output(blob, kind="vid", ext="mp4", staged=None, directory=None):
    """File the result under OUTPUT_ROOT/videos or OUTPUT_ROOT/images with a timestamped name."""
    directory = directory or VIDEO_DIR
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    name = f"{kind}_{stamp}.{ext}"
    path = os.path.join(directory, name)
    i = 1
    while os.path.exists(path):
        name = f"{kind}_{stamp}_{i}.{ext}"
        path = os.path.join(directory, name)
        i += 1
    with open(path, "wb") as f:
        f.write(blob)
    if staged and os.path.isfile(staged):
        try:
            if os.path.getsize(staged) == len(blob):
                os.remove(staged)
        except OSError:
            pass
    return name, path


def probe_fps(path, default=24.0):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=20).stdout.strip()
        num, _, den = out.partition("/")
        val = float(num) / float(den or 1)
        return val if val > 0 else default
    except Exception:
        return default


# ---------- HTTP ----------
class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        if code == 200 and ctype.startswith("application/json") and getattr(self, "_body", None) is not None:
            self._record_meta(body)
        jid = getattr(self, "_job", None)
        if jid:
            with JOBS_LOCK:
                JOBS[jid].update({"state": "done", "code": code, "body": body,
                                  "ctype": ctype, "t1": time.time()})
            return
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record_meta(self, body):
        """Tag every finished output with what made it, so the gallery can show and reuse it."""
        try:
            d = json.loads(body)
            url = d.get("url") or ""
            if not (url.startswith("/outputs/") or url.startswith("/images/") or url.startswith("/audio/")):
                return
            req = self._body or {}
            mode = req.get("mode") or ""
            label = str(req.get("label") or "").strip()
            if not label:
                label = {"t2v": "Video", "i2v": "Image to Video", "ref": "Reference", "orbit": "360 Orbit",
                         "swap": "Character Swap", "edit": "Edit", "angles": "Angles",
                         "song": "Song", "sfx": "Sound FX"}.get(mode, mode or "render")
            meta = {"label": label[:80], "mode": mode, "move": req.get("move"), "seed": d.get("seed"),
                    "prompt": str(req.get("prompt") or req.get("subject") or req.get("scene") or "")[:400],
                    "took": d.get("took"), "made": time.strftime("%Y-%m-%d %H:%M")}
            with open(os.path.join(META_DIR, os.path.basename(url) + ".json"), "w") as f:
                json.dump(meta, f)
        except Exception:
            pass

    def _authed(self):
        if not PASSWORD:
            return True
        hdr = self.headers.get("Authorization", "")
        if hdr.startswith("Basic "):
            try:
                _, _, pw = base64.b64decode(hdr[6:]).decode().partition(":")
                if pw == PASSWORD:
                    return True
            except Exception:
                pass
        body = b'{"error": "password required"}'
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Backlot"')
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return False

    def do_GET(self):
        if not self._authed():
            return
        p = urllib.parse.urlparse(self.path)
        if p.path.startswith("/api/jobs/"):
            jid = p.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                j = dict(JOBS.get(jid) or {})
            if not j:
                return self._send(404, json.dumps({"error": "no such job"}))
            if j["state"] != "done":
                return self._send(202, json.dumps({"job": jid, "state": "running",
                                                   "elapsed": round(time.time() - j["t0"], 1),
                                                   "progress": j.get("progress")}))
            return self._send(j["code"], j["body"], j["ctype"])
        if p.path == "/api/jobs":
            with JOBS_LOCK:
                out = [{"job": k, "state": j["state"], "path": j["path"],
                        "elapsed": round((j.get("t1") or time.time()) - j["t0"], 1)}
                       for k, j in JOBS.items()]
            return self._send(200, json.dumps(out))
        if p.path == "/favicon.ico":
            with open(os.path.join(HERE, "icon", "png", "backlot-32.png"), "rb") as f:
                return self._send(200, f.read(), "image/png")
        if p.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if p.path == "/api/cinema":
            return self._send(200, json.dumps({k: v for k, v in CINEMA.items() if not k.startswith("_")}))
        if p.path == "/api/moves":
            out = []
            for m in MOVES.values():
                pv = os.path.join(PREVIEW_DIR, "moves", m["id"] + ".mp4")
                d = {k: v for k, v in m.items() if k != "preview_subject"}
                d["preview"] = f"/previews/moves/{m['id']}.mp4" if os.path.isfile(pv) else None
                d["poster"] = (f"/previews/moves/{m['id']}.jpg"
                               if os.path.isfile(pv[:-4] + ".jpg") else None)
                out.append(d)
            return self._send(200, json.dumps({"moves": out}))
        if p.path.startswith("/previews/"):
            rel = os.path.normpath(urllib.parse.unquote(p.path[len("/previews/"):]))
            fp = os.path.join(PREVIEW_DIR, rel)
            if rel.startswith("..") or not os.path.isfile(fp):
                return self._send(404, json.dumps({"error": "not found"}))
            mime = "video/mp4" if fp.endswith(".mp4") else "image/jpeg"
            with open(fp, "rb") as f:
                return self._send(200, f.read(), mime)
        if p.path == "/api/heartbeat":
            HEARTBEAT["last"] = time.time()
            HEARTBEAT["seen"] = True
            return self._send(200, json.dumps({"ok": True}))
        if p.path == "/api/status":
            return self._send(200, json.dumps({
                "comfy": comfy_alive(),
                "idle": {"window_seen": HEARTBEAT["seen"],
                         "since_heartbeat": round(time.time() - HEARTBEAT["last"], 1),
                         "jobs": ACTIVE_JOBS["n"]}}))
        if p.path == "/api/chains":
            # Every chained project gets its OWN latent folder so two projects can
            # never overwrite each other's slots. Folder name IS the project id.
            base = COMFY_STAGING
            out = []
            try:
                for d in sorted(os.listdir(base)):
                    if not d.startswith("h3_context"):
                        continue
                    full = os.path.join(base, d)
                    if not os.path.isdir(full):
                        continue
                    slots = sorted(
                        int(f.rsplit("_", 1)[1].split(".")[0])
                        for f in os.listdir(full) if f.endswith(".safetensors") and "_" in f)
                    out.append({"id": d, "slots": len(slots),
                                "highest": (max(slots) if slots else 0),
                                "next": (max(slots) + 1 if slots else 1)})
            except FileNotFoundError:
                pass
            return self._send(200, json.dumps({"chains": out}), "application/json")

        if p.path == "/api/gallery":
            kind = (urllib.parse.parse_qs(p.query).get("kind") or ["video"])[0]
            if kind == "image":
                folder, want = IMAGE_DIR, (".png", ".jpg", ".jpeg", ".webp")
            elif kind == "audio":
                folder, want = AUDIO_DIR, (".mp3", ".wav")
            else:
                folder, want = VIDEO_DIR, (".mp4", ".webm", ".gif")
            try:
                files = sorted([f for f in os.listdir(folder)
                                if f.lower().endswith(want)], reverse=True)[:60]
            except OSError:
                files = []
            tags = {}
            for f in files:
                mp = os.path.join(META_DIR, f + ".json")
                if os.path.isfile(mp):
                    try:
                        with open(mp) as fh:
                            tags[f] = json.load(fh)
                    except (OSError, ValueError):
                        pass
            return self._send(200, json.dumps({"files": files, "tags": tags}))
        if p.path.startswith("/images/"):
            name = os.path.basename(urllib.parse.unquote(p.path[len("/images/"):]))
            fp = os.path.join(IMAGE_DIR, name)
            if os.path.isfile(fp):
                mime = "image/png" if name.lower().endswith(".png") else "image/jpeg"
                with open(fp, "rb") as f:
                    return self._send(200, f.read(), mime)
            return self._send(404, json.dumps({"error": "not found"}))
        if p.path.startswith("/audio/"):
            name = os.path.basename(urllib.parse.unquote(p.path[len("/audio/"):]))
            fp = os.path.join(AUDIO_DIR, name)
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    return self._send(200, f.read(), "audio/mpeg" if name.endswith(".mp3") else "audio/wav")
            return self._send(404, json.dumps({"error": "not found"}))
        if p.path == "/api/audio/loras":
            return self._send(200, json.dumps([{"file": k, **v} for k, v in YUE2_LORAS.items()]))
        if p.path.startswith("/outputs/"):
            name = os.path.basename(urllib.parse.unquote(p.path[len("/outputs/"):]))
            fp = os.path.join(VIDEO_DIR, name)
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    return self._send(200, f.read(), "video/mp4")
            return self._send(404, json.dumps({"error": "not found"}))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if not self._authed():
            return
        p = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        if p.path == "/api/closing":
            # pagehide beacon: the window is going away. Pull the heartbeat clock back so
            # the idle watchdog frees the card in CLOSE_GRACE seconds unless a window
            # (a reload, or another tab) heartbeats first.
            self.rfile.read(length)
            HEARTBEAT["last"] = min(HEARTBEAT["last"], time.time() - (IDLE_FREE_AFTER - CLOSE_GRACE))
            return self._send(200, json.dumps({"ok": True}))
        body = json.loads(self.rfile.read(length)) if length else {}

        if p.path in JOB_PATHS and self.headers.get("X-Backlot-Async") == "1":
            jid = _new_job(p.path)
            # The worker gets its own shallow copy of this handler so its _send() records
            # into the job while this request answers 202 on the real socket.
            worker = __import__("copy").copy(self)
            worker._job = jid

            def run():
                JOB_CTX.jid = jid
                try:
                    worker._dispatch(p, body)
                    with JOBS_LOCK:
                        if JOBS[jid]["state"] != "done":   # handler returned without _send
                            JOBS[jid].update({"state": "done", "code": 500, "t1": time.time(),
                                              "ctype": "application/json",
                                              "body": b'{"error": "no response from handler"}'})
                except Exception as e:
                    with JOBS_LOCK:
                        JOBS[jid].update({"state": "done", "code": 500, "t1": time.time(),
                                          "ctype": "application/json",
                                          "body": json.dumps({"error": str(e)}).encode()})

            __import__("threading").Thread(target=run, daemon=True).start()
            return self._send(202, json.dumps({"job": jid, "state": "running"}))
        return self._dispatch(p, body)

    def _dispatch(self, p, body):
        self._body = body if isinstance(body, dict) else {}
        if p.path == "/api/cinema/compose":
            try:
                if not str(body.get("scene") or "").strip() and not any(str(x).strip() for x in (body.get("shots") or [])):
                    return self._send(400, json.dumps({"error": "describe the scene first"}))
                prompt, cuts = compose_cinema_prompt(body)
                return self._send(200, json.dumps({"prompt": prompt, "cuts": cuts, "shots": len(cuts)}))
            except Exception as e:
                return self._send(400, json.dumps({"error": str(e)}))
        if p.path == "/api/video":
            try:
                if not comfy_up():
                    return self._send(503, json.dumps(
                        {"error": BACKEND_DOWN}))
                prompt = (body.get("prompt") or "").strip()
                mode = (body.get("mode") or "t2v").lower()
                # ---- Moves (Oct 20 2026): a preset composes the prompt + LoRA settings ----
                move_id = (body.get("move") or "").strip()
                move = None
                if move_id:
                    move = MOVES.get(move_id)
                    if not move:
                        return self._send(400, json.dumps({"error": f"unknown move: {move_id}"}))
                    if mode not in ("t2v", "i2v"):
                        return self._send(400, json.dumps({"error": "Moves work in Text and Image modes"}))
                    subject = (body.get("subject") or prompt).strip()
                    if not subject:
                        return self._send(400, json.dumps({"error": "describe the subject first"}))
                    prompt = compose_move_prompt(move, subject)
                    body = dict(body)
                    body["camera_motion"] = bool(move.get("lora"))
                    body["camera_strength"] = float(move.get("strength") or 1.0)
                    body.setdefault("seconds", move.get("seconds", 5))
                if not prompt and mode not in ("orbit", "swap"):   # orbit and swap write their own prompt
                    return self._send(400, json.dumps({"error": "prompt required"}))
                width = int(body.get("width", 1344))
                height = int(body.get("height", 768))
                seconds = float(body.get("seconds", 5.0))
                seed = int(body.get("seed", -1))
                if seed < 0:
                    seed = int.from_bytes(os.urandom(4), "big")

                # ---- LTX-2.5 (Oct 20 2026): the fast second video model ----
                # Distilled 22B, two-stage pixel pipeline from ComfyUI's own template (see
                # workflows/build_ltx25_workflow.py). Text and Image to Video only; Moves, Cinema,
                # Reference, Orbit and chaining stay on H3 because their LoRAs and prompt format are H3's.
                if body.get("model") == "ltx25":
                    if mode not in ("t2v", "i2v"):
                        return self._send(400, json.dumps({"error": "LTX-2.5 does Text and Image to Video here"}))
                    seconds = max(2.0, min(LTX_MAX_SECONDS, seconds))
                    width = max(256, round(width / 64) * 64)
                    height = max(256, round(height / 64) * 64)
                    first_frame = None
                    try:
                        if mode == "i2v":
                            img = body.get("image") or ""
                            if not img:
                                return self._send(400, json.dumps({"error": "an input image is required for Image to Video"}))
                            from PIL import Image
                            import io
                            im = Image.open(io.BytesIO(base64.b64decode(img.split(",")[-1]))).convert("RGB")
                            sw, sh = im.size
                            if sw / sh > width / height:
                                cw, ch = round(sh * width / height), sh
                            else:
                                cw, ch = sw, round(sw * height / width)
                            x, y = (sw - cw) // 2, (sh - ch) // 2
                            im = im.crop((x, y, x + cw, y + ch)).resize((width, height), Image.LANCZOS)
                            buf = io.BytesIO()
                            im.save(buf, "PNG")
                            first_frame = upload_to_comfy(buf.getvalue(), f"backlot_ltx_i2v_{int(time.time())}.png")
                        wf, frames = build_ltx25(prompt, width=width, height=height, seconds=seconds,
                                                 fps=LTX_FPS, seed=seed, image=first_frame, prefix="video/LTX25")
                        t0 = time.time()
                        blob, ext, staged = submit_and_wait(wf, timeout=1800, exts=(".mp4", ".webm", ".mkv"))
                        name, path = save_output(blob, kind="ltx-" + mode, ext=ext, staged=staged)
                        return self._send(200, json.dumps({
                            "url": f"/outputs/{name}", "file": path, "seed": seed, "frames": frames,
                            "seconds": round(frames / LTX_FPS, 2), "took": round(time.time() - t0, 1),
                            "mode": mode, "model": "ltx25", "width": width, "height": height}))
                    finally:
                        if first_frame:
                            try:
                                discard_upload(first_frame)
                            except OSError:
                                pass

                # Camera Motion LoRA. Its trigger word has to LEAD the prompt.
                # 🔴 For i2v the builder prepends MiniMax's required alignment
                # line, so the trigger ends up second. Untested in that combo.
                camera = bool(body.get("camera_motion", False))
                cam_strength = float(body.get("camera_strength", 1.0))
                if camera and not prompt.lower().startswith(CAMERA_TRIGGER):
                    prompt = f"{CAMERA_TRIGGER}, {prompt}"

                # ---- 360 Orbit (Oct 10 2026, the user: "drop in an ORB 360 button") ----
                # MatlowAI ORB360 LoRA on Ref2VA. 1 photo, or 3 = front / 120 / 240 degrees clockwise.
                # The author's own caption is the prompt and its timing is fixed at 124 frames
                # (5.17 s, milestones at frames 41/82/123), so prompt + length are not user-set.
                # Sizes = the author's tested ones. Burger test: identity held all the way round;
                # with ONE photo it invents the hidden back side (3 photos fix that).
                if mode == "orbit":
                    imgs = body.get("images") or []
                    if len(imgs) not in (1, 3):
                        return self._send(400, json.dumps(
                            {"error": "360 Orbit takes 1 photo, or 3 (front, then 120 and 240 degrees clockwise)"}))
                    with open(os.path.join(HERE, "prompts", ORB360_PROMPTS[len(imgs)])) as f:
                        prompt = f.read()
                    width, height = ORB360_SIZES[("land" if width > height else "port" if height > width else "square")]
                    names = []
                    for i, img in enumerate(imgs):
                        raw = base64.b64decode(str(img).split(",")[-1])
                        names.append(upload_to_comfy(raw, f"mmstudio_orbit{i}_{int(time.time())}.png"))
                    wf, frames = build_ref2va(
                        prompt, ref_images=names, width=width, height=height,
                        seconds=ORB360_FRAMES / H3_FPS, steps=H3_STEPS, seed=seed,
                        low_vram_chunks=H3_HEAD_CHUNKS, fps=H3_FPS, clip_cpu=True,
                        prefix="video/MiniMaxH3Orbit", extra_loras=((ORB360_LORA, 1.0),))
                    t0 = time.time()
                    blob, ext, staged = submit_and_wait(wf, timeout=3600)
                    name, path = save_output(blob, kind="orbit", ext=ext, staged=staged)
                    for nm in names:
                        try:
                            discard_upload(nm)
                        except OSError:
                            pass
                    return self._send(200, json.dumps({
                        "url": f"/outputs/{name}", "file": path, "seed": seed,
                        "frames": frames, "seconds": round(frames / H3_FPS, 2),
                        "took": round(time.time() - t0, 1), "mode": mode, "refs": len(names),
                        "width": width, "height": height}))

                # ---- Character Swap (Apps tab, Oct 20 2026) ----
                if mode == "swap":
                    vid = body.get("video") or ""
                    img = body.get("image") or ""
                    if not vid or not img:
                        return self._send(400, json.dumps(
                            {"error": "Character Swap needs a source video and a character image"}))
                    vraw = base64.b64decode(str(vid).split(",")[-1])
                    iraw = base64.b64decode(str(img).split(",")[-1])
                    stamp = int(time.time())
                    vname = upload_to_comfy(vraw, f"backlot_swap_src_{stamp}.mp4")
                    iname = upload_to_comfy(iraw, f"backlot_swap_char_{stamp}.png")
                    # length follows the source clip (capped); probe it from a local copy
                    tmp = stage_local(vraw, f"swap_probe_{stamp}.mp4")
                    try:
                        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                              "-of", "csv=p=0", tmp], capture_output=True, text=True).stdout.strip()
                        src_seconds = float(out) if out else seconds
                    except Exception:
                        src_seconds = seconds
                    finally:
                        cleanup_local(tmp)
                    seconds = max(2.0, min(SWAP_MAX_SECONDS, src_seconds))
                    who = (body.get("subject") or "the person").strip()
                    prompt = (f"subject_definitions:\n<Subject 1> is the character in <Picture 1>: keep this face, "
                              f"hair, skin and clothing exactly.\n<Subject 2> is the performance in <Video 1>: every movement, "
                              f"gesture, camera move, timing and the spoken audio.\n\nsummary:\nReplace {who} in <Video 1> with "
                              f"<Subject 1>, keeping <Subject 2> unchanged.\n\nretention_analysis:\n<Subject 2> is fully preserved: "
                              f"framing, camera motion, body movement, lip movement and the soundtrack carry over unaltered. "
                              f"<Subject 1> replaces the face, hair and clothing of {who} and nothing else changes.\n\n"
                              f"detailed_description:\n[Shot 1] The same shot as <Video 1> from first frame to last, with "
                              f"<Subject 1> in place of {who}, matching every pose and expression in sync with the original.\n\n"
                              f"overall_soundscape:\nIdentical to <Video 1>.\n\nnon_diegetic_music:\nN/A")
                    wf, frames = build_ref2va(
                        prompt, ref_images=[iname], width=width, height=height,
                        seconds=seconds, steps=H3_STEPS, seed=seed, scheduler="normal",
                        low_vram_chunks=H3_HEAD_CHUNKS, fps=H3_FPS, clip_cpu=True,
                        prefix="video/BacklotSwap", extra_loras=((SWAP_LORA, 1.0),))
                    wf["9"]["inputs"]["sampler_name"] = "er_sde"
                    wf["60"] = {"class_type": "LoadVideo", "inputs": {"file": vname}}
                    wf["61"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["60", 0]}}
                    wf["8"]["inputs"]["ref_videos.ref_video_0"] = ["61", 0]
                    wf["8"]["inputs"]["ref_video_audios.ref_video_audio_0"] = ["61", 1]
                    t0 = time.time()
                    blob, ext, staged = submit_and_wait(wf, timeout=3600)
                    name, path = save_output(blob, kind="swap", ext=ext, staged=staged)
                    for nm in (vname, iname):
                        try:
                            discard_upload(nm)
                        except OSError:
                            pass
                    return self._send(200, json.dumps({
                        "url": f"/outputs/{name}", "file": path, "seed": seed, "frames": frames,
                        "seconds": round(frames / H3_FPS, 2), "took": round(time.time() - t0, 1),
                        "mode": mode, "width": width, "height": height}))

                # ---- Ref2VA: its own DiT, own turbo LoRA, own builder ----
                if mode == "ref":
                    imgs = body.get("images") or []
                    if not imgs:
                        return self._send(400, json.dumps(
                            {"error": "Reference mode needs at least one image"}))
                    if len(imgs) > 9:
                        return self._send(400, json.dumps(
                            {"error": "Ref2VA accepts at most 9 reference images"}))
                    # Reference AUDIO is optional. Each item is {name, data};
                    # the ORIGINAL extension has to survive the upload because
                    # ComfyUI's LoadAudio resolves the decoder by file suffix.
                    auds = body.get("audios") or []
                    if len(auds) > 3:
                        return self._send(400, json.dumps(
                            {"error": "Ref2VA accepts at most 3 reference audio clips"}))
                    names = []
                    for i, img in enumerate(imgs):
                        raw = base64.b64decode(str(img).split(",")[-1])
                        names.append(upload_to_comfy(
                            raw, f"mmstudio_ref{i}_{int(time.time())}.png"))
                    anames = []
                    for i, aud in enumerate(auds):
                        data = aud.get("data") if isinstance(aud, dict) else aud
                        orig = (aud.get("name") if isinstance(aud, dict) else "") or ""
                        ext = os.path.splitext(orig)[1].lower()
                        if ext not in (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"):
                            ext = ".wav"
                        raw = base64.b64decode(str(data).split(",")[-1])
                        anames.append(upload_to_comfy(
                            raw, f"mmstudio_refaud{i}_{int(time.time())}{ext}"))
                    # Motion Context chaining. 0 = off; 1 = first clip of a chain
                    # (saves its latent); 2+ = continue clip N-1 in LATENT space.
                    # 🔴 Chaining through a PNG loses the join: the returned first
                    # frame is REINTERPRETED (measured 14-16% pixel difference) and
                    # a still frame carries no velocity, so motion restarts.
                    try:
                        chain_index = int(body.get("chain_index") or 0)
                    except (TypeError, ValueError):
                        chain_index = 0
                    if chain_index < 0:
                        return self._send(400, json.dumps(
                            {"error": "chain_index must be 0 or greater"}))
                    # Folder name for this project's chain. Type a plain name like
                    # "captain jacks" — spaces become hyphens, junk is dropped, and
                    # the h3_context prefix is added so all chains group together.
                    proj = str(body.get("chain_project") or "h3_context").strip()
                    proj = re.sub(r"\s+", "-", proj)
                    proj = re.sub(r"[^A-Za-z0-9_-]", "", proj)
                    proj = re.sub(r"-{2,}", "-", proj).strip("-") or "h3_context"
                    if not proj.startswith("h3_context"):
                        proj = "h3_context-" + proj
                    # 🔴 PRE-FLIGHT: the chain is strictly sequential. Clip N loads
                    # slot N-1, so a gap fails INSIDE ComfyUI only after the models
                    # have loaded — the user burned 186s on that. Check it here instead
                    # and say exactly which slots exist. (2026-09-22)
                    if chain_index >= 2 and COMFY_STAGING:
                        ctx_dir = os.path.join(COMFY_STAGING, proj)
                        want = "_%05d.safetensors" % (chain_index - 1)
                        have = sorted(f for f in os.listdir(ctx_dir)
                                      if f.endswith(".safetensors")) \
                            if os.path.isdir(ctx_dir) else []
                        if not any(f.endswith(want) for f in have):
                            slots = sorted({int(f.rsplit("_", 1)[1].split(".")[0])
                                            for f in have if "_" in f})
                            msg = (f"Chain gap: clip {chain_index} needs clip "
                                   f"{chain_index - 1}'s latent and it does not exist. ")
                            msg += (f"Saved so far: clip(s) {slots}. Render clip "
                                    f"{(max(slots) + 1) if slots else 1} next."
                                    if slots else
                                    "Nothing is saved yet - start the chain at 1.")
                            return self._send(400, json.dumps({"error": msg}))
                    wf, frames = build_ref2va(
                        prompt, ref_images=names, ref_audios=anames,
                        chain_index=chain_index, context_prefix=proj + "/clip",
                        width=width, height=height,
                        seconds=seconds, steps=H3_STEPS, seed=seed,
                        low_vram_chunks=H3_HEAD_CHUNKS, fps=H3_FPS, clip_cpu=True,
                        prefix="video/MiniMaxH3Ref",
                        # 🔴 The Camera Motion LoRA used to be t2v/i2v ONLY. The trigger
                        # word was still being prepended to the prompt in EVERY mode
                        # (see above), so ticking the box on a reference render polluted
                        # the prompt with "camera motion," and loaded no LoRA at all —
                        # a silent no-op. build_ref2va chains extra_loras after the
                        # ref2v turbo LoRA, same as build_h3 does. Wired 2026-09-20.
                        extra_loras=((CAMERA_LORA, cam_strength),) if camera else ())
                    t0 = time.time()
                    # References add context tokens and attention is quadratic:
                    # 2 refs measured 725s vs 452s for plain t2v at the same size.
                    blob, ext, staged = submit_and_wait(wf, timeout=3600)
                    name, path = save_output(blob, kind="ref", ext=ext, staged=staged)
                    for nm in names + anames:   # don't litter ComfyUI's input dir
                        try:
                            discard_upload(nm)
                        except OSError:
                            pass
                    return self._send(200, json.dumps({
                        "url": f"/outputs/{name}", "file": path, "seed": seed,
                        "frames": frames, "seconds": round(frames / H3_FPS, 2),
                        "took": round(time.time() - t0, 1), "mode": mode,
                        "refs": len(names), "audio_refs": len(anames),
                        "chain_index": chain_index,
                        "camera_motion": camera,
                    }))

                first_frame = None
                start_crop = None
                if mode == "i2v":
                    img = body.get("image") or ""
                    if not img:
                        return self._send(400, json.dumps(
                            {"error": "an input image is required for Image to Video"}))
                    raw = base64.b64decode(img.split(",")[-1])
                    # Oct 5 2026 (the user): H3's MiniMaxH3ImageToVideo STRETCHES the first frame
                    # to the canvas ("plain stretch", only the last frame is cover-cropped), so a
                    # 3:2 still in a 16:9 video came out ~17% wider. Center-crop it to the video's
                    # exact shape and size here so nothing is ever stretched.
                    from PIL import Image
                    import io
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                    sw, sh = im.size
                    if sw / sh > width / height:      # too wide: trim the sides
                        cw, ch = round(sh * width / height), sh
                    else:                             # too tall: trim top and bottom
                        cw, ch = sw, round(sw * height / width)
                    x, y = (sw - cw) // 2, (sh - ch) // 2
                    im = im.crop((x, y, x + cw, y + ch)).resize((width, height), Image.LANCZOS)
                    start_crop = {"from": [sw, sh], "crop": [x, y, cw, ch], "to": [width, height],
                                  "trimmed": (cw, ch) != (sw, sh)}
                    buf = io.BytesIO()
                    im.save(buf, "PNG")
                    first_frame = upload_to_comfy(buf.getvalue(), f"mmstudio_i2v_{int(time.time())}.png")

                wf, frames = build_h3(
                    prompt, width=width, height=height, seconds=seconds,
                    steps=H3_STEPS, seed=seed, turbo=True,
                    low_vram_chunks=H3_HEAD_CHUNKS, fps=H3_FPS,
                    clip_cpu=True, first_frame=first_frame,
                    extra_lora=CAMERA_LORA if camera else None,
                    extra_lora_strength=cam_strength)
                t0 = time.time()
                blob, ext, staged = submit_and_wait(wf, timeout=1800)
                name, path = save_output(blob, kind="vid", ext=ext, staged=staged)
                return self._send(200, json.dumps({
                    "url": f"/outputs/{name}", "file": path, "seed": seed,
                    "frames": frames, "seconds": round(frames / H3_FPS, 2),
                    "took": round(time.time() - t0, 1), "mode": mode,
                    "camera_motion": camera, "start_crop": start_crop,
                }))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))

        if p.path == "/api/free":
            vlo_free()
            freed = comfy_free()
            return self._send(200, json.dumps({
                "ok": True, "freed_mib": freed, "used_mib": gpu_used_mib()}))

        if p.path == "/api/audio":
            try:
                engine = body.get("engine", "song")
                seed = int(body.get("seed", -1))
                if seed < 0:
                    seed = int.from_bytes(os.urandom(4), "big")
                if engine == "sfx":
                    prompt = (body.get("prompt") or "").strip()
                    if not prompt:
                        return self._send(400, json.dumps({"error": "describe the sound first"}))
                    seconds = max(1, min(SFX_MAX_SECONDS, int(float(body.get("seconds", 6)))))
                    wf = build_sa3(prompt, seconds, seed)
                    label = prompt
                elif engine == "song":
                    style = (body.get("style") or "").strip()
                    lyrics = (body.get("lyrics") or "").strip()
                    if not style:
                        return self._send(400, json.dumps({"error": "give the song a style (genre, mood, voice, tempo)"}))
                    if not lyrics:
                        return self._send(400, json.dumps({"error": "write some lyrics, or section tags like [verse] [chorus] for an instrumental"}))
                    plan = body.get("plan", "off")
                    if plan not in ("off", "full", "melody"):
                        return self._send(400, json.dumps({"error": "plan must be off, full or melody"}))
                    seconds = max(10, min(SONG_MAX_SECONDS, int(float(body.get("seconds", 60)))))
                    loras = []
                    for item in (body.get("loras") or [])[:6]:
                        name = str(item.get("file", "")) if isinstance(item, dict) else str(item)
                        if name in YUE2_LORAS:
                            strength = float(item.get("strength", YUE2_LORAS[name]["default"])) if isinstance(item, dict) else YUE2_LORAS[name]["default"]
                            loras.append((name, max(0.0, min(1.5, strength))))
                    wf = build_yue2(style, lyrics, seconds, seed, plan=plan, loras=loras)
                    label = style
                else:
                    return self._send(400, json.dumps({"error": "engine must be song or sfx"}))
                if not comfy_up():
                    return self._send(503, json.dumps({"error": BACKEND_DOWN}))
                t0 = time.time()
                blob, ext, _ = submit_and_wait(wf, timeout=1800, exts=(".wav", ".flac"))
                title = re.sub(r"\s+", " ", str(body.get("title") or "")).strip()[:80]
                if title:
                    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "song"
                    name, i = f"{slug}.mp3", 2
                    while os.path.exists(os.path.join(AUDIO_DIR, name)):
                        name, i = f"{slug}-{i}.mp3", i + 1
                    label = title
                else:
                    name = f"{engine}_{time.strftime('%Y-%m-%d_%H%M%S')}.mp3"
                path = os.path.join(AUDIO_DIR, name)
                wav_to_mp3(blob, path, title=title or None)
                return self._send(200, json.dumps({
                    "url": f"/audio/{name}", "file": path, "seed": seed, "seconds": seconds, "engine": engine,
                    "title": title, "label": label[:80], "took": round(time.time() - t0, 1)}))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))

        if p.path == "/api/image" and body.get("model") == "moss":
            # MOSS-Image (see MOSS_* above). text = plain prompt; layout = the poster form.
            try:
                prompt = (body.get("prompt") or "").strip()
                mode = body.get("mode", "text")
                if mode not in ("text", "layout"):
                    return self._send(400, json.dumps({"error": "MOSS supports text or layout"}))
                seed = int(body.get("seed", -1))
                if seed < 0:
                    seed = int.from_bytes(os.urandom(4), "big")
                steps = max(15, min(50, int(body.get("steps", MOSS_STEPS))))
                width, height = int(body.get("width", 1024)), int(body.get("height", 1408))
                if width * height > MOSS_MAX_PIXELS:
                    k = (MOSS_MAX_PIXELS / (width * height)) ** 0.5
                    width, height = width * k, height * k
                width = max(256, round(width / 16) * 16)
                height = max(256, round(height / 16) * 16)
                if mode == "layout":
                    text = moss_layout_prompt(prompt, body.get("layout") or {}, width, height)
                    if text is None:
                        return self._send(400, json.dumps({"error": "fill in at least one layout field"}))
                else:
                    if not prompt:
                        return self._send(400, json.dumps({"error": "prompt required"}))
                    text = prompt
                if not comfy_up():
                    return self._send(503, json.dumps({"error": BACKEND_DOWN}))
                wf = build_moss(text, width, height, steps=steps, seed=seed)
                t0 = time.time()
                blob, ext, staged = submit_and_wait(wf, timeout=1800, exts=(".png",))
                kind = "moss-layout" if mode == "layout" else "moss"
                name, path = save_output(blob, kind=kind, ext=ext, staged=staged, directory=IMAGE_DIR)
                return self._send(200, json.dumps({
                    "url": f"/images/{name}", "file": path, "seed": seed, "steps": steps,
                    "width": width, "height": height, "mode": mode, "model": "moss",
                    "denoise": None, "took": round(time.time() - t0, 1)}))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))

        if p.path == "/api/image" and body.get("model") == "qwen21":
            # Qwen Image 2.1 + Kijai turbo LoRA (ComfyUI 0.38+; on 0.42+ the same backend as H3). Two modes:
            #   text = text-to-image at the aspect buttons' size (multiples of 32);
            #   edit = instruction edit of 1..10 reference images, output in image 1's shape
            #          at ~1 MP (the encoder sizes the latent off the first reference).
            # transparent=True asks for real RGBA using Qwen's documented prompt wording.
            up_names = []
            try:
                from PIL import Image
                import io
                prompt = (body.get("prompt") or "").strip()
                if not prompt:
                    return self._send(400, json.dumps({"error": "prompt required"}))
                mode = body.get("mode", "text")
                if mode not in ("text", "edit"):
                    return self._send(400, json.dumps({"error": "Qwen 2.1 supports text or edit"}))
                transparent = bool(body.get("transparent", False))
                seed = int(body.get("seed", -1))
                if seed < 0:
                    seed = int.from_bytes(os.urandom(4), "big")
                width = max(256, round(int(body.get("width", 1728)) / 32) * 32)
                height = max(256, round(int(body.get("height", 1152)) / 32) * 32)

                raws = []
                if mode == "edit":
                    if body.get("image"):
                        raws.append(base64.b64decode(str(body["image"]).split(",")[-1]))
                    elif body.get("source"):
                        cand = os.path.join(IMAGE_DIR, os.path.basename(body["source"]))
                        if os.path.isfile(cand):
                            with open(cand, "rb") as f:
                                raws.append(f.read())
                    if not raws:
                        return self._send(400, json.dumps({"error": "add an input image first"}))
                    for extra in (body.get("extra_images") or [])[:QWEN21_MAX_REFS - 1]:
                        raws.append(base64.b64decode(str(extra).split(",")[-1]))
                if transparent:
                    if mode == "text":
                        prompt = ("This is an RGBA image with transparency. " + prompt +
                                  " The image has alpha channel and the background is transparent.")
                    else:
                        prompt += " Output a PNG image with a transparent background."

                comfy_free()   # H3/Krea weights off the card before Qwen loads
                if not vlo_up():
                    return self._send(503, json.dumps(
                        {"error": f"Qwen backend is not reachable at {VLO}"}))
                stamp = int(time.time())
                for i, raw in enumerate(raws, 1):
                    im = Image.open(io.BytesIO(raw))
                    im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
                    buf = io.BytesIO()
                    im.save(buf, "PNG")
                    name = upload_to_comfy(buf.getvalue(), f"mmstudio_qwen21_{stamp}_{i}.png", backend=QWEN)
                    up_names.append(name)
                    if i == 1:
                        ar = im.width / im.height
                        width = round((1024 * 1024 * ar) ** 0.5 / 32) * 32
                        height = round((1024 * 1024 / ar) ** 0.5 / 32) * 32

                wf = {
                    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN21_UNET, "weight_dtype": "default"}},
                    "2": {"class_type": "LoraLoaderModelOnly", "inputs": {
                        "model": ["1", 0], "lora_name": QWEN21_LORA, "strength_model": 1.0}},
                    "3": {"class_type": "QwenImage21Cache", "inputs": {"model": ["2", 0], "device": "auto", "dtype": "default"}},
                    "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN21_TE, "type": "qwen_image"}},
                    "5": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN21_VAE}},
                    "6": {"class_type": "TextEncodeQwenImage21", "inputs": {
                        "clip": ["4", 0], "prompt": prompt, "negative_prompt": "", "resolution": 1024}},
                    "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
                    "8": {"class_type": "KSampler", "inputs": {
                        "model": ["3", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0],
                        "seed": seed, "steps": QWEN21_STEPS, "cfg": 1.0, "sampler_name": "euler",
                        "scheduler": "simple", "denoise": 1.0}},
                    "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["5", 0]}},
                    "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "qwen21/MiniMaxStudio"}},
                }
                if up_names:
                    wf["6"]["inputs"]["vae"] = ["5", 0]
                    for i, n in enumerate(up_names, 1):
                        wf[f"2{i:02d}"] = {"class_type": "LoadImage", "inputs": {"image": n}}
                        wf["6"]["inputs"][f"images.image_{i}"] = [f"2{i:02d}", 0]
                    wf["8"]["inputs"]["latent_image"] = ["6", 2]
                    del wf["7"]

                t0 = time.time()
                blob, ext, staged = submit_and_wait(wf, timeout=1800, exts=(".png",), backend=QWEN)
                if not transparent:
                    # Qwen 2.1 always decodes 4 channels; flatten opaque results so they act
                    # like every other still (upscale, vertical, H3 first frame).
                    im = Image.open(io.BytesIO(blob))
                    if im.mode == "RGBA":
                        buf = io.BytesIO()
                        im.convert("RGB").save(buf, "PNG")
                        blob = buf.getvalue()
                kind = "qwen21" + ("-edit" if mode == "edit" else "") + ("-transparent" if transparent else "")
                name, path = save_output(blob, kind=kind, ext=ext, staged=staged, directory=IMAGE_DIR)
                with Image.open(path) as im:
                    width, height = im.size
                return self._send(200, json.dumps({
                    "url": f"/images/{name}", "file": path, "seed": seed, "steps": QWEN21_STEPS,
                    "width": width, "height": height, "mode": mode, "model": "qwen21",
                    "transparent": transparent, "refs": len(up_names), "denoise": None,
                    "took": round(time.time() - t0, 1)}))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))
            finally:
                for n in up_names:
                    try:
                        discard_upload(n, QWEN)
                    except OSError:
                        pass

        if p.path == "/api/image":
            try:
                # Angles = the Qwen Edit graph + the Multiple Angles LoRA, prompt built from the pick.
                angle = None
                if body.get("mode") == "angles":
                    az, el, di = body.get("azimuth"), body.get("elevation"), body.get("distance")
                    if az not in ANGLE_AZIMUTHS or el not in ANGLE_ELEVATIONS or di not in ANGLE_DISTANCES:
                        return self._send(400, json.dumps({"error": "pick an angle, a height and a distance"}))
                    angle = f"{az} {el} {di}"
                    body["prompt"], body["mode"] = "<sks> " + angle, "edit"
                if not comfy_up():
                    return self._send(503, json.dumps(
                        {"error": BACKEND_DOWN}))
                prompt = (body.get("prompt") or "").strip()
                if not prompt:
                    return self._send(400, json.dumps({"error": "prompt required"}))
                width = int(body.get("width", 1728))
                height = int(body.get("height", 1152))
                steps = max(KREA_STEP_MIN, min(KREA_STEP_MAX, int(body.get("steps", 24))))
                seed = int(body.get("seed", -1))
                if seed < 0:
                    seed = int.from_bytes(os.urandom(4), "big")
                weighting = bool(body.get("prompt_weight", False))

                # Oct 4 2026: three modes. "text" = plain text-to-image. "img2img" = encode
                # the input image and redraw only the last `denoise` fraction of the schedule.
                # "reference" = Krea 2's built-in reference input (Kontext-style): a fresh
                # image from empty noise that LOOKS AT the input image through attention.
                mode = body.get("mode", "text")
                if mode not in ("text", "img2img", "reference", "identity", "edit"):
                    return self._send(400, json.dumps({"error": "mode must be text, img2img, reference, identity or edit"}))
                denoise = max(0.05, min(1.0, float(body.get("denoise", 0.55))))
                up_name = None
                if mode != "text":
                    from PIL import Image
                    import io
                    raw = None
                    if body.get("image"):
                        raw = base64.b64decode(str(body["image"]).split(",")[-1])
                    elif body.get("source"):
                        cand = os.path.join(IMAGE_DIR, os.path.basename(body["source"]))
                        if os.path.isfile(cand):
                            with open(cand, "rb") as f:
                                raw = f.read()
                    if not raw:
                        return self._send(400, json.dumps({"error": "add an input image first"}))
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                    if mode == "img2img" and body.get("match_source", True):
                        # Keep the input's shape at ~2 MP (the bake-off winner), multiples of 16.
                        ar = im.width / im.height
                        width = max(256, round((KREA_TARGET_PIXELS * ar) ** 0.5 / 16) * 16)
                        height = max(256, round((KREA_TARGET_PIXELS / ar) ** 0.5 / 16) * 16)
                    buf = io.BytesIO()
                    im.save(buf, "PNG")
                    up_name = upload_to_comfy(buf.getvalue(), f"mmstudio_{mode}_{int(time.time())}.png")
                    if mode == "edit":
                        # Qwen edits at ~1 MP in the input's own shape (ImageScaleToTotalPixels below).
                        ar = im.width / im.height
                        width = max(256, round((1_000_000 * ar) ** 0.5 / 16) * 16)
                        height = max(256, round((1_000_000 / ar) ** 0.5 / 16) * 16)

                # 🔴 Ask the latent for 2x so the 8x VAE decode lands on the size
                # the user actually picked. See KREA_LATENT_SCALE.
                wf = build_krea(
                    prompt,
                    width=width * KREA_LATENT_SCALE,
                    height=height * KREA_LATENT_SCALE,
                    steps=steps, seed=seed, prefix="krea/MiniMaxStudio",
                    prompt_weight=weighting,
                    weight_strength=float(body.get("weight_strength", 1.0)))
                # Scheduler gets the size we actually want out, not the 2x latent size.
                wf["6"]["inputs"]["width"], wf["6"]["inputs"]["height"] = width, height
                if up_name:
                    wf["20"] = {"class_type": "LoadImage", "inputs": {"image": up_name}}
                if mode == "img2img":
                    # VAEEncode needs no 2x trick: qwen_image_vae encodes at 8x, which is
                    # exactly the 16-channel latent Krea 2 samples in.
                    wf["21"] = {"class_type": "ImageScale", "inputs": {
                        "image": ["20", 0], "upscale_method": "lanczos",
                        "width": width, "height": height, "crop": "center"}}
                    wf["22"] = {"class_type": "VAEEncode", "inputs": {
                        "pixels": ["21", 0], "vae": ["3", 0]}}
                    # Stretch the schedule so the part that actually runs is still `steps`
                    # long, then keep only its low-noise tail (same as KSampler's denoise).
                    total = min(KREA_STEP_MAX, max(steps, round(steps / denoise)))
                    wf["6"]["inputs"]["steps"] = total
                    wf["23"] = {"class_type": "SplitSigmasDenoise", "inputs": {
                        "sigmas": ["6", 0], "denoise": denoise}}
                    wf["10"]["inputs"]["sigmas"] = ["23", 1]
                    wf["10"]["inputs"]["latent_image"] = ["22", 0]
                    del wf["5"]
                elif mode == "reference":
                    # STYLE reference: a brand-new picture from the prompt, drawn in the look
                    # of the input image. Mirrors ComfyUI's official template
                    # image_krea2_turbo_int8_image_style_reference (ostris LoRA, Qwen edit
                    # encoder that also SEES the image, index_timestep_zero, CFGGuider cfg 1).
                    # 🔴 Without the LoRA, Krea 2 Turbo + ReferenceLatent = red/blue diagonal
                    # artifacts (Oct 4 test). It is NOT an instruction editor either way.
                    wf["30"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                        "model": ["1", 0], "lora_name": KREA_STYLE_LORA,
                        "strength_model": float(body.get("style_strength", 1.0))}}
                    wf["4"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
                        "clip": ["2", 0], "prompt": prompt, "vae": ["3", 0], "image1": ["20", 0]}}
                    wf["25"] = {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
                        "conditioning": ["4", 0],
                        "reference_latents_method": body.get("ref_method", KREA_REF_METHOD)}}
                    wf["26"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["25", 0]}}
                    wf["8"] = {"class_type": "CFGGuider", "inputs": {
                        "model": ["30", 0], "positive": ["25", 0], "negative": ["26", 0], "cfg": 1.0}}
                if mode == "identity":
                    # Krea 2 Identity Edit v1.2 (conradlocke LoRA + lbouaraba/comfyui-krea2edit nodes),
                    # wired exactly like the pack's krea2_identity_edit.json. Same face + outfit in a
                    # new scene from a plain-English instruction. Oct 4 test: insightface likeness
                    # 0.81 vs the source (wide-shot H3 Holmes fell to 0.20). The author says 1 MP is
                    # the sweet spot, so the chosen aspect is scaled to ~1 MP; Turbo 10 steps, cfg 1.
                    ar = width / height
                    iw = round((KREA_ID_PIXELS * ar) ** 0.5 / 16) * 16
                    ih = round((KREA_ID_PIXELS / ar) ** 0.5 / 16) * 16
                    width, height = iw, ih
                    id_steps = int(body.get("id_steps", 10))
                    wf = {
                        "1": wf["1"], "2": wf["2"], "3": wf["3"],
                        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {
                            "model": ["1", 0], "lora_name": KREA_ID_LORA, "strength_model": 1.0}},
                        "20": wf["20"],
                        "22": {"class_type": "VAEEncode", "inputs": {"pixels": ["20", 0], "vae": ["3", 0]}},
                        "7": {"class_type": "EmptySD3LatentImage", "inputs": {
                            "width": iw, "height": ih, "batch_size": 1}},
                        "8": {"class_type": "Krea2EditModelPatch", "inputs": {
                            "model": ["4", 0], "source_latent": ["22", 0],
                            "ref_boost": float(body.get("ref_boost", 4.0)), "ref_boost_a": 1.0,
                            "fit_mode": "fit", "vae": ["3", 0], "source_image": ["20", 0],
                            "target_latent": ["7", 0]}},
                        "9": {"class_type": "Krea2EditGroundedEncode", "inputs": {
                            "clip": ["2", 0], "prompt": prompt, "image": ["20", 0], "grounding_px": 768}},
                        "10": {"class_type": "Krea2EditGroundedEncode", "inputs": {
                            "clip": ["2", 0], "prompt": "", "image": ["20", 0], "grounding_px": 768}},
                        "11": {"class_type": "KSampler", "inputs": {
                            "model": ["8", 0], "positive": ["9", 0], "negative": ["10", 0],
                            "latent_image": ["7", 0], "seed": seed, "steps": id_steps, "cfg": 1.0,
                            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
                        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
                        "13": {"class_type": "SaveImage", "inputs": {
                            "images": ["12", 0], "filename_prefix": "krea/MiniMaxStudio_identity"}},
                    }
                    steps = id_steps
                elif mode == "edit":
                    # Oct 6 2026 (the user: "the qwen edits are great"). Qwen-Image-Edit-2511, Q4_K_M GGUF + the
                    # 4-step Lightning LoRA. An INSTRUCTION editor: relight, swap backgrounds, remove things,
                    # change the angle, keep everything else. ~20-30 s at 1 MP. 🔴 Never name an object that
                    # isn't in the photo ("clear the island" on a kitchen with no island BUILT one).
                    wf = {
                        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": QWEN_EDIT_GGUF}},
                        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {
                            "model": ["1", 0], "lora_name": QWEN_EDIT_LORA, "strength_model": 1.0}},
                        "3": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["2", 0], "shift": 3.1}},
                        "4": {"class_type": "CFGNorm", "inputs": {"model": ["3", 0], "strength": 1.0}},
                        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_EDIT_TE, "type": "qwen_image"}},
                        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
                        "7": {"class_type": "LoadImage", "inputs": {"image": up_name}},
                        "8": {"class_type": "ImageScaleToTotalPixels", "inputs": {
                            "image": ["7", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}},
                        "9": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
                            "clip": ["5", 0], "prompt": prompt, "vae": ["6", 0], "image1": ["8", 0]}},
                        "10": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
                            "clip": ["5", 0], "prompt": "", "vae": ["6", 0], "image1": ["8", 0]}},
                        "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["6", 0]}},
                        "12": {"class_type": "KSampler", "inputs": {
                            "model": ["4", 0], "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["11", 0],
                            "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                            "denoise": 1.0}},
                        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["6", 0]}},
                        "14": {"class_type": "SaveImage", "inputs": {
                            "images": ["13", 0], "filename_prefix": "krea/MiniMaxStudio_qwenedit"}},
                    }
                    if angle:
                        wf["2a"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                            "model": ["2", 0], "lora_name": QWEN_ANGLES_LORA, "strength_model": 1.0}}
                        wf["3"]["inputs"]["model"] = ["2a", 0]
                    steps = 4
                t0 = time.time()
                try:
                    blob, ext, staged = submit_and_wait(
                        wf, timeout=3600, exts=(".png", ".jpg", ".jpeg", ".webp"))
                finally:
                    if up_name:
                        try:
                            discard_upload(up_name)
                        except OSError:
                            pass
                kind = {"text": "krea", "img2img": "krea-img2img", "reference": "krea-ref",
                        "identity": "krea-identity", "edit": "qwen-angles" if angle else "qwen-edit"}[mode]
                name, path = save_output(blob, kind=kind, ext=ext,
                                         staged=staged, directory=IMAGE_DIR)
                return self._send(200, json.dumps({
                    "url": f"/images/{name}", "file": path, "seed": seed,
                    "steps": steps, "width": width, "height": height, "mode": mode,
                    "denoise": denoise if mode == "img2img" else None,
                    "angle": angle,
                    "took": round(time.time() - t0, 1),
                }))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))

        if p.path == "/api/vertical":
            # Krea renders most photoreal in landscape (Oct 4 2026 finding), so make the wide
            # still and crop it vertical here: PIL crop at the chosen spot, then RealESRGAN x4
            # and a lanczos resize to the standard phone size. Built Oct 4 2026.
            try:
                from PIL import Image
                import io
                targets = {"9:16": (1080, 1920), "4:5": (1080, 1350)}
                ratio = body.get("ratio", "9:16")
                if ratio not in targets:
                    return self._send(400, json.dumps({"error": "ratio must be 9:16 or 4:5"}))
                src = os.path.join(IMAGE_DIR, os.path.basename(body.get("name") or ""))
                if not os.path.isfile(src):
                    return self._send(400, json.dumps({"error": "pick an image first"}))
                tw, th = targets[ratio]
                pos = max(0.0, min(1.0, float(body.get("pos", 0.5))))
                im = Image.open(src).convert("RGB")
                w, h = im.size
                if w / h > tw / th:      # wider than the target: slide a full-height window sideways
                    cw, ch = round(h * tw / th), h
                    x, y = round((w - cw) * pos), 0
                else:                    # already taller: slide a full-width window up/down
                    cw, ch = w, round(w * th / tw)
                    x, y = 0, round((h - ch) * pos)
                buf = io.BytesIO()
                im.crop((x, y, x + cw, y + ch)).save(buf, "PNG")
                if not comfy_up():
                    return self._send(503, json.dumps(
                        {"error": BACKEND_DOWN}))
                up_name = upload_to_comfy(buf.getvalue(), f"mmstudio_vertical_{int(time.time())}.png")
                # Oct 5 2026: SeedVR2 instead of RealESRGAN x4 (ESRGAN smoothed skin to plastic).
                svr, svr_out = seedvr2_nodes(["1", 0], tw)
                wf = {
                    "1": {"class_type": "LoadImage", "inputs": {"image": up_name}},
                    **svr,
                    "4": {"class_type": "ImageScale", "inputs": {
                        "image": svr_out, "upscale_method": "lanczos",
                        "width": tw, "height": th, "crop": "disabled"}},
                    "5": {"class_type": "SaveImage", "inputs": {
                        "images": ["4", 0], "filename_prefix": "krea/MiniMaxStudio_vertical"}},
                }
                t0 = time.time()
                blob, ext, staged = submit_and_wait(wf, timeout=600, exts=(".png",))
                name, path = save_output(blob, kind="krea-vertical", ext=ext,
                                         staged=staged, directory=IMAGE_DIR)
                try:
                    discard_upload(up_name)
                except OSError:
                    pass
                return self._send(200, json.dumps({
                    "url": f"/images/{name}", "file": path, "width": tw, "height": th,
                    "crop": [x, y, cw, ch], "took": round(time.time() - t0, 1),
                }))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))

        if p.path == "/api/image-upscale":
            # Oct 5 2026 (the user): SeedVR2 upscale for ANY still: a Studio image by name, or an
            # uploaded file. size "4k" = short edge 2160, "2x" = double the short edge.
            try:
                from PIL import Image
                import io
                raw, base = None, "upload"
                if body.get("image"):
                    raw = base64.b64decode(str(body["image"]).split(",")[-1])
                    base = os.path.splitext(os.path.basename(body.get("filename") or "upload"))[0]
                elif body.get("name"):
                    cand = os.path.join(IMAGE_DIR, os.path.basename(body["name"]))
                    if os.path.isfile(cand):
                        with open(cand, "rb") as f:
                            raw = f.read()
                if not raw:
                    return self._send(400, json.dumps({"error": "pick or upload an image first"}))
                im = Image.open(io.BytesIO(raw))
                im = im.convert("RGB")   # drops alpha + palette; SeedVR2 wants plain RGB
                w, h = im.size
                size = body.get("size", "4k")
                short = 2160 if size == "4k" else min(w, h) * 2
                short = min(short, round(SEEDVR2_MAX_LONG * min(w, h) / max(w, h)))
                short = max(64, int(short))
                if not comfy_up():
                    return self._send(503, json.dumps(
                        {"error": BACKEND_DOWN}))
                buf = io.BytesIO()
                im.save(buf, "PNG")
                up_name = upload_to_comfy(buf.getvalue(), f"mmstudio_seedvr2_{int(time.time())}.png")
                svr, svr_out = seedvr2_nodes(["1", 0], short)
                wf = {
                    "1": {"class_type": "LoadImage", "inputs": {"image": up_name}},
                    **svr,
                    "5": {"class_type": "SaveImage", "inputs": {
                        "images": svr_out, "filename_prefix": "krea/MiniMaxStudio_seedvr2"}},
                }
                t0 = time.time()
                blob, ext, staged = submit_and_wait(wf, timeout=900, exts=(".png",))
                name, path = save_output(blob, kind="seedvr2-" + size, ext=ext,
                                         staged=staged, directory=IMAGE_DIR)
                try:
                    discard_upload(up_name)
                except OSError:
                    pass
                ow, oh = Image.open(path).size
                return self._send(200, json.dumps({
                    "url": f"/images/{name}", "file": path, "name": name, "width": ow, "height": oh,
                    "source": [w, h], "took": round(time.time() - t0, 1),
                }))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))

        if p.path == "/api/upscale":
            try:
                if not comfy_up():
                    return self._send(503, json.dumps(
                        {"error": BACKEND_DOWN}))
                # Either a browsed/uploaded file, or the app's own last output.
                src_name = None
                src_path = None
                if body.get("video"):
                    raw = base64.b64decode(str(body["video"]).split(",")[-1])
                    fname = body.get("filename") or f"upscale_{int(time.time())}.mp4"
                    src_name = upload_to_comfy(raw, f"mmstudio_up_{fname}")
                    src_path = stage_local(raw, src_name)
                elif body.get("last"):
                    cand = os.path.join(VIDEO_DIR, os.path.basename(body["last"]))
                    if not os.path.isfile(cand):
                        return self._send(400, json.dumps({"error": "last output not found"}))
                    with open(cand, "rb") as f:
                        src_name = upload_to_comfy(f.read(), f"mmstudio_up_{os.path.basename(cand)}")
                    src_path = cand
                else:
                    return self._send(400, json.dumps({"error": "no video supplied"}))

                fps = probe_fps(src_path)
                interpolate = bool(body.get("interpolate", False))
                # Oct 5 2026 (the user): SeedVR2 is the default upscaler; "fast" keeps RealESRGAN x2.
                fast = bool(body.get("fast", False))
                if fast:
                    wf, out_fps = build_postfx(
                        src_name, per_batch=2, rife=interpolate, multiplier=2,
                        fps=fps, prefix="postfx/MiniMaxStudio")
                else:
                    wf, out_fps = build_seedvr2_video(
                        src_name, rife=interpolate, multiplier=2,
                        fps=fps, prefix="postfx/MiniMaxStudio_seedvr2")
                t0 = time.time()
                blob, ext, staged = submit_and_wait(wf, timeout=3600)
                name, path = save_output(blob, kind="up" if fast else "up-seedvr2", ext=ext, staged=staged)
                discard_upload(src_name)
                cleanup_local(src_path)
                return self._send(200, json.dumps({
                    "url": f"/outputs/{name}", "file": path, "upscaler": "esrgan" if fast else "seedvr2",
                    "fps": out_fps, "took": round(time.time() - t0, 1),
                }))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))

        if p.path == "/api/slowmo":
            # Oct 10 2026 (the user: "slow down the rotation ... so it's double the length").
            # RIFE x2 invents an in-between frame for every pair, then the clip is saved at the
            # SOURCE frame rate, so it plays twice as long and stays smooth (a plain setpts
            # slow-down would stutter at 12 real frames a second). Audio is stretched to match
            # with atempo=0.5, which keeps the pitch. No upscale, so it is quick.
            src_path = None
            try:
                if not comfy_up():
                    return self._send(503, json.dumps(
                        {"error": BACKEND_DOWN}))
                if body.get("video"):
                    raw = base64.b64decode(str(body["video"]).split(",")[-1])
                    fname = os.path.basename(body.get("filename") or "clip.mp4")
                elif body.get("last"):
                    cand = os.path.join(VIDEO_DIR, os.path.basename(body["last"]))
                    if not os.path.isfile(cand):
                        return self._send(400, json.dumps({"error": "last output not found"}))
                    with open(cand, "rb") as f:
                        raw = f.read()
                    fname = os.path.basename(cand)
                else:
                    return self._send(400, json.dumps({"error": "no video supplied"}))
                src_name = upload_to_comfy(raw, f"mmstudio_slow_{int(time.time())}_{fname}")
                src_path = stage_local(raw, src_name)
                fps = probe_fps(src_path)
                wf, _ = build_postfx(src_name, upscale=False, rife=True, multiplier=2,
                                     fps=fps / 2, keep_audio=False, prefix="postfx/MiniMaxStudio_slow")
                t0 = time.time()
                blob, ext, staged = submit_and_wait(wf, timeout=3600)
                import tempfile
                with tempfile.TemporaryDirectory() as td:
                    vid = os.path.join(td, "v." + ext)
                    with open(vid, "wb") as f:
                        f.write(blob)
                    has_audio = subprocess.run(
                        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                         "-of", "csv=p=0", src_path], capture_output=True, text=True).stdout.strip() != ""
                    out = os.path.join(td, "out.mp4")
                    if has_audio:
                        cmd = ["ffmpeg", "-v", "error", "-y", "-i", vid, "-i", src_path, "-map", "0:v", "-map", "1:a",
                               "-c:v", "copy", "-filter:a", "atempo=0.5", "-c:a", "aac", "-b:a", "192k", "-shortest", out]
                    else:
                        cmd = ["ffmpeg", "-v", "error", "-y", "-i", vid, "-c:v", "copy", out]
                    subprocess.run(cmd, check=True, timeout=600)
                    with open(out, "rb") as f:
                        final = f.read()
                if staged and os.path.isfile(staged):
                    try:
                        os.remove(staged)
                    except OSError:
                        pass
                name, path = save_output(final, kind="slow2x", ext="mp4")
                dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                            "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip() or 0)
                return self._send(200, json.dumps({
                    "url": f"/outputs/{name}", "file": path, "fps": fps, "seconds": round(dur, 2),
                    "audio": has_audio, "took": round(time.time() - t0, 1)}))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))
            finally:
                if src_path:
                    discard_upload(os.path.basename(src_path))
                    cleanup_local(src_path)

        return self._send(404, json.dumps({"error": "not found"}))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8801"))
    print(f"Backlot on http://localhost:{port}")
    print(f"Comfy API -> {COMFY}" + (f"   (Qwen -> {VLO})" if VLO != COMFY else ""))
    print(f"Videos    -> {VIDEO_DIR}")
    print(f"Images    -> {IMAGE_DIR}")
    import threading
    threading.Thread(target=_idle_watchdog, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
