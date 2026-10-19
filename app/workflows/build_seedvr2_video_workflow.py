#!/usr/bin/env python3
"""Build a SeedVR2 POST-FX ComfyUI API workflow for H3 output. Norm 2026-10-05.

Sibling of build_postfx_workflow.py (RealESRGAN x2), which is left untouched per
house rule: never edit a working workflow in place. Oct 5 2026:
"SeedVR2 beats ESRGAN hands down...we should use it for both images and videos".

  LoadVideo -> GetVideoComponents -> (IMAGE, AUDIO)
  SeedVR2 3B fp8 (short side `short`, temporal batches) -> [RIFE VFI] -> CreateVideo -> SaveVideo

Settings measured on the 4070 Ti Super (16 GB), Sherlock R10-pipe-v5, 1344x768 x 110 frames:
  batch 21 at 1080p OOMs unless DiT blocks are swapped to system RAM.
  batch 21 / overlap 4 / swap 32 -> 1890x1080, 330 s, peak 12.4 GB. (batch 9 / swap 16: 338 s, 10.9 GB,
  but twice the background shimmer.) Batch seams show as small jumps every ~17 frames; bigger
  batches = fewer seams. batch_size must be 4n+1.
"""
import json, argparse

DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
VAE = "ema_vae_fp16.safetensors"
RIFE_CKPT = "rife49.pth"


def build(video_file, short=1080, batch=21, overlap=4, swap=32, rife=True, rife_ckpt=RIFE_CKPT,
          multiplier=2, fps=24.0, upscale=True, prefix="postfx/MiniMaxH3_seedvr2", keep_audio=True,
          seed=42):
    wf = {}
    wf["1"] = {"class_type": "LoadVideo", "inputs": {"file": video_file}}
    wf["2"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}}
    images, audio = ["2", 0], ["2", 1]

    if upscale:
        wf["50"] = {"class_type": "SeedVR2LoadDiTModel", "inputs": {
            "model": DIT, "device": "cuda:0", "blocks_to_swap": int(swap),
            "offload_device": "cpu" if swap else "none"}}
        wf["51"] = {"class_type": "SeedVR2LoadVAEModel", "inputs": {
            "model": VAE, "device": "cuda:0", "encode_tiled": True, "decode_tiled": True}}
        wf["52"] = {"class_type": "SeedVR2VideoUpscaler", "inputs": {
            "image": images, "dit": ["50", 0], "vae": ["51", 0], "seed": int(seed),
            "resolution": int(short), "max_resolution": 0, "batch_size": int(batch),
            "uniform_batch_size": True, "color_correction": "lab", "temporal_overlap": int(overlap)}}
        images = ["52", 0]

    out_fps = float(fps)
    if rife:
        wf["5"] = {"class_type": "RIFE VFI", "inputs": {
            "ckpt_name": rife_ckpt, "frames": images, "clear_cache_after_n_frames": 10,
            "multiplier": multiplier, "fast_mode": True, "ensemble": True, "scale_factor": 1.0,
            "dtype": "float16", "batch_size": 1, "torch_compile": False}}
        images = ["5", 0]
        out_fps = float(fps) * multiplier

    cv = {"images": images, "fps": out_fps}
    if keep_audio:
        cv["audio"] = audio
    wf["6"] = {"class_type": "CreateVideo", "inputs": cv}
    wf["7"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["6", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}}
    return wf, out_fps


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video", help="filename inside ComfyUI-h3/ComfyUI/input/")
    p.add_argument("-o", "--out", required=True, help="workflow json to write")
    p.add_argument("--short", type=int, default=1080, help="target length of the SHORT edge")
    p.add_argument("--batch", type=int, default=21, help="frames per temporal batch (4n+1)")
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--swap", type=int, default=32, help="DiT blocks parked in system RAM (0-36)")
    p.add_argument("--no-upscale", action="store_true")
    p.add_argument("--no-rife", action="store_true")
    p.add_argument("--rife-ckpt", default=RIFE_CKPT)
    p.add_argument("--multiplier", type=int, default=2)
    p.add_argument("--fps", type=float, default=24.0, help="SOURCE fps")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--prefix", default="postfx/MiniMaxH3_seedvr2")
    a = p.parse_args()
    if (a.batch - 1) % 4:
        p.error("--batch must be 4n+1 (5, 9, 13, 17, 21...)")
    wf, out_fps = build(a.video, short=a.short, batch=a.batch, overlap=a.overlap, swap=a.swap,
                        rife=not a.no_rife, rife_ckpt=a.rife_ckpt, multiplier=a.multiplier,
                        fps=a.fps, upscale=not a.no_upscale, prefix=a.prefix,
                        keep_audio=not a.no_audio)
    with open(a.out, "w") as f:
        json.dump(wf, f, indent=1)
    print(f"wrote {a.out}: {len(wf)} nodes, output fps {out_fps}")
