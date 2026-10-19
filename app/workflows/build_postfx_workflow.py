#!/usr/bin/env python3
"""Build a POST-FX ComfyUI API workflow for H3 output. Norm 2026-09-11.

Separate from build_h3_workflow.py by design (house rule: never edit a
working workflow in place). This graph never loads the DiT — it takes an already
rendered mp4 and runs it through a PIXEL-space upscaler and RIFE interpolation.

  LoadVideo -> GetVideoComponents -> (IMAGE, AUDIO)
  UpscaleModelLoader(RealESRGAN_x2) -> ImageUpscaleWithModelBatched(per_batch)
  -> RIFE VFI(multiplier) -> CreateVideo(fps * multiplier, audio) -> SaveVideo

All schemas verified against the live server on :8189 (Sep 11 2026).
The LTX upscalers on disk are LATENT models and CANNOT be used here — pixel only.
"""
import json, argparse

UPSCALER = "RealESRGAN_x2.pth"
RIFE_CKPT = "rife49.pth"


def build(video_file, upscale_model=UPSCALER, per_batch=2, precision="float16",
          rife=True, rife_ckpt=RIFE_CKPT, multiplier=2, fps=24.0,
          clear_cache_after_n_frames=10, fast_mode=True, ensemble=True,
          rife_dtype="float16", scale_factor=1.0, rife_batch=1, upscale=True,
          prefix="postfx/MiniMaxH3", keep_audio=True):
    wf = {}
    wf["1"] = {"class_type": "LoadVideo", "inputs": {"file": video_file}}
    wf["2"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}}
    images = ["2", 0]
    audio = ["2", 1]

    if upscale:
        wf["3"] = {"class_type": "UpscaleModelLoader",
                   "inputs": {"model_name": upscale_model}}
        # Batched variant (KJNodes) so VRAM stays flat no matter the frame count.
        wf["4"] = {"class_type": "ImageUpscaleWithModelBatched", "inputs": {
            "upscale_model": ["3", 0], "images": images,
            "per_batch": per_batch, "precision": precision}}
        images = ["4", 0]

    out_fps = float(fps)
    if rife:
        wf["5"] = {"class_type": "RIFE VFI", "inputs": {
            "ckpt_name": rife_ckpt, "frames": images,
            "clear_cache_after_n_frames": clear_cache_after_n_frames,
            "multiplier": multiplier, "fast_mode": fast_mode,
            "ensemble": ensemble, "scale_factor": scale_factor,
            "dtype": rife_dtype, "batch_size": rife_batch,
            "torch_compile": False}}
        images = ["5", 0]
        out_fps = float(fps) * multiplier

    cv = {"images": images, "fps": out_fps}
    if keep_audio:
        cv["audio"] = audio
    wf["6"] = {"class_type": "CreateVideo", "inputs": cv}
    wf["7"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["6", 0], "filename_prefix": prefix,
        "format": "auto", "codec": "auto"}}
    return wf, out_fps


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video", help="filename inside ComfyUI-h3/ComfyUI/input/")
    p.add_argument("-o", "--out", required=True, help="workflow json to write")
    p.add_argument("--upscale-model", default=UPSCALER)
    p.add_argument("--per-batch", type=int, default=2,
                   help="frames per upscaler GPU call. 2 measured FASTER and 2.5x\n                        leaner than 8 on the 16GB card (78.9s vs 85.1s, 6.2 vs 15.9 GiB)")
    p.add_argument("--precision", default="float16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--no-upscale", action="store_true")
    p.add_argument("--no-rife", action="store_true")
    p.add_argument("--rife-ckpt", default=RIFE_CKPT)
    p.add_argument("--multiplier", type=int, default=2)
    p.add_argument("--rife-batch", type=int, default=1,
                   help="interpolation tasks per GPU call; higher = more VRAM")
    p.add_argument("--fps", type=float, default=24.0, help="SOURCE fps")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--prefix", default="postfx/MiniMaxH3")
    a = p.parse_args()
    wf, out_fps = build(a.video, upscale_model=a.upscale_model,
                        per_batch=a.per_batch, precision=a.precision,
                        upscale=not a.no_upscale, rife=not a.no_rife,
                        rife_ckpt=a.rife_ckpt, multiplier=a.multiplier,
                        rife_batch=a.rife_batch,
                        fps=a.fps, prefix=a.prefix, keep_audio=not a.no_audio)
    with open(a.out, "w") as f:
        json.dump(wf, f, indent=1)
    print(f"wrote {a.out}: {len(wf)} nodes, output fps {out_fps}")
