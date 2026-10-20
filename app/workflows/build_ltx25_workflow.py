#!/usr/bin/env python3
"""LTX-2.5 (distilled) text-to-video and image-to-video graphs, API format.

Transcribed node for node from ComfyUI 0.42.1's own templates (video_ltx2_5_t2v.json and
video_ltx2_5_i2v.json, prompt enhancer off). Two-stage pixel pipeline: stage 1 samples at half
size (8 fixed sigmas, euler_ancestral, CFG 1 video and audio), the spatial latent upsampler doubles
it, stage 2 refines with 3 sigmas, then tiled VAE decode + audio VAE decode -> CreateVideo (mp4
with sound). Image-to-video = the same graph with LTXVImgToVideoInplace at both stages (0.7 then
1.0 strength), the image pre-sized to the output canvas here so no resize node is needed.
Norm 2026-10-20.
"""
DIT = "ltx-2.5-22b-distilled-transformer-w4a8_convrot.safetensors"       # 12.5 GB, fits the card
DIT_INT8 = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"  # 21.5 GB, the template's file
TE = "gemma4-12b-with-proj-ltx-2.5-w4a8_convrot.safetensors"
TE_INT8 = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VAE = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
NEG = "pc game, console game, video game, cartoon, childish, ugly"
SIGMAS_1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS_2 = "0.85, 0.7250, 0.4219, 0.0"


def snap(v, m=64):
    return max(m, int(round(v / m)) * m)


def build(prompt, width=1280, height=720, seconds=5.0, fps=24, seed=42, image=None,
          dit=DIT, te=TE, prefix="video/LTX25", image_strength=0.7):
    """image = filename already in ComfyUI's input dir (or an uploaded asset name), sized to
    width x height. Returns (workflow, frame_count). Output size is a multiple of 64 so the
    half-size stage stays a multiple of 32."""
    width, height = snap(width), snap(height)
    length = int(round(seconds * fps)) + 1          # the template's "a * b + 1"
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": dit, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "ltxv", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "ups": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPSCALER}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "negt": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": NEG}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["pos", 0], "negative": ["negt", 0], "frame_rate": float(fps)}},
        "guider": {"class_type": "LTXVDualCFGGuider", "inputs": {
            "model": ["unet", 0], "positive": ["cond", 0], "negative": ["cond", 1], "video_cfg": 1.0, "audio_cfg": 1.0}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "sig1": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS_1}},
        "sig2": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS_2}},
        "noise1": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "noise2": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "lat_v": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": width // 2, "height": height // 2, "length": length, "batch_size": 1}},
        "lat_a": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": length, "frame_rate": float(fps), "batch_size": 1, "audio_vae": ["avae", 0]}},
    }
    stage1_video = ["lat_v", 0]
    if image:
        wf["img"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        wf["pre"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["img", 0], "img_compression": 18}}
        wf["i2v1"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["vae", 0], "image": ["pre", 0], "latent": ["lat_v", 0], "strength": image_strength, "bypass": False}}
        stage1_video = ["i2v1", 0]
    wf["cat1"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": stage1_video, "audio_latent": ["lat_a", 0]}}
    wf["s1"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["noise1", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sig1", 0], "latent_image": ["cat1", 0]}}
    wf["sep1"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["s1", 0]}}
    wf["up"] = {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["sep1", 0], "upscale_model": ["ups", 0], "vae": ["vae", 0]}}
    stage2_video = ["up", 0]
    if image:
        wf["i2v2"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["vae", 0], "image": ["pre", 0], "latent": ["up", 0], "strength": 1.0, "bypass": False}}
        stage2_video = ["i2v2", 0]
    wf["cat2"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": stage2_video, "audio_latent": ["sep1", 1]}}
    wf["s2"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["noise2", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sig2", 0], "latent_image": ["cat2", 0]}}
    wf["sep2"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["s2", 0]}}
    wf["dec"] = {"class_type": "VAEDecodeTiled", "inputs": {
        "samples": ["sep2", 0], "vae": ["vae", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 16}}
    wf["adec"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["sep2", 1], "audio_vae": ["avae", 0]}}
    wf["vid"] = {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "audio": ["adec", 0], "fps": float(fps)}}
    wf["save"] = {"class_type": "SaveVideo", "inputs": {"video": ["vid", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}}
    return wf, length


if __name__ == "__main__":
    import argparse, json
    a = argparse.ArgumentParser()
    a.add_argument("prompt"); a.add_argument("--width", type=int, default=1280); a.add_argument("--height", type=int, default=720)
    a.add_argument("--seconds", type=float, default=5.0); a.add_argument("--seed", type=int, default=42); a.add_argument("--image")
    a.add_argument("--int8", action="store_true", help="use the template's int8 DiT + TE")
    o = a.parse_args()
    wf, n = build(o.prompt, o.width, o.height, o.seconds, seed=o.seed, image=o.image,
                  dit=DIT_INT8 if o.int8 else DIT, te=TE_INT8 if o.int8 else TE)
    print(json.dumps(wf, indent=1)); import sys; print(n, "frames", file=sys.stderr)
