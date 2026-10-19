#!/usr/bin/env python3
"""Build a Krea 2 Turbo text-to-image workflow. Norm 2026-09-11.

NEW FILE — touches nothing that already works.

Krea 2 is Flux2-family, NOT the Z-Image shape:
  UNETLoader(krea2_turbo_fp8_scaled) -> MODEL
  CLIPLoader(qwen3vl_4b_fp8_scaled, type="krea2") -> CLIP
  VAELoader(qwen_image_vae)
  CLIPTextEncode -> CONDITIONING
  EmptyFlux2LatentImage + Flux2Scheduler(steps,w,h) -> SIGMAS
  SamplerCustomAdvanced(RandomNoise, BasicGuider, KSamplerSelect, sigmas)
  -> VAEDecode -> SaveImage

🔴 The CLIP type is "krea2". qwen_image / flux2 are different types on the same loader.
🔴 Turbo is distilled: BasicGuider (no CFG) and few steps. Do not add a negative prompt.
🔴 For per-token weighting use Krea2PromptWeight — ComfyUI's normal (word:1.2) syntax does
   NOTHING here, because the Qwen3-VL encoder ignores it.
"""
import json, argparse

DIT  = "krea2_turbo_fp8_scaled.safetensors"
TE   = "qwen3vl_4b_fp8_scaled.safetensors"
VAE  = "qwen_image_vae.safetensors"


def build(prompt, width=1024, height=1024, steps=8, seed=42,
          sampler="euler", prefix="krea/Krea2",
          prompt_weight=False, weight_strength=1.0):
    """prompt_weight=True swaps CLIPTextEncode for KJNodes' Krea2PromptWeight, which is
    the ONLY way to weight tokens on this model. Syntax: (word:2) emphasise,
    (word:-1) actively remove. It returns a patched MODEL as well as CONDITIONING."""
    wf = {}
    wf["1"] = {"class_type": "UNETLoader", "inputs": {
        "unet_name": DIT, "weight_dtype": "default"}}
    wf["2"] = {"class_type": "CLIPLoader", "inputs": {
        "clip_name": TE, "type": "krea2"}}
    wf["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE}}
    if prompt_weight:
        wf["4"] = {"class_type": "Krea2PromptWeight", "inputs": {
            "clip": ["2", 0], "model": ["1", 0], "text": prompt,
            "strength": float(weight_strength)}}
        model_src, cond_src = ["4", 0], ["4", 1]
    else:
        wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {
            "text": prompt, "clip": ["2", 0]}}
        model_src, cond_src = ["1", 0], ["4", 0]
    wf["5"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {
        "width": width, "height": height, "batch_size": 1}}
    wf["6"] = {"class_type": "Flux2Scheduler", "inputs": {
        "steps": steps, "width": width, "height": height}}
    wf["7"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}}
    wf["8"] = {"class_type": "BasicGuider", "inputs": {
        "model": model_src, "conditioning": cond_src}}
    wf["9"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    wf["10"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["9", 0], "guider": ["8", 0], "sampler": ["7", 0],
        "sigmas": ["6", 0], "latent_image": ["5", 0]}}
    wf["11"] = {"class_type": "VAEDecode", "inputs": {
        "samples": ["10", 0], "vae": ["3", 0]}}
    wf["12"] = {"class_type": "SaveImage", "inputs": {
        "images": ["11", 0], "filename_prefix": prefix}}
    return wf


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-weight", action="store_true",
                   help="enable (word:2) / (word:-1) token weighting")
    p.add_argument("--weight-strength", type=float, default=1.0)
    p.add_argument("--out", default="krea.json")
    a = p.parse_args()
    json.dump(build(a.prompt, a.width, a.height, a.steps, a.seed,
                    prompt_weight=a.prompt_weight,
                    weight_strength=a.weight_strength),
              open(a.out, "w"), indent=2)
    print(f"wrote {a.out}  {a.width}x{a.height} steps={a.steps}")
