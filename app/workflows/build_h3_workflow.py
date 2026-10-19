#!/usr/bin/env python3
"""Build a MiniMax H3 ComfyUI API-format workflow. Norm 2026-09-11.

Node graph, all schemas verified against the live server on :8189:
  UnetLoaderGGUF -> LoraLoaderModelOnly(turbo) -> MiniMaxH3SigmaShift(12/3)
                 -> MiniMaxLowVRAMAttention -> MODEL
  CLIPLoaderGGUF(type=minimax) -> CLIP
  VAELoader x2 (video fp16 + audio fp32)
  MiniMaxH3ImageToVideo(clip, vae, prompt, w, h, length) -> (CONDITIONING, LATENT)
  SamplerCustomAdvanced(RandomNoise, BasicGuider, KSamplerSelect, BasicScheduler)
  -> VAEDecode(video vae) + VAEDecodeAudio(audio vae) -> CreateVideo(24fps) -> SaveVideo
"""
import json, argparse

# HiFi INT8 repack (MiniMax-H3-16G-HiFi): better quality and ~13% faster than the Q4 GGUF in our A/B.
# Rollback: set DIT = DIT_Q4 (or restore build_h3_workflow.py.bak-pre-hifi-2026-10-14).
DIT_Q4 = "MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf"
DIT   = "minimax_h3_fl2va_dit_16g.safetensors"
TE    = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
VVAE  = "minimax_h3_video_vae_fp16.safetensors"
AVAE  = "minimax_h3_audio_vae_fp32.safetensors"
TURBO = "minimax_h3_turbo_4step_ckpt600_ema_V4.safetensors"

# Turbo LoRA profiles: (lora file, sampler, shift_video, shift_audio). Added 2026-09-24.
# "legacy" is the proven Sep 11 setup and emits a byte-identical graph. Its LoRA is a
# DAMAGED pruned repack (its own metadata: "distillation behaviour may be degraded or
# broken"). The lightx2v files load intact on the pruned DiT. See minimax-video SKILL.md.
TURBO_PROFILES = {
    "legacy": (TURBO, "res_multistep", 12.0, 3.0),
    # lx768 / lx544 (lightx2v 8-step FL2V LoRAs) lost the Sep 24 A/B and were DELETED Oct 9 2026.
}
DEFAULT_TURBO_PROFILE = "legacy"  # 2026-09-24: reverted after A/B (legacy won detail + audio)

# Optional extra LoRA, chained AFTER the turbo LoRA. Off by default: when
# extra_lora is None this builder emits a byte-identical graph to the proven one.
# Camera Motion LoRA (Jojocodex): trigger word "camera motion" MUST lead the
# prompt, strength 0.8-1.0 (0.8 gentle, 1.0 definite, >1.2 goes unstable).
CAMERA_LORA = "camera_motion_h3_lora_v1_3000_pruned.safetensors"
CAMERA_TRIGGER = "camera motion"

def snap(n):
    """H3 frame grid: n %% 17 == 5."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n

ALIGN_I2V = ("For the target video, at 0.00 seconds into the target video, "
             "<Picture 1> (from [Shot 1]) is fully referenced.")
ALIGN_FL2V = ("How the reference pictures align with the target video — Picture 1 "
              "(from Shot 1) aligns with the 0.00-second mark of the target video; "
              "Picture 2 (from Shot 1) aligns with the {dur}-second mark of the target video.")

def build(prompt, width=864, height=480, seconds=5.0, steps=4, seed=42,
          turbo=True, low_vram_chunks=4, fps=24, prefix="video/MiniMaxH3",
          clip_cpu=False, first_frame=None, last_frame=None,
          extra_lora=None, extra_lora_strength=1.0, turbo_profile=None):
    lora_file, sampler, shift_v, shift_a = TURBO_PROFILES[turbo_profile or DEFAULT_TURBO_PROFILE]
    length = snap(round(seconds * fps))
    wf = {}
    wf["1"]  = ({"class_type":"UnetLoaderGGUF","inputs":{"unet_name":DIT}} if DIT.endswith(".gguf")
                else {"class_type":"UNETLoader","inputs":{"unet_name":DIT,"weight_dtype":"default"}})
    model = ["1",0]
    if turbo:
        wf["2"] = {"class_type":"LoraLoaderModelOnly","inputs":{
            "model":model,"lora_name":lora_file,"strength_model":1.0}}
        model = ["2",0]
    if extra_lora:
        # Chained after turbo so the turbo path above is untouched.
        wf["2b"] = {"class_type":"LoraLoaderModelOnly","inputs":{
            "model":model,"lora_name":extra_lora,
            "strength_model":float(extra_lora_strength)}}
        model = ["2b",0]
    # H3 denoises video and audio on two different flow schedules.
    wf["3"] = {"class_type":"MiniMaxH3SigmaShift","inputs":{
        "model":model,"shift_video":shift_v,"shift_audio":shift_a}}
    model = ["3",0]
    if low_vram_chunks:
        wf["4"] = {"class_type":"MiniMaxLowVRAMAttention","inputs":{
            "model":model,"head_chunks":low_vram_chunks}}
        model = ["4",0]

    wf["5"] = {"class_type":"CLIPLoaderGGUF","inputs":{"clip_name":TE,"type":"minimax"}}
    clip = ["5",0]
    if clip_cpu:
        # Qwen3-VL-32B Q4_K is 13.58 GiB and ComfyUI-GGUF dequantizes on-GPU.
        # On a 15.57 GiB card that leaves ~0.8 GiB of scratch and it OOMs in
        # dequantize_blocks_Q4_K. The encoder runs ONCE per prompt, so pushing
        # it to CPU costs one slow pass and leaves the whole card for the DiT.
        wf["5b"] = {"class_type":"SelectCLIPDevice","inputs":{"clip":["5",0],"device":"cpu"}}
        clip = ["5b",0]
    wf["6"] = {"class_type":"VAELoader","inputs":{"vae_name":VVAE}}
    wf["7"] = {"class_type":"VAELoader","inputs":{"vae_name":AVAE}}

    # 🔴 MiniMax's spec REQUIRES an alignment instruction as the first line when
    # reference frames are supplied, then ONE blank line before the core fields.
    # Without it the model is getting out-of-distribution input.
    if first_frame and last_frame:
        prompt = ALIGN_FL2V.format(dur=f"{length/fps:.2f}") + "\n\n" + prompt
    elif first_frame or last_frame:
        prompt = ALIGN_I2V + "\n\n" + prompt

    n = 18
    wf["8"] = {"class_type":"MiniMaxH3ImageToVideo","inputs":{
        "clip":clip,"vae":["6",0],"prompt":prompt,
        "width":width,"height":height,"length":length}}
    if first_frame:
        wf[str(n)] = {"class_type":"LoadImage","inputs":{"image":first_frame}}
        wf["8"]["inputs"]["first_frame"] = [str(n),0]; n += 1
    if last_frame:
        wf[str(n)] = {"class_type":"LoadImage","inputs":{"image":last_frame}}
        wf["8"]["inputs"]["last_frame"] = [str(n),0]; n += 1

    wf["9"]  = {"class_type":"KSamplerSelect","inputs":{"sampler_name":sampler}}
    wf["10"] = {"class_type":"BasicScheduler","inputs":{
        "model":model,"scheduler":"simple","steps":steps,"denoise":1.0}}
    wf["11"] = {"class_type":"BasicGuider","inputs":{"model":model,"conditioning":["8",0]}}
    wf["12"] = {"class_type":"RandomNoise","inputs":{"noise_seed":seed}}
    wf["13"] = {"class_type":"SamplerCustomAdvanced","inputs":{
        "noise":["12",0],"guider":["11",0],"sampler":["9",0],
        "sigmas":["10",0],"latent_image":["8",1]}}

    wf["14"] = {"class_type":"VAEDecode","inputs":{"samples":["13",0],"vae":["6",0]}}
    wf["15"] = {"class_type":"VAEDecodeAudio","inputs":{"samples":["13",0],"vae":["7",0]}}
    wf["16"] = {"class_type":"CreateVideo","inputs":{
        "images":["14",0],"fps":float(fps),"audio":["15",0]}}
    wf["17"] = {"class_type":"SaveVideo","inputs":{
        "video":["16",0],"filename_prefix":prefix,"format":"auto","codec":"auto"}}
    return wf, length

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--width", type=int, default=864)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-turbo", action="store_true")
    p.add_argument("--turbo-profile", choices=sorted(TURBO_PROFILES), default=None,
                   help="turbo LoRA + sampler + shift set (default: DEFAULT_TURBO_PROFILE)")
    p.add_argument("--clip-cpu", action="store_true",
                   help="run the text encoder on CPU (avoids the GGUF dequant OOM)")
    p.add_argument("--first-frame", help="filename already uploaded to ComfyUI input/")
    p.add_argument("--last-frame", help="filename already uploaded to ComfyUI input/")
    p.add_argument("--camera-motion", action="store_true",
                   help="add the Camera Motion LoRA (prepends its trigger word)")
    p.add_argument("--camera-strength", type=float, default=1.0)
    p.add_argument("--head-chunks", type=int, default=4,
                   help="MiniMaxLowVRAMAttention chunks; higher = less VRAM, slower (max 56)")
    p.add_argument("--out", default="h3.json")
    a = p.parse_args()
    _prompt = a.prompt
    if a.camera_motion and not _prompt.lower().startswith(CAMERA_TRIGGER):
        _prompt = f"{CAMERA_TRIGGER}, {_prompt}"
    wf, length = build(_prompt, a.width, a.height, a.seconds, a.steps,
                       a.seed, turbo=not a.no_turbo, clip_cpu=a.clip_cpu,
                       low_vram_chunks=a.head_chunks,
                       first_frame=a.first_frame, last_frame=a.last_frame,
                       extra_lora=CAMERA_LORA if a.camera_motion else None,
                       extra_lora_strength=a.camera_strength,
                       turbo_profile=a.turbo_profile)
    json.dump(wf, open(a.out,"w"), indent=2)
    print(f"wrote {a.out}")
    mode = ("FL2V (first+last)" if a.first_frame and a.last_frame
            else "I2V (one frame)" if (a.first_frame or a.last_frame) else "T2V (text only)")
    print(f"  mode: {mode}")
    print(f"  {a.width}x{a.height}  length={length} frames "
          f"({length/24:.2f}s @24fps)  steps={a.steps}  turbo={not a.no_turbo}")
