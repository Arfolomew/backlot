#!/usr/bin/env python3
"""Build a MiniMax H3 **Ref2VA** (reference-to-video) workflow. Norm 2026-09-11.

SEPARATE FILE by house rule — build_h3_workflow.py (the proven FL2VA/i2v
path) is not touched by any of this.

Ref2VA keeps a CHARACTER, PRODUCT or STYLE across a scene the model invents, rather
than interpolating between keyframes the way FL2VA does. Up to 9 images / 3 videos /
3 audio clips, 12 files total.

  UnetLoaderGGUF(Ref2VA DiT) -> LoraLoaderModelOnly(ref2v turbo) -> SigmaShift
    -> MiniMaxLowVRAMAttention -> MODEL
  CLIPLoaderGGUF -> SelectCLIPDevice(cpu) -> CLIP
  MiniMaxH3ReferenceToVideo(clip, vae, audio_vae, prompt, w, h, length,
                            ref_image_size, ref_image_N...) -> (CONDITIONING, LATENT)
  SamplerCustomAdvanced -> VAEDecode + VAEDecodeAudio -> CreateVideo -> SaveVideo

🔴 The FL2VA turbo LoRA does NOT work on Ref2VA — it needs its own ref2v turbo LoRA.
🔴 The text encoder MUST stay on CPU; it is 13.58 GB and ComfyUI-GGUF dequantizes
   on-GPU, so it OOMs on CUDA.
🔴 Ref2VA's prompt format is DIFFERENT from FL2VA's: six sections, 350-500 words,
   field name `detailed_description`. Spec: h3-official-prompt-spec/ref-en.txt
"""
import json, argparse

# Oct 14 2026: HiFi INT8 repack (same source as the FL2VA switch). Rollback: DIT = DIT_Q4.
DIT_Q4 = "MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf"
DIT   = "minimax_h3_ref2va_dit_16g.safetensors"
TE    = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
VVAE  = "minimax_h3_video_vae_fp16.safetensors"
AVAE  = "minimax_h3_audio_vae_fp32.safetensors"
TURBO = "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
REALISM = "h3-realism-people-t2v-i2v-r2v.safetensors"  # fal, trigger word: r34l1sm


def snap(n):
    """H3 frame grid: n %% 17 == 5."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def build(prompt, ref_images=(), ref_audios=(), width=1344, height=768, seconds=5.0, steps=8,
          seed=42, turbo=True, low_vram_chunks=16, fps=24,
          prefix="video/MiniMaxH3Ref", clip_cpu=True, ref_image_size="match",
          scheduler="simple", ref_index_base=0, extra_loras=(),
          chain_index=0, context_length="22", audio_context_length=24,
          context_prefix="h3_context/clip"):
    """ref_images / ref_audios: filenames already sitting in the ComfyUI input/ directory."""
    if len(ref_images) > 9:
        raise ValueError("Ref2VA accepts at most 9 reference images")
    if len(ref_audios) > 3:
        raise ValueError("Ref2VA accepts at most 3 reference audio clips")
    # Motion Context: only these windows are whole numbers of latent steps.
    if str(context_length) not in ("5", "22", "39", "56"):
        raise ValueError("context_length must be one of 5, 22, 39, 56")
    if chain_index < 0:
        raise ValueError("chain_index must be >= 0")
    length = snap(round(seconds * fps))

    wf = {}
    wf["1"] = ({"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": DIT}} if DIT.endswith(".gguf")
               else {"class_type": "UNETLoader", "inputs": {"unet_name": DIT, "weight_dtype": "default"}})
    model = ["1", 0]
    if turbo:
        wf["2"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model, "lora_name": TURBO, "strength_model": 1.0}}
        model = ["2", 0]
    # Optional stacked LoRAs, e.g. fal Realism People. Chained after turbo so the
    # step-distillation still owns the schedule. Node ids 2a, 2b, ... to avoid
    # renumbering the proven graph.
    for i, (lname, lstr) in enumerate(extra_loras):
        nid = f"2{chr(ord('a') + i)}"
        wf[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model, "lora_name": lname, "strength_model": float(lstr)}}
        model = [nid, 0]
    wf["3"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {
        "model": model, "shift_video": 12.0, "shift_audio": 3.0}}
    model = ["3", 0]
    if low_vram_chunks:
        wf["4"] = {"class_type": "MiniMaxLowVRAMAttention", "inputs": {
            "model": model, "head_chunks": low_vram_chunks}}
        model = ["4", 0]

    wf["5"] = {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": TE, "type": "minimax"}}
    clip = ["5", 0]
    if clip_cpu:
        wf["5b"] = {"class_type": "SelectCLIPDevice", "inputs": {"clip": clip, "device": "cpu"}}
        clip = ["5b", 0]

    wf["6"] = {"class_type": "VAELoader", "inputs": {"vae_name": VVAE}}
    wf["7"] = {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}}

    ref_node = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": clip, "vae": ["6", 0], "audio_vae": ["7", 0], "prompt": prompt,
        "width": width, "height": height, "length": length,
        "ref_image_size": ref_image_size}}
    n = 20
    for i, fn in enumerate(ref_images):
        wf[str(n)] = {"class_type": "LoadImage", "inputs": {"image": fn}}
        # 🔴 AUTOGROW_V3 API key is a DOTTED PATH: "<input_id>.<prefix><i>", i.e.
        # "ref_images.ref_image_0". comfy_api/latest/_io.py builds it with
        # finalize_prefix(), which is ".".join(prefix_list). Zero-based.
        # A flat "ref_image_0" VALIDATES FINE and then dies at execution with
        # "unexpected keyword argument" — acceptance never proved the wiring.
        ref_node["inputs"][f"ref_images.ref_image_{i + ref_index_base}"] = [str(n), 0]
        n += 1
    # Reference AUDIO — voice-timbre / soundscape conditioning. Same AUTOGROW_V3
    # dotted-path rule as images: "ref_audios.ref_audio_0", zero-based. Node ids
    # start at 40 so they can never collide with the 20-28 image block.
    # 🔴 This is the STANDALONE ref_audios input. The node also exposes
    # ref_video_audios (a soundtrack PAIRED to a same-numbered ref_video); HF
    # discussion #64 claims cloning only works properly through that path. Pairing
    # needs a ref_video too, so it is deliberately not built here yet.
    a = 40
    for i, fn in enumerate(ref_audios):
        wf[str(a)] = {"class_type": "LoadAudio", "inputs": {"audio": fn}}
        ref_node["inputs"][f"ref_audios.ref_audio_{i}"] = [str(a), 0]
        a += 1
    wf["8"] = ref_node

    wf["9"]  = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    wf["10"] = {"class_type": "BasicScheduler", "inputs": {
        "model": model, "scheduler": scheduler, "steps": steps, "denoise": 1.0}}
    # ---- Motion Context chaining (nodes 50+; images use 20-28, audio 40-42) ----
    # 🔴 Chaining through a PNG loses the join: the first frame comes back
    # REINTERPRETED (measured 14-16% mean pixel difference on a roller-rink
    # clips 2026-09-22), brightness drifts down every link, and a single still
    # frame carries NO VELOCITY, so motion restarts from a standstill. This pins
    # N consecutive frames straight out of the previous clip's LATENT instead.
    guider_cond = ["8", 0]      # default: straight off the reference node
    video_images = ["14", 0]    # default: straight off the decode
    video_audio = ["15", 0]
    if chain_index >= 2:
        wf["50"] = {"class_type": "MiniMaxH3MotionContextLoadLatent", "inputs": {
            "latent_path": context_prefix.split("/")[0] or "h3_context",
            "clip_index": chain_index - 1}}
        wf["51"] = {"class_type": "MiniMaxH3MotionContext", "inputs": {
            "conditioning": ["8", 0], "vae": ["6", 0], "latent": ["8", 1],
            "context_length": str(context_length),
            "audio_context_length": int(audio_context_length),
            "context_latent": ["50", 0]}}
        guider_cond = ["51", 0]

    wf["11"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": guider_cond}}
    wf["12"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    wf["13"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["12", 0], "guider": ["11", 0], "sampler": ["9", 0],
        "sigmas": ["10", 0], "latent_image": ["8", 1]}}
    wf["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["6", 0]}}
    wf["15"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["7", 0]}}

    if chain_index >= 1:
        # Save THIS clip's sampler latent so the next one can continue from it.
        wf["52"] = {"class_type": "MiniMaxH3MotionContextSaveLatent", "inputs": {
            "latent": ["13", 0], "filename_prefix": context_prefix,
            "clip_index": chain_index}}
    if chain_index >= 2:
        # Strip the pinned head back off, picture and sound together.
        wf["53"] = {"class_type": "MiniMaxH3MotionContextTrim", "inputs": {
            "images": ["14", 0], "trim_frames": ["51", 1],
            "audio": ["15", 0], "fps": float(fps), "match_tail": True}}
        video_images = ["53", 0]
        video_audio = ["53", 1]

    wf["16"] = {"class_type": "CreateVideo", "inputs": {
        "images": video_images, "fps": float(fps), "audio": video_audio}}
    wf["17"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["16", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}}
    return wf, length


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--ref", action="append", default=[],
                   help="reference image filename in ComfyUI input/ (repeatable, max 9)")
    p.add_argument("--ref-audio", action="append", default=[], dest="ref_audio",
                   help="reference audio filename in ComfyUI input/ (repeatable, max 3)")
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scheduler", default="simple", choices=["simple", "normal", "beta"])
    p.add_argument("--ref-image-size", default="match", choices=["match", "max"])
    p.add_argument("--head-chunks", type=int, default=16)
    p.add_argument("--realism", type=float, default=None, metavar="STRENGTH",
                   help="stack the fal Realism People LoRA at this strength "
                        "(1.0 intended, 0.6-0.8 lighter). Put 'r34l1sm' at the "
                        "head of the prompt when you use it.")
    p.add_argument("--lora", action="append", default=[], metavar="NAME:STRENGTH",
                   help="stack an arbitrary extra LoRA (repeatable)")
    p.add_argument("--chain-index", type=int, default=0, dest="chain_index", metavar="N",
                   help="Motion Context chaining. 0=off. 1=first clip of a chain "
                        "(saves its latent). 2+=continue from clip N-1 (loads, pins, trims). "
                        "Clip N always saves to slot N and loads slot N-1.")
    p.add_argument("--context-length", default="22", dest="context_length",
                   choices=["5", "22", "39", "56"],
                   help="frames of the previous clip's picture to pin. 22 is nearly "
                        "seamless; 5 is just barely fluid; longer windows come off the "
                        "FRONT of the delivered clip (default: 22)")
    p.add_argument("--audio-context", type=int, default=24, dest="audio_context_length",
                   metavar="FRAMES",
                   help="frames of tail audio to pin. Multiples of 24 are whole "
                        "seconds; 24 pins the last second (default: 24)")
    p.add_argument("--context-prefix", default="h3_context/clip", dest="context_prefix",
                   help="where chain latents live under the ComfyUI output folder")
    p.add_argument("--out", default="ref2va.json")
    a = p.parse_args()
    extra = []
    if a.realism is not None:
        extra.append((REALISM, a.realism))
    for spec in a.lora:
        name, _, strength = spec.rpartition(":")
        if not name:
            p.error(f"--lora expects NAME:STRENGTH, got {spec!r}")
        extra.append((name, float(strength)))
    wf, length = build(a.prompt, ref_images=a.ref, ref_audios=a.ref_audio,
                       width=a.width, height=a.height,
                       seconds=a.seconds, steps=a.steps, seed=a.seed,
                       scheduler=a.scheduler, ref_image_size=a.ref_image_size,
                       low_vram_chunks=a.head_chunks, extra_loras=extra,
                       chain_index=a.chain_index, context_length=a.context_length,
                       audio_context_length=a.audio_context_length,
                       context_prefix=a.context_prefix)
    json.dump(wf, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")
    print(f"  {a.width}x{a.height}  length={length} ({length/24:.2f}s @24fps)  "
          f"refs={len(a.ref)}  audio_refs={len(a.ref_audio)}  "
          f"steps={a.steps}  scheduler={a.scheduler}")
    if extra:
        print("  stacked LoRAs: " + ", ".join(f"{n}@{v}" for n, v in extra))
    if a.chain_index == 0:
        print("  chaining: OFF (pass --chain-index 1 on the first clip, 2 on the next, ...)")
    elif a.chain_index == 1:
        print(f"  chaining: clip 1 of a chain - saves latent to slot 1, nothing to continue from")
    else:
        print(f"  chaining: clip {a.chain_index} - continues slot {a.chain_index - 1}, "
              f"pins {a.context_length} frames + {a.audio_context_length} audio frames, "
              f"saves slot {a.chain_index}")
