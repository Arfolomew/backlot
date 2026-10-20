#!/bin/bash
# Download the model files into a ComfyUI install.
#   setup/download-models.sh /path/to/ComfyUI          required set (video + Krea stills)
#   setup/download-models.sh /path/to/ComfyUI --all    plus every optional mode
# Uses huggingface-cli (pip install huggingface_hub). Re-running skips files that exist.
set -euo pipefail
COMFY="${1:?usage: download-models.sh /path/to/ComfyUI [--all]}"
ALL="${2:-}"
M="$COMFY/models"
[ -d "$M" ] || { echo "$M does not exist; is $COMFY a ComfyUI checkout?" >&2; exit 1; }
command -v huggingface-cli >/dev/null || { echo "pip install huggingface_hub   (needed for huggingface-cli)" >&2; exit 1; }

get() {  # get <repo> <path-in-repo> <models-subfolder>
  local repo="$1" path="$2" sub="$3" name
  name="$(basename "$path")"
  mkdir -p "$M/$sub"
  if [ -s "$M/$sub/$name" ]; then echo "have   $sub/$name"; return; fi
  echo "fetch  $sub/$name  <- $repo/$path"
  huggingface-cli download "$repo" "$path" --local-dir "/tmp/backlot-dl" >/dev/null
  mv "/tmp/backlot-dl/$path" "$M/$sub/$name"
}

# ---- required: MiniMax H3 ----
get ml-resources/MiniMax-H3-16G-HiFi diffusion_models/minimax_h3_fl2va_dit_16g.safetensors    diffusion_models
get ml-resources/MiniMax-H3-16G-HiFi diffusion_models/minimax_h3_ref2va_dit_16g.safetensors   diffusion_models
get Abiray/MiniMax-H3-GGUF          text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf          text_encoders
get Comfy-Org/MiniMax-H3            vae/minimax_h3_video_vae_fp16.safetensors                  vae
get Comfy-Org/MiniMax-H3            vae/minimax_h3_audio_vae_fp32.safetensors                  vae
get Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI minimax_h3_turbo_4step_ckpt600_ema_V4.safetensors loras
get lightx2v/Minimax-h3-Turbo       minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors loras
# ---- required: Krea 2 Turbo ----
get Comfy-Org/Krea-2                diffusion_models/krea2_turbo_fp8_scaled.safetensors        diffusion_models
get Comfy-Org/Krea-2                text_encoders/qwen3vl_4b_fp8_scaled.safetensors            text_encoders
get Comfy-Org/Krea-2                vae/qwen_image_vae.safetensors                             vae

if [ "$ALL" = "--all" ]; then
  get Jojocodex/minimax-h3-Camera-Motion-lora camera_motion_h3_lora_v1_3000_pruned.safetensors loras
  get MATLOWAI/MiniMax-H3-ORB360-CardSpin     minimax_h3_orb360_step1500.safetensors            loras/h3-tier2
  get akatz-ai/MiniMax-H3-Character-Swap-LoRA h3_character_swap_v1.5.safetensors            loras
  get Comfy-Org/Krea-2                        loras/krea2_style_reference.safetensors           loras
  get conradlocke/krea2-identity-edit         krea2_identity_edit_v1_2.safetensors              loras
  get unsloth/Qwen-Image-Edit-2511-GGUF       qwen-image-edit-2511-Q4_K_M.gguf                  unet
  get lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors loras
  get Comfy-Org/Qwen-Image_ComfyUI            split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors text_encoders
  get fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA qwen-image-edit-2511-multiple-angles-lora.safetensors loras
  get Comfy-Org/Qwen-Image-2.1                diffusion_models/qwen_image_2.1_int8_convrot.safetensors diffusion_models
  get Comfy-Org/Qwen-Image-2.1                loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors loras
  get Comfy-Org/Qwen-Image-2.1                text_encoders/qwen3vl_8b_int8_convrot.safetensors  text_encoders
  get Comfy-Org/Qwen-Image-2.1                vae/qwen_image_2.1_vae_bf16.safetensors            vae
  get numz/SeedVR2_comfyUI                    seedvr2_ema_3b_fp8_e4m3fn.safetensors             SEEDVR2
  get numz/SeedVR2_comfyUI                    ema_vae_fp16.safetensors                          SEEDVR2
  get ai-forever/Real-ESRGAN                  RealESRGAN_x2.pth                                 upscale_models
fi
echo "done."
