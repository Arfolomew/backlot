# Model files

Every model is open weights. Put each file in the ComfyUI folder named in the last column
(`ComfyUI/models/<folder>/`). Sizes are rounded. `setup/download-models.sh` fetches the
required set with `huggingface-cli`.

## Required: video (MiniMax H3)

| File | Source | Size | Folder |
|---|---|---|---|
| `minimax_h3_fl2va_dit_16g.safetensors` | [ml-resources/MiniMax-H3-16G-HiFi](https://huggingface.co/ml-resources/MiniMax-H3-16G-HiFi) `diffusion_models/` (mirror of ModelScope `l1ngzi/MiniMax-H3-16G-HiFi`) | 28.4 GB | `diffusion_models` |
| `minimax_h3_ref2va_dit_16g.safetensors` | same repo, `diffusion_models/` | 28.4 GB | `diffusion_models` |
| `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | [Abiray/MiniMax-H3-GGUF](https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf) `text_encoders/` | 14.6 GB | `text_encoders` |
| `minimax_h3_video_vae_fp16.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) `vae/` | 5.2 GB | `vae` |
| `minimax_h3_audio_vae_fp32.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) `vae/` | 0.6 GB | `vae` |
| `minimax_h3_turbo_4step_ckpt600_ema_V4.safetensors` | [Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI](https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt600_ema_V4.safetensors) | 0.6 GB | `loras` |
| `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors) | 2.0 GB | `loras` |

The two 16G DiTs are INT8 "HiFi" repacks that fit a 16 GB card. The original Q4 GGUFs
(`Abiray/MiniMax-H3-Pruned-GGUF`, 11.6 GB each, folder `unet`) still work: set `DIT = DIT_Q4`
in `app/workflows/build_h3_workflow.py`. Do not substitute the same-named text-encoder GGUF from
other repos; it is a different, larger quant.

## Required: stills (Krea 2 Turbo)

| File | Source | Size | Folder |
|---|---|---|---|
| `krea2_turbo_fp8_scaled.safetensors` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors) | 13.1 GB | `diffusion_models` |
| `qwen3vl_4b_fp8_scaled.safetensors` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) | 5.2 GB | `text_encoders` |
| `qwen_image_vae.safetensors` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors) | 0.25 GB | `vae` |

## Optional: extra modes

| Mode | File | Source | Size | Folder |
|---|---|---|---|---|
| Camera motion | `camera_motion_h3_lora_v1_3000_pruned.safetensors` | [Jojocodex/minimax-h3-Camera-Motion-lora](https://huggingface.co/Jojocodex/minimax-h3-Camera-Motion-lora/resolve/main/camera_motion_h3_lora_v1_3000_pruned.safetensors) | 0.16 GB | `loras` |
| 360 orbit | `h3-tier2/minimax_h3_orb360_step1500.safetensors` | [MATLOWAI/MiniMax-H3-ORB360-CardSpin](https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/resolve/main/minimax_h3_orb360_step1500.safetensors) | 0.6 GB | `loras/h3-tier2` |
| Krea style reference | `krea2_style_reference.safetensors` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_style_reference.safetensors) | 0.46 GB | `loras` |
| Krea identity edit | `krea2_identity_edit_v1_2.safetensors` | [conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/krea2_identity_edit_v1_2.safetensors) | 1.8 GB | `loras` |
| Instruction edit | `qwen-image-edit-2511-Q4_K_M.gguf` | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q4_K_M.gguf) | 13.3 GB | `unet` |
| Instruction edit | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors) | 0.85 GB | `loras` |
| Instruction edit | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) | 9.4 GB | `text_encoders` |
| Camera angles | `qwen-image-edit-2511-multiple-angles-lora.safetensors` | [fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/main/qwen-image-edit-2511-multiple-angles-lora.safetensors) | 0.3 GB | `loras` |
| Qwen Image 2.1 | `qwen_image_2.1_int8_convrot.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors) | 7.3 GB | `diffusion_models` |
| Qwen Image 2.1 | `qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors) | 0.9 GB | `loras` |
| Qwen Image 2.1 | `qwen3vl_8b_int8_convrot.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors) | 9.4 GB | `text_encoders` |
| Qwen Image 2.1 | `qwen_image_2.1_vae_bf16.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors) | 0.7 GB | `vae` |
| Upscale (SeedVR2) | `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | [numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp8_e4m3fn.safetensors) | 3.4 GB | `SEEDVR2` |
| Upscale (SeedVR2) | `ema_vae_fp16.safetensors` | [numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/ema_vae_fp16.safetensors) | 0.5 GB | `SEEDVR2` |
| Upscale (fast) | `RealESRGAN_x2.pth` | [ai-forever/Real-ESRGAN](https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth) | 67 MB | `upscale_models` |
| Interpolation | `rife49.pth` | [Fannovel16/ComfyUI-Frame-Interpolation release](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife49.pth) (the node downloads it on first use) | 21 MB | `custom_nodes/comfyui-frame-interpolation/ckpts/rife/` |

## Disk budget

Required set: about 98 GB. Everything: about 170 GB.
