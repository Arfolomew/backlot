#!/bin/bash
# Clone the node packs the studio's graphs use into a ComfyUI install.
#   setup/install-nodes.sh /path/to/ComfyUI
set -euo pipefail
COMFY="${1:?usage: install-nodes.sh /path/to/ComfyUI}"
NODES="$COMFY/custom_nodes"
[ -d "$NODES" ] || { echo "$NODES does not exist; is $COMFY a ComfyUI checkout?" >&2; exit 1; }

PY="$COMFY/../venv/bin/python"; [ -x "$PY" ] || PY="$COMFY/venv/bin/python"; [ -x "$PY" ] || PY="python3"

clone() {  # clone <repo-url> <dir>
  if [ -d "$NODES/$2" ]; then echo "have   $2"; else
    echo "clone  $2"; git clone -q --depth 1 "$1" "$NODES/$2"
  fi
  if [ -f "$NODES/$2/requirements.txt" ]; then
    "$PY" -m pip install -q -r "$NODES/$2/requirements.txt" || echo "  (requirements for $2 failed; check manually)"
  fi
}

clone https://github.com/city96/ComfyUI-GGUF                         ComfyUI-GGUF
clone https://github.com/kijai/ComfyUI-KJNodes                       ComfyUI-KJNodes
# Multishot from Arfolomew's fork: one-line fix so audio guides and reference audio can share a graph
# (upstream jlucasmcrell/ComfyUI-H3-Multishot crashes Character Swap with a [412, 32] / [824, 32] shape mismatch).
clone https://github.com/Arfolomew/ComfyUI-H3-Multishot             ComfyUI-H3-Multishot
clone https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context       ComfyUI-H3-Motion-Context
clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation      comfyui-frame-interpolation
clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler          ComfyUI-SeedVR2_VideoUpscaler
clone https://github.com/lbouaraba/comfyui-krea2edit                 comfyui-krea2edit
# Character Swap: akatz's H3 Relay (SAM-masked swap + windowed continuation), pinned to the tested commit.
if [ ! -d "$NODES/h3-relay" ]; then
  echo "clone  h3-relay"; git clone -q https://github.com/akatz-ai/h3-relay "$NODES/h3-relay"
  git -C "$NODES/h3-relay" checkout -q 8c313cc6bd23298be3b475d95a3af4cde6e7f10b
else echo "have   h3-relay"; fi
echo "done. Restart ComfyUI."

# Audio tab: one small node that saves 32-bit float WAV (the stock SaveAudio clips YuE2's overs).
cp "$(dirname "$0")/save_audio_float_wav.py" "$NODES/save_audio_float_wav.py" && echo "copied save_audio_float_wav.py"
