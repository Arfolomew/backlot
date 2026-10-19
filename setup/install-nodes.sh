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
clone https://github.com/jlucasmcrell/ComfyUI-H3-Multishot           ComfyUI-H3-Multishot
clone https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context       ComfyUI-H3-Motion-Context
clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation      comfyui-frame-interpolation
clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler          ComfyUI-SeedVR2_VideoUpscaler
clone https://github.com/lbouaraba/comfyui-krea2edit                 comfyui-krea2edit
echo "done. Restart ComfyUI."
