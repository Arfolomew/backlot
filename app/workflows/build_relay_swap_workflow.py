"""Character Swap, masked (Oct 26 2026, Jake's new default after the viking crowd A/B).

akatz's full H3 Character Swap graph (h3_relay_character_swap.api.json, from
akatz-labs/h3-character-swap-tool, relay 8c313cc): SAM 3.1 masks ONE person and only that mask is
regenerated, so bystanders, framing and the rest of the frame stay the source's. Long clips run as
124-frame windows with 18 frames of history. Needs the h3-relay custom node pack on the backend.

Changes for a 16 GB card: our Ref2VA DiT, the Q4 GGUF text encoder on CPU, swap LoRA V1.5, and the
sampling workspace floor 17 GiB -> 0 (ComfyUI's own estimate).
"""
import copy
import json
import os

from .build_ref2va_workflow import DIT, TE

SWAP_LORA = "h3_character_swap_v1.5.safetensors"
SELECTIONS = ("largest", "leftmost", "rightmost")
with open(os.path.join(os.path.dirname(__file__), "h3_relay_character_swap.api.json")) as _f:
    GRAPH = json.load(_f)


def prompt_for(description=""):
    """akatz: a generic 'replace the person' keeps much of the source identity; describe the character."""
    desc = (description or "").strip().rstrip(".")
    who = f"the character in <Picture 1>: {desc}" if desc else "the character in <Picture 1>, keeping their face, hair, skin and clothing"
    return (f"Replace the selected person in <Video 1> with {who}. Preserve the source performance, "
            f"every movement and gesture, the camera motion, the scene and the audio timing.")


def build(video, image, prompt, seed=42, seconds=0.0, resolution="768", selection="largest",
          expand_px=64, prefix="video/BacklotSwap"):
    """video/image = names already uploaded to the backend. seconds 0 = the whole clip."""
    wf = copy.deepcopy(GRAPH)
    wf.pop("mask_preview", None)
    wf["model"]["inputs"]["unet_name"] = DIT
    wf["text"] = {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": TE, "type": "minimax"}}
    wf["text_device"]["inputs"]["device"] = "cpu"
    wf["adapter"]["inputs"]["lora_name"] = SWAP_LORA
    wf["sampling_memory"]["inputs"]["activation_memory_gib"] = 0.0
    wf["source"]["inputs"]["file"] = video
    wf["reference"]["inputs"]["image"] = image
    wf["prepare"]["inputs"].update(resolution=str(resolution), duration_seconds=float(seconds))
    wf["select"]["inputs"].update(selection=selection if selection in SELECTIONS else "largest", index=0)
    wf["expand_mask"]["inputs"]["expansion_px"] = int(expand_px)
    wf["plan"]["inputs"]["seed"] = int(seed)
    wf["reference_conditioning"]["inputs"]["prompt"] = prompt
    wf["save"]["inputs"]["filename_prefix"] = prefix
    return wf
