"""SaveAudioFloatWAV: write the decoder output as 32-bit float WAV, no clamping.

Added for Audio Studio (2026-10-03). The stock SaveAudio writes 16-bit FLAC, and YuE2's
decoder peaks slightly over full scale, so every stock save clipped a few samples.
Float WAV keeps the overs intact; Audio Studio's post step does the gain and limiting.
Pure numpy, no extra dependencies.
"""
import os
import struct

import numpy as np
import folder_paths


def _write_float_wav(path, x, sr):
    """x: (frames, channels) float32. WAVE_FORMAT_IEEE_FLOAT with a fact chunk."""
    x = np.ascontiguousarray(x, dtype="<f4")
    frames, ch = x.shape
    data = x.tobytes()
    fmt = struct.pack("<HHIIHHH", 3, ch, sr, sr * ch * 4, ch * 4, 32, 0)
    fact = struct.pack("<I", frames)
    riff_size = 4 + (8 + len(fmt)) + (8 + len(fact)) + (8 + len(data))
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", riff_size) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        f.write(b"fact" + struct.pack("<I", len(fact)) + fact)
        f.write(b"data" + struct.pack("<I", len(data)) + data)


class SaveAudioFloatWAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",),
                             "filename_prefix": ("STRING", {"default": "audio/float"})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def save(self, audio, filename_prefix):
        out_dir = folder_paths.get_output_directory()
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        sr = int(audio["sample_rate"])
        results = []
        for wf in audio["waveform"].cpu():
            fn = f"{filename}_{counter:05}_.wav"
            _write_float_wav(os.path.join(folder, fn), wf.float().numpy().T, sr)
            results.append({"filename": fn, "subfolder": subfolder, "type": "output"})
            counter += 1
        return {"ui": {"audio": results}}


NODE_CLASS_MAPPINGS = {"SaveAudioFloatWAV": SaveAudioFloatWAV}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveAudioFloatWAV": "Save Audio (32-bit float WAV)"}
