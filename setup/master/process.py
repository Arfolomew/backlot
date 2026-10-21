#!/usr/bin/env python3
"""Audio Studio post step for one take. Runs in post/.venv (CPU only, GPU stays free).

  process.py RAW_FLOAT_WAV OUT_BASE [--target -10]

Writes OUT_BASE.flac          raw take, 24-bit, peak-safe (gain only, so nothing clips)
       OUT_BASE-master.flac   mastered, 24-bit, -10 LUFS / -1.5 dBTP
       OUT_BASE-master.mp3    mastered, LAME V0
Prints one JSON line: {"raw_peak_db", "master": qc, "mp3": qc, "score": {...}}.
Score = Meta Audiobox Aesthetics on the RAW take (it barely reacts to mastering, so rank raw).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf

a = argparse.ArgumentParser()
a.add_argument("src"); a.add_argument("out_base"); a.add_argument("--target", type=float, default=-10.0)
o = a.parse_args()

x, sr = sf.read(o.src, dtype="float32", always_2d=True)
peak = float(np.abs(x).max()) or 1.0
gain = min(1.0, 10 ** (-1.0 / 20) / peak)          # only turn DOWN, to a -1 dBFS peak
sf.write(o.out_base + ".flac", x * gain, sr, subtype="PCM_24")
out = {"raw_peak_db": round(20 * np.log10(peak), 2)}

from master import master
out.update(master(o.src, o.out_base + "-master", target=o.target))

try:
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    from audiobox_aesthetics.infer import initialize_predictor
    r = initialize_predictor().forward([{"path": torch.from_numpy(x.T.copy()), "sample_rate": sr}])[0]
    out["score"] = {k: round(float(v), 2) for k, v in r.items()}
    out["score"]["overall"] = round((out["score"]["CE"] + out["score"]["PQ"]) / 2, 2)
except Exception as e:                                # scoring is a bonus, never fail the take over it
    out["score"] = None
    out["score_error"] = str(e)[:300]
print(json.dumps(out))
