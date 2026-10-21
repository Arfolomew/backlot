#!/usr/bin/env python3
"""Vocal-delta rebalance: out = mix + (g - 1) * vocals_est.
Never resums separated stems, so separation errors only leak in at (g - 1) scale.
Optional: EQ applied only to the instrumental residual (mix - vocals)."""
import sys, numpy as np, soundfile as sf
from math import gcd
from scipy.signal import resample_poly
from pedalboard import Pedalboard, PeakFilter, HighpassFilter

def load_at(path, sr_out):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sr_out:                                   # audio-separator writes 44.1 kHz stems
        g = gcd(sr, sr_out); x = resample_poly(x, sr_out // g, sr // g, axis=0).astype(np.float32)
    return x

def rebalance(mix_path, vocals_path, out_path, vocal_db=2.0, inst_mud_db=-2.0, processed_vocals_path=None):
    """processed_vocals_path: e.g. a de-essed copy of the stem; it replaces the vocal (mix - stem + processed)."""
    mix, sr = sf.read(mix_path, dtype="float32", always_2d=True)
    voc = load_at(vocals_path, sr); n = min(len(mix), len(voc)); mix, voc = mix[:n], voc[:n]
    inst = mix - voc                                   # residual instrumental, sums back exactly
    if inst_mud_db:
        inst = Pedalboard([HighpassFilter(25), PeakFilter(300, inst_mud_db, 0.8)])(inst.T, sr).T
    if processed_vocals_path:
        voc = load_at(processed_vocals_path, sr)[:n]
    out = inst + voc * 10 ** (vocal_db / 20)
    sf.write(out_path, out, sr, subtype="FLOAT")      # float: may exceed 0 dBFS, mastering handles it
    return out

if __name__ == "__main__":
    # rebalance.py MIX VOCALS OUT [VOCAL_DB] [PROCESSED_VOCALS]
    rebalance(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 2.0,
              processed_vocals_path=sys.argv[5] if len(sys.argv) > 5 else None)
