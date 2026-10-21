#!/usr/bin/env python3
"""YuE2 post-production master chain (tested 2026-10-02 on YuE2 48 kHz output).

  python master.py IN OUT_BASENAME [--target -10] [--ceiling -1.5] [--reference ref.wav]
                   [--vocals vocals_stem.wav --vocal-db 2] [--clip-db 3]

Stages: float load -> (vocal-delta rebalance) -> (Matchering EQ/width match, no limiter)
-> headroom -> corrective EQ + glue -> leveler -> 4x-oversampled soft clip -> true-peak
brickwall -> 24-bit WAV + FLAC + MP3 V0 + AAC -> QC gate.
Needs: pip install pedalboard pyloudnorm soundfile scipy numpy matchering ; ffmpeg on PATH.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile
import numpy as np, soundfile as sf, pyloudnorm as pyln
from scipy.signal import resample_poly
from pedalboard import (Pedalboard, HighpassFilter, PeakFilter, HighShelfFilter, LowpassFilter,
                        Compressor, BrickwallLimiter)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qc import qc
from rebalance import rebalance

def softclip(y, ceil_db=-0.3, os_factor=4):
    """tanh clipper, oversampled so the harmonics it makes don't alias back down."""
    c = 10 ** (ceil_db / 20)
    u = resample_poly(y, os_factor, 1, axis=0)
    return resample_poly(c * np.tanh(u / c), 1, os_factor, axis=0).astype(np.float32)

def loud_stage(x, sr, gain_db, clip_db, ceiling_db):
    y = x * 10 ** ((gain_db - clip_db) / 20)
    y = BrickwallLimiter(ceiling_db=-3.0, release_ms=300, lookahead_ms=5)(y.T, sr).T    # slow leveler
    y = softclip(y * 10 ** ((clip_db + 1.5) / 20))                                     # eats transients
    return BrickwallLimiter(ceiling_db=ceiling_db, release_ms=60, lookahead_ms=3,
                            true_peak=True)(y.T, sr).T                                   # TP ceiling

def master(src, out_base, target=-10.0, ceiling=-1.5, reference=None, vocals=None,
           vocal_db=2.0, clip_db=3.0, eq=True):
    tmp = tempfile.mkdtemp(prefix="yue2m-")
    if vocals:
        p = os.path.join(tmp, "rebalanced.wav"); rebalance(src, vocals, p, vocal_db); src = p
    if reference:
        import matchering as mg
        cfg = mg.Config(); cfg.internal_sample_rate = 48000
        p = os.path.join(tmp, "matched.wav")
        mg.process(target=src, reference=reference, config=cfg,
                   results=[mg.Result(p, subtype="FLOAT", use_limiter=False, normalize=False)])
        src = p
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    meter = pyln.Meter(sr)
    x = pyln.normalize.loudness(x, meter.integrated_loudness(x), -18.0).astype(np.float32)
    if eq:
        x = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=28),
            PeakFilter(cutoff_frequency_hz=250, gain_db=-2.0 if not reference else -1.0, q=0.9),
            PeakFilter(cutoff_frequency_hz=3200, gain_db=1.5 if not reference else 0.0, q=0.8),
            HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.0 if not reference else 0.0, q=0.7),
            LowpassFilter(cutoff_frequency_hz=19000),
            Compressor(threshold_db=-20, ratio=1.8, attack_ms=30, release_ms=150),
        ])(x.T, sr).T
    g = target + 18.0                                       # first guess
    for _ in range(6):
        y = loud_stage(x, sr, g, clip_db, ceiling)
        err = target - meter.integrated_loudness(y)
        if abs(err) < 0.25: break
        g = min(g + err, 24.0)
    # Audio Studio copy: the 24-bit WAV is an intermediate in the temp dir; outputs are FLAC + MP3 only.
    wav = os.path.join(tmp, "master.wav")
    sf.write(wav, y, sr, subtype="PCM_24")
    run = lambda *a: subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, *a], check=True)
    run("-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24", out_base + ".flac")
    run("-c:a", "libmp3lame", "-q:a", "0", out_base + ".mp3")
    rep = {"master": qc(wav, target), "mp3": qc(out_base + ".mp3", target, tp_max=-0.5), "drive_db": round(g, 1)}
    shutil.rmtree(tmp, ignore_errors=True)
    return rep

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("src"); a.add_argument("out_base")
    a.add_argument("--target", type=float, default=-10.0); a.add_argument("--ceiling", type=float, default=-1.5)
    a.add_argument("--reference"); a.add_argument("--vocals"); a.add_argument("--vocal-db", type=float, default=2.0)
    a.add_argument("--clip-db", type=float, default=3.0); a.add_argument("--no-eq", action="store_true")
    o = a.parse_args()
    r = master(o.src, o.out_base, o.target, o.ceiling, o.reference, o.vocals, o.vocal_db, o.clip_db, not o.no_eq)
    print(json.dumps(r, indent=1))
    sys.exit(1 if r["master"]["fails"] else 0)
