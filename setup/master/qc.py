#!/usr/bin/env python3
"""Release QC gate. Exit 1 if any hard check fails."""
import sys, json, subprocess, numpy as np

def ebur128(path):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
                          "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    t = err[err.rfind("Summary:"):]
    g = lambda k: float(t.split(k)[1].split()[0])
    return g("I:"), g("LRA:"), g("Peak:")

def decode(path):
    """ffmpeg decode to float so MP3/AAC/FLAC/WAV all work (soundfile can't read .m4a)."""
    sr = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                             "stream=sample_rate", "-of", "csv=p=0", path], capture_output=True, text=True).stdout)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64), sr

def qc(path, target=-10.0, tol=1.0, tp_max=-1.0):
    x, sr = decode(path)
    I, LRA, TP = ebur128(path)
    a = np.abs(x)
    clip_runs = int(np.sum((a[1:] >= 0.999) & (a[:-1] >= 0.999)))     # consecutive full-scale samples
    dc = float(np.abs(x.mean(0)).max())
    L, R = x[:, 0], x[:, -1]
    corr = float(np.corrcoef(L, R)[0, 1]) if x.shape[1] == 2 else 1.0
    mono_loss = 10 * np.log10((((L + R) / 2) ** 2).mean() / (((L ** 2).mean() + (R ** 2).mean()) / 2 + 1e-20) + 1e-20)  # 0 = mono-safe, more negative = more cancels
    m = x.mean(1); n = 8192
    P = np.mean([np.abs(np.fft.rfft(m[i:i + n] * np.hanning(n))) ** 2 for i in range(0, len(m) - n, n * 4)], 0)
    f = np.fft.rfftfreq(n, 1 / sr); band = lambda lo, hi: 10 * np.log10(P[(f >= lo) & (f < hi)].mean() + 1e-30)
    hf_cliff = band(14000, 16000) - band(18000, 20000)                  # big number = hard lowpass wall
    env = 20 * np.log10(np.abs(m) + 1e-9)
    head = int(np.argmax(env > -60)) / sr; tail = (len(m) - int(np.argmax(env[::-1] > -60))) / sr
    r = {"LUFS_I": I, "LRA": LRA, "TP_dBTP": TP, "clipped_runs": clip_runs, "dc": round(dc, 5),
         "LR_corr": round(corr, 3), "mono_fold_dB": round(mono_loss, 2), "hf_cliff_dB": round(hf_cliff, 1),
         "head_silence_s": round(head, 2), "tail_silence_s": round(len(m) / sr - tail, 2)}
    fails = []
    if TP > tp_max: fails.append(f"true peak {TP} > {tp_max}")
    if abs(I - target) > tol: fails.append(f"loudness {I} not within {tol} of {target}")
    if clip_runs: fails.append(f"{clip_runs} clipped sample runs")
    if dc > 0.001: fails.append("DC offset")
    if corr < 0.2: fails.append("stereo correlation < 0.2 (mono-compat risk)")
    r["fails"] = fails
    return r

if __name__ == "__main__":
    r = qc(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else -10.0)
    print(json.dumps(r, indent=1)); sys.exit(1 if r["fails"] else 0)
