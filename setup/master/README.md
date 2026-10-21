# Mastering + quality score (optional)

After every song render, Backlot can master the raw take and score it, the way a release would be prepared:

- `<name>.flac` the raw take, 24-bit, peak-safe
- `<name>-master.flac` and `<name>-master.mp3` mastered to -10 LUFS / -1.5 dBTP
- a quality score (Meta Audiobox Aesthetics on the raw take) shown in the gallery as "quality N"

It runs on the CPU in its own environment so the GPU stays free. Set it up once:

    python3 -m venv setup/master/.venv
    setup/master/.venv/bin/pip install -r setup/master/requirements.txt

Backlot uses it automatically when `setup/master/.venv/bin/python` exists, or set `BACKLOT_MASTER_PY`
to another interpreter. Without it, songs are loudness-normalized to MP3 only.
