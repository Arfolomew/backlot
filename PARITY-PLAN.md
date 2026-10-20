# Backlot parity build (branch `higgsfield-parity`) — status and next steps

Goal: ship the Higgsfield *workflow* on a 16 GB card before the Oct 30 9am PT deadline, push to the
live link early if done. Source of truth for what to copy: `research/higgsfield/HIGGSFIELD-TEARDOWN.md`.

## Done (Oct 20, 2 to 2:35 AM)
- **Moves tab** (commit 8b3d529): 30 presets in `app/moves.json` (24 camera moves on the Camera Motion
  LoRA, 6 prompt effects), `GET /api/moves`, `/previews/`, `move` + `subject` fields on `/api/video`.
  Verified: push-in preview rendered end to end (140 s at 864x480), UI smoke test 30 cards / 9 filters /
  preview plays / 0 JS errors. **Previews rendered: push-in, push-in-close, pull-back** (3 of 30).
- **Cinema Studio tab** (next commit): `app/cinema.json` fragments, `POST /api/cinema/compose`,
  `GET /api/cinema`. Verified: composer unit test, API, UI smoke (7 pickers, cut list, prompt box fills,
  0 errors). **A live 3-shot render was started but cancelled** (Jake: stop for tonight). Not yet seen
  on screen whether H3 honors the cut times at 864x480.

## Oct 20 afternoon
- 03:44 the rig rebooted uncleanly; Storage came up dirty and would not mount, so every H3 render
  failed until 15:42. Jake ran ntfsfix + mount. The two HiFi DiTs (28 GB each) and h3-tier2 LoRAs
  are now REAL FILES on the main drive (symlinks removed); judging no longer depends on Storage.
- Apps tab built + verified (6 apps; SeedVR2 upscale ran through it in 10.5 s). Character Swap
  mode added to /api/video (not yet rendered end to end).
- Cinema Studio VERIFIED: calm 3-shot lighthouse test at 864x480, 8 s, 320 s render. Cuts landed
  at 2.3 s and 5.3 s exactly as composed, same subject across cuts. Sample saved to
  app/previews/cinema/calm-lighthouse-1960s.mp4.
- Gallery tags + README sections committed. Tag end-to-end check still pending (GPU was busy).
- Preview batch restarted 15:47, ~140 s each. Spot check at 10/30: crash-zoom, orbit-left, tracking-side
  read correctly; dolly-zoom-in renders as push-in then pull-out, not a vertigo warp (re-prompt later,
  low priority); freeze froze the window, not the rose (same). earth-zoom-out, disintegration, fpv, aerial,
  handheld, crash-zoom, orbit, tracking all correct.
- 16:33 MiniMax Studio's idle watchdog stopped comfyui-h3 mid-batch (4 previews failed, re-run in pass 2).
  Fixed in minimax-studio/server.py: the watchdog skips the stop while :8189/queue has work. Test scripts for Character Swap + chaotic Cinema are in tools/tests/.

- 16:50 all 30 previews on disk (pass 2 re-rendered the 4 that failed). Committed (39 MB).
- 17:05 chaotic Cinema test: 6 shots in 8 s at 864x480 in 206 s, cuts where composed, same drummer
  throughout. Gallery TAGS verified end to end on it (meta JSON + tooltip). Narration v2 generated
  (8 lines, ends 118.1 s of 120) -> demo/voice/vo-v2.wav. Full 5-tab UI smoke passed, 0 JS errors.
- 17:06 Character Swap end-to-end render started (tools/tests/swap_test.py, ~20 min).

## Next, in order
1. `PORT=8805 ... .venv/bin/python app/server.py` (env in this file's "Dev instance" below), then
   `.venv/bin/python tools/render_move_previews.py --server http://127.0.0.1:8805` for the other 27
   previews (~135 s each, ~1 h). Eyeball a few (dolly zoom, FPV, whip pan, the 3 effects).
2. Cinema: render the calm 3-shot lighthouse test and a chaotic one; confirm cuts land where the
   prompt says. If H3 ignores cut times at short clips, fall back to rendering shots separately and
   concatenating (ffmpeg concat, audio included).
3. **Apps tab**: grid of one-input tools that already exist as modes/workflows: Angles (Qwen Multiple
   Angles LoRA), Face Fix (CodeFormer in shot-caller), Character Swap (akatz LoRA, build_charswap),
   4K Upscale (SeedVR2), 360 Orbit, Cutout (transparent). Mostly UI over existing endpoints.
4. **Gallery tags**: tag every output with its mode/move/app and add a re-run button.
5. README: new sections + the credits comparison (Higgsfield Plus = 640 five-second 720p clips for
   $59/mo vs Backlot on your own card). Demo video v2 (motion-studio) showing Moves + Cinema + Apps.
6. Deploy: merge to main, push, clean-clone install test, restart backlot-live (`systemctl --user
   restart backlot-live`), replace the hosted demo mp4, follow-up #OpenSourceHiggsfield post.

## Dev instance
```
cd ~/Desktop/Projects/backlot
export PORT=8805 BACKLOT_COMFY_URL=http://127.0.0.1:8199 BACKLOT_COMFYUI_URL=http://127.0.0.1:8189 \
  BACKLOT_QWEN_URL=http://127.0.0.1:8198 BACKLOT_QWEN_COMFYUI_URL=http://127.0.0.1:8191 \
  BACKLOT_COMFY_OUTPUT_DIR=$HOME/ComfyUI-h3/ComfyUI/output
nohup .venv/bin/python app/server.py > dev-8805.log 2>&1 &
```
Shares the proxy (:8199) and ComfyUI-h3 (:8189) with the live instance on :8802; renders queue on
the same GPU. Never restart :8802 or :8189 while judges might be looking. Stop dev with
`fuser -k 8805/tcp` (never pkill -f, it kills the tool shell).

## Open decisions
- Commit the 30 preview mp4s (~27 MB) into the repo, or host them on demos.onpointfl.com and fetch
  on first run? Leaning: commit the .jpg posters (small) and host the mp4s.
- Ref2VA support in Cinema (character stills as references) needs the six-section Ref2VA prompt
  format; v1 is t2v/i2v only.
