# Backlot

**The open-source Higgsfield that runs on a 16 GB card.**

A local, open-weights video and image studio on your own ComfyUI, driven through the
**Comfy Developer Platform API** (Comfy SDK). One window, no node graphs, the GPU you already own.

Built for the Comfy [Open Source Higgsfield](https://www.comfy.org/open-source-higgsfield) challenge.

## What it does

**Video (MiniMax H3, 24 fps, with sound)**
- Text to video, image to video (the first frame is center-cropped, never stretched).
- Reference mode: up to 9 reference images and 3 reference audio clips keep a person, product or
  place consistent across new shots (Ref2VA).
- 360 Orbit: one photo (or three) becomes a full turntable spin (ORB360 LoRA).
- Latent chaining: continue a clip from its last latent, so a scene can run as long as you like
  without a visible join (Motion Context).
- Camera Motion LoRA, SeedVR2 or RealESRGAN upscaling, RIFE interpolation, 2x slow motion.

**Stills (Krea 2 Turbo, Qwen Image 2.1, Qwen Image Edit 2511)**
- Text to image, image to image, style reference, identity edit (same face and outfit in a new
  scene), instruction edit, and camera angles (re-shoot a photo from 96 camera positions).
- Transparent PNG output, vertical crops for phones, SeedVR2 4K upscale.

Every render is sent as a job through the Comfy SDK. Locally the SDK talks to
[`comfy-api-proxy`](https://pypi.org/project/comfy-api-proxy/), which serves the Comfy API in front
of a plain ComfyUI. Point `BACKLOT_COMFY_URL` at Comfy Cloud and nothing else changes.

All models are open weights. No partner nodes, no closed APIs. Full list with download links in
[docs/models.md](docs/models.md).

## Requirements

- Linux (tested on Ubuntu 24.04) or macOS, Python 3.10+.
- A ComfyUI install (0.35 or newer) with these node packs in `custom_nodes/`:
  - [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
  - [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
  - [jlucasmcrell/ComfyUI-H3-Multishot](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot)
  - [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
  - [Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)
  - [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
  - [lbouaraba/comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) (identity edit only)

  `setup/install-nodes.sh /path/to/ComfyUI` clones all of them.
- An NVIDIA GPU with 16 GB VRAM. Everything here was measured on an RTX 4070 Ti SUPER (16 GB).
  More VRAM only makes it faster.
- About 100 GB of disk for the required models, 170 GB for all of them.
- `ffmpeg` and `ffprobe` on PATH (slow motion and upscale need them).

## Setup

```bash
git clone https://github.com/<you>/backlot
cd backlot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# node packs + models into your ComfyUI
setup/install-nodes.sh ~/ComfyUI
setup/download-models.sh ~/ComfyUI            # required set (~100 GB)
setup/download-models.sh ~/ComfyUI --all      # every optional mode too
```

Start ComfyUI with these flags (they are what keeps H3 inside 16 GB):

```bash
python main.py --listen 127.0.0.1 --port 8188 --disable-pinned-memory --fast-disk --reserve-vram 0.4 --use-ck-attention
```

Then:

```bash
./start.sh          # starts the Comfy API proxy on :8199 and the studio on :8801, opens it
./stop.sh           # stops both (ComfyUI keeps running)
```

If your ComfyUI is somewhere else: `COMFYUI_URL=http://127.0.0.1:8189 ./start.sh`.

## Configuration

Everything is an environment variable; `start.sh` sets the first two for you.

| Variable | Default | Meaning |
|---|---|---|
| `BACKLOT_COMFY_URL` | `http://127.0.0.1:8199` | Comfy API endpoint (the local proxy, or Comfy Cloud). |
| `BACKLOT_COMFYUI_URL` | `http://127.0.0.1:8188` | The plain ComfyUI behind it. Used only by the Free VRAM button. |
| `BACKLOT_QWEN_URL` | same as `BACKLOT_COMFY_URL` | Optional second endpoint for Qwen Image 2.1 if you run it on another ComfyUI. |
| `BACKLOT_OUTPUT_DIR` | `./output` | Where videos and images are filed (`videos/`, `images/`). |
| `BACKLOT_COMFY_OUTPUT_DIR` | unset | ComfyUI's output folder. Lets latent chaining list the clips it has saved. |
| `PORT` | `8801` | App port. |

## How it is built

```
app/server.py           the HTTP app: one route per feature, builds a graph, submits it
app/comfy_backend.py    the Comfy SDK adapter: assets in, jobs out, nothing else touches ComfyUI
app/workflows/          one graph builder per model family (H3, Ref2VA, Krea 2, SeedVR2, post-fx)
app/index.html          the UI, plain HTML and JavaScript, no build step
app/prompts/            the 360 Orbit prompts
docs/models.md          every model file, where it comes from, where it goes
setup/                  node-pack and model download scripts
```

Inputs are uploaded with `client.assets.from_bytes()` and dropped into `LoadImage` /
`LoadVideo` / `LoadAudio` nodes as asset references. Graphs are submitted with
`client.submit()`, waited on with `job.wait()`, and the Save node's output is read back with
`output.to_bytes()`. The app never writes into ComfyUI's input folder and never reads its output
folder, which is what makes it run unchanged against the cloud.

## Measured timings (RTX 4070 Ti SUPER, 16 GB)

| Job | Time |
|---|---|
| Krea 2 Turbo still, 1152x768, 12 steps | about 55 s cold (model load), 15 s warm |
| SeedVR2 2x upscale of that still | about 16 s |
| H3 text to video, 5 s at 1344x768, 8 steps | about 7.5 min |
| H3 reference video, 2 refs, 5 s | about 12 min |
| 360 Orbit, 1 photo, 5.2 s | about 8 min |

H3 is slow on 16 GB because the 13.6 GB text encoder has to run on the CPU. That is the trade
the whole project is built around: it works on the card most people actually have.

## Recipes that took real renders to find

- H3's Qwen3-VL text encoder must stay on the CPU (`clip_cpu`) or dequantization OOMs on a 16 GB card.
- Krea 2's `EmptyFlux2LatentImage` is a //16 latent but the VAE decodes at 8x: ask the latent for
  2x the size, and hand `Flux2Scheduler` the real size or skin goes waxy.
- The Camera Motion LoRA's trigger words must lead the prompt.
- Latent chaining must stay sequential; chaining through a PNG loses the join (about 15% pixel drift).
- SeedVR2 keeps pores, lashes and lettering where RealESRGAN smooths them to plastic.

## License

MIT. Model weights carry their own licenses; see each source in [docs/models.md](docs/models.md).
