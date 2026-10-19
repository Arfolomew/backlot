"""One ComfyUI backend, driven through the Comfy SDK.

Everything the app renders goes through the Comfy Developer Platform API (``comfy-sdk``):
inputs are uploaded as assets, graphs are submitted as jobs, results come back as job
outputs. Locally that API is served by ``comfy-api-proxy`` sitting in front of a plain
ComfyUI, so the same code runs unchanged against Comfy Cloud.

The app never writes into ComfyUI's input folder and never reads its output folder.
"""
import json
import os
import subprocess
import threading
import time
import urllib.request

from comfy_sdk import Comfy, ComfyError, RetryPolicy

SUCCESS_STATES = ("success", "completed", "succeeded")


class Backend:
    """A Comfy API endpoint (the proxy or the cloud) plus, optionally, the direct ComfyUI
    URL behind it, used only for the VRAM "free" convenience that API v2 does not cover."""

    def __init__(self, api_url, direct_url=None, label="ComfyUI"):
        self.api_url = api_url.rstrip("/")
        self.direct_url = (direct_url or "").rstrip("/") or None
        self.label = label
        self._client = None
        self._lock = threading.Lock()
        self._uploads = {}   # filename -> Asset handle, until the job that uses it is sent

    # ---- client ----
    def client(self):
        with self._lock:
            if self._client is None:
                # The SDK reads COMFY_BASE_URL at construction, so set it per backend.
                os.environ["COMFY_BASE_URL"] = self.api_url
                self._client = Comfy(
                    timeout=120.0,
                    client_info="backlot",
                    retry=RetryPolicy(collect_max_elapsed=4 * 3600.0))
            return self._client

    def alive(self):
        """The Comfy API answers, and (when we know it) so does the ComfyUI behind it. The
        proxy's health route does not probe upstream, so a dead ComfyUI would otherwise look fine."""
        try:
            with urllib.request.urlopen(f"{self.api_url}/api/v2/health", timeout=3) as r:
                if r.status != 200:
                    return False
        except Exception:
            return False
        if self.direct_url:
            try:
                with urllib.request.urlopen(f"{self.direct_url}/system_stats", timeout=3) as r:
                    return r.status == 200
            except Exception:
                return False
        return True

    # ---- inputs ----
    def upload(self, raw, filename):
        """Stage bytes as an input asset. Returns the filename to put in a LoadImage /
        LoadVideo / LoadAudio node; it is swapped for the asset reference at submit."""
        safe = os.path.basename(filename).replace("/", "_") or "upload.bin"
        asset = self.client().assets.from_bytes(raw, filename=safe)
        self._uploads[safe] = asset
        return safe

    def discard(self, filename):
        """Forget (and delete) a staged input once its job is done."""
        asset = self._uploads.pop(filename, None)
        if asset is not None:
            try:
                asset.delete()
            except Exception:
                pass

    def _bind_uploads(self, graph):
        """Replace every input whose value is a staged filename with its asset handle."""
        if not self._uploads:
            return graph
        for node in graph.values():
            inputs = node.get("inputs") if isinstance(node, dict) else None
            if not isinstance(inputs, dict):
                continue
            for k, v in list(inputs.items()):
                if isinstance(v, str) and v in self._uploads:
                    inputs[k] = self._uploads[v]
        return graph

    # ---- jobs ----
    def run(self, graph, timeout=1800, exts=(".mp4", ".webm", ".gif")):
        """Submit an API-format graph, wait for it, return (bytes, ext) of the first output
        whose name ends with one of ``exts``."""
        client = self.client()
        wf = client.workflows.from_json(self._bind_uploads(graph))
        try:
            job = client.submit(wf)
        except ComfyError as e:
            raise RuntimeError(f"{self.label} rejected the workflow: {_describe(e)}") from None
        try:
            job.wait(timeout=timeout)
        except TimeoutError:
            try:
                job.cancel()
            except Exception:
                pass
            raise RuntimeError("generation timed out") from None
        if str(job.status).lower() not in SUCCESS_STATES:
            raise RuntimeError(f"generation failed: {job.status} {_describe(job.error)}".strip())
        # Prefer what the Save* node wrote: a LoadVideo node echoes our own input as an
        # output too, and that must never be mistaken for the result.
        savers = {nid for nid, node in graph.items()
                  if isinstance(node, dict) and str(node.get("class_type", "")).startswith(("Save", "VHS_VideoCombine"))}
        want = tuple(e.lower() for e in exts)
        ranked = sorted(job.outputs, key=lambda o: 0 if o.node_id in savers else 1)
        for out in ranked:
            name = (out.name or "").lower()
            if name.endswith(want):
                return out.to_bytes(), name.rsplit(".", 1)[-1]
        raise RuntimeError("finished but produced no output file")

    # ---- VRAM ----
    def free(self):
        """Ask ComfyUI to drop its resident models. Needs the direct URL; harmless without it.
        Returns MiB freed, or None if nothing could be measured."""
        if not self.direct_url:
            return None
        before = gpu_used_mib()
        data = json.dumps({"unload_models": True, "free_memory": True}).encode()
        req = urllib.request.Request(f"{self.direct_url}/free", data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30):
                pass
        except Exception:
            return None
        time.sleep(2.0)
        after = gpu_used_mib()
        if before is None or after is None:
            return 0
        return max(0, before - after)


def _describe(err):
    if err is None:
        return ""
    for attr in ("message", "detail", "details"):
        v = getattr(err, attr, None)
        if v:
            return str(v)[:400]
    return str(err)[:400]


def gpu_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return None
