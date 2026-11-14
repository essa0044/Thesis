#!/usr/bin/env python3
"""Load the VLM in Ollama the way the missions use it, report how much of it is on the GPU, and time a few calls.

    python scripts/vlm_warmup.py                       # before Gazebo starts (sim/run_series.sh does this)
    python scripts/vlm_warmup.py --max-median 30       # exit code 1 if the median call is slower than 30 s

Ollama decides at load time how many layers go to the GPU, from the GPU memory free at that moment. Loaded while
Gazebo is running on a 4 GB laptop GPU, Qwen3-VL-4B ended up 54 % on the CPU and every call took more than 60 s
(12 Nov). So: unload, load again with the same num_ctx and keep_alive the missions use (a different num_ctx makes Ollama
reload), then time calls on a 1920 x 1080 test image (a red box on grey, sent at --max-side like the missions).
"""
import _path  # noqa: F401
import argparse
import json
import statistics
import sys
import time
import urllib.request

import numpy as np

from neurodrone.vlm import encode_jpeg
from sim_mission import ask_ollama


def post(url, body, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def gpu_share(base, model):
    with urllib.request.urlopen(base + "/api/ps", timeout=10) as r:
        for m in json.loads(r.read()).get("models", []):
            if m.get("name") == model or m.get("model") == model:
                size, vram = m.get("size") or 0, m.get("size_vram") or 0
                return (vram / size if size else None), size
    return None, None


def test_image():
    img = np.full((1080, 1920, 3), 200, np.uint8)
    img[600:1080] = 170  # floor
    img[520:640, 1100:1220] = (180, 30, 30)  # red box
    img[540:660, 600:680] = (30, 60, 180)  # blue barrel
    return img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--max-side", type=int, default=896)
    ap.add_argument("--num-gpu", type=int, default=None, help="model layers on the GPU (0 = CPU only)")
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--no-reload", action="store_true", help="time the model as it is loaded now")
    ap.add_argument("--max-median", type=float, default=None, help="exit 1 if the median call takes longer (s)")
    a = ap.parse_args(argv)
    base = a.ollama.rstrip("/")
    if not a.no_reload:
        post(base + "/api/generate", {"model": a.model, "keep_alive": 0})
        time.sleep(2)
        t0 = time.monotonic()
        opts = {"num_ctx": a.num_ctx}
        if a.num_gpu is not None:
            opts["num_gpu"] = a.num_gpu
        post(base + "/api/generate", {"model": a.model, "keep_alive": "60m", "prompt": "", "options": opts})
        print(f"model loaded in {time.monotonic() - t0:.1f} s", flush=True)
    share, size = gpu_share(base, a.model)
    if share is not None:
        print(f"on the GPU: {100 * share:.0f} % of {size / 1e9:.1f} GB (rest on the CPU)", flush=True)
    jpeg, _ = encode_jpeg(test_image(), max_side=a.max_side, quality=90)
    lat = []
    for i in range(a.calls):
        text, t = ask_ollama(a.ollama, a.model, jpeg, "the red box", num_ctx=a.num_ctx, num_gpu=a.num_gpu)
        lat.append(t)
        print(f"call {i + 1}: {t:.1f} s  {text.strip()[:80]}", flush=True)
    med = statistics.median(lat)
    print(f"median {med:.1f} s", flush=True)
    print(json.dumps(dict(gpu_share=share, size_bytes=size, latencies_s=[round(x, 2) for x in lat], num_ctx=a.num_ctx,
                          num_gpu=a.num_gpu,
                          max_side=a.max_side, model=a.model)))
    if a.max_median is not None and med > a.max_median:
        print(f"TOO SLOW: median {med:.1f} s > {a.max_median} s", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
