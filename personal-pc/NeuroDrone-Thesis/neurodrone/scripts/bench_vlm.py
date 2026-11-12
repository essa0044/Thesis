#!/usr/bin/env python3
"""Benchmark a VLM server for the thesis: latency, tokens, parse rate, localisation accuracy and
(on a Jetson) power / energy per query. Run once per model x quantisation x image size and compare.

    # serve a model first (any OpenAI-compatible server), e.g. llama.cpp:
    #   llama-server -m qwen2.5-vl-3b-q4.gguf --mmproj mmproj.gguf --port 8080 -ngl 99
    python scripts/bench_vlm.py --base-url http://127.0.0.1:8080/v1 --model qwen2.5-vl-3b \
        --images ./bench_images --instruction "the red door" -n 30 --power --out qwen3b_q4.json

Optional ground truth, so accuracy is measured and not just speed (x = box centre, 0..1):
    labels.json:  {"door1.jpg": {"found": true, "x": 0.62}, "empty_hall.jpg": {"found": false}}

Power: measured over the whole module. An idle baseline is recorded first (--idle-s) and energy
per query uses the *increment* over idle. It is a rough figure (the sampler's resolution is ~0.5 s,
and the model server itself idles above zero), so report it as such.
"""
import _path  # noqa: F401
import argparse
import glob
import json
import os
import time

import numpy as np
from PIL import Image

from neurodrone.telemetry import PowerSampler
from neurodrone.vlm import OpenAICompatVLM, encode_jpeg, parse_goal


def pct(a):
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--images", required=True, help="directory of .jpg/.png test frames")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("-n", type=int, default=20, help="measured queries (images are cycled)")
    ap.add_argument("--warmup", type=int, default=2, help="unmeasured queries (model load / cache warm-up)")
    ap.add_argument("--labels")
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--hfov", type=float, default=69.0, help="to express x error in degrees")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--coords", default="auto", choices=["auto", "norm1000", "pixels", "unit"],
                    help="the model's box convention (Qwen VLMs in Ollama answer in norm1000)")
    ap.add_argument("--max-tokens", type=int, default=80,
                    help="answer length cap; raise it for models that think before answering")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--idle-s", type=float, default=5.0)
    ap.add_argument("--out")
    a = ap.parse_args()

    paths = sorted(p for ext in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(a.images, f"*.{ext}")))
    if not paths:
        raise SystemExit(f"no images found in {a.images}")
    labels = json.load(open(a.labels)) if a.labels else {}
    client = OpenAICompatVLM(a.base_url, a.model, a.api_key, timeout_s=a.timeout, max_tokens=a.max_tokens)
    frames = {p: np.asarray(Image.open(p).convert("RGB")) for p in paths}

    power = PowerSampler(interval_s=0.5) if a.power else None
    idle_w = None
    if power:
        power.start()
        time.sleep(a.idle_s)
        idle_w = float(np.mean(power.samples)) if power.samples else None
    n_idle = len(power.samples) if power else 0

    lat, enc_ms, ptok, ctok, parsed_ok, found_ok, xerr, errors = [], [], [], [], 0, [], [], 0
    samples = []  # raw answers, so a failed run can be diagnosed afterwards
    for i in range(a.warmup + a.n):
        p = paths[i % len(paths)]
        t0 = time.perf_counter()
        jpeg, size = encode_jpeg(frames[p], a.max_side)
        e_ms = (time.perf_counter() - t0) * 1e3
        try:
            res = client.query(jpeg, a.instruction)
        except Exception as e:
            if i >= a.warmup:
                errors += 1
            print(f"  query {i} failed: {e!r}")
            continue
        if i < a.warmup:
            continue
        lat.append(res.latency_s)
        enc_ms.append(e_ms)
        if res.prompt_tokens is not None:
            ptok.append(res.prompt_tokens)
        if res.completion_tokens is not None:
            ctok.append(res.completion_tokens)
        g = parse_goal(res.text, size, a.coords)
        parsed_ok += g is not None
        samples.append({"image": os.path.basename(p), "latency_s": round(res.latency_s, 3),
                        "text": res.text.strip()[:300], "parsed": g is not None})
        lab = labels.get(os.path.basename(p))
        if lab is not None and g is not None:
            found_ok.append(g.found == lab["found"])
            if g.found and lab["found"] and g.x is not None and "x" in lab:
                xerr.append(abs(g.x - lab["x"]))
        print(f"  [{i - a.warmup + 1:3d}/{a.n}] {os.path.basename(p):24s} {res.latency_s:6.2f}s  ->  {res.text.strip()[:70]!r}")

    n = len(lat)
    out = {"model": a.model, "base_url": a.base_url, "image_max_side": a.max_side, "instruction": a.instruction,
           "max_tokens": a.max_tokens, "coords": a.coords, "queries": n, "errors": errors}
    if n:
        out["latency_s"] = pct(lat)
        out["jpeg_encode_ms_mean"] = float(np.mean(enc_ms))
        out["parse_success"] = parsed_ok / n
        if ptok:
            out["prompt_tokens_mean"] = float(np.mean(ptok))
        if ctok:
            out["completion_tokens_mean"] = float(np.mean(ctok))
        if found_ok:
            out["found_accuracy"] = float(np.mean(found_ok))
        if xerr:
            out["x_mae_norm"] = float(np.mean(xerr))
            out["x_mae_deg"] = float(np.mean(xerr) * a.hfov)
    if power:
        power.stop()
        active = power.samples[n_idle:]
        if active and idle_w is not None:
            mean_active = float(np.mean(active))
            out["power_w"] = {"idle": idle_w, "active": mean_active, "increment": mean_active - idle_w,
                              "source": power.source}
            if n:
                out["energy_per_query_j_above_idle"] = (mean_active - idle_w) * float(np.mean(lat))
        elif not power.samples:
            out["power_w"] = "no power source available (not a Jetson, or jtop/tegrastats missing)"
    print(json.dumps(out, indent=2))
    out["samples"] = samples
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
