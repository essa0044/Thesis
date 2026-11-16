#!/usr/bin/env python3
"""VLM grounding benchmark on PUBLIC datasets, scored with the datasets' own labels.

    python scripts/bench_grounding.py --data ..\\data --out ..\\results\\vlm\\full \
        --models qwen3-vl:2b qwen2.5vl:3b --tasks refcoco:testA refcoco:testB refcoco:val grefcoco:val refdrone:test

Datasets (see the workstation day plan for sources and licences):
  refcoco   lmms-eval copy of RefCOCO (UNC splits): one box per expression        -> accuracy at IoU >= 0.5
  grefcoco  gRefCOCO (CVPR 2023): single-target AND no-target expressions          -> also "not found" accuracy
  refdrone  RefDrone (drone photos, VisDrone2019): single-target and no-target     -> accuracy by target size

Protocol (fixed so anyone can rerun it):
  * samples are drawn with a fixed seed and the exact list is saved next to the results
  * the first `--calib` samples of every task are a CALIBRATION set, left out of all scores: the
    coordinate convention of each model (0-1000, pixels of the image sent, or 0-1) is chosen there,
    by best mean IoU, and then frozen for the scored samples
  * every raw answer is saved (JSONL), so a run can be re-scored or audited later; a run that is
    stopped resumes where it stopped
  * prompt "drone" = exactly the prompt the drone uses (neurodrone.vlm.build_prompt);
    prompt "native" = the model family's own grounding prompt, to check our prompt doesn't handicap it
"""
import _path  # noqa: F401
import argparse
import base64
import io
import json
import math
import os
import random
import sys
import time
import urllib.request
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from neurodrone.vlm import SYSTEM_PROMPT, build_prompt, encode_jpeg, parse_goal

# How a model's numbers map to the image. "pixels_q25" / "pixels_q3": pixels of the image as the server
# resized it internally (Qwen smart_resize with patch grid 28 / 32 and at least 1024 image tokens, which
# Ollama >= 0.30 applies; Qwen2.5-VL answers in those pixels). Chosen per model on the calibration samples.
COORD_CANDIDATES = ("norm1000", "pixels", "unit", "pixels_q25", "pixels_q3")
RESIZE = {"pixels_q25": (28, 1024 * 28 * 28, 12845056), "pixels_q3": (32, 1024 * 32 * 32, 16777216)}


def smart_resize(w: int, h: int, factor: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    """Qwen2.5-VL / Qwen3-VL image resize (transformers' smart_resize); returns (width, height)."""
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return wb, hb


# ============================================================================ samples
# A sample: dict(uid, task, image_key, expr, boxes (list of [x1,y1,x2,y2] normalised 0-1; empty = no target),
#                img_w, img_h, area_frac)
def _rng(seed: int, task: str) -> random.Random:
    return random.Random(f"{seed}:{task}")


def load_refcoco(root: str, split: str, n: int, seed: int, all_expr: bool = False) -> Tuple[List[dict], callable]:
    """n objects, one of their expressions each (drawn at random); all_expr: every expression of every
    object, shuffled (the way published RefCOCO scores are counted)."""
    import pyarrow.parquet as pq
    d = os.path.join(root, "refcoco")
    files = sorted(f for f in os.listdir(d) if f.startswith(split + "-") and f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet in {d}")
    meta, images = [], {}
    for f in files:
        t = pq.read_table(os.path.join(d, f), columns=["question_id", "image", "answer", "bbox", "file_name"])
        for r in t.to_pylist():
            img = r["image"]
            b = img["bytes"] if isinstance(img, dict) else img
            images.setdefault(r["file_name"], b)
            meta.append(r)
    rng = _rng(seed, f"refcoco:{split}")
    out = []
    if all_expr:
        from PIL import Image
        size = {}
        for i, r in enumerate(meta):
            if r["file_name"] not in size:
                size[r["file_name"]] = Image.open(io.BytesIO(images[r["file_name"]])).size
            W, H = size[r["file_name"]]
            x, y, w, h = r["bbox"]
            for j, expr in enumerate(a for a in (r["answer"] or []) if a):
                out.append(dict(uid=f"refcoco:{split}:{r['question_id']}:e{j}", task=f"refcoco:{split}",
                                image_key=r["file_name"], expr=expr, boxes=[[x / W, y / H, (x + w) / W, (y + h) / H]],
                                img_w=W, img_h=H, area_frac=(w * h) / (W * H)))
        rng.shuffle(out)
        return out[:n], lambda k: images[k]
    picks = rng.sample(range(len(meta)), min(n, len(meta)))
    for i in picks:
        r = meta[i]
        from PIL import Image
        W, H = Image.open(io.BytesIO(images[r["file_name"]])).size
        x, y, w, h = r["bbox"]
        answers = [a for a in (r["answer"] or []) if a]
        expr = answers[rng.randrange(len(answers))]
        out.append(dict(uid=f"refcoco:{split}:{r['question_id']}:{expr[:40]}", task=f"refcoco:{split}",
                        image_key=r["file_name"], expr=expr, boxes=[[x / W, y / H, (x + w) / W, (y + h) / H]],
                        img_w=W, img_h=H, area_frac=(w * h) / (W * H)))
    return out, lambda k: images[k]


def load_grefcoco(root: str, split: str, n: int, seed: int) -> Tuple[List[dict], callable]:
    """Half no-target, half with target(s). gRefCOCO's val/test splits hold only no-target and
    MULTI-target expressions (single-target ones are RefCOCO's, tested above); for a multi-target
    expression a box counts as correct if it matches any one of the objects (IoU >= 0.5) - the drone
    needs one valid target."""
    d = os.path.join(root, "grefcoco")
    refs = json.load(open(os.path.join(d, "grefs(unc).json")))
    inst = json.load(open(os.path.join(d, "instances.json")))
    anns = {a["id"]: a for a in inst["annotations"]}
    imgs = {i["id"]: i for i in inst["images"]}
    refs = [r for r in refs if r["split"] == split]
    target = [r for r in refs if not r["no_target"] and r["ann_id"] and all(a in anns for a in r["ann_id"])]
    empty = [r for r in refs if r["no_target"]]
    rng = _rng(seed, f"grefcoco:{split}")
    chosen = rng.sample(target, min(n // 2, len(target))) + rng.sample(empty, min(n - n // 2, len(empty)))
    rng.shuffle(chosen)
    out = []
    for r in chosen:
        im = imgs[r["image_id"]]
        W, H = im["width"], im["height"]
        s = r["sentences"][rng.randrange(len(r["sentences"]))]["sent"]
        if r["no_target"]:
            boxes, area = [], 0.0
        else:
            bb = [anns[a]["bbox"] for a in r["ann_id"]]
            boxes = [[x / W, y / H, (x + w) / W, (y + h) / H] for x, y, w, h in bb]
            area = max(w * h for _, _, w, h in bb) / (W * H)
        out.append(dict(uid=f"grefcoco:{split}:{r['ref_id']}", task=f"grefcoco:{split}", image_key=r["file_name"],
                        expr=s, boxes=boxes, img_w=W, img_h=H, area_frac=area, n_targets=len(boxes)))
    cache = os.path.join(root, "coco_train2014_cache")
    rc_dir = os.path.join(root, "refcoco")
    rc_images: Dict[str, bytes] = {}

    def get(key: str) -> bytes:
        p = os.path.join(cache, key)
        if os.path.exists(p):
            return open(p, "rb").read()
        if not rc_images and os.path.isdir(rc_dir):  # RefCOCO's own copies of the same COCO images
            import pyarrow.parquet as pq
            for f in sorted(os.listdir(rc_dir)):
                if f.endswith(".parquet"):
                    for row in pq.read_table(os.path.join(rc_dir, f), columns=["image", "file_name"]).to_pylist():
                        img = row["image"]
                        rc_images.setdefault(row["file_name"], img["bytes"] if isinstance(img, dict) else img)
        b = rc_images.get(key)
        if b is None:  # last resort: the COCO image server
            with urllib.request.urlopen("http://images.cocodataset.org/train2014/" + key, timeout=60) as r:
                b = r.read()
        os.makedirs(cache, exist_ok=True)
        with open(p, "wb") as f:
            f.write(b)
        return b

    return out, get


def load_refdrone(root: str, split: str, n: int, seed: int, n_empty: Optional[int] = None) -> Tuple[List[dict], callable]:
    d = os.path.join(root, "refdrone")
    js = json.load(open(os.path.join(d, f"RefDrone_{split}_mdetr.json")))
    by_img: Dict[int, list] = {}
    for a in js.get("annotations", js.get("annotaions", [])):
        by_img.setdefault(a["image_id"], []).append(a)
    single, empty = [], []
    for im in js["images"]:
        a = [x for x in by_img.get(im["id"], []) if not x.get("empty", 0) and x.get("bbox") and x["bbox"][2] > 0]
        if len(a) == 1:
            single.append((im, a[0]))
        elif len(a) == 0:
            empty.append((im, None))
    rng = _rng(seed, f"refdrone:{split}")
    n_empty = n // 3 if n_empty is None else n_empty
    chosen = rng.sample(single, min(n, len(single))) + rng.sample(empty, min(n_empty, len(empty)))
    rng.shuffle(chosen)
    out = []
    for im, a in chosen:
        W, H = im["width"], im["height"]
        if a is None:
            boxes, area = [], 0.0
        else:
            x, y, w, h = a["bbox"]
            boxes, area = [[x / W, y / H, (x + w) / W, (y + h) / H]], (w * h) / (W * H)
        out.append(dict(uid=f"refdrone:{split}:{im['id']}", task=f"refdrone:{split}", image_key=str(im["id"]),
                        expr=im["caption"], boxes=boxes, img_w=W, img_h=H, area_frac=area))
    need = {s["image_key"] for s in out}
    tsv = os.path.join(d, f"refdrone_{split}_base64.tsv")
    images: Dict[str, bytes] = {}
    with open(tsv, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        ii, im_col = header.index("index"), header.index("image")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[ii] in need:
                images[parts[ii]] = base64.b64decode(parts[im_col])
    missing = need - set(images)
    if missing:
        raise RuntimeError(f"{len(missing)} RefDrone images not found in {tsv}")
    return out, lambda k: images[k]


LOADERS = {"refcoco": load_refcoco, "grefcoco": load_grefcoco, "refdrone": load_refdrone}


# ============================================================================ model client
NATIVE = {
    "qwen": "Locate \"{e}\" in the image and output its bounding box as JSON: "
            "[{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"{e}\"}}]. If it is not in the image, output []."
}


class Client:
    def __init__(self, base_url: str, model: str, max_tokens: int = 256, timeout: float = 300.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.max_tokens, self.timeout = model, max_tokens, timeout

    def ask(self, jpeg: bytes, expr: str, prompt: str) -> dict:
        if prompt == "drone":
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
                        {"type": "text", "text": build_prompt(expr)}]}]
        else:
            msgs = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
                {"type": "text", "text": NATIVE["qwen"].format(e=expr.replace('"', "'"))}]}]
        body = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens, "messages": msgs}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        lat = time.monotonic() - t0
        u = data.get("usage") or {}
        return dict(text=data["choices"][0]["message"].get("content") or "", latency_s=lat,
                    prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"))


class OllamaClient:
    """Ollama's native /api/chat (--api ollama): same system + user prompt as Client, plus a fixed num_ctx (the
    missions' 2048, so a benchmark does not make Ollama reload the model with another context) and Ollama's own
    timing split - prefill (reading the prompt, including the image) and decode (writing the answer). Added for the
    Jetson (stage 1); the workstation numbers of Chapter 4 used Client (OpenAI-compatible endpoint)."""

    def __init__(self, base_url: str, model: str, max_tokens: int = 256, timeout: float = 300.0, num_ctx: int = 2048):
        self.base = base_url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.model, self.max_tokens, self.timeout, self.num_ctx = model, max_tokens, timeout, num_ctx

    def ask(self, jpeg: bytes, expr: str, prompt: str) -> dict:
        from neurodrone.ollama import timing_from_response
        img = base64.b64encode(jpeg).decode()
        if prompt == "drone":
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(expr), "images": [img]}]
        else:
            msgs = [{"role": "user", "content": NATIVE["qwen"].format(e=expr.replace('"', "'")), "images": [img]}]
        body = {"model": self.model, "stream": False, "think": False, "keep_alive": "60m", "messages": msgs,
                "options": {"temperature": 0, "num_predict": self.max_tokens, "num_ctx": self.num_ctx}}
        req = urllib.request.Request(self.base + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        lat = time.monotonic() - t0
        t = timing_from_response(data)
        return dict(text=(data.get("message") or {}).get("content") or "", latency_s=lat,
                    prompt_tokens=t["prefill_tokens"], completion_tokens=t["decode_tokens"],
                    prefill_s=t["prefill_s"], decode_s=t["decode_s"], load_s=t["load_s"])


# ============================================================================ scoring
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def score(text: str, sent_size: Tuple[int, int], coords: str, gt_boxes: List[list]) -> dict:
    if coords in RESIZE:
        sent_size, coords = smart_resize(sent_size[0], sent_size[1], *RESIZE[coords]), "pixels"
    p = parse_goal(text, sent_size, coords)
    if p is None:
        return dict(valid=False, found=None, iou=0.0, centre_in=False, x_err=None, box=None)
    if not p.found:
        return dict(valid=True, found=False, iou=0.0, centre_in=False, x_err=None, box=None)
    if p.w is None:  # a point, not a box
        box = None
        best = 0.0
    else:
        box = [p.x - p.w / 2, p.y - p.h / 2, p.x + p.w / 2, p.y + p.h / 2]
        best = max((iou(box, g) for g in gt_boxes), default=0.0)
    cin = any(g[0] <= p.x <= g[2] and g[1] <= p.y <= g[3] for g in gt_boxes)
    xe = min((abs(p.x - (g[0] + g[2]) / 2) for g in gt_boxes), default=None)
    return dict(valid=True, found=True, iou=best, centre_in=cin, x_err=xe, box=box, cx=p.x, cy=p.y)


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def summarise(rows: List[dict], coords: str) -> dict:
    pos = [r for r in rows if r["boxes"]]
    neg = [r for r in rows if not r["boxes"]]
    s = {"n": len(rows), "n_target": len(pos), "n_no_target": len(neg), "coords": coords}
    if rows:
        s["valid_rate"] = sum(r["sc"]["valid"] for r in rows) / len(rows)
        lat = np.array([r["latency_s"] for r in rows])
        s["latency_s_median"] = float(np.median(lat))
        s["latency_s_p90"] = float(np.percentile(lat, 90))
        pt = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
        s["prompt_tokens_median"] = float(np.median(pt)) if pt else None
        for k in ("prefill_s", "decode_s"):  # only with --api ollama
            v = [r[k] for r in rows if r.get(k) is not None]
            if v:
                s[k + "_median"] = float(np.median(v))
    if pos:
        k = sum(r["sc"]["iou"] >= 0.5 for r in pos)
        s["acc_iou50"] = k / len(pos)
        s["acc_iou50_ci95"] = wilson(k, len(pos))
        k2 = sum(r["sc"]["centre_in"] for r in pos)
        s["centre_in_box"] = k2 / len(pos)
        s["centre_in_box_ci95"] = wilson(k2, len(pos))
        s["said_not_found_on_target"] = sum(r["sc"]["found"] is False for r in pos) / len(pos)
        xe = [r["sc"]["x_err"] for r in pos if r["sc"]["x_err"] is not None]
        s["x_err_median"] = float(np.median(xe)) if xe else None
        buckets = {"small (<1% of image)": (0, 0.01), "medium (1-10%)": (0.01, 0.10), "large (>10%)": (0.10, 2)}
        s["by_size"] = {}
        for name, (lo, hi) in buckets.items():
            b = [r for r in pos if lo <= r["area_frac"] < hi]
            if b:
                kb = sum(r["sc"]["iou"] >= 0.5 for r in b)
                s["by_size"][name] = {"n": len(b), "acc_iou50": kb / len(b), "ci95": wilson(kb, len(b)),
                                      "centre_in_box": sum(r["sc"]["centre_in"] for r in b) / len(b)}
        multi = [r for r in pos if r.get("n_targets", 1) > 1]
        if multi:
            km = sum(r["sc"]["iou"] >= 0.5 for r in multi)
            s["multi_target"] = {"n": len(multi), "acc_iou50_any": km / len(multi), "ci95": wilson(km, len(multi))}
    if neg:
        k = sum(r["sc"]["found"] is False for r in neg)
        s["no_target_correct"] = k / len(neg)
        s["no_target_correct_ci95"] = wilson(k, len(neg))
        s["no_target_false_found"] = sum(r["sc"]["found"] is True for r in neg) / len(neg)
    return s


# ============================================================================ main
def run_task(client: Client, model: str, task: str, samples: List[dict], get_img, a, out_dir: str) -> dict:
    tag = f"{model.replace(':', '_').replace('/', '_')}__{task.replace(':', '_')}__{a.prompt}__{a.max_side}"
    path = os.path.join(out_dir, tag + ".jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
                if not r.get("error"):  # failed requests (server down, model missing) are asked again
                    done[r["uid"]] = r
            except json.JSONDecodeError:
                pass
    n_err, ok_any = 0, False
    with open(path, "a", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            if s["uid"] in done:
                continue
            from PIL import Image
            img = np.asarray(Image.open(io.BytesIO(get_img(s["image_key"]))).convert("RGB"))
            jpeg, sent = encode_jpeg(img, max_side=a.max_side, quality=90)
            try:
                res = client.ask(jpeg, s["expr"], a.prompt)
            except Exception as e:  # keep going; the error is recorded (and retried on the next run)
                body = ""
                if hasattr(e, "read"):
                    try:
                        body = " " + e.read().decode(errors="replace")[:300]
                    except Exception:  # noqa: BLE001
                        pass
                res = dict(text="", latency_s=float("nan"), prompt_tokens=None, completion_tokens=None, error=(repr(e) + body)[:600])
                n_err += 1
            else:
                ok_any = True
            r = dict(s, **res, sent_w=sent[0], sent_h=sent[1], calib=i < a.calib, model=model, prompt=a.prompt, max_side=a.max_side)
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[s["uid"]] = r
            if n_err >= 10 and not ok_any:  # the model is not answering at all: do not burn through the task
                print(f"  {model} {task}: GIVING UP, the first 10 requests failed, e.g. {res['error'][:300]}", flush=True)
                break
            if (i + 1) % 25 == 0:
                print(f"  {model} {task}: {i + 1}/{len(samples)}", flush=True)
    rows = [done[s["uid"]] for s in samples if s["uid"] in done]
    calib = [r for r in rows if r["calib"] and r["boxes"]]
    best, best_iou = "norm1000", -1.0
    for c in COORD_CANDIDATES:
        m = float(np.mean([score(r["text"], (r["sent_w"], r["sent_h"]), c, r["boxes"])["iou"] for r in calib])) if calib else 0.0
        if m > best_iou:
            best, best_iou = c, m
    scored = [dict(r, sc=score(r["text"], (r["sent_w"], r["sent_h"]), best, r["boxes"])) for r in rows if not r["calib"]]
    summ = summarise(scored, best)
    summ.update(model=model, task=task, prompt=a.prompt, max_side=a.max_side, calib_mean_iou=best_iou,
                errors=sum(1 for r in rows if r.get("error")))
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=["refcoco:testA", "refcoco:testB", "refcoco:val", "grefcoco:val", "refdrone:test"])
    ap.add_argument("--n", type=int, default=500, help="scored samples per task (calibration samples come on top)")
    ap.add_argument("--calib", type=int, default=50)
    ap.add_argument("--full", action="store_true",
                    help="every sample of each task instead of --n (RefCOCO: every expression); the first --calib still only calibrate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1024, help="longest image side sent to the VLM")
    ap.add_argument("--prompt", choices=["drone", "native"], default="drone")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--server-note", default="", help="free text stored with the results, e.g. 'ollama 0.40.13, Vulkan'")
    ap.add_argument("--api", choices=["openai", "ollama"], default="openai",
                    help="openai: /v1/chat/completions (the Chapter 4 runs); ollama: native /api/chat with --num-ctx and "
                         "the prefill/decode split (the Jetson runs)")
    ap.add_argument("--num-ctx", type=int, default=2048, help="context length with --api ollama (the missions' value)")
    a = ap.parse_args(argv)
    server = a.server_note
    try:  # Ollama reports its version here; other servers simply do not
        with urllib.request.urlopen(a.base_url.rstrip("/").rsplit("/v1", 1)[0] + "/api/version", timeout=5) as r:
            server = (json.loads(r.read()).get("version", "") + " " + server).strip()
    except Exception:  # noqa: BLE001
        pass
    os.makedirs(a.out, exist_ok=True)
    summaries = []
    sumfile = os.path.join(a.out, "summary.json")
    if os.path.exists(sumfile):
        summaries = json.load(open(sumfile))
    for task in a.tasks:
        ds, split = task.split(":")
        t0 = time.time()
        n = 10 ** 9 if a.full else a.n + a.calib
        samples, get_img = LOADERS[ds](a.data, split, n, a.seed, **({"all_expr": True} if (a.full and ds == "refcoco") else {}))
        with open(os.path.join(a.out, f"samples__{task.replace(':', '_')}__seed{a.seed}.json"), "w") as f:
            json.dump([{k: s[k] for k in ("uid", "image_key", "expr", "boxes")} for s in samples], f)
        print(f"{task}: {len(samples)} samples loaded in {time.time() - t0:.0f}s", flush=True)
        for model in a.models:
            client = (OllamaClient(a.base_url, model, a.max_tokens, num_ctx=a.num_ctx) if a.api == "ollama"
                      else Client(a.base_url, model, a.max_tokens))
            s = run_task(client, model, task, samples, get_img, a, a.out)
            s["server"] = server
            summaries = [x for x in summaries if not (x["model"] == model and x["task"] == task and x["prompt"] == a.prompt and x["max_side"] == a.max_side)]
            summaries.append(s)
            with open(sumfile, "w") as f:
                json.dump(summaries, f, indent=1)
            acc = s.get("acc_iou50")
            print(f"== {model:22s} {task:14s} acc@0.5={acc if acc is None else round(acc, 3)} "
                  f"no-target-ok={s.get('no_target_correct')} valid={s.get('valid_rate', 0):.2f} "
                  f"lat={s.get('latency_s_median', float('nan')):.2f}s coords={s['coords']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
