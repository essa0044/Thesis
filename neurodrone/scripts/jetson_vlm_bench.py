#!/usr/bin/env python3
"""Stage 1 of the deployment plan: the VLM on the Jetson, no camera, no drone (thesis Chapter 6).

    python scripts/jetson_vlm_bench.py all --data ~/nd_data --out ~/nd_results/stage1     # everything, ~4-5 h
    python scripts/jetson_vlm_bench.py latency --calls 50 --out ...                        # one phase only
    python scripts/jetson_vlm_bench.py report --out ~/nd_results/stage1                    # rebuild the report

Phases (each writes into --out; `all` runs them in this order):
  info      board (L4T/JetPack, power mode, memory), Ollama version, model digest and quantisation
  cold      unload the model, load it with the missions' settings, time the first answer (cold start)
  latency   --calls answers (default 50), a NEW picture every call (Ollama reuses its work on a picture it has
            just seen); per call: total time, prefill (reading the prompt incl. the image) and decode (writing the
            answer) from Ollama's own timers, tokens
  soak      answers back to back for --minutes (default 30) while tegrastats logs memory, temperatures, GPU clock and
            power once a second: does the board throttle, how much memory is left
  accuracy  scripts/bench_grounding.py at 896 px through Ollama's native API (num_ctx 2048): RefDrone test (all
            1,304 queries) and gRefCOCO val (seed 0, 500 + 50 calibration) - the same samples as on the workstation
  report    report.md + summary.json with the stage-1 pass checks:
            median answer <= 10 s; accuracy within 2 points of the workstation; >= 1 GB memory free; no throttling

Before: JetPack 6.2, MAXN SUPER power mode (`sudo nvpmodel -m <id>`, then `sudo jetson_clocks`), Ollama for Jetson
(curl -fsSL https://ollama.com/install.sh | sh), `ollama pull qwen3-vl:4b-instruct`, and the data
(python scripts/get_vlm_data.py --root ~/nd_data). Nothing else should run on the board.
"""
import _path  # noqa: F401
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

from neurodrone import jetson as J
from neurodrone import ollama as O
from neurodrone.vlm import encode_jpeg, parse_goal

# Workstation results to compare with (Chapter 4; Qwen3-VL-4B-Instruct q4_K_M, drone prompt, accuracy at IoU 0.5)
REFERENCE = {
    "refdrone:test": (0.683, "RefDrone test, all queries, 896 px (workstation job 131)"),
    "grefcoco:val": (0.889, "gRefCOCO val, seed 0, 500 samples (workstation, Report 13.1)"),
}


# ----------------------------------------------------------------------------- pictures
def synthetic_picture(k) -> np.ndarray:
    """1920 x 1080 test picture, different for every k: a red box and a blue barrel on a floor, plus noise."""
    rng = np.random.default_rng(k)
    img = np.full((1080, 1920, 3), 200, np.uint8)
    img[600:1080] = 170
    x, y = int(rng.integers(300, 1500)), int(rng.integers(450, 700))
    img[y:y + 120, x:x + 120] = (180, 30, 30)
    x2 = int(rng.integers(100, 1700))
    img[540:660, x2:x2 + 80] = (30, 60, 180)
    return np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape, dtype=np.int16), 0, 255).astype(np.uint8)


class Pictures:
    """New picture every call: synthetic, or real photos from a folder (each reuse gets fresh noise, so Ollama
    cannot reuse its work on it)."""

    _instances = 0

    def __init__(self, folder=None, seed=0):
        Pictures._instances += 1
        self.base = (int(time.time() * 1000), Pictures._instances)  # every Pictures object draws its own pictures
        self.files = []
        if folder:
            self.files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                                if f.lower().endswith((".jpg", ".jpeg", ".png")))
            if not self.files:
                raise SystemExit(f"no .jpg/.png in {folder}")
        self.k = seed

    def next(self) -> np.ndarray:
        self.k += 1
        if not self.files:
            return synthetic_picture((*self.base, self.k))
        from PIL import Image
        img = np.asarray(Image.open(self.files[self.k % len(self.files)]).convert("RGB"))
        if self.k >= len(self.files):
            rng = np.random.default_rng(self.k)
            img = np.clip(img.astype(np.int16) + rng.integers(-2, 3, img.shape, dtype=np.int16), 0, 255).astype(np.uint8)
        return img


# ----------------------------------------------------------------------------- phases
def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def phase_info(a):
    info = dict(board=J.board_info(), server=O.server_info(a.ollama, a.model), model=a.model, num_ctx=a.num_ctx,
                max_side=a.max_side, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        info["loaded"] = O.loaded_info(a.ollama, a.model)
    except Exception as e:  # noqa: BLE001
        info["loaded_error"] = repr(e)
    write_json(os.path.join(a.out, "info.json"), info)
    print(json.dumps(info, indent=1, default=str), flush=True)
    return info


def one_call(a, pics, k):
    jpeg, sent = encode_jpeg(pics.next(), max_side=a.max_side, quality=90)
    r = O.chat(a.ollama, a.model, jpeg, a.instruction, timeout=a.timeout, num_ctx=a.num_ctx, num_gpu=a.num_gpu)
    p = parse_goal(r["text"], sent, "norm1000")
    r.update(k=k, t=time.time(), sent_w=sent[0], sent_h=sent[1], parsed=p is not None,
             found=None if p is None else p.found, text=r["text"][:200])
    return r


def phase_cold(a):
    tl = J.TegraLogger(path=os.path.join(a.out, "tegrastats_cold.jsonl")).start()
    t_unload = time.time()
    O.unload(a.ollama, a.model)
    time.sleep(3)
    free_before = J.board_info().get("MemAvailable_mb")
    load_s = O.load(a.ollama, a.model, num_ctx=a.num_ctx, num_gpu=a.num_gpu)
    loaded = O.loaded_info(a.ollama, a.model)
    first = one_call(a, Pictures(a.images, seed=999), 0)
    free_after = J.board_info().get("MemAvailable_mb")
    tl.stop()
    out = dict(load_s=load_s, first_answer=first, loaded=loaded, mem_available_mb_before=free_before,
               mem_available_mb_after=free_after, tegrastats=J.summarize(tl.between(t_unload, time.time())))
    write_json(os.path.join(a.out, "cold.json"), out)
    print(f"cold start: load {load_s:.1f} s, first answer {first['latency_s']:.1f} s "
          f"(prefill {first.get('prefill_s')}, decode {first.get('decode_s')}); "
          f"GPU share {loaded.get('gpu_share')}", flush=True)
    return out


def calls_summary(rows):
    def med(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return float(statistics.median(v)) if v else None

    def p90(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return float(np.percentile(v, 90)) if v else None
    return dict(n=len(rows), latency_s_median=med("latency_s"), latency_s_p90=p90("latency_s"),
                latency_s_max=max((r["latency_s"] for r in rows), default=None),
                prefill_s_median=med("prefill_s"), decode_s_median=med("decode_s"),
                prefill_tokens_median=med("prefill_tokens"), decode_tokens_median=med("decode_tokens"),
                prefill_tok_per_s_median=med("prefill_tok_per_s"), decode_tok_per_s_median=med("decode_tok_per_s"),
                reload_calls=sum(1 for r in rows if (r.get("load_s") or 0) > 1.0),
                parsed_share=(sum(r["parsed"] for r in rows) / len(rows)) if rows else None,
                errors=sum(1 for r in rows if r.get("error")))


def run_calls(a, name, n_calls=None, minutes=None):
    pics = Pictures(a.images)
    tl = J.TegraLogger(path=os.path.join(a.out, f"tegrastats_{name}.jsonl")).start()
    tl.wait_first()
    rows, t_start, k = [], time.time(), 0
    with open(os.path.join(a.out, f"{name}_calls.jsonl"), "w") as f:
        while True:
            if n_calls is not None and k >= n_calls:
                break
            if minutes is not None and time.time() - t_start >= 60 * minutes:
                break
            k += 1
            try:
                r = one_call(a, pics, k)
            except Exception as e:  # noqa: BLE001 - keep going, record it
                r = dict(k=k, t=time.time(), error=repr(e)[:300], latency_s=None, parsed=False)
                time.sleep(2)
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 10 == 0 or k <= 3:
                lat = r.get("latency_s")
                print(f"[{name}] call {k}: {'error' if lat is None else f'{lat:.2f} s'} "
                      f"prefill={r.get('prefill_s')} decode={r.get('decode_s')}", flush=True)
    tl.stop()
    ok = [r for r in rows if r.get("latency_s") is not None]
    out = dict(calls=calls_summary(ok), errors=len(rows) - len(ok), duration_s=time.time() - t_start,
               tegrastats=J.summarize(tl.records), tegrastats_available=tl.available)
    out["throttled"] = J.throttled(out["tegrastats"])
    if minutes is not None:  # per-5-minute medians: does the answer get slower as the board warms up?
        bins = {}
        for r in ok:
            bins.setdefault(int((r["t"] - t_start) // 300), []).append(r["latency_s"])
        out["latency_by_5min"] = {f"{5 * b}-{5 * b + 5} min": float(statistics.median(v)) for b, v in sorted(bins.items())}
    write_json(os.path.join(a.out, f"{name}.json"), out)
    c = out["calls"]
    print(f"[{name}] {c['n']} answers: median {c['latency_s_median']} s (p90 {c['latency_s_p90']}), prefill "
          f"{c['prefill_s_median']} s, decode {c['decode_s_median']} s; throttled={out['throttled']}", flush=True)
    return out


def phase_accuracy(a):
    import bench_grounding as B
    acc_dir = os.path.join(a.out, "accuracy")
    common = ["--data", os.path.expanduser(a.data), "--out", acc_dir, "--models", a.model, "--max-side", str(a.max_side),
              "--api", "ollama", "--num-ctx", str(a.num_ctx), "--base-url", a.ollama, "--server-note", "Jetson stage 1"]
    if "refdrone:test" in a.tasks:
        B.main(common + ["--tasks", "refdrone:test", "--full"])
    rest = [t for t in a.tasks if t != "refdrone:test"]
    if rest:
        B.main(common + ["--tasks", *rest, "--n", str(a.n)])
    return json.load(open(os.path.join(acc_dir, "summary.json")))


# ----------------------------------------------------------------------------- report
def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def checks(a, latency, soak, acc):
    """The stage-1 pass checks (Chapter 6, Table 6.2). Each: (name, value, limit, ok or None if not measured)."""
    out = []
    med = (latency or {}).get("calls", {}).get("latency_s_median")
    out.append(("median answer time (s)", med, f"<= {a.max_median}", None if med is None else med <= a.max_median))
    for s in acc or []:
        ref = REFERENCE.get(s.get("task"))
        if ref and s.get("acc_iou50") is not None and s.get("max_side") == a.max_side:
            d = s["acc_iou50"] - ref[0]
            out.append((f"accuracy {s['task']} vs workstation {ref[0]:.3f}", round(s["acc_iou50"], 3),
                        f">= {ref[0] - a.max_acc_drop:.3f}", d >= -a.max_acc_drop))
    ts = (soak or latency or {}).get("tegrastats", {})
    free = ts.get("ram_free_mb_min")
    out.append(("smallest free memory (MB)", free, f">= {a.min_free_mb}", None if free is None else free >= a.min_free_mb))
    thr = (soak or {}).get("throttled")
    out.append(("throttling during the soak", thr, "no", None if thr is None else not thr))
    return out


def phase_report(a):
    info, cold, lat, soak = (_load(os.path.join(a.out, f)) for f in ("info.json", "cold.json", "latency.json", "soak.json"))
    acc = _load(os.path.join(a.out, "accuracy", "summary.json"))
    ch = checks(a, lat, soak, acc)
    L = [f"# Stage 1: VLM on the Jetson - {a.model}", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{a.out}`.", ""]
    if info:
        b, s = info.get("board", {}), info.get("server", {})
        L += ["## Set-up", "", f"- Board: {b.get('model', '?')}; {b.get('l4t_release', '?')}; JetPack {b.get('jetpack', '?')}",
              f"- Power mode: {(b.get('nvpmodel') or '?').splitlines()[0] if b.get('nvpmodel') else '?'}; "
              f"memory {b.get('MemTotal_mb', '?')} MB, swap {b.get('SwapTotal_mb', '?')} MB",
              f"- Ollama {s.get('ollama_version', '?')}; model digest {str(s.get('digest', '?'))[:12]}, "
              f"{s.get('quantization', '?')}, {s.get('parameter_size', '?')}; num_ctx {info.get('num_ctx')}, "
              f"images {info.get('max_side')} px", ""]
    L += ["## Pass checks", "", "| Check | Measured | Limit | Result |", "|---|---|---|---|"]
    for name, val, lim, ok in ch:
        L.append(f"| {name} | {val if val is not None else 'not measured'} | {lim} | "
                 f"{'pass' if ok else ('FAIL' if ok is False else '-')} |")
    L.append("")
    if cold:
        fa = cold["first_answer"]
        L += ["## Cold start", "", f"Model load {cold['load_s']:.1f} s; first answer {fa['latency_s']:.1f} s "
              f"(prefill {fa.get('prefill_s')}, decode {fa.get('decode_s')}); share on the GPU "
              f"{cold.get('loaded', {}).get('gpu_share')}; available memory {cold.get('mem_available_mb_before')} -> "
              f"{cold.get('mem_available_mb_after')} MB.", ""]
    for name, d in (("Latency", lat), ("Soak", soak)):
        if not d:
            continue
        c, t = d["calls"], d.get("tegrastats", {})
        L += [f"## {name}", "", f"- {c['n']} answers ({d.get('errors', 0)} errors) in {d['duration_s'] / 60:.1f} min",
              f"- answer time median {c['latency_s_median']} s, p90 {c['latency_s_p90']} s, max {c['latency_s_max']} s",
              f"- prefill (prompt + image) median {c['prefill_s_median']} s for {c['prefill_tokens_median']} tokens; "
              f"decode median {c['decode_s_median']} s for {c['decode_tokens_median']} tokens "
              f"({c['decode_tok_per_s_median']} tokens/s)",
              f"- calls that reloaded the model: {c['reload_calls']}; answers parsed: {c['parsed_share']}",
              f"- memory used max {t.get('ram_used_mb_max')} of {t.get('ram_total_mb')} MB (swap max "
              f"{t.get('swap_used_mb_max')} MB); temperature max {t.get('temp_max_c')} C; power mean "
              f"{t.get('power_mean_w')} W, max {t.get('power_max_w')} W",
              f"- busy GPU clock first / last 5 min: {t.get('gpu_freq_first_mhz')} / {t.get('gpu_freq_last_mhz')} MHz "
              f"(max {t.get('gpu_freq_max_mhz')}); throttled: {d.get('throttled')}"]
        if d.get("latency_by_5min"):
            L.append("- median answer time per 5 min: " + ", ".join(f"{k}: {v:.2f} s" for k, v in d["latency_by_5min"].items()))
        L.append("")
    if acc:
        L += ["## Accuracy (drone prompt, IoU >= 0.5)", "", "| Task | Px | n | Accuracy (95 % CI) | Not found when absent | "
              "Workstation | Median answer | Prefill / decode |", "|---|---|---|---|---|---|---|---|"]
        for s in acc:
            ci = s.get("acc_iou50_ci95") or (None, None)
            ref = REFERENCE.get(s["task"], (None, ""))[0]
            L.append(f"| {s['task']} | {s['max_side']} | {s.get('n_target')} | "
                     f"{s.get('acc_iou50', 0):.3f} ({ci[0]:.2f}-{ci[1]:.2f}) | {s.get('no_target_correct')} | {ref} | "
                     f"{s.get('latency_s_median', 0):.2f} s | {s.get('prefill_s_median')} / {s.get('decode_s_median')} s |")
        L.append("")
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(L))
    write_json(os.path.join(a.out, "summary.json"), dict(checks=[dict(name=n, value=v, limit=l, ok=o) for n, v, l, o in ch]))
    print("\n".join(L), flush=True)
    return ch


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["all", "info", "cold", "latency", "soak", "accuracy", "report"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--num-gpu", type=int, default=None)
    ap.add_argument("--max-side", type=int, default=896)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--instruction", default="the red box")
    ap.add_argument("--images", default=None, help="folder of real photos instead of synthetic pictures")
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--data", default="~/nd_data", help="benchmark data (scripts/get_vlm_data.py)")
    ap.add_argument("--tasks", nargs="+", default=["refdrone:test", "grefcoco:val"])
    ap.add_argument("--n", type=int, default=500, help="scored samples for tasks other than RefDrone (seed 0)")
    ap.add_argument("--skip-accuracy", action="store_true", help="with `all`: leave out the accuracy phase")
    ap.add_argument("--max-median", type=float, default=10.0)
    ap.add_argument("--max-acc-drop", type=float, default=0.02)
    ap.add_argument("--min-free-mb", type=float, default=1024.0)
    a = ap.parse_args(argv)
    a.out = os.path.expanduser(a.out)
    os.makedirs(a.out, exist_ok=True)
    ph = a.phase
    if ph in ("all", "info"):
        phase_info(a)
    if ph in ("all", "cold"):
        phase_cold(a)
    if ph in ("all", "latency"):
        run_calls(a, "latency", n_calls=a.calls)
    if ph in ("all", "soak"):
        run_calls(a, "soak", minutes=a.minutes)
    if ph == "accuracy" or (ph == "all" and not a.skip_accuracy):
        phase_accuracy(a)
    if ph in ("all", "report"):
        ch = phase_report(a)
        return 1 if any(ok is False for *_, ok in ch) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
