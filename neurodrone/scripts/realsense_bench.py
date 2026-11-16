#!/usr/bin/env python3
"""Stage 3 of the deployment plan: the RealSense D435i on a table or tripod with the Jetson, no drone (Chapter 6).

    python scripts/realsense_bench.py depth   --out ~/nd_results/stage3/depth  [--emitter both]
    python scripts/realsense_bench.py vlm     --out ~/nd_results/stage3/vlm --instructions "the red box" "the blue barrel"
    python scripts/realsense_bench.py vlm-score ~/nd_results/stage3/vlm/review.csv
    python scripts/realsense_bench.py guard   --out ~/nd_results/stage3/guard --duration 60
    python scripts/realsense_bench.py timing  --out ~/nd_results/stage3/timing --duration 60 [--vlm-load]

depth      Camera facing a flat target (wall, board) square on. For each distance you measured with the tape (type it,
           press Enter) the script takes --frames pictures and reports, for the centre of the image: measured depth,
           error against the tape, noise (frame to frame), spread across the target and the share of pixels with a
           depth. With --emitter both every distance is measured with the IR dot projector on and off (the visual
           odometry wants it off). At the end: error and noise against distance, and the noise model sigma = a * Z^2
           compared with the one the cloud SIH test assumed (a = 0.01).
vlm        The lab's objects in front of the camera. For each target and each position you type the tape distance (or
           'absent' if the target is not in view); the script asks the VLM --shots times (a new picture each time),
           draws the box, takes the distance from the aligned depth in the box exactly as the mission does, and writes
           review.csv + index.html. Open index.html, fill the column 'correct' in review.csv (y = box on the named
           object, n = on something else), then run vlm-score.
vlm-score  Found rate on present targets, wrong-object rate, "not found" rate on absent targets, distance from box +
           depth against the tape.
guard      The mission's depth guard on the live depth (corridor 0.8 m, stop 1.2 m, time to contact 2 s at --speed),
           levelled with the camera's IMU. Move an object towards the camera (or carry the camera towards it): the
           script beeps and logs the distance when the guard first blocks; type the tape distance when it beeps.
           Point it at an empty floor/wall to count false stops.
timing     Frame age (camera time stamp to now), frame rate, frames missed - optionally while the VLM answers non-stop
           (--vlm-load) - and the camera attitude from the IMU (check: level tripod -> roll and pitch near 0).

Needs pyrealsense2 (librealsense) and, for vlm, Ollama with the model loaded (scripts/jetson_vlm_bench.py cold).
"""
import _path  # noqa: F401
import argparse
import csv
import html
import json
import math
import os
import sys
import threading
import time
import warnings

import numpy as np

from neurodrone import mission as M
from neurodrone import ollama as O
from neurodrone.realsense import attitude_from_accel
from neurodrone.vlm import encode_jpeg, parse_goal


# ----------------------------------------------------------------------------- pure helpers (tested)
def roi_slice(shape, frac=0.1, centre=(0.5, 0.5)):
    h, w = shape[:2]
    hw, hh = max(1, int(w * frac / 2)), max(1, int(h * frac / 2))
    cx, cy = int(w * centre[0]), int(h * centre[1])
    return slice(max(0, cy - hh), min(h, cy + hh)), slice(max(0, cx - hw), min(w, cx + hw))


def depth_stats(frames, frac=0.1):
    """Centre region of several depth pictures (m, NaN = none) -> measured distance (median over frames of the
    per-frame median), noise (median over pixels of the frame-to-frame std), spread across the region (std of the
    time-averaged region) and fill (share of pixels with a depth)."""
    sl = roi_slice(frames[0].shape, frac)
    stack = np.stack([f[sl] for f in frames]).astype(np.float64)
    valid = np.isfinite(stack) & (stack > 0)
    fill = float(valid.mean())
    per_frame = [np.median(s[v]) for s, v in zip(stack, valid) if v.sum() >= 5]
    if not per_frame:
        return dict(measured_m=None, noise_m=None, spread_m=None, fill=fill, n_frames=len(frames))
    all_valid = valid.all(axis=0)
    noise = float(np.median(stack[:, all_valid].std(axis=0))) if all_valid.sum() >= 5 else None
    mean_img = np.where(valid, stack, np.nan)
    with warnings.catch_warnings():  # pixels with no depth in any frame: NaN, that is fine
        warnings.simplefilter("ignore", RuntimeWarning)
        avg = np.nanmean(mean_img, axis=0)
    spread = float(np.nanstd(avg)) if np.isfinite(avg).sum() >= 5 else None
    return dict(measured_m=float(np.median(per_frame)), noise_m=noise, spread_m=spread, fill=fill, n_frames=len(frames))


def noise_fit(rows):
    """sigma = a * Z^2 by least squares over the rows with a noise value; bias and RMS of the error."""
    pts = [(r["tape_m"], r["noise_m"]) for r in rows if r.get("noise_m") is not None and r.get("tape_m")]
    errs = [r["measured_m"] - r["tape_m"] for r in rows if r.get("measured_m") is not None and r.get("tape_m")]
    out = {}
    if pts:
        z, s = np.array(pts).T
        out["a_sigma_z2"] = float(np.sum(s * z ** 2) / np.sum(z ** 4))
    if errs:
        e = np.array(errs)
        out.update(bias_m=float(e.mean()), rms_error_m=float(np.sqrt((e ** 2).mean())), max_abs_error_m=float(np.abs(e).max()))
    return out


def box_distance(depth_m, cam, box):
    """Distance of the object in a normalised box, as the mission computes it (median depth in the central half),
    on depth aligned to the colour picture (so the colour camera model serves both)."""
    return M.box_depth(depth_m, cam, cam, box)


def score_review(rows):
    """review.csv rows (with 'correct' filled) -> found rate, wrong-object rate, not-found-when-absent, distance error."""
    present = [r for r in rows if r.get("absent", "").lower() not in ("1", "true", "y", "yes")]
    absent = [r for r in rows if r not in present]
    out = dict(n_present=len(present), n_absent=len(absent))
    found = [r for r in present if r["found"] == "True"]
    if present:
        right = [r for r in found if r.get("correct", "").strip().lower() in ("y", "yes", "1")]
        wrong = [r for r in found if r.get("correct", "").strip().lower() in ("n", "no", "0")]
        unreviewed = len(found) - len(right) - len(wrong)
        out.update(found_rate=len(found) / len(present), correct_rate=len(right) / len(present),
                   wrong_object_rate=len(wrong) / len(present), unreviewed=unreviewed)
        errs = [float(r["dist_m"]) - float(r["tape_m"]) for r in right if r.get("dist_m") and r.get("tape_m")]
        if errs:
            e = np.array(errs)
            out.update(dist_bias_m=float(e.mean()), dist_rms_m=float(np.sqrt((e ** 2).mean())), n_dist=len(errs))
    if absent:
        out["not_found_when_absent"] = sum(r["found"] == "False" for r in absent) / len(absent)
    by = {}
    for r in present:
        by.setdefault(r["instruction"], []).append(r)
    out["by_target"] = {k: dict(n=len(v), found_rate=sum(x["found"] == "True" for x in v) / len(v),
                                correct_rate=sum(x.get("correct", "").strip().lower() in ("y", "yes", "1") for x in v) / len(v))
                        for k, v in by.items()}
    return out


def level_att(accel):
    return attitude_from_accel(accel) if accel is not None else (0.0, 0.0)


# ----------------------------------------------------------------------------- camera
def open_camera(a, imu=False):
    from neurodrone.realsense import RealSenseCameras
    cams = RealSenseCameras(color=tuple(a.color), depth=tuple(a.depth), emitter=a.emitter != "off", imu=imu)
    cams.wait()
    print(f"camera {cams.device}  colour {cams.intrinsics['width']}x{cams.intrinsics['height']} "
          f"fx {cams.intrinsics['fx']:.1f}  depth scale {cams.depth_scale}", flush=True)
    return cams


def grab(cams, n, after=None):
    frames, last = [], (after if after is not None else (cams.frame().idx if cams.frame() else 0))
    while len(frames) < n:
        f = cams.wait_new(last)
        if f is None:
            raise TimeoutError("camera stopped delivering frames")
        frames.append(f)
        last = f.idx
    return frames


def save_jpg(path, rgb, box=None, text=None):
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb)
    if box:
        w, h = im.size
        ImageDraw.Draw(im).rectangle([box[0] * w, box[1] * h, box[2] * w, box[3] * h], outline=(255, 0, 0), width=4)
    if text:
        ImageDraw.Draw(im).text((10, 10), text, fill=(255, 255, 0))
    im.save(path, quality=88)


# ----------------------------------------------------------------------------- subcommands
def cmd_depth(a, cams, ask=input):
    os.makedirs(a.out, exist_ok=True)
    rows, modes = [], (["on", "off"] if a.emitter == "both" else [a.emitter])
    print("Camera square on to a flat target. Type the tape distance in m and press Enter; empty line = done.", flush=True)
    while True:
        s = ask("tape distance (m): ").strip()
        if not s:
            break
        try:
            tape = float(s.replace(",", "."))
        except ValueError:
            print("  not a number", flush=True)
            continue
        for mode in modes:
            cams.set_emitter(mode == "on")
            time.sleep(a.settle)
            fr = grab(cams, a.frames)
            st = depth_stats([f.depth_m for f in fr], a.roi)
            st.update(tape_m=tape, emitter=mode, error_m=None if st["measured_m"] is None else st["measured_m"] - tape)
            tag = f"{tape:.2f}m_emitter_{mode}"
            save_jpg(os.path.join(a.out, f"rgb_{tag}.jpg"), fr[-1].rgb)
            np.save(os.path.join(a.out, f"depth_{tag}.npy"), fr[-1].depth_m.astype(np.float32))
            rows.append(st)
            print(f"  emitter {mode}: measured {st['measured_m']} m, error {st['error_m']}, noise {st['noise_m']}, "
                  f"spread {st['spread_m']}, fill {st['fill']:.2f}", flush=True)
    keys = ["tape_m", "emitter", "measured_m", "error_m", "noise_m", "spread_m", "fill", "n_frames"]
    with open(os.path.join(a.out, "depth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    summ = {m: noise_fit([r for r in rows if r["emitter"] == m]) for m in modes}
    summ["sim_assumption_a"] = 0.01
    json.dump(dict(rows=rows, fit=summ, camera=cams.meta()), open(os.path.join(a.out, "depth.json"), "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1), flush=True)
    return rows, summ


def cmd_vlm(a, cams, ask=input):
    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True)
    review = os.path.join(a.out, "review.csv")
    keys = ["instruction", "position", "tape_m", "absent", "shot", "found", "box", "dist_m", "latency_s", "prefill_s",
            "decode_s", "frame", "text", "correct"]
    new = not os.path.exists(review)
    fh = open(review, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=keys)
    if new:
        w.writeheader()
    pos = 0
    print("For each position type the tape distance to the target in m, 'absent' if it is not in view, "
          "or an empty line to go to the next target.", flush=True)
    for instr in a.instructions:
        while True:
            s = ask(f"[{instr}] tape distance (m) / absent / empty = next target: ").strip().lower()
            if not s:
                break
            absent = s == "absent"
            try:
                tape = None if absent else float(s.replace(",", "."))
            except ValueError:
                print("  not a number", flush=True)
                continue
            pos += 1
            last = cams.frame().idx if cams.frame() else 0
            for shot in range(a.shots):
                f = grab(cams, 1, after=last)[0]
                last = f.idx + a.skip  # the next shot uses a later picture
                jpeg, sent = encode_jpeg(f.rgb, max_side=a.max_side, quality=90)
                r = O.chat(a.ollama, a.model, jpeg, instr, num_ctx=a.num_ctx)
                p = parse_goal(r["text"], sent, "norm1000")
                box = dist = None
                if p is not None and p.found and p.w is not None:
                    box = [p.x - p.w / 2, p.y - p.h / 2, p.x + p.w / 2, p.y + p.h / 2]
                    dist = box_distance(f.depth_m, cams.cam, box)
                name = f"p{pos:03d}_s{shot}_{instr.replace(' ', '_')[:30]}.jpg"
                save_jpg(os.path.join(a.out, "frames", name), f.rgb, box, f"{instr} | tape {tape} | depth {dist}")
                row = dict(instruction=instr, position=pos, tape_m=tape, absent=absent, shot=shot,
                           found=bool(p and p.found), box=None if box is None else [round(v, 4) for v in box],
                           dist_m=None if dist is None else round(dist, 3), latency_s=round(r["latency_s"], 3),
                           prefill_s=r.get("prefill_s"), decode_s=r.get("decode_s"), frame=os.path.join("frames", name),
                           text=r["text"][:200], correct="")
                w.writerow(row)
                fh.flush()
                print(f"  shot {shot}: found={row['found']} dist={row['dist_m']} ({r['latency_s']:.1f} s)", flush=True)
    fh.close()
    write_review_html(review)
    print(f"Now open {os.path.join(a.out, 'index.html')}, fill 'correct' (y/n) in {review}, then run vlm-score.", flush=True)
    return review


def write_review_html(review):
    rows = list(csv.DictReader(open(review)))
    d = os.path.dirname(review)
    cells = []
    for i, r in enumerate(rows):
        cap = html.escape(f"#{i + 2} {r['instruction']} | pos {r['position']} | tape {r['tape_m'] or 'absent'} | "
                          f"found {r['found']} | depth {r['dist_m']}")
        cells.append(f'<figure><img src="{html.escape(r["frame"])}" loading="lazy"><figcaption>{cap}</figcaption></figure>')
    page = ("<!doctype html><meta charset=utf-8><title>VLM review</title><style>body{font-family:sans-serif;margin:16px}"
            "figure{display:inline-block;width:420px;margin:6px}img{width:100%}figcaption{font-size:12px}</style>"
            "<p>#n = line in review.csv. Fill 'correct': y = the red box is on the named object, n = on something else."
            "</p>" + "".join(cells))
    open(os.path.join(d, "index.html"), "w").write(page)


def cmd_vlm_score(path):
    rows = list(csv.DictReader(open(path)))
    s = score_review(rows)
    json.dump(s, open(os.path.join(os.path.dirname(path), "vlm_score.json"), "w"), indent=1)
    print(json.dumps(s, indent=1), flush=True)
    return s


def guard_loop(a, cams, out_path, duration_s):
    """Run the mission's guard on every new depth picture; log, beep and ask for the tape distance on each new block."""
    v = np.array([a.speed, 0.0, 0.0])
    recs, blocked_before, events = [], False, []
    t_end, last = time.monotonic() + duration_s, 0
    with open(out_path, "w") as fh:
        while time.monotonic() < t_end:
            f = cams.wait_new(last, timeout_s=2.0)
            if f is None:
                break
            last = f.idx
            depth, age = cams.depth()
            att = level_att(cams.accel_mean()) if a.imu else (a.roll, a.pitch)
            g = M.guard(depth, cams.cam, v, age, radius_m=a.radius, stop_dist_m=a.stop_dist, ttc_s=a.ttc, att_deg=att)
            rec = dict(t=time.time(), idx=f.idx, dist_m=None if not math.isfinite(g.dist_ahead_m) else round(g.dist_ahead_m, 3),
                       ttc_s=None if not math.isfinite(g.ttc_s) else round(g.ttc_s, 2), blocked=g.blocked, reason=g.reason,
                       age_s=round(age, 3), roll=round(att[0], 1), pitch=round(att[1], 1))
            recs.append(rec)
            fh.write(json.dumps(rec) + "\n")
            if g.blocked and not blocked_before:
                print(f"\aBLOCKED: obstacle at {rec['dist_m']} m ({g.reason})", flush=True)
                events.append(dict(rec))
            blocked_before = g.blocked
    return recs, events


def guard_summary(recs, events):
    if not recs:
        return dict(n=0)
    d = [r["dist_m"] for r in recs if r["dist_m"] is not None]
    return dict(n=len(recs), blocked_share=sum(r["blocked"] for r in recs) / len(recs), block_events=len(events),
                first_block_dist_m=[e["dist_m"] for e in events], dist_min_m=min(d) if d else None,
                age_median_s=float(np.median([r["age_s"] for r in recs])))


def cmd_guard(a, cams, ask=input):
    os.makedirs(a.out, exist_ok=True)
    print(f"Guard running for {a.duration:.0f} s (corridor {a.radius} m, stop {a.stop_dist} m, TTC {a.ttc} s at "
          f"{a.speed} m/s). Move an object towards the camera; note the tape distance at each BLOCKED.", flush=True)
    recs, events = guard_loop(a, cams, os.path.join(a.out, "guard.jsonl"), a.duration)
    s = guard_summary(recs, events)
    if events and not a.no_ask:
        tapes = ask("tape distances at the BLOCKED beeps, comma separated (empty = skip): ").strip()
        if tapes:
            s["tape_at_block_m"] = [float(x) for x in tapes.replace(";", ",").split(",") if x.strip()]
    json.dump(s, open(os.path.join(a.out, "guard_summary.json"), "w"), indent=1)
    print(json.dumps(s, indent=1), flush=True)
    return s


def cmd_timing(a, cams):
    os.makedirs(a.out, exist_ok=True)
    stop = threading.Event()
    vlm_lat = []

    def vlm_load():  # a new camera picture every call, as in flight
        while not stop.is_set():
            f = cams.frame()
            if f is None:
                time.sleep(0.1)
                continue
            jpeg, _ = encode_jpeg(f.rgb, max_side=a.max_side, quality=90)
            try:
                r = O.chat(a.ollama, a.model, jpeg, "the red box", num_ctx=a.num_ctx)
                vlm_lat.append(r["latency_s"])
            except Exception as e:  # noqa: BLE001
                print(f"vlm error {e!r}", flush=True)
                time.sleep(1)

    th = threading.Thread(target=vlm_load, daemon=True) if a.vlm_load else None
    if th:
        th.start()
    ages, idx, nos, t0, last = [], [], [], time.monotonic(), 0
    while time.monotonic() - t0 < a.duration:
        f = cams.wait_new(last, timeout_s=2.0)
        if f is None:
            break
        last = f.idx
        ages.append(time.time() - f.depth_ts_ms / 1000.0)
        idx.append(time.monotonic())
        nos.append(f.depth_no)
    stop.set()
    att = level_att(cams.accel_mean()) if a.imu else None
    nos = np.array(nos)
    step = float(np.median(np.diff(nos))) if len(nos) > 2 else 1.0
    s = dict(frames=len(ages), fps=(len(idx) - 1) / (idx[-1] - idx[0]) if len(idx) > 1 else None,
             age_median_s=float(np.median(ages)) if ages else None, age_p99_s=float(np.percentile(ages, 99)) if ages else None,
             age_max_s=float(max(ages)) if ages else None,
             camera_frames_missed=int(np.sum(np.maximum(0, np.diff(nos) - step)) // max(1, step)) if len(nos) > 2 else None,
             processing_dropped=getattr(cams, "n_dropped", None), vlm_load=bool(a.vlm_load),
             vlm_answers=len(vlm_lat), vlm_median_s=float(np.median(vlm_lat)) if vlm_lat else None,
             imu_roll_pitch_deg=att)
    json.dump(s, open(os.path.join(a.out, f"timing{'_vlm' if a.vlm_load else ''}.json"), "w"), indent=1)
    print(json.dumps(s, indent=1), flush=True)
    return s


# ----------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["depth", "vlm", "vlm-score", "guard", "timing"])
    ap.add_argument("review", nargs="?", help="review.csv (vlm-score)")
    ap.add_argument("--out", default="stage3")
    ap.add_argument("--color", type=int, nargs=3, default=[1280, 720, 15], metavar=("W", "H", "FPS"))
    ap.add_argument("--depth", type=int, nargs=3, default=[848, 480, 30], metavar=("W", "H", "FPS"))
    ap.add_argument("--emitter", choices=["on", "off", "both"], default="on")
    ap.add_argument("--settle", type=float, default=1.0, help="s to wait after switching the emitter")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--roi", type=float, default=0.1, help="centre region, fraction of the image width/height")
    ap.add_argument("--instructions", nargs="+", default=["the red box"])
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--skip", type=int, default=5, help="frames between two shots")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--max-side", type=int, default=896)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--speed", type=float, default=0.5, help="assumed forward speed for the time to contact (m/s)")
    ap.add_argument("--radius", type=float, default=0.8)
    ap.add_argument("--stop-dist", type=float, default=1.2)
    ap.add_argument("--ttc", type=float, default=2.0)
    ap.add_argument("--no-imu", dest="imu", action="store_false", help="do not level with the IMU (use --roll/--pitch)")
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--no-ask", action="store_true")
    ap.add_argument("--vlm-load", action="store_true", help="timing: the VLM answers non-stop meanwhile")
    return ap


def main(argv=None, cams=None, ask=input):
    a = build_parser().parse_args(argv)
    if a.cmd == "vlm-score":
        return 0 if cmd_vlm_score(a.review) else 1
    a.out = os.path.expanduser(a.out)
    own = cams is None
    cams = cams or open_camera(a, imu=a.imu or a.cmd in ("guard", "timing"))
    try:
        if a.cmd == "depth":
            cmd_depth(a, cams, ask)
        elif a.cmd == "vlm":
            cmd_vlm(a, cams, ask)
        elif a.cmd == "guard":
            cmd_guard(a, cams, ask)
        elif a.cmd == "timing":
            cmd_timing(a, cams)
    finally:
        if own:
            cams.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
