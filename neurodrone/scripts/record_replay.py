#!/usr/bin/env python3
"""Stage 4 of the deployment plan: record the RealSense while carrying it (with the Jetson) along the planned paths,
then replay the recording through the mission code offline (thesis Chapter 6).

    python scripts/record_replay.py record --out ~/nd_rec/run01 --notes "red box straight ahead, pole at 3 m"
        while it records, type a mark + Enter whenever you stand at a measured point, e.g.
            d=4.0            target 4.0 m away (tape)
            d=2.5 o=1.5      target 2.5 m, obstacle 1.5 m ahead
            absent           target not in view here
        and q + Enter to stop.
    python scripts/record_replay.py info   ~/nd_rec/run01
    python scripts/record_replay.py replay ~/nd_rec/run01 --instruction "the red box" --out ~/nd_rec/run01/replay

replay  goes through the recording in time order and, as in flight:
        - every frame: the depth guard (corridor, stop distance, time to contact at --speed, levelled with the IMU)
        - every --vlm-period s of recording time: a VLM answer on that picture -> box -> edge rule -> depth in the box
          -> 3-D goal -> 2-of-3 consensus (the mission's own functions)
        The camera position is unknown when carried by hand (no drone, no odometry yet): goals are given in the frame
        of the camera at that moment, unless --pose gives a CSV (t_wall, north, east, down, roll, pitch, yaw) - e.g.
        the visual odometry from stage 6. So compare distances at the marks, where you stood still.
        Output: replay.jsonl (every decision), marks.csv (truth vs what the stack saw at each mark), summary.json.
        The recording is plain files (JPEG, 16-bit PNG, CSV), so it replays on the laptop too (Ollama there).
"""
import _path  # noqa: F401
import argparse
import bisect
import csv
import json
import math
import os
import sys
import threading
import time
import types

import numpy as np

from neurodrone import mission as M
from neurodrone import ollama as O
from neurodrone.realsense import Recording, attitude_from_accel, parse_mark
from neurodrone.vlm import encode_jpeg, parse_goal


# ----------------------------------------------------------------------------- record
def cmd_record(a, open_cams=None, lines=None):
    """Record until q + Enter (or --duration). open_cams(on_frame, on_imu) -> camera (tests pass a fake one)."""
    from neurodrone.realsense import Recorder
    os.makedirs(a.out, exist_ok=True)
    rec_box = {}

    def on_frame(f):
        r = rec_box.get("r")
        if r is not None:
            r.add_frame(f)

    def on_imu(x):
        r = rec_box.get("r")
        if r is not None:
            r.add_imu(x)
    if open_cams is None:
        def open_cams(of, oi):
            from neurodrone.realsense import RealSenseCameras
            c = RealSenseCameras(color=tuple(a.color), depth=tuple(a.depth), emitter=not a.emitter_off, imu=True,
                                 on_frame=of, on_imu=oi)
            c.wait()
            return c
    cams = open_cams(on_frame, on_imu)
    rec = Recorder(a.out, dict(cams.meta(), notes=a.notes), every=a.every)
    rec_box["r"] = rec
    print(f"recording to {a.out} - type a mark (d=4.0, o=1.5, absent, any text) + Enter; q + Enter to stop", flush=True)
    t0 = time.time()
    stop = threading.Event()

    def read_lines():
        src = lines if lines is not None else iter(sys.stdin.readline, "")
        for line in src:
            line = line.strip()
            if line.lower() in ("q", "quit", "stop"):
                break
            if line:
                rec.mark(line)
                print(f"  mark at {time.time() - t0:.1f} s: {line}", flush=True)
        stop.set()
    threading.Thread(target=read_lines, daemon=True).start()
    try:
        while not stop.is_set() and (a.duration is None or time.time() - t0 < a.duration):
            time.sleep(0.5)
            if a.verbose:
                print(f"  {rec.n_written} frames written, {rec.n_dropped} dropped", flush=True)
    except KeyboardInterrupt:
        pass
    rec_box.pop("r", None)
    rec.close()
    cams.stop()
    print(f"done: {rec.n_written} frames, {rec.n_dropped} dropped by the writer", flush=True)
    return Recording(a.out).summary()


# ----------------------------------------------------------------------------- replay
def load_pose(path):
    """CSV t_wall, north, east, down, roll, pitch, yaw -> interpolating function t -> state (or None outside)."""
    rows = sorted((tuple(float(r[k]) for k in ("t_wall", "north", "east", "down", "roll", "pitch", "yaw")))
                  for r in csv.DictReader(open(path)))
    arr = np.array(rows)

    def at(t):
        if t < arr[0, 0] or t > arr[-1, 0]:
            return None
        v = [np.interp(t, arr[:, 0], arr[:, k]) for k in range(1, 7)]
        return types.SimpleNamespace(north_m=v[0], east_m=v[1], down_m=v[2], roll_deg=v[3], pitch_deg=v[4], yaw_deg=v[5])
    return at


def default_ask(a):
    def ask(rgb):
        jpeg, sent = encode_jpeg(rgb, max_side=a.max_side, quality=90)
        r = O.chat(a.ollama, a.model, jpeg, a.instruction, num_ctx=a.num_ctx)
        return r["text"], sent, r["latency_s"]
    return ask


def replay(rec: Recording, a, ask=None, out_dir=None):
    """Run the guard on every frame and the VLM goal chain every --vlm-period s of recording time."""
    ask = ask or default_ask(a)
    pose_at = load_pose(a.pose) if a.pose else None
    cons = M.Consensus(need=a.consensus_need, window=a.consensus_window, tol_m=a.consensus_tol)
    watchdog = M.DepthWatchdog()
    v = np.array([a.speed, 0.0, 0.0])
    out_dir = out_dir or a.out
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "replay.jsonl"), "w")
    guard_recs, vlm_recs = [], []
    next_vlm_t, accepted = None, None
    for i in range(len(rec)):
        f = rec.load(i)
        acc = rec.accel_mean(f.t_wall)
        roll, pitch = attitude_from_accel(acc) if acc is not None else (0.0, 0.0)
        st = pose_at(f.t_wall) if pose_at else None
        if st is None:
            st = types.SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-a.alt, roll_deg=roll, pitch_deg=pitch, yaw_deg=0.0)
        age = max(0.0, f.t_wall - f.depth_ts_ms / 1000.0) if f.depth_ts_ms > 1e9 else 0.0  # age when it arrived
        frozen = watchdog.update(f.depth_m, st.north_m, st.east_m) if pose_at else False
        g = M.guard(f.depth_m, rec.cam, v, float("inf") if frozen else age, radius_m=a.radius, stop_dist_m=a.stop_dist,
                    ttc_s=a.ttc, att_deg=(st.roll_deg, st.pitch_deg))
        gr = dict(kind="guard", t=f.t_wall, idx=f.idx, dist_m=None if not math.isfinite(g.dist_ahead_m) else round(g.dist_ahead_m, 3),
                  blocked=g.blocked, reason=g.reason, roll=round(st.roll_deg, 1), pitch=round(st.pitch_deg, 1),
                  age_s=round(age, 3), frozen=frozen)
        guard_recs.append(gr)
        log.write(json.dumps(gr) + "\n")
        if a.no_vlm:
            continue
        if next_vlm_t is None:
            next_vlm_t = f.t_wall
        if f.t_wall + 1e-6 < next_vlm_t:
            continue
        next_vlm_t = f.t_wall + a.vlm_period
        text, sent, lat = ask(f.rgb)
        p = parse_goal(text, sent, "norm1000")
        box = dist = goal = None
        edge = False
        if p is not None and p.found and p.w is not None:
            box = [p.x - p.w / 2, p.y - p.h / 2, p.x + p.w / 2, p.y + p.h / 2]
            edge = M.box_at_edge(box)
            dist = None if edge else M.box_depth(f.depth_m, rec.cam, rec.cam, box)
            if dist is not None:
                goal = M.goal_from_box(box, dist, rec.cam, st, cam_offset_frd=tuple(a.cam_offset), standoff_m=a.standoff)
        acc_goal = cons.add(goal)
        if acc_goal is not None:
            accepted = acc_goal
        bearing = None
        if box is not None:
            r = rec.cam.ray((box[0] + box[2]) / 2 * rec.cam.width, (box[1] + box[3]) / 2 * rec.cam.height)
            bearing = math.degrees(math.atan2(r[1], r[0]))
        vr = dict(kind="vlm", t=f.t_wall, idx=f.idx, text=text[:200], found=bool(p and p.found), box=box, at_edge=edge,
                  dist_m=dist, bearing_deg=bearing, goal=None if goal is None else goal.tolist(),
                  accepted=None if acc_goal is None else acc_goal.tolist(), latency_s=lat)
        vlm_recs.append(vr)
        log.write(json.dumps(vr) + "\n")
        print(f"[{f.t_wall - rec.frames[0]['t_wall']:6.1f} s] found={vr['found']} dist={dist if dist is None else round(dist, 2)} "
              f"edge={edge} accepted={acc_goal is not None}  guard={gr['dist_m']} blocked={g.blocked}", flush=True)
    log.close()
    marks = mark_table(rec, guard_recs, vlm_recs, a.mark_window)
    keys = ["t_s", "label", "d", "o", "absent", "vlm_n", "vlm_found", "vlm_dist_median", "dist_error", "guard_dist",
            "guard_error", "guard_blocked"]
    with open(os.path.join(out_dir, "marks.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for m in marks:
            w.writerow({k: m.get(k) for k in keys})
    summ = replay_summary(rec, guard_recs, vlm_recs, marks, accepted)
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float), flush=True)
    return summ


def mark_table(rec, guard_recs, vlm_recs, window_s):
    """For each mark: truth from the label, the VLM answers within +-window_s and the guard at the nearest frame."""
    t0 = rec.frames[0]["t_wall"] if rec.frames else 0.0
    g_t = [g["t"] for g in guard_recs]
    out = []
    for t, label in rec.marks:
        m = parse_mark(label)
        row = dict(t_s=round(t - t0, 2), label=label, d=m.get("d"), o=m.get("o"), absent=m.get("absent", False))
        near = [v for v in vlm_recs if abs(v["t"] - t) <= window_s]
        row["vlm_n"] = len(near)
        if near:
            row["vlm_found"] = sum(v["found"] for v in near) / len(near)
            ds = [v["dist_m"] for v in near if v["dist_m"] is not None]
            if ds:
                row["vlm_dist_median"] = round(float(np.median(ds)), 3)
                if row["d"] is not None:
                    row["dist_error"] = round(row["vlm_dist_median"] - row["d"], 3)
        if g_t:
            gr = guard_recs[_nearest(g_t, t)]
            row.update(guard_dist=gr["dist_m"], guard_blocked=gr["blocked"])
            if row["o"] is not None and gr["dist_m"] is not None:
                row["guard_error"] = round(gr["dist_m"] - row["o"], 3)
        out.append(row)
    return out


def _nearest(ts, t):
    i = bisect.bisect_left(ts, t)
    return min((j for j in (i - 1, i) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - t))


def replay_summary(rec, guard_recs, vlm_recs, marks, accepted):
    s = dict(recording=rec.summary(), frames=len(guard_recs),
             guard_blocked_share=(sum(g["blocked"] for g in guard_recs) / len(guard_recs)) if guard_recs else None,
             vlm_calls=len(vlm_recs), vlm_found_share=(sum(v["found"] for v in vlm_recs) / len(vlm_recs)) if vlm_recs else None,
             vlm_edge_share=(sum(v["at_edge"] for v in vlm_recs) / len(vlm_recs)) if vlm_recs else None,
             goal_accepted=None if accepted is None else [round(x, 3) for x in accepted], marks=len(marks))
    de = [m["dist_error"] for m in marks if m.get("dist_error") is not None]
    if de:
        s.update(target_dist_bias_m=float(np.mean(de)), target_dist_rms_m=float(np.sqrt(np.mean(np.square(de)))))
    ge = [m["guard_error"] for m in marks if m.get("guard_error") is not None]
    if ge:
        s.update(obstacle_dist_bias_m=float(np.mean(ge)), obstacle_dist_rms_m=float(np.sqrt(np.mean(np.square(ge)))))
    ab = [m for m in marks if m.get("absent") and m.get("vlm_n")]
    if ab:
        s["found_when_absent"] = float(np.mean([m["vlm_found"] for m in ab]))
    return s


# ----------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["record", "info", "replay"])
    ap.add_argument("path", nargs="?", help="recording folder (info, replay)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--color", type=int, nargs=3, default=[1280, 720, 15], metavar=("W", "H", "FPS"))
    ap.add_argument("--depth", type=int, nargs=3, default=[848, 480, 15], metavar=("W", "H", "FPS"))
    ap.add_argument("--emitter-off", action="store_true", help="record with the IR dot projector off")
    ap.add_argument("--every", type=int, default=1, help="record every n-th frame")
    ap.add_argument("--duration", type=float, default=None, help="stop after this many s (default: when you type q)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--instruction", default="the red box")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--max-side", type=int, default=896)
    ap.add_argument("--vlm-period", type=float, default=2.0, help="s of recording time between VLM answers")
    ap.add_argument("--no-vlm", action="store_true", help="replay only the guard")
    ap.add_argument("--pose", default=None, help="CSV t_wall,north,east,down,roll,pitch,yaw (e.g. from the VIO)")
    ap.add_argument("--alt", type=float, default=1.2, help="height of the camera when carried (m), without --pose")
    ap.add_argument("--cam-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="camera in the body frame (m)")
    ap.add_argument("--standoff", type=float, default=1.5)
    ap.add_argument("--speed", type=float, default=0.5, help="assumed forward speed for the time to contact (m/s)")
    ap.add_argument("--radius", type=float, default=0.8)
    ap.add_argument("--stop-dist", type=float, default=1.2)
    ap.add_argument("--ttc", type=float, default=2.0)
    ap.add_argument("--consensus-need", type=int, default=2)
    ap.add_argument("--consensus-window", type=int, default=3)
    ap.add_argument("--consensus-tol", type=float, default=1.5)
    ap.add_argument("--mark-window", type=float, default=3.0, help="VLM answers within +- this many s count for a mark")
    return ap


def main(argv=None, open_cams=None, lines=None, ask=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "record":
        if not a.out:
            raise SystemExit("record needs --out")
        a.out = os.path.expanduser(a.out)
        print(json.dumps(cmd_record(a, open_cams, lines), indent=1), flush=True)
        return 0
    if not a.path:
        raise SystemExit(f"{a.cmd} needs the recording folder")
    rec = Recording(os.path.expanduser(a.path))
    if a.cmd == "info":
        print(json.dumps(rec.summary(), indent=1), flush=True)
        for t, label in rec.marks:
            print(f"  mark {t - rec.frames[0]['t_wall']:7.1f} s  {label}", flush=True)
        return 0
    a.out = os.path.expanduser(a.out or os.path.join(a.path, "replay"))
    replay(rec, a, ask)
    return 0


if __name__ == "__main__":
    sys.exit(main())
