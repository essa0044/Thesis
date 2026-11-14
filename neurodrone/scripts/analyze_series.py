#!/usr/bin/env python3
"""Results table for a series of simulated missions (scripts/sim_mission.py logs, sim/run_series.sh).

    python scripts/analyze_series.py results/series/*.jsonl --worlds sim/worlds --out results/series/summary

Per run (from the JSONL log and the world file the run used):
  outcome          arrived / timeout / stopped / error (from the log's "end" record)
  t_goal_s         OFFBOARD start -> first accepted VLM goal
  t_arrive_s       OFFBOARD start -> first "arrived" step
  standoff_m       final horizontal distance from the drone's centre to the target's surface (aim: 1.5 m)
  right_target     the object closest to the final position is the one named in the instruction
  pole_surface_m   closest horizontal distance from the drone's centre to any pole's surface during the flight
  prop_clear_m     pole_surface_m - PROP_RADIUS_M (X500 V2: arm 0.25 m + half a 13" propeller = ~0.42 m);
                   < 0 means a propeller would have touched the pole
  blocked_s, avoid_s   time spent stopped by the guard / going around
  vlm_n, vlm_found, vlm_first_s, vlm_median_s, vlm_p90_s   VLM calls, share that found the target, latency of the
                   first call (model loading) and of the others
  success          arrived and right_target and 0.5 <= standoff_m <= 2.5 and prop_clear_m >= 0 (or no pole)
Positions: the world files are ENU (x east, y north); PX4 logs NED, so north = y, east = x (drone spawned at 0, 0).
Writes <out>.csv (one row per run), <out>.json (runs + groups) and prints a Markdown table per world x target.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

PROP_RADIUS_M = 0.42
STANDOFF_OK = (0.5, 2.5)


def load_world(path):
    """{name: dict(kind, n, e, half=(hn, he) for boxes | r for cylinders)} for every model except the ground."""
    s = open(path).read()
    out = {}
    for m in re.finditer(r'<model name="([^"]+)">(.*?)</model>', s, re.S):
        name, body = m.group(1), m.group(2)
        if "ground" in name:
            continue
        pose = re.search(r"<pose>([^<]+)</pose>", body)
        if not pose:
            continue
        x, y = (float(v) for v in pose.group(1).split()[:2])
        box = re.search(r"<box>\s*<size>([^<]+)</size>", body)
        cyl = re.search(r"<cylinder>\s*<radius>([^<]+)</radius>", body)
        if box:
            sx, sy = (float(v) for v in box.group(1).split()[:2])
            out[name] = dict(kind="box", n=y, e=x, hn=sy / 2, he=sx / 2)
        elif cyl:
            out[name] = dict(kind="cyl", n=y, e=x, r=float(cyl.group(1)))
    return out


def surface_dist(obj, n, e):
    """Horizontal distance from (n, e) to the object's surface (0 inside)."""
    if obj["kind"] == "cyl":
        return max(0.0, math.hypot(n - obj["n"], e - obj["e"]) - obj["r"])
    dn = max(0.0, abs(n - obj["n"]) - obj["hn"])
    de = max(0.0, abs(e - obj["e"]) - obj["he"])
    return math.hypot(dn, de)


def target_name(instruction, objects):
    key = re.sub(r"^(the|a|an)\s+", "", instruction.strip().lower()).replace(" ", "_")
    return key if key in objects else None


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def analyze_run(path, worlds_dir):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    start = next((r for r in rows if r.get("kind") == "start"), {})
    end = next((r for r in rows if r.get("kind") == "end"), None)
    off = next((r for r in rows if r.get("kind") == "offboard"), None)
    ctl = [r for r in rows if r.get("kind") == "ctl"]
    vlm = [r for r in rows if r.get("kind") == "vlm"]
    world = start.get("world")
    instr = start.get("instruction") or (start.get("settings") or {}).get("instruction", "")
    res = dict(run=os.path.basename(path), world=world, instruction=instr,
               search=(start.get("settings") or {}).get("search", "sweep"),
               outcome=end["outcome"] if end else "error")
    t0 = off["t"] if off else (ctl[0]["t"] if ctl else None)
    wpath = os.path.join(worlds_dir, f"{world}.sdf") if world else None
    objects = load_world(wpath) if wpath and os.path.exists(wpath) else {}
    poles = {k: v for k, v in objects.items() if "pole" in k}
    targets = {k: v for k, v in objects.items() if "pole" not in k}
    tname = target_name(instr, objects)
    acc = [r for r in vlm if r.get("accepted")]
    if off is None:  # never reached OFFBOARD: takeoff or arming failed in the simulator, not the stack
        res["outcome"] = "sim_error"
    res["t_goal_s"] = round(acc[0]["t"] - t0, 1) if acc and t0 is not None else None
    arr = [r for r in ctl if r.get("phase") == "arrived"]
    res["t_arrive_s"] = round(arr[0]["t"] - t0, 1) if arr and t0 is not None else None
    res["duration_s"] = end.get("duration_s") if end else None
    last = None
    if end and end.get("n") is not None:
        last = (end["n"], end["e"])
    elif ctl:
        last = (ctl[-1]["n"], ctl[-1]["e"])
    if last and tname:
        res["standoff_m"] = round(surface_dist(objects[tname], *last), 2)
        closest = min(targets, key=lambda k: surface_dist(targets[k], *last)) if targets else None
        res["right_target"] = closest == tname
    else:
        res["standoff_m"], res["right_target"] = None, None
    if poles and ctl:
        d = min(surface_dist(p, r["n"], r["e"]) for r in ctl for p in poles.values())
        res["pole_surface_m"] = round(d, 2)
        res["prop_clear_m"] = round(d - PROP_RADIUS_M, 2)
    else:
        res["pole_surface_m"] = res["prop_clear_m"] = None
    dt = statistics.median([b["t"] - a["t"] for a, b in zip(ctl, ctl[1:])]) if len(ctl) > 1 else 0.05
    res["blocked_s"] = round(sum(r.get("phase") == "blocked" for r in ctl) * dt, 1)
    res["avoid_s"] = round(sum(r.get("phase") == "avoid" for r in ctl) * dt, 1)
    lat = [r["latency_s"] for r in vlm if r.get("latency_s") is not None]
    res["vlm_n"] = len(vlm)
    res["vlm_found"] = round(sum(r.get("box") is not None for r in vlm) / len(vlm), 2) if vlm else None
    res["vlm_first_s"] = round(lat[0], 2) if lat else None
    rest = lat[1:]
    res["vlm_median_s"] = round(statistics.median(rest), 2) if rest else None
    res["vlm_p90_s"] = round(pct(rest, 0.9), 2) if rest else None
    ok_standoff = res["standoff_m"] is not None and STANDOFF_OK[0] <= res["standoff_m"] <= STANDOFF_OK[1]
    no_hit = res["prop_clear_m"] is None or res["prop_clear_m"] >= 0
    res["success"] = bool(res["outcome"] == "arrived" and res["right_target"] and ok_standoff and no_hit)
    return res


def summarize(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["world"], r["instruction"], r["search"]), []).append(r)
    out = []
    for (world, instr, search), rs_all in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rs = [r for r in rs_all if r["outcome"] != "sim_error"]
        if not rs:
            out.append(dict(world=world, instruction=instr, search=search, runs=0, success=0, arrived=0, right_target=0,
                            t_arrive_median_s=None, standoff_median_m=None, prop_clear_min_m=None,
                            avoid_median_s=None, vlm_median_s=None, vlm_calls_median=None,
                            sim_errors=len(rs_all)))
            continue
        def med(k):
            v = [r[k] for r in rs if r[k] is not None]
            return round(statistics.median(v), 2) if v else None

        def mn(k):
            v = [r[k] for r in rs if r[k] is not None]
            return round(min(v), 2) if v else None
        lat = [r["vlm_median_s"] for r in rs if r["vlm_median_s"] is not None]
        out.append(dict(world=world, instruction=instr, search=search, runs=len(rs), success=sum(r["success"] for r in rs),
                        arrived=sum(r["outcome"] == "arrived" for r in rs),
                        right_target=sum(bool(r["right_target"]) for r in rs),
                        t_arrive_median_s=med("t_arrive_s"), standoff_median_m=med("standoff_m"),
                        prop_clear_min_m=mn("prop_clear_m"), avoid_median_s=med("avoid_s"),
                        vlm_median_s=round(statistics.median(lat), 2) if lat else None,
                        vlm_calls_median=med("vlm_n"), sim_errors=len(rs_all) - len(rs)))
    return out


def markdown(groups):
    head = ["world", "target", "search", "runs", "success", "arrived", "right target", "time to goal (s, median)",
            "stand-off (m, median)", "min prop clearance (m)", "going around (s, median)", "VLM latency (s, median)",
            "VLM calls (median)", "sim errors (not counted)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for g in groups:
        f = lambda v: "–" if v is None else str(v)  # noqa: E731
        lines.append("| " + " | ".join([f(g["world"]), f(g["instruction"]), f(g["search"]), f(g["runs"]),
                                         f"{g['success']}/{g['runs']}", f"{g['arrived']}/{g['runs']}",
                                         f"{g['right_target']}/{g['runs']}", f(g["t_arrive_median_s"]),
                                         f(g["standoff_median_m"]), f(g["prop_clear_min_m"]),
                                         f(g["avoid_median_s"]), f(g["vlm_median_s"]), f(g["vlm_calls_median"]),
                                         f(g["sim_errors"])]) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="mission JSONL logs (globs allowed)")
    ap.add_argument("--worlds", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim", "worlds"))
    ap.add_argument("--out", default=None, help="write <out>.csv and <out>.json")
    a = ap.parse_args(argv)
    paths = sorted({p for g in a.logs for p in glob.glob(g)})
    runs = [analyze_run(p, a.worlds) for p in paths]
    runs = [r for r in runs if r["world"]]  # --no-fly or unfinished logs without a start record are skipped
    groups = summarize(runs)
    print(markdown(groups))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        if runs:
            with open(a.out + ".csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
                w.writeheader()
                w.writerows(runs)
        with open(a.out + ".json", "w") as f:
            json.dump(dict(prop_radius_m=PROP_RADIUS_M, standoff_ok_m=STANDOFF_OK, runs=runs, groups=groups), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
