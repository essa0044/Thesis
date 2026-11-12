#!/usr/bin/env python3
"""Closed-loop check of the depth guard against PX4's built-in simulator (SIH), no Gazebo, no VLM.

    # terminal 1: PX4 SITL with the SIH quadrotor (see sim/run_sih.sh)
    # terminal 2:
    python scripts/sim_sih_guard.py --out results/sih

PX4 flies the real flight stack (estimator, controllers, OFFBOARD velocity setpoints over MAVSDK). SIH has no cameras,
so the depth image is rendered here from the drone's pose in a small world of axis-aligned boxes (ray casting, with
the RealSense-like noise of scripts/eval_tartanair.py: sigma = 0.01 * Z^2, 5 % dropout) and fed to
neurodrone.mission.guard, the same function the mission script uses. Every case is flown with several seeds.

Cases (the drone starts at 2 m, flies north at --speed m/s towards a point 8 m away; obstacles are placed relative to
where it starts):
  wall    : a wall across the path, near face --wall-dist m ahead     -> must stop before it, never touch it
  free    : nothing in the way                                          -> must not stop, must reach the 8 m point
  beside  : a pillar 2.5 m to the side of the path, same distance       -> must not stop (outside the corridor)
Writes <out>/sih_guard_summary.json (settings, versions, one row per run) and <out>/<case>_<seed>.jsonl step logs.
"""
import _path  # noqa: F401
import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

from neurodrone import mission as M
from neurodrone.types import VelocityCmd

CAM = M.Pinhole(160, 120, 1.274)  # depth camera of x500_depth (hfov 1.274 rad), 1/4 resolution: enough for the guard
CAM_OFFSET_FRD = np.array([0.12, -0.03, -0.24])  # as in mission.goal_from_box


def make_world(case, wall_dist, start_ned):
    """Boxes (centre NED, half size) relative to the start position."""
    n0, e0, d0 = start_ned
    if case == "wall":  # 6 m wide, 1 m thick, from the ground to 4 m
        return [((n0 + wall_dist + 0.5, e0, -1.5), (0.5, 3.0, 2.5))]
    if case == "beside":
        return [((n0 + wall_dist + 0.5, e0 + 2.5, -1.5), (0.5, 0.5, 2.5))]
    return []


def render_depth(state, boxes, cam, rng, noise_c=0.01, dropout=0.05, dmin=0.3, dmax=10.0):
    """Depth image (m, NaN = no return) seen by the forward camera: boxes + the ground plane, then sensor noise."""
    R = M.R_ned_from_frd(state.roll_deg, state.pitch_deg, state.yaw_deg)
    pos = np.array([state.north_m, state.east_m, state.down_m]) + R @ CAM_OFFSET_FRD
    u, v = np.meshgrid(np.arange(cam.width) + 0.5, np.arange(cam.height) + 0.5)
    rays = np.stack([np.ones_like(u), (u - cam.width / 2) / cam.f, (v - cam.height / 2) / cam.f], -1)  # FRD, x = 1
    d = rays @ R.T  # NED direction per pixel, z of the FRD ray (x-component) = 1 -> depth = t
    best = np.full(u.shape, np.inf)
    for c, h in boxes:
        lo, hi = np.array(c) - h, np.array(c) + h
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - pos) / d, (hi - pos) / d
        t1, t2 = np.where(np.isnan(t1), -np.inf, t1), np.where(np.isnan(t2), np.inf, t2)
        tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
        hit = (tf >= np.maximum(tn, 0)) & (tn > 0)
        best = np.where(hit, np.minimum(best, tn), best)
    with np.errstate(divide="ignore", invalid="ignore"):  # ground at down = 0
        tg = np.where(d[..., 2] > 1e-6, (0.0 - pos[2]) / d[..., 2], np.inf)
    best = np.minimum(best, np.where(tg > 0, tg, np.inf))
    z = best.copy()  # t is the distance along the ray scaled so the x-component is 1 -> already the forward depth
    z[(z < dmin) | (z > dmax)] = np.nan
    z = z + rng.normal(0.0, 1.0, z.shape) * noise_c * z * z
    z[rng.random(z.shape) < dropout] = np.nan
    return z.astype(np.float32)


async def run_case(veh, case, seed, a, outdir):
    rng = np.random.default_rng(seed)
    await veh.arm_and_takeoff(a.alt)
    s = await veh.get_state()
    start = (s.north_m, s.east_m, s.down_m)
    yaw0 = s.yaw_deg
    if abs(yaw0) > 10:
        print(f"  warning: start yaw {yaw0:.0f} deg, the world is built along north", flush=True)
    boxes = make_world(case, a.wall_dist, start)
    target_n = start[0] + a.goal_dist
    await veh.start_offboard()
    log = open(os.path.join(outdir, f"{case}_{seed}.jsonl"), "w")
    dt = 1.0 / a.rate
    t0 = time.monotonic()
    stopped_at, first_block, first_block_dist, blocked_steps = None, None, None, 0
    min_gap = math.inf  # smallest distance between the camera and the wall's near face
    wall_face = start[0] + a.wall_dist
    outcome = "timeout"
    while time.monotonic() - t0 < a.max_time:
        t_step = time.monotonic()
        s = await veh.get_state()
        depth = render_depth(s, boxes, CAM, rng)
        v = np.array([a.speed, 0.0, 0.0])
        g = M.guard(depth, CAM, v, 0.0, radius_m=a.radius, stop_dist_m=a.stop_dist, ttc_s=a.ttc)
        cmd = VelocityCmd(0.0, 0.0, 0.0, 0.0) if g.blocked else VelocityCmd(a.speed, 0.0, 0.0, 0.0)
        # hold the start altitude (vz in NED, + = down) and the heading
        cmd.vz = float(np.clip(0.8 * (start[2] - s.down_m), -0.5, 0.5))
        # PX4 does not hold the heading in OFFBOARD body-velocity mode: P control on the yaw error (deg/s, + = clockwise)
        err = (s.yaw_deg - yaw0 + 180.0) % 360.0 - 180.0
        cmd.yaw_rate = float(np.clip(-1.5 * err, -30.0, 30.0))
        await veh.send_velocity(cmd)
        t = time.monotonic() - t0
        if case == "wall":
            min_gap = min(min_gap, wall_face - s.north_m)
        if g.blocked:
            blocked_steps += 1
            if first_block is None:
                first_block, first_block_dist = t, wall_face - s.north_m
        log.write(json.dumps(dict(t=round(t, 3), n=s.north_m, e=s.east_m, d=s.down_m, yaw=s.yaw_deg,
                                  guard_dist=None if not math.isfinite(g.dist_ahead_m) else g.dist_ahead_m,
                                  blocked=g.blocked, vx_cmd=cmd.vx)) + "\n")
        if case == "wall" and first_block is not None and t - first_block > 3.0:
            outcome = "stopped"
            break
        if case != "wall" and s.north_m >= target_n:
            outcome = "reached"
            break
        await asyncio.sleep(max(0.0, dt - (time.monotonic() - t_step)))
    log.close()
    s = await veh.get_state()
    for _ in range(40):  # brake and hold
        await veh.send_velocity(VelocityCmd(0, 0, 0, 0))
        await asyncio.sleep(0.05)
    s = await veh.get_state()
    await veh.land()
    res = dict(case=case, seed=seed, outcome=outcome, start_ned=[round(x, 3) for x in start],
               final_n=round(s.north_m - start[0], 3), final_e=round(s.east_m - start[1], 3),
               blocked_steps=blocked_steps, first_block_s=None if first_block is None else round(first_block, 2),
               gap_at_first_block_m=None if first_block_dist is None else round(first_block_dist, 2),
               final_gap_to_wall_m=round(wall_face - s.north_m, 2) if case == "wall" else None,
               min_gap_to_wall_m=round(min_gap, 2) if case == "wall" else None)
    if case == "wall":
        res["pass"] = outcome == "stopped" and min_gap > 0.3
    elif case == "free":
        res["pass"] = outcome == "reached" and blocked_steps == 0
    else:
        res["pass"] = outcome == "reached" and blocked_steps == 0
    return res


def versions():
    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import mavsdk
        v["mavsdk"] = getattr(mavsdk, "__version__", "?")
    except Exception:  # noqa: BLE001
        pass
    try:
        from importlib.metadata import version
        v["mavsdk_pkg"] = version("mavsdk")
    except Exception:  # noqa: BLE001
        pass
    px4 = os.environ.get("PX4_DIR")
    if px4 and os.path.isdir(os.path.join(px4, ".git")):
        v["px4"] = subprocess.run(["git", "-C", px4, "describe", "--tags", "--always"], capture_output=True, text=True).stdout.strip()
        v["px4_commit"] = subprocess.run(["git", "-C", px4, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    try:
        v["neurodrone_commit"] = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    return v


async def main_async(a):
    from neurodrone.vehicle import Px4Vehicle
    os.makedirs(a.out, exist_ok=True)
    veh = Px4Vehicle(a.connect)
    await veh.connect()
    print("connected to PX4", flush=True)
    rows = []
    try:
        for case in a.cases:
            for seed in range(a.seeds):
                print(f"case {case} seed {seed}", flush=True)
                r = await run_case(veh, case, seed, a, a.out)
                rows.append(r)
                print("  ", r, flush=True)
                await asyncio.sleep(4.0)  # let it land
                with open(os.path.join(a.out, "sih_guard_summary.json"), "w") as f:
                    json.dump(dict(settings=vars(a), versions=versions(), runs=rows), f, indent=1)
    finally:
        await veh.close()
    n = len(rows)
    print(f"passed {sum(r['pass'] for r in rows)} of {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="udpin://0.0.0.0:14540")
    ap.add_argument("--out", default="results/sih")
    ap.add_argument("--cases", nargs="+", default=["wall", "free", "beside"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--alt", type=float, default=2.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--wall-dist", type=float, default=8.0, help="distance from the start to the wall's near face (m)")
    ap.add_argument("--goal-dist", type=float, default=6.0, help="free / beside: distance to fly (m)")
    ap.add_argument("--radius", type=float, default=0.6)
    ap.add_argument("--stop-dist", type=float, default=1.2)
    ap.add_argument("--ttc", type=float, default=2.0)
    ap.add_argument("--max-time", type=float, default=30.0)
    a = ap.parse_args()
    asyncio.run(main_async(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
