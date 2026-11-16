#!/usr/bin/env python3
"""Stage 6 of the deployment plan: is PX4's position from the visual-inertial odometry good enough to fly on, and
does it stay good while the VLM runs on the same Jetson (thesis Chapter 6)?

    # VIO running (jetson/README.md, stage 6), the Pixhawk on USB, propellers OFF for `drift`
    python scripts/vio_check.py drift --laps 3 --out ~/nd_results/stage6/drift
    python scripts/vio_check.py log --duration 120 --out ~/nd_results/stage6/hold            # hover / hold, pilot flies
    python scripts/vio_check.py log --duration 300 --vlm-load --ros-hz-container isaac_ros_dev-aarch64-container \
        --out ~/nd_results/stage6/hold_vlm                                                   # the same with the VLM busy

drift  Put the drone on a floor mark and press Enter; carry it around the room (a few metres, turns included) and
       back onto the same mark; press Enter again. The difference of PX4's position estimate between the two (2 s
       averages) is the drift of that lap. There is no motion capture: the floor mark is the ground truth.
log    Records PX4's position and attitude at 20 Hz for --duration s while the pilot hovers or holds position:
       spread of the position (how still it holds), how often the estimate updates, gaps, and PX4's
       "local position ok" flag. --vlm-load: meanwhile the VLM answers non-stop on new pictures (the stage-6 pass
       check: VIO keeps >= 30 Hz and its tracking while the VLM runs). --ros-hz-container: also measure the VIO's own
       output rate with `ros2 topic hz` inside that Docker container.
"""
import _path  # noqa: F401
import argparse
import asyncio
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time

import numpy as np


# ----------------------------------------------------------------------------- pure helpers (tested)
def mean_pose(samples):
    a = np.array([[s["n"], s["e"], s["d"]] for s in samples])
    yaw = math.degrees(math.atan2(np.mean([math.sin(math.radians(s["yaw"])) for s in samples]),
                                  np.mean([math.cos(math.radians(s["yaw"])) for s in samples])))
    return a.mean(axis=0), yaw


def drift(start, end):
    p0, y0 = mean_pose(start)
    p1, y1 = mean_pose(end)
    d = p1 - p0
    dyaw = (y1 - y0 + 180) % 360 - 180
    return dict(horizontal_m=float(math.hypot(d[0], d[1])), vertical_m=float(d[2]), yaw_deg=float(dyaw),
                dn=float(d[0]), de=float(d[1]))


def log_summary(recs):
    if len(recs) < 2:
        return dict(n=len(recs))
    p = np.array([[r["n"], r["e"], r["d"]] for r in recs])
    t = np.array([r["t"] for r in recs])
    changed = [i for i in range(1, len(recs)) if (recs[i]["n"], recs[i]["e"], recs[i]["d"]) != (recs[i - 1]["n"], recs[i - 1]["e"], recs[i - 1]["d"])]
    upd_t = t[changed] if changed else np.array([])
    gaps = np.diff(upd_t) if len(upd_t) > 1 else np.array([])
    c = p.mean(axis=0)
    hor = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
    ok = [r["local_ok"] for r in recs if r.get("local_ok") is not None]
    return dict(n=len(recs), duration_s=float(t[-1] - t[0]), horizontal_spread_rms_m=float(np.sqrt((hor ** 2).mean())),
                horizontal_max_m=float(hor.max()), vertical_spread_m=float(p[:, 2].std()),
                position_update_hz=float(len(upd_t) / (t[-1] - t[0])) if len(upd_t) else 0.0,
                longest_gap_s=float(gaps.max()) if len(gaps) else None,
                local_position_ok_share=(sum(ok) / len(ok)) if ok else None)


_HZ = re.compile(r"average rate:\s*([0-9.]+)")


def parse_ros_hz(text):
    return [float(x) for x in _HZ.findall(text)]


def ros_hz(container, topic, seconds, domain_id=1):
    """`ros2 topic hz` inside the VIO's Docker container for `seconds`; returns the reported average rates."""
    cmd = ["docker", "exec", container, "bash", "-lc",
           f"source /opt/ros/humble/setup.bash && ROS_DOMAIN_ID={domain_id} timeout {int(seconds)} ros2 topic hz {topic}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 30)
        return parse_ros_hz(r.stdout + r.stderr)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"ros2 topic hz failed: {e!r}", flush=True)
        return []


# ----------------------------------------------------------------------------- PX4 side
class Px4Pose:
    """Position, attitude and the local-position-ok flag from PX4 over MAVSDK (the Pixhawk on USB)."""

    def __init__(self, connect):
        from neurodrone.vehicle import Px4Vehicle
        self.veh, self.local_ok = Px4Vehicle(connect), None

    async def start(self):
        await self.veh.connect()
        t = self.veh._sys.telemetry

        async def health():
            try:
                async for h in t.health():
                    self.local_ok = bool(h.is_local_position_ok)
            except Exception:  # noqa: BLE001
                pass
        self._h = asyncio.create_task(health())

    async def sample(self):
        s = await self.veh.get_state()
        return dict(t=time.time(), n=s.north_m, e=s.east_m, d=s.down_m, yaw=s.yaw_deg, roll=s.roll_deg,
                    pitch=s.pitch_deg, local_ok=self.local_ok)

    async def close(self):
        self._h.cancel()
        await self.veh.close()


async def collect(pose, seconds, rate=20.0, fh=None):
    out, t_end = [], time.monotonic() + seconds
    while time.monotonic() < t_end:
        r = await pose.sample()
        out.append(r)
        if fh:
            fh.write(json.dumps(r) + "\n")
        await asyncio.sleep(1.0 / rate)
    return out


def vlm_load_thread(a, stop, lat):
    from jetson_vlm_bench import synthetic_picture
    from neurodrone import ollama as O
    from neurodrone.vlm import encode_jpeg

    def run():
        k = 0
        while not stop.is_set():
            k += 1
            jpeg, _ = encode_jpeg(synthetic_picture(int(time.time() * 1000) % 1_000_000 + k), max_side=896, quality=90)
            try:
                lat.append(O.chat(a.ollama, a.model, jpeg, "the red box", num_ctx=a.num_ctx)["latency_s"])
            except Exception as e:  # noqa: BLE001
                print(f"vlm error {e!r}", flush=True)
                time.sleep(1)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


# ----------------------------------------------------------------------------- commands
async def cmd_drift(a, pose, ask=input):
    laps = []
    with open(os.path.join(a.out, "drift.jsonl"), "w") as fh:
        for lap in range(1, a.laps + 1):
            await asyncio.to_thread(ask, f"lap {lap}: drone on the floor mark, still - Enter: ")
            start = await collect(pose, a.average_s, fh=fh)
            await asyncio.to_thread(ask, f"lap {lap}: carry it around and put it back on the same mark - Enter: ")
            end = await collect(pose, a.average_s, fh=fh)
            d = drift(start, end)
            d["lap"] = lap
            d["local_ok_end"] = end[-1].get("local_ok")
            laps.append(d)
            print(f"  lap {lap}: horizontal drift {d['horizontal_m']:.3f} m, vertical {d['vertical_m']:+.3f} m, "
                  f"yaw {d['yaw_deg']:+.1f} deg", flush=True)
    s = dict(laps=laps, horizontal_m_median=statistics.median(l["horizontal_m"] for l in laps) if laps else None)
    json.dump(s, open(os.path.join(a.out, "drift.json"), "w"), indent=1)
    return s


async def cmd_log(a, pose):
    stop, lat, th = threading.Event(), [], None
    if a.vlm_load:
        th = vlm_load_thread(a, stop, lat)
    hz_box = {}
    if a.ros_hz_container:
        hz_th = threading.Thread(target=lambda: hz_box.update(rates=ros_hz(a.ros_hz_container, a.ros_topic, a.duration)),
                                 daemon=True)
        hz_th.start()
    with open(os.path.join(a.out, "log.jsonl"), "w") as fh:
        recs = await collect(pose, a.duration, fh=fh)
    stop.set()
    s = log_summary(recs)
    s.update(vlm_load=a.vlm_load, vlm_answers=len(lat), vlm_median_s=statistics.median(lat) if lat else None)
    if a.ros_hz_container:
        hz_th.join(timeout=60)
        rates = hz_box.get("rates") or []
        s.update(vio_topic=a.ros_topic, vio_rate_hz_min=min(rates) if rates else None,
                 vio_rate_hz_last=rates[-1] if rates else None, vio_rate_ok=(min(rates) >= a.min_vio_hz) if rates else None)
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(s, indent=1), flush=True)
    return s


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["drift", "log"])
    ap.add_argument("--out", default="stage6")
    ap.add_argument("--connect", default="serial:///dev/ttyACM0:57600")
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--average-s", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--vlm-load", action="store_true")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--ros-hz-container", default=None, help="Docker container running the VIO (Isaac ROS)")
    ap.add_argument("--ros-topic", default="/visual_slam/tracking/vo_pose_covariance")
    ap.add_argument("--min-vio-hz", type=float, default=30.0)
    return ap


async def amain(a, pose=None, ask=input):
    own = pose is None
    if own:
        pose = Px4Pose(a.connect)
        await pose.start()
    try:
        if a.cmd == "drift":
            return await cmd_drift(a, pose, ask)
        return await cmd_log(a, pose)
    finally:
        if own:
            await pose.close()


def main(argv=None, pose=None, ask=input):
    a = build_parser().parse_args(argv)
    a.out = os.path.expanduser(a.out)
    os.makedirs(a.out, exist_ok=True)
    asyncio.run(amain(a, pose, ask))
    return 0


if __name__ == "__main__":
    sys.exit(main())
