#!/usr/bin/env python3
"""VLM-goal mission in PX4 SITL + Gazebo (x500_depth): the thesis stack without the SNN.

    # terminal 1 (PX4-Autopilot):  PX4_GZ_WORLD=nd_test make px4_sitl gz_x500_depth
    # terminal 2:
    python scripts/sim_mission.py --instruction "the red box" --ollama http://<workstation-ip>:11434

Loop: Gazebo RGB -> Qwen3-VL (Ollama) box -> distance from the depth image -> NED goal, accepted when 2 of the last 3
answers agree -> turn to it, fly at <= --vmax; every control step the depth guard checks the corridor along the
commanded motion and stops the drone if an obstacle is < --stop-dist m away or < --ttc s from contact.
Logs every step to a JSONL file; --save-frames writes the VLM images with the box drawn.

--no-fly: cameras + VLM + guard only (prints what it would do; PX4 need not be running).
"""
import _path  # noqa: F401
import argparse
import asyncio
import base64
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from neurodrone import mission as M
from neurodrone.telemetry import JsonlLogger
from neurodrone.types import VehicleState, VelocityCmd
from neurodrone.vlm import build_prompt, encode_jpeg, parse_goal


def ask_ollama(url, model, jpeg, instruction, timeout=240, num_ctx=2048, num_gpu=None):
    """One VLM call. num_ctx must be the same in every call (and in scripts/vlm_warmup.py): a different value makes
    Ollama reload the model. The timeout is long on purpose: when the client gives up, Ollama still finishes that
    request, so the next one waits behind it and times out too."""
    body = {"model": model, "stream": False, "think": False, "keep_alive": "60m",
            "options": {"temperature": 0, "num_predict": 128, "num_ctx": num_ctx},
            "messages": [{"role": "user", "content": build_prompt(instruction), "images": [base64.b64encode(jpeg).decode()]}]}
    if num_gpu is not None:  # model layers on the GPU (0 = CPU only); must match the load, or Ollama reloads
        body["options"]["num_gpu"] = int(num_gpu)
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:  # models without a thinking switch reject "think"
        if e.code == 400 and "think" in e.read().decode(errors="replace").lower():
            body.pop("think")
            req = urllib.request.Request(req.full_url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
        else:
            raise
    return (data.get("message") or {}).get("content") or "", time.monotonic() - t0


class VlmWorker(threading.Thread):
    """Every --vlm-period s: latest RGB + depth + the vehicle state at capture time -> goal -> consensus."""

    def __init__(self, a, cams, get_state, log):
        super().__init__(daemon=True)
        self.a, self.cams, self.get_state, self.log = a, cams, get_state, log
        self.cons = M.Consensus(need=a.consensus_need, window=a.consensus_window, tol_m=a.consensus_tol)
        self.goal = None
        self.last = {}
        self.stop = False

    def run(self):
        from neurodrone.gz_camera import DEPTH_CAM, RGB_CAM
        k = 0
        while not self.stop:
            t0 = time.monotonic()
            rgb, age = self.cams.rgb()
            t_capture = t0 - (age if age != float("inf") else 0.0)
            depth, dage = self.cams.depth()
            state = self.get_state()
            if rgb is None or depth is None or state is None:
                time.sleep(0.2)
                continue
            jpeg, sent = encode_jpeg(rgb, max_side=self.a.max_side, quality=90)
            try:
                text, lat = ask_ollama(self.a.ollama, self.a.model, jpeg, self.a.instruction,
                                       timeout=getattr(self.a, "vlm_timeout", 240.0),
                                       num_ctx=getattr(self.a, "num_ctx", 2048),
                                       num_gpu=getattr(self.a, "num_gpu", None))
            except Exception as e:  # noqa: BLE001
                self.log.log("vlm_error", error=repr(e)[:300])
                print(f"[vlm] error: {e!r}", flush=True)
                time.sleep(1.0)
                continue
            p = parse_goal(text, sent, "norm1000")
            goal, box, dist = None, None, None
            if p is not None and p.found and p.w is not None:
                box = [p.x - p.w / 2, p.y - p.h / 2, p.x + p.w / 2, p.y + p.h / 2]
                dist = M.box_depth(depth, DEPTH_CAM, RGB_CAM, box)
                if dist is not None:
                    goal = M.goal_from_box(box, dist, RGB_CAM, state, standoff_m=self.a.standoff)
            acc = self.cons.add(goal)
            if acc is not None:
                self.goal = acc
            self.last = dict(found=bool(p and p.found), box=box, dist=dist, latency_s=lat, accepted=acc is not None,
                             t_capture=t_capture, yaw_capture=getattr(state, "yaw_deg", None))
            self.log.log("vlm", text=text[:300], box=box, dist_m=dist, goal=None if goal is None else goal.tolist(),
                         accepted=None if acc is None else acc.tolist(), latency_s=lat, rgb_age_s=age)
            print(f"[vlm] {lat:.2f} s  found={self.last['found']}  dist={dist if dist is None else round(dist, 2)}  "
                  f"goal={'-' if goal is None else np.round(goal, 1).tolist()}  accepted={acc is not None}", flush=True)
            if self.a.save_frames:
                save_frame(self.a.save_frames, k, rgb, box, text)
            k += 1
            time.sleep(max(0.0, self.a.vlm_period - (time.monotonic() - t0)))


def save_frame(d, k, rgb, box, text):
    from PIL import Image, ImageDraw
    os.makedirs(d, exist_ok=True)
    im = Image.fromarray(rgb)
    if box:
        w, h = im.size
        ImageDraw.Draw(im).rectangle([box[0] * w, box[1] * h, box[2] * w, box[3] * h], outline=(255, 0, 0), width=5)
    im.save(os.path.join(d, f"vlm_{k:04d}.jpg"), quality=85)


async def main_async(a):
    from neurodrone.gz_camera import DEPTH_CAM, GzCameras
    cams = GzCameras(a.rgb_topic, a.depth_topic)
    print(f"cameras: {cams.rgb_topic}, {cams.depth_topic}", flush=True)
    cams.wait()
    print(f"camera subscriptions throttled: {cams.throttled}", flush=True)
    world = cams.rgb_topic.split("/")[2] if cams.rgb_topic.startswith("/world/") else None
    log = JsonlLogger(a.log)
    log.log("start", instruction=a.instruction, model=a.model, world=world, no_fly=a.no_fly,
            settings={k: v for k, v in vars(a).items()})
    latest = {"state": None}

    if a.no_fly:
        latest["state"] = VehicleState(down_m=-a.alt)
        w = VlmWorker(a, cams, lambda: latest["state"], log)
        w.start()
        while True:
            depth, dage = cams.depth()
            g = M.guard(depth, DEPTH_CAM, np.array([1.0, 0, 0]), dage, radius_m=a.radius, stop_dist_m=a.stop_dist, ttc_s=a.ttc)
            print(f"[guard] ahead {g.dist_ahead_m:.2f} m  blocked={g.blocked} {g.reason}", flush=True)
            await asyncio.sleep(1.0)

    from neurodrone.vehicle import Px4Vehicle
    veh = Px4Vehicle(a.connect)
    await veh.connect()
    print("connected to PX4", flush=True)
    await veh.arm_and_takeoff(a.alt)
    latest["state"] = await veh.get_state()
    await veh.start_offboard()
    print(f"OFFBOARD at {a.alt} m; looking for: {a.instruction}", flush=True)
    w = VlmWorker(a, cams, lambda: latest["state"], log)
    w.start()
    t_start = time.monotonic()
    outcome = "error"
    log.log("offboard", alt_m=a.alt)
    if not a.search_yaw:
        searcher = None
    elif a.search == "active":
        searcher = M.ActiveSearcher(max_rate=a.search_yaw)
    else:
        searcher = M.Searcher(max_rate=a.search_yaw)
    avoider = M.Avoider(max_angle_deg=a.avoid_angle, speed=a.avoid_speed)
    dt = 1.0 / a.rate
    last_print = 0.0
    arrived_since = None
    try:
        while True:
            t0 = time.monotonic()
            s = await veh.get_state()
            latest["state"] = s
            cmd = M.steer(w.goal, s, vmax=a.vmax)
            depth, dage = cams.depth()
            if a.no_avoid:
                g = M.guard(depth, DEPTH_CAM, np.array([cmd.vx, cmd.vy, 0.0]), dage, radius_m=a.radius,
                            stop_dist_m=a.stop_dist, ttc_s=a.ttc)
                cmd = M.apply_guard(cmd, g)
            else:
                cmd, g = avoider.step(cmd, depth, DEPTH_CAM, dage, time.monotonic(), radius_m=a.radius,
                                      stop_dist_m=a.stop_dist, ttc_s=a.ttc)
            if cmd.phase == "search" and a.search_yaw:
                yr, note = searcher.step(s.yaw_deg, time.monotonic(), w.last)
                cmd = M.Command(0, 0, 0, yr, "search", note)
            await veh.send_velocity(VelocityCmd(cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate))
            log.log("ctl", phase=cmd.phase, note=cmd.note, vx=cmd.vx, vy=cmd.vy, yaw_rate=cmd.yaw_rate, n=s.north_m, e=s.east_m,
                    d=s.down_m, yaw=s.yaw_deg, guard_dist=g.dist_ahead_m, guard_ttc=g.ttc_s, guard_blocked=g.blocked,
                    goal=None if w.goal is None else w.goal.tolist())
            if cmd.phase == "arrived":
                arrived_since = arrived_since or time.monotonic()
                if time.monotonic() - arrived_since > a.hover_s:
                    print("ARRIVED - landing", flush=True)
                    outcome = "arrived"
                    break
            else:
                arrived_since = None
            if time.monotonic() - t_start > a.max_time:
                print(f"TIMEOUT after {a.max_time:.0f} s - landing", flush=True)
                outcome = "timeout"
                break
            if time.monotonic() - last_print > 1.0:
                last_print = time.monotonic()
                print(f"[ctl] {cmd.phase:8s} pos=({s.north_m:+.1f},{s.east_m:+.1f},{-s.down_m:.1f}) yaw={s.yaw_deg:+.0f} "
                      f"v=({cmd.vx:.2f},{cmd.vy:+.2f}) ahead={g.dist_ahead_m:.1f} m {cmd.note}", flush=True)
            await asyncio.sleep(max(0.0, dt - (time.monotonic() - t0)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("stopped by user - landing", flush=True)
        outcome = "stopped"
    finally:
        s = latest["state"]
        log.log("end", outcome=outcome, duration_s=round(time.monotonic() - t_start, 2),
                n=getattr(s, "north_m", None), e=getattr(s, "east_m", None), d=getattr(s, "down_m", None),
                goal=None if w.goal is None else w.goal.tolist())
        w.stop = True
        try:
            await veh.land()
        finally:
            await veh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", required=True, help='what to fly to, e.g. "the red box"')
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--connect", default="udpin://0.0.0.0:14540")
    ap.add_argument("--rgb-topic", default=None, help="default: found with `gz topic -l` (IMX214 .../image)")
    ap.add_argument("--depth-topic", default="/depth_camera")
    ap.add_argument("--alt", type=float, default=2.0)
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--vlm-period", type=float, default=1.5)
    ap.add_argument("--max-side", type=int, default=896)
    ap.add_argument("--vlm-timeout", type=float, default=240.0, help="s to wait for one VLM answer")
    ap.add_argument("--num-ctx", type=int, default=2048, help="Ollama context length (smaller = less GPU memory)")
    ap.add_argument("--num-gpu", type=int, default=None,
                    help="model layers Ollama puts on the GPU (0 = CPU only; default: Ollama decides)")
    ap.add_argument("--standoff", type=float, default=1.5, help="stop this far before the target (m)")
    ap.add_argument("--radius", type=float, default=0.8,
                    help="guard corridor radius from the drone's centre (m); X500 propeller tips reach ~0.4 m")
    ap.add_argument("--no-avoid", action="store_true", help="only stop at obstacles, never go around")
    ap.add_argument("--avoid-angle", type=float, default=30.0, help="largest go-around angle (deg), inside the camera view")
    ap.add_argument("--avoid-speed", type=float, default=0.5, help="speed while going around (m/s)")
    ap.add_argument("--stop-dist", type=float, default=1.2)
    ap.add_argument("--ttc", type=float, default=2.0)
    ap.add_argument("--consensus-need", type=int, default=2)
    ap.add_argument("--consensus-window", type=int, default=3)
    ap.add_argument("--consensus-tol", type=float, default=1.5)
    ap.add_argument("--search-yaw", type=float, default=0.0, help="while no goal is accepted: turn-stop-look search, turning at up to this many deg/s (0 = just hover)")
    ap.add_argument("--search", choices=["sweep", "active"], default="sweep",
                    help="sweep: fixed list of headings; active: the VLM answers choose the next view (active sensing)")
    ap.add_argument("--hover-s", type=float, default=3.0, help="hover this long at the goal, then land")
    ap.add_argument("--max-time", type=float, default=180.0, help="give up and land after this many s in OFFBOARD")
    ap.add_argument("--no-fly", action="store_true")
    ap.add_argument("--save-frames", default=None)
    ap.add_argument("--log", default=f"mission_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    a = ap.parse_args()
    try:
        asyncio.run(main_async(a))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
