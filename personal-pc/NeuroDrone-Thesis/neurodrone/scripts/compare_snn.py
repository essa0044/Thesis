#!/usr/bin/env python3
"""Compare the two looming detectors on IDENTICAL synthetic event streams:
the fully spiking detector (snn.impl = spiking, the default) and the hybrid baseline (snn.impl = hybrid).

    python scripts/compare_snn.py                    # everything, ~5-10 min
    python scripts/compare_snn.py --no-closed-loop   # skip the closed-loop flights
    python scripts/compare_snn.py --out results/snn_compare.json

Reports, per detector:
  * detection: time-to-contact (s) at the first alarm, flying at 1.5 m/s at five obstacles
  * false alarms: worst "frontal urgency" in seven scenes with nothing on a collision course (1.0 = alarm)
  * packet cutting: first-alarm time-to-contact when the same stream is cut into packets differently
  * CPU time per 10 ms packet (this computer, not the Jetson)
  * spiking detector only: spikes per layer and synaptic operations per second of flight
  * closed loop: outcome of every scenario in neurodrone/closedloop.py
Synthetic simulator only: these numbers check logic and compare the two designs, they do not
predict performance on a real sensor.
"""
import _path  # noqa: F401
import argparse
import json
import math
import time

import numpy as np

from neurodrone.closedloop import ClosedLoopSim, PacketCutter, default_scenarios
from neurodrone.config import SNNConfig
from neurodrone.reflex import make_snn
from neurodrone.sim import Box, Camera, DVSSimulator, PointMassDrone, World
from neurodrone.types import VelocityCmd

SPEED = 1.5
DETECT = {"wall 3 m": [Box(x=14, y=0, w=3, h=3)],
          "box 1.5 m": [Box(x=14, y=0, w=1.5, h=1.5)],
          "blank wall 3 m": [Box(x=14, y=0, w=3, h=3, texture="blank")],
          "pole 0.4 m": [Box(x=14, y=0, w=0.4, h=3)],
          "pole 0.15 m": [Box(x=14, y=0, w=0.15, h=3)]}
FALSE = {"hover + 3 deg shake, object ahead": ([Box(x=14, y=0, w=2, h=2)], dict(speed=0.0, jitter_deg=3.0, dur=5)),
         "hover + 3 deg shake, object to the side": ([Box(x=14, y=2, w=2, h=2)], dict(speed=0.0, jitter_deg=3.0, dur=6)),
         "forward + 5 deg shake, empty": ([], dict(jitter_deg=5.0, dur=8)),
         "forward + 5 deg shake + 10x noise, empty": ([], dict(jitter_deg=5.0, noise_hz=3.0, dur=6)),
         "turning 30 deg/s, empty": ([], dict(speed=0.0, yaw_rate_deg_s=30.0, dur=6)),
         "forward + 10x noise, empty": ([], dict(noise_hz=3.0, dur=8)),
         "object passing on the left": ([Box(x=14, y=-3.0, w=1.5, h=3)], dict(dur=8))}
IMPLS = ("spiking", "hybrid")


def record(boxes, speed=SPEED, dur=8.0, jitter_deg=0.0, yaw_rate_deg_s=0.0, noise_hz=0.3, seed=0):
    """Sensor frames (2 ms) of a constant-speed flight, with the obstacle clearance after each frame."""
    cam = Camera()
    world = World(boxes, cam)
    dvs = DVSSimulator(cam.width, cam.height, noise_hz=noise_hz, seed=seed)
    drone = PointMassDrone()
    drone.v[:] = [speed, 0.0, 0.0]
    fu, t_us, out = ClosedLoopSim.FRAME_US, 0, []
    while t_us < dur * 1e6:
        x, y, z, yaw = drone.pose
        yc = yaw + math.radians(jitter_deg) * math.sin(2 * math.pi * t_us / 1e6)
        ev = dvs.step(world.render((x, y, z, yc)), t_us, t_us + fu)
        t_us += fu
        drone.step(VelocityCmd(vx=speed, yaw_rate=yaw_rate_deg_s), fu / 1e6)
        out.append((ev, world.clearance(drone.pose)))
    return out


def run(impl, frames, aligned=False, seed=0):
    """Feed one recorded stream through a detector. Returns (clearance at first alarm, max frontal,
    CPU ms per packet, detector)."""
    cam = Camera()
    det = make_snn(cam.width, cam.height, SNNConfig(impl=impl))
    cutter = (PacketCutter(ClosedLoopSim.PACKET_US, phase_us=0, jitter_us=0, seed=seed) if aligned
              else PacketCutter(ClosedLoopSim.PACKET_US, seed=seed))
    first, mx, cpu = None, 0.0, []
    for ev, clr in frames:
        for pkt in cutter.push(ev, ev.t_end_us):
            rep = det.process(pkt)
            cpu.append(rep.proc_s * 1e3)
            mx = max(mx, rep.frontal)
            if rep.alarm and first is None:
                first = clr
    return first, mx, cpu, det


def ttc(clr):
    return None if clr is None else round(clr / SPEED, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-closed-loop", action="store_true")
    ap.add_argument("--out", help="write all numbers to this JSON file")
    a = ap.parse_args()
    res = {i: {"detect_ttc_s": {}, "false_alarm_max": {}, "cut_ttc_s": {}, "cpu_ms": [], "closed_loop": {}} for i in IMPLS}
    spk = {"seconds": 0.0, "events": 0, "layer1_spikes": 0, "layer2_spikes": 0, "layer3_spikes": 0, "synops": 0}
    neurons = None

    print("1/4 detection (time-to-contact at first alarm, s; higher = earlier warning)", flush=True)
    for name, boxes in DETECT.items():
        frames = record(boxes)
        for i in IMPLS:
            first, _, cpu, det = run(i, frames)
            res[i]["detect_ttc_s"][name] = ttc(first)
            res[i]["cpu_ms"] += cpu
            if i == "spiking":
                neurons = det.neuron_counts
                spk["seconds"] += frames[-1][0].t_end_us / 1e6
                for k in ("events", "layer1_spikes", "layer2_spikes", "layer3_spikes", "synops"):
                    spk[k] += det.stats[k]
        print(f"   {name:16s} " + "  ".join(f"{i}: {res[i]['detect_ttc_s'][name] or 'miss'}" for i in IMPLS), flush=True)
        if name == "wall 3 m":  # packet-cutting sensitivity on the same stream
            for i in IMPLS:
                res[i]["cut_ttc_s"] = {"jittered (default)": res[i]["detect_ttc_s"][name],
                                       "aligned with sensor frames": ttc(run(i, frames, aligned=True)[0]),
                                       "jittered, other seed": ttc(run(i, frames, seed=7)[0])}

    print("2/4 false alarms (worst frontal urgency; 1.0 = alarm)", flush=True)
    for name, (boxes, kw) in FALSE.items():
        frames = record(boxes, **kw)
        for i in IMPLS:
            _, mx, cpu, _ = run(i, frames)
            res[i]["false_alarm_max"][name] = round(mx, 2)
            res[i]["cpu_ms"] += cpu
        print(f"   {name:42s} " + "  ".join(f"{i}: {res[i]['false_alarm_max'][name]:.2f}" for i in IMPLS), flush=True)

    print("3/4 packet cutting (wall 3 m, time-to-contact at first alarm, s)")
    for i in IMPLS:
        print(f"   {i:8s} " + "  ".join(f"{k}: {v}" for k, v in res[i]["cut_ttc_s"].items()))

    if not a.no_closed_loop:
        print("4/4 closed loop (whole system flying, oracle VLM)", flush=True)
        for sc in default_scenarios():
            row = []
            for i in IMPLS:
                r = ClosedLoopSim(sc, SNNConfig(impl=i)).run()
                out = "COLLISION" if r.collided else ("arrived" if r.arrived else "-")
                res[i]["closed_loop"][sc.name] = {"result": out, "min_clearance_m": round(r.min_clearance, 2),
                                                  "first_alarm_clearance_m": None if r.first_alarm_clearance is None
                                                  else round(r.first_alarm_clearance, 1),
                                                  "sim_time_s": round(r.t_end, 1), "dodges": int(r.modes.get("AVOID", 0) > 0)}
                row.append(f"{i}: {out} (min clearance {r.min_clearance:.1f} m)")
            print(f"   {sc.name:22s} " + "  ".join(row), flush=True)

    print("\nsummary")
    for i in IMPLS:
        c = np.array(res[i]["cpu_ms"])
        res[i]["cpu_ms"] = {"mean": round(float(c.mean()), 3), "p95": round(float(np.percentile(c, 95)), 3),
                            "max": round(float(c.max()), 3)}
        print(f"   {i:8s} CPU per packet: mean {c.mean():.2f} ms, p95 {np.percentile(c, 95):.2f} ms (this computer)")
    s = spk["seconds"]
    res["spiking"]["activity_per_s"] = {k: round(spk[k] / s) for k in spk if k != "seconds"}
    res["spiking"]["neurons"] = neurons
    print(f"   spiking  neurons: {neurons}")
    print("   spiking  per second of approach flight: " +
          ", ".join(f"{k} {v:,}" for k, v in res["spiking"]["activity_per_s"].items()))
    if a.out:
        import os
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"   written to {a.out}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"   ({time.time() - t0:.0f} s)")
