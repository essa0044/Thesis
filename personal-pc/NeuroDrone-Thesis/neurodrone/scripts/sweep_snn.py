#!/usr/bin/env python3
"""Sweep looming-detector thresholds on the synthetic simulator: detection time-to-contact vs false alarms.

Pick the MOST SENSITIVE setting that still keeps a safety factor (~1.5-2x) above the false-alarm
cliff, as the shipped defaults do. Synthetic results only: repeat this on real recordings with
scripts/replay_snn.py before trusting any threshold.

    python scripts/sweep_snn.py                                   # fully spiking detector (default)
    python scripts/sweep_snn.py --theta0 4 6 8 --beta 0.08 0.12 --rate 15 20
    python scripts/sweep_snn.py --impl hybrid --theta 2.5 4 6 --rate 10 15 20
"""
import _path  # noqa: F401
import argparse
import dataclasses
import itertools

from neurodrone.closedloop import open_loop_trial
from neurodrone.config import SNNConfig, SpikingConfig
from neurodrone.sim import Box

ap = argparse.ArgumentParser()
ap.add_argument("--impl", choices=["spiking", "hybrid"], default="spiking")
ap.add_argument("--theta", type=float, nargs="+", default=[2.5, 4.0, 6.0], help="hybrid: neuron threshold (z units)")
ap.add_argument("--theta0", type=float, nargs="+", default=[4.0, 6.0, 8.0], help="spiking: base threshold")
ap.add_argument("--beta", type=float, nargs="+", default=[0.12], help="spiking: adaptive threshold")
ap.add_argument("--rate", type=float, nargs="+", default=[15.0], help="alarm spike-rate threshold")
ap.add_argument("--speed", type=float, default=1.5)
args = ap.parse_args()

DET = {"wall 3m": ([Box(x=14, y=0, w=3, h=3)], {}),
       "box 1.5m": ([Box(x=14, y=0, w=1.5, h=1.5)], {}),
       "blank 3m": ([Box(x=14, y=0, w=3, h=3, texture="blank")], {}),
       "pole 0.4m": ([Box(x=14, y=0, w=0.4, h=3)], {}),
       "pole 0.15m": ([Box(x=14, y=0, w=0.15, h=3)], {})}
FA = {"stationary+jitter, object": ([Box(x=14, y=2, w=2, h=2)], dict(speed=0.0, jitter_deg=3.0, dur=6)),
      "fwd+jitter 5deg empty": ([], dict(jitter_deg=5.0, dur=8)),
      "yaw 30deg/s empty": ([], dict(speed=0.0, yaw_rate_deg_s=30.0, dur=6)),
      "noise 3Hz/px empty": ([], dict(noise_hz=3.0, dur=8)),
      "object passing left": ([Box(x=14, y=-3.0, w=1.5, h=3)], dict(dur=8))}

if args.impl == "hybrid":
    grid = [(f"theta={th:<4} rate={ra:<4}", SNNConfig(impl="hybrid", theta=th, rate_alarm=ra))
            for th, ra in itertools.product(args.theta, args.rate)]
else:
    grid = [(f"theta0={t0:<4} beta={b:<5} rate={ra:<4}",
             SNNConfig(impl="spiking", spiking=dataclasses.replace(SpikingConfig(), theta0=t0, beta=b, rate_alarm=ra)))
            for t0, b, ra in itertools.product(args.theta0, args.beta, args.rate)]

print(f"detector: {args.impl}")
print("detection columns: time-to-contact (s) at first alarm, higher = earlier warning; 'miss' = never alarmed")
print(f"{'setting':32s} | " + " ".join(f"{k:>10}" for k in DET) + " | worst false-alarm urgency")
for label, cfg in grid:
    cols = []
    for boxes, kw in DET.values():
        first, _ = open_loop_trial(boxes, cfg, speed=args.speed, **kw)
        cols.append("miss" if first is None else f"{first / args.speed:.1f}")
    worst = max(open_loop_trial(b, cfg, **kw)[1] for b, kw in FA.values())
    flag = "  <-- FALSE ALARM" if worst >= 1.0 else ""
    print(f"{label:32s} | " + " ".join(f"{c:>10}" for c in cols) + f" | {worst:.2f}{flag}", flush=True)
