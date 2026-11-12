#!/usr/bin/env python3
"""Write a synthetic recording (approach to a wall) to .npz for trying replay_snn.py / dry runs."""
import _path  # noqa: F401
import argparse

import numpy as np

from neurodrone.closedloop import open_loop_trial
from neurodrone.events import save_events_npz
from neurodrone.sim import Box, Camera

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="demo_events.npz")
ap.add_argument("--dur", type=float, default=9.0)
args = ap.parse_args()

rec = []
first, _ = open_loop_trial([Box(x=14, y=0, w=3, h=3)], speed=1.5, dur=args.dur, record=rec)
cam = Camera()
x = np.concatenate([e.x for e in rec]); y = np.concatenate([e.y for e in rec])
t = np.concatenate([e.t_us for e in rec]); p = np.concatenate([e.p for e in rec])
o = np.argsort(t, kind="stable")
save_events_npz(args.out, x[o], y[o], t[o], p[o], cam.width, cam.height)
print(f"wrote {args.out}: {t.size} events over {args.dur:.0f} s ({cam.width}x{cam.height}); "
      f"wall is at 14 m, first alarm expected at {first:.1f} m clearance")
