#!/usr/bin/env python3
"""Run the synthetic closed-loop scenarios and print a summary table.

    python scripts/sim_closed_loop.py                 # all scenarios
    python scripts/sim_closed_loop.py approach_target # one

Logic check on an idealised simulator, not a performance prediction."""
import _path  # noqa: F401
import argparse
import time

from neurodrone.closedloop import ClosedLoopSim, default_scenarios
from neurodrone.config import SNNConfig
from neurodrone.reflex import IMPLS

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--impl", choices=IMPLS, default=SNNConfig().impl, help="looming detector")
args = ap.parse_args()

print(f"detector: {args.impl}")
print(f"{'scenario':24s} {'result':10s} {'min clr':>8s} {'1st SNN alarm':>14s} {'sim s':>6s} {'cpu s':>6s}  ended")
for sc in default_scenarios():
    if args.names and sc.name not in args.names:
        continue
    t0 = time.time()
    r = ClosedLoopSim(sc, SNNConfig(impl=args.impl), seed=args.seed).run()
    res = "COLLISION" if r.collided else ("arrived" if r.arrived else "-")
    fa = f"{r.first_alarm_clearance:.1f} m" if r.first_alarm_clearance is not None else "none"
    note = "   <-- known limitation (SNN alone cannot see it)" if sc.name == "blank_wall_snn_only" else ""
    print(f"{r.name:24s} {res:10s} {r.min_clearance:8.2f} {fa:>14s} {r.t_end:6.1f} {time.time() - t0:6.0f}  {r.ended}{note}")
