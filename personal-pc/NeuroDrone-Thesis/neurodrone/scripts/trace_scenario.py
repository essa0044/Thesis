#!/usr/bin/env python3
"""Print the coordinator's mode transitions (and the VLM goals) for one closed-loop scenario.

    python scripts/trace_scenario.py dodge_obstacle --goals

The tool that exposed most of the coordinator's design bugs: aggregate numbers hide *why* a
scenario ended the way it did; the transition list shows it (e.g. the drone dodging its own
target because the VLM box it used was 3 s stale)."""
import _path  # noqa: F401
import argparse

from neurodrone.closedloop import ClosedLoopSim, default_scenarios
from neurodrone.config import SNNConfig
from neurodrone.reflex import IMPLS

ap = argparse.ArgumentParser()
ap.add_argument("scenario", choices=[s.name for s in default_scenarios()])
ap.add_argument("--goals", action="store_true", help="also print each VLM goal as it ARRIVES at the coordinator")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--impl", choices=IMPLS, default=SNNConfig().impl, help="looming detector")
a = ap.parse_args()

sc = next(s for s in default_scenarios() if s.name == a.scenario)
sim = ClosedLoopSim(sc, SNNConfig(impl=a.impl), seed=a.seed)
events, brain = [], sim.brain
orig_step, orig_goal = brain.control_step, brain.on_goal


def step(now, veh):
    dec = orig_step(now, veh)
    events.append((now, "decision", dec))
    return dec


def goal(g):
    events.append((g.captured_at + g.latency_s, "goal", g))
    return orig_goal(g)


brain.control_step, brain.on_goal = step, goal
r = sim.run()

last = None
for t, kind, x in events:
    if kind == "decision" and x.mode != last:
        c = x.cmd
        print(f"t={t:6.1f}s  {x.mode:10s} vx={c.vx:+.2f} vy={c.vy:+.2f} yaw={c.yaw_rate:+5.1f}  {x.note}")
        last = x.mode
    elif kind == "goal" and a.goals:
        shown = f"x={x.x:.2f} w={x.w:.2f}" if x.kind == "target" else "not visible"
        print(f"t={t:6.1f}s    goal arrives: {x.kind} {shown}  (captured t={x.captured_at:.1f}s)")
print(f"\nresult: {'COLLISION' if r.collided else ('arrived' if r.arrived else 'no arrival')} | ended: {r.ended} | "
      f"sim time {r.t_end:.1f}s | final pose x={r.final_pose[0]:.1f} y={r.final_pose[1]:.1f} yaw={r.final_pose[2]:.0f}deg")
