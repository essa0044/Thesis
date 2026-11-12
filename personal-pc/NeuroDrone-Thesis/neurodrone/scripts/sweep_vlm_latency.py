#!/usr/bin/env python3
"""Thesis experiment (in simulation): what does VLM latency / call period cost in mission time?

For each latency the VLM is queried every max(period_floor, latency) seconds (a real server
cannot be called faster than it answers). The oracle VLM is perfectly accurate, so this isolates
the *timing* cost only. goal_ttl is widened to cover the latency, otherwise goals would expire
before they arrive.

    python scripts/sweep_vlm_latency.py --scenario dodge_obstacle --latencies 0.5 1 2 4 8
"""
import _path  # noqa: F401
import argparse
import dataclasses

from neurodrone.closedloop import ClosedLoopSim, default_scenarios

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="approach_target")
ap.add_argument("--latencies", type=float, nargs="+", default=[0.5, 1, 2, 4, 8])
ap.add_argument("--period-floor", type=float, default=1.0)
ap.add_argument("--max-sim-s", type=float, default=150.0)
args = ap.parse_args()

base = {s.name: s for s in default_scenarios()}[args.scenario]
print(f"scenario: {args.scenario}   (oracle VLM, ideal sensors: only the TIMING cost is measured)")
print(f"{'latency s':>9} {'period s':>8} | {'result':9s} {'mission s':>9} {'dodges':>6} {'idle %':>7}  (idle = braking/waiting on the VLM)")
for lat in args.latencies:
    period = max(args.period_floor, lat)
    coord = dataclasses.replace(base.coord, goal_ttl_s=max(base.coord.goal_ttl_s, 2.0 * (lat + period)),
                                adjudicate_timeout_s=max(base.coord.adjudicate_timeout_s, 3.0 * (lat + period)))
    sc = dataclasses.replace(base, vlm_latency_s=lat, vlm_period_s=period, duration_s=args.max_sim_s, coord=coord)
    r = ClosedLoopSim(sc).run()
    dodges = sum(1 for (a, b) in zip(r.path, r.path[1:]) if b[4] == "AVOID" and a[4] != "AVOID")
    total = sum(r.modes.values())
    idle = 100.0 * (r.modes["ADJUDICATE"] + r.modes["NO_GOAL"] + r.modes["SETTLE"]) / max(total, 1)
    res = "COLLISION" if r.collided else ("arrived" if r.arrived else "timeout")
    print(f"{lat:9.1f} {period:8.1f} | {res:9s} {r.t_end:9.1f} {dodges:6d} {idle:7.0f}", flush=True)
