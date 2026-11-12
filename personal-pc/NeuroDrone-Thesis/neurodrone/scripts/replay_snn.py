#!/usr/bin/env python3
"""Replay a recorded event stream through the looming detector offline: the tool for tuning thresholds
on REAL data. The detector is chosen by snn.impl in the config (spiking = default, hybrid = baseline).

    python scripts/replay_snn.py recording.npz --out trace.csv [--config my.yaml]

Workflow: (1) record obstacle-free flights -> any alarm is a false alarm; count them and widen
theta until none remain with a >=2x margin. (2) record approaches to known obstacles -> note the
time-to-contact at the first alarm; that number sets cruise speed via neurodrone/safety.py.

.npz needs arrays x, y, t (int64 microseconds, sorted), p (0/1) and scalars width, height.
Converting an iniVation .aedat4 file: iterate dv_processing.io.MonoCameraRecording batches and
save batch.numpy() fields (timestamp, x, y, polarity) with neurodrone.events.save_events_npz."""
import _path  # noqa: F401
import argparse
import csv

import numpy as np

from neurodrone.config import load_config
from neurodrone.events import ReplayEventSource
from neurodrone.reflex import make_snn

ap = argparse.ArgumentParser()
ap.add_argument("recording")
ap.add_argument("--config")
ap.add_argument("--out", default="snn_trace.csv")
ap.add_argument("--packet-us", type=int, default=10_000)
args = ap.parse_args()

cfg = load_config(args.config)
src = ReplayEventSource(args.recording, packet_us=args.packet_us)
snn = make_snn(src.width, src.height, cfg.snn)
nb = snn.n_bands
head = ["t_s", "n_events", "urg_left", "urg_centre", "urg_right", "frontal", "alarm", "proc_ms"] + \
       [f"rate_{n}" for n in ["L", "C", "R"] + [f"band{k}" for k in range(nb)]]
alarms, in_alarm, n_pkts, t0, proc, max_rate = [], False, 0, None, [], np.zeros(3 + nb)
with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(head)
    while True:
        pk = src.read()
        if pk is None:
            break
        rep = snn.process(pk)
        t0 = pk.t_start_us if t0 is None else t0
        t = (rep.t_us - t0) / 1e6
        w.writerow([f"{t:.3f}", rep.n_events, *[f"{u:.3f}" for u in rep.urgency], f"{rep.frontal:.3f}",
                    int(rep.alarm), f"{rep.proc_s * 1e3:.3f}", *[f"{r:.1f}" for r in snn.rate]])
        if rep.alarm and not in_alarm:
            alarms.append(t)
        in_alarm = rep.alarm
        proc.append(rep.proc_s * 1e3)
        max_rate = np.maximum(max_rate, snn.rate)
        n_pkts += 1
ra = cfg.snn.spiking.rate_alarm if cfg.snn.impl == "spiking" else cfg.snn.rate_alarm
th = f"theta0={cfg.snn.spiking.theta0}, beta={cfg.snn.spiking.beta}" if cfg.snn.impl == "spiking" else f"theta={cfg.snn.theta}"
print(f"detector: {cfg.snn.impl} | {n_pkts} packets, {src.width}x{src.height}, deltas={snn.deltas} px, {th}, rate_alarm={ra}")
print(f"alarm onsets at t = {', '.join(f'{a:.2f}s' for a in alarms) if alarms else 'none'}")
print(f"SNN processing time per packet: mean {np.mean(proc):.2f} ms, p95 {np.percentile(proc, 95):.2f} ms, max {np.max(proc):.2f} ms")
print(f"peak spike-rate per neuron (alarm at {ra}): " +
      ", ".join(f"{n}={r:.0f}" for n, r in zip(["L", "C", "R"] + [f"band{k}" for k in range(nb)], max_rate)))
if hasattr(snn, "stats"):
    st = snn.stats
    print(f"spikes: layer1 {st['layer1_spikes']}, layer2 {st['layer2_spikes']}, layer3 {st['layer3_spikes']} | "
          f"synaptic operations {st['synops']} ({st['synops'] / max(st['events'], 1):.1f} per event)")
print(f"trace written to {args.out}")
