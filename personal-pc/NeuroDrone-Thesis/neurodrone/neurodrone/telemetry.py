"""Structured logging + Jetson power sampling + log summary (the feasibility numbers)."""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np


# ----------------------------------------------------------------------------- logging
def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class JsonlLogger:
    """Thread-safe JSON-lines logger. Every record has t (monotonic s), wall (epoch s), kind."""

    def __init__(self, path: str):
        self._f = open(path, "a", buffering=1)
        self._lock = threading.Lock()

    def log(self, kind: str, **fields) -> None:
        rec = {"t": round(time.monotonic(), 4), "wall": round(time.time(), 3), "kind": kind, **fields}
        line = json.dumps(rec, default=_json_default)
        with self._lock:
            self._f.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._f.close()


class NullLogger:
    def log(self, kind: str, **fields) -> None:
        pass

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------- power
# Formats differ between Jetson modules / JetPack versions. These patterns are written from
# memory of typical tegrastats output and are UNVERIFIED against your board: run `tegrastats`
# once, compare with the pattern, and fix if needed (or rely on jtop, tried first).
_RAIL_TOTAL = re.compile(r"\bVDD_IN (\d+)mW")  # Orin Nano / Orin NX: module input power
_RAIL_POM = re.compile(r"\bPOM_5V_IN (\d+)/(\d+)")  # older Nano / Xavier: instantaneous/average mW
_RAIL_AGX = [re.compile(p + r" (\d+)mW") for p in (r"\bVDD_GPU_SOC", r"\bVDD_CPU_CV", r"\bVIN_SYS_5V0")]


def parse_tegrastats_line(line: str) -> Optional[float]:
    """Total power in watts from one tegrastats line, or None if no known rail is present."""
    m = _RAIL_TOTAL.search(line)
    if m:
        return int(m.group(1)) / 1000.0
    m = _RAIL_POM.search(line)
    if m:
        return int(m.group(1)) / 1000.0
    parts = [r.search(line) for r in _RAIL_AGX]
    if all(parts):
        return sum(int(p.group(1)) for p in parts) / 1000.0
    return None


class PowerSampler:
    """Background sampler of total module power. Tries jtop (jetson-stats) then tegrastats.
    On a non-Jetson machine it simply reports no data."""

    def __init__(self, interval_s: float = 0.5, logger=None):
        self.interval_s = interval_s
        self.logger = logger or NullLogger()
        self.samples: List[float] = []
        self.source: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="power", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _record(self, w: float) -> None:
        self.samples.append(w)
        self.logger.log("power", watts=w)

    def _run(self) -> None:
        try:
            from jtop import jtop  # type: ignore

            with jtop(interval=self.interval_s) as j:
                self.source = "jtop"
                while j.ok() and not self._stop.is_set():
                    self._record(float(j.power["tot"]["power"]) / 1000.0)  # mW -> W
            return
        except Exception:
            pass
        try:
            self._proc = subprocess.Popen(["tegrastats", "--interval", str(int(self.interval_s * 1000))],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.source = "tegrastats"
            for line in self._proc.stdout:  # type: ignore
                if self._stop.is_set():
                    break
                w = parse_tegrastats_line(line)
                if w is not None:
                    self._record(w)
        except Exception:
            self.source = None

    def stats(self) -> Optional[Dict[str, float]]:
        if not self.samples:
            return None
        a = np.array(self.samples)
        return {"mean_w": float(a.mean()), "max_w": float(a.max()), "n": int(a.size), "source": self.source}


# ----------------------------------------------------------------------------- summary
def _pct(a: np.ndarray) -> Dict[str, float]:
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()), "n": int(a.size)}


def summarize_log(path: str) -> Dict[str, object]:
    """Feasibility numbers from a run log: VLM latency, SNN processing time, mode time-share,
    avoidance events, power."""
    recs = defaultdict(list)
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[r.get("kind")].append(r)
    out: Dict[str, object] = {"records": {k: len(v) for k, v in recs.items()}}
    v = recs.get("vlm_call", [])
    if v:
        ok = [r for r in v if r.get("ok")]
        lat = [r["latency_s"] for r in v if r.get("latency_s") is not None]
        if lat:  # a log of only failed calls has no latencies
            out["vlm_latency_s"] = _pct(np.array(lat))
        out["vlm_parse_success"] = len(ok) / len(v)
    th = recs.get("threat", [])
    if th:
        out["snn_proc_ms"] = _pct(np.array([r["proc_ms"] for r in th]))
        out["alarm_packets"] = int(sum(1 for r in th if r.get("alarm")))
    dec = recs.get("decision", [])
    if dec:
        c = Counter(r["mode"] for r in dec)
        out["mode_share"] = {k: n / len(dec) for k, n in c.most_common()}
    p = recs.get("power", [])
    if p:
        w = np.array([r["watts"] for r in p])
        out["power_w"] = {"mean": float(w.mean()), "max": float(w.max()), "n": int(w.size)}
    return out
