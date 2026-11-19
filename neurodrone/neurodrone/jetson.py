"""Jetson health while the VLM runs: memory, temperatures, clocks, GPU load and power from `tegrastats`.

A typical Orin line (JetPack 6) looks like

  11-20-2026 10:00:00 RAM 5123/7620MB (lfb 2x4MB) SWAP 12/3810MB (cached 0MB) CPU [35%@1510,20%@1510,...]
  EMC_FREQ 31%@3199 GR3D_FREQ 97%@[1020] NVENC off ... cpu@55.4C soc2@52.1C soc0@53.0C gpu@57.3C tj@57.9C
  soc1@51.8C VDD_IN 14520mW/12010mW VDD_CPU_GPU_CV 6120mW/5000mW VDD_SOC 3010mW/2800mW

The parser is written to tolerate the small differences between JetPack versions (GR3D_FREQ "97%@[1020]" or
"97%@1020", temperatures "gpu@57.3C" or "GPU@57C", rails "VDD_IN 14520mW/12010mW" or "VDD_IN 14520/12010").
CHECK it against the real board on day one: `tegrastats --interval 1000 | head -3` and
`python -c "from neurodrone.jetson import parse_tegrastats as p; import sys; print(p(sys.stdin.readline()))"`.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from typing import Dict, List, Optional

import numpy as np

_RAM = re.compile(r"\bRAM (\d+)/(\d+)MB")
_SWAP = re.compile(r"\bSWAP (\d+)/(\d+)MB")
_CPU = re.compile(r"\bCPU \[([^\]]*)\]")
_GPU = re.compile(r"\bGR3D_FREQ (\d+)%(?:@\[?(\d+)(?:,\d+)*\]?)?")
_EMC = re.compile(r"\bEMC_FREQ (\d+)%(?:@(\d+))?")
_TEMP = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@(-?\d+(?:\.\d+)?)C\b")
_RAIL = re.compile(r"\b(VDD_[A-Z0-9_]+|VIN_[A-Z0-9_]+|POM_[A-Z0-9_]+) (\d+)(?:mW)?/(\d+)(?:mW)?")


def parse_tegrastats(line: str) -> Optional[dict]:
    """One tegrastats line -> dict (only the fields found), or None if it is not a tegrastats line."""
    out: dict = {}
    m = _RAM.search(line)
    if m:
        out["ram_used_mb"], out["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    m = _SWAP.search(line)
    if m:
        out["swap_used_mb"], out["swap_total_mb"] = int(m.group(1)), int(m.group(2))
    m = _CPU.search(line)
    if m:
        loads, freqs = [], []
        for c in m.group(1).split(","):
            c = c.strip()
            if c == "off" or not c:
                continue
            mm = re.match(r"(\d+)%(?:@(\d+))?", c)
            if mm:
                loads.append(int(mm.group(1)))
                if mm.group(2):
                    freqs.append(int(mm.group(2)))
        out["cpu_load_pct"] = loads
        if freqs:
            out["cpu_freq_mhz"] = freqs
    m = _GPU.search(line)
    if m:
        out["gpu_load_pct"] = int(m.group(1))
        if m.group(2):
            out["gpu_freq_mhz"] = int(m.group(2))
    m = _EMC.search(line)
    if m:
        out["emc_load_pct"] = int(m.group(1))
        if m.group(2):
            out["emc_freq_mhz"] = int(m.group(2))
    temps = {k.lower(): float(v) for k, v in _TEMP.findall(line)}
    if temps:
        out["temp_c"] = temps
    rails = {k: int(now) for k, now, _avg in _RAIL.findall(line)}
    if rails:
        out["power_mw"] = rails
    if "ram_used_mb" not in out and "gpu_load_pct" not in out:
        return None
    return out


def total_power_w(rec: dict) -> Optional[float]:
    """Board input power: VDD_IN (Orin Nano / NX), else POM_5V_IN (older modules)."""
    p = rec.get("power_mw") or {}
    for k in ("VDD_IN", "POM_5V_IN"):
        if k in p:
            return p[k] / 1000.0
    return None


def max_temp(rec: dict) -> Optional[float]:
    t = rec.get("temp_c") or {}
    if "tj" in t:  # junction temperature: the one the board throttles on
        return t["tj"]
    vals = [v for k, v in t.items() if k not in ("pmic",) and v > -200]  # PMIC reads a fixed 50 C / -256 C on some
    return max(vals) if vals else None


class TegraLogger:
    """Runs `tegrastats` in the background and keeps every parsed line (and writes them to a JSONL file).
    On a machine without tegrastats it records nothing (self.available is False)."""

    def __init__(self, interval_ms: int = 1000, path: Optional[str] = None, cmd: Optional[List[str]] = None):
        self.interval_ms, self.path = interval_ms, path
        self.cmd = cmd or ["tegrastats", "--interval", str(interval_ms)]
        self.records: List[dict] = []
        self.available = False
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._marks: List[tuple] = []

    def start(self) -> "TegraLogger":
        try:
            self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except (FileNotFoundError, PermissionError):
            return self
        self.available = True
        self._thread = threading.Thread(target=self._run, name="tegrastats", daemon=True)
        self._thread.start()
        return self

    def wait_first(self, timeout_s: float = 3.0) -> bool:
        """Wait until the first line has been parsed (so a short phase is not measured before the log runs)."""
        t0 = time.monotonic()
        while self.available and not self.records and time.monotonic() - t0 < timeout_s:
            time.sleep(0.02)
        return bool(self.records)

    def mark(self, label: str) -> None:
        """Mark a phase (e.g. 'soak start') so the summary can split the record."""
        self._marks.append((time.time(), label))

    def _run(self) -> None:
        f = open(self.path, "a") if self.path else None
        try:
            for line in self._proc.stdout:  # type: ignore[union-attr]
                if self._stop.is_set():
                    break
                rec = parse_tegrastats(line)
                if rec is None:
                    continue
                rec["t"] = time.time()
                self.records.append(rec)
                if f:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
        finally:
            if f:
                f.close()

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=3)

    def between(self, t0: float, t1: float) -> List[dict]:
        return [r for r in self.records if t0 <= r["t"] <= t1]


def summarize(records: List[dict], window_s: float = 300.0) -> Dict[str, object]:
    """Peak memory, temperatures, power, and whether the GPU clock fell over time (throttling): the median GPU
    clock while busy (load >= 50 %) in the first and the last `window_s` of the record."""
    if not records:
        return {"n": 0}
    out: Dict[str, object] = {"n": len(records), "duration_s": round(records[-1]["t"] - records[0]["t"], 1)}
    ram = [r["ram_used_mb"] for r in records if "ram_used_mb" in r]
    if ram:
        total = next(r["ram_total_mb"] for r in records if "ram_total_mb" in r)
        out.update(ram_total_mb=total, ram_used_mb_max=max(ram), ram_free_mb_min=total - max(ram),
                   ram_used_mb_median=float(np.median(ram)))
    swap = [r["swap_used_mb"] for r in records if "swap_used_mb" in r]
    if swap:
        out["swap_used_mb_max"] = max(swap)
    temps = [t for t in (max_temp(r) for r in records) if t is not None]
    if temps:
        out["temp_max_c"] = max(temps)
        out["temp_median_c"] = float(np.median(temps))
    pw = [p for p in (total_power_w(r) for r in records) if p is not None]
    if pw:
        out.update(power_mean_w=float(np.mean(pw)), power_max_w=max(pw))
    busy = [r for r in records if r.get("gpu_load_pct", 0) >= 50 and "gpu_freq_mhz" in r]
    if busy:
        out["gpu_busy_share"] = len(busy) / len(records)
        out["gpu_freq_max_mhz"] = max(r["gpu_freq_mhz"] for r in busy)
        t0, t1 = busy[0]["t"], busy[-1]["t"]
        first = [r["gpu_freq_mhz"] for r in busy if r["t"] <= t0 + window_s]
        last = [r["gpu_freq_mhz"] for r in busy if r["t"] >= t1 - window_s]
        if first and last:
            f0, f1 = float(np.median(first)), float(np.median(last))
            out.update(gpu_freq_first_mhz=f0, gpu_freq_last_mhz=f1, gpu_freq_drop=(f0 - f1) / f0 if f0 else None)
    return out


def throttled(summary: dict, max_drop: float = 0.05, max_temp_c: float = 95.0) -> Optional[bool]:
    """True if the busy GPU clock fell by more than max_drop over the run or the junction got close to the
    throttling point (Orin: software throttling from ~99 C). None if there is no data."""
    drop, temp = summary.get("gpu_freq_drop"), summary.get("temp_max_c")
    if drop is None and temp is None:
        return None
    return bool((drop is not None and drop > max_drop) or (temp is not None and temp >= max_temp_c))


def board_info() -> dict:
    """JetPack / L4T release, power mode and memory, read from the board (missing items are skipped)."""
    info: dict = {}
    try:
        info["l4t_release"] = open("/etc/nv_tegra_release").readline().strip()
    except OSError:
        pass
    try:
        info["model"] = open("/proc/device-tree/model").read().strip("\x00\n ")
    except OSError:
        pass
    for key, cmd in (("nvpmodel", ["nvpmodel", "-q"]), ("jetpack", ["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"])):
        try:
            info[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        with open("/proc/meminfo") as f:
            mem = {l.split(":")[0]: int(l.split()[1]) for l in f if l.split(":")[0] in ("MemTotal", "MemAvailable", "SwapTotal")}
        info.update({k + "_mb": v // 1024 for k, v in mem.items()})
    except OSError:
        pass
    return info
