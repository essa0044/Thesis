"""Event sources. Anything that can produce `EventPacket`s plugs into the pipeline.

* ReplayEventSource - plays back an .npz recording (offline tuning, regression tests).
* DavisLiveSource   - live iniVation DAVIS/DVXplorer via dv-processing.   [UNTESTED: needs hardware]
* For other sensors (SynSense Speck, Prophesee, ...) subclass EventSource and implement
  read(); everything downstream only sees EventPacket.

npz format: arrays x, y, t (int64 microseconds, sorted), p (0/1), scalars width, height.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Optional

import numpy as np

from .types import EventPacket


class EventSource(ABC):
    width: int
    height: int

    @abstractmethod
    def read(self, timeout_s: float = 0.1) -> Optional[EventPacket]:
        """Next packet (may be empty), or None if nothing is available within timeout_s."""

    def close(self) -> None:
        pass


def save_events_npz(path: str, x, y, t, p, width: int, height: int) -> None:
    np.savez_compressed(path, x=np.asarray(x, np.int16), y=np.asarray(y, np.int16),
                        t=np.asarray(t, np.int64), p=np.asarray(p, np.int8),
                        width=int(width), height=int(height))


class ReplayEventSource(EventSource):
    def __init__(self, path: str, packet_us: int = 10_000, realtime: bool = False):
        d = np.load(path)
        self._x, self._y, self._t, self._p = d["x"], d["y"], d["t"].astype(np.int64), d["p"]
        self.width, self.height = int(d["width"]), int(d["height"])
        self.packet_us = int(packet_us)
        self.realtime = realtime
        self._i = 0
        self._t_cur = int(self._t[0]) if self._t.size else 0
        self._t_final = int(self._t[-1]) if self._t.size else 0
        self._wall0: Optional[float] = None
        self.finished = False

    def read(self, timeout_s: float = 0.1) -> Optional[EventPacket]:
        if self.finished:
            return None
        t0, t1 = self._t_cur, self._t_cur + self.packet_us
        if self.realtime:
            if self._wall0 is None:
                self._wall0 = time.monotonic() - t0 / 1e6
            delay = (self._wall0 + t1 / 1e6) - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, timeout_s))
                if delay > timeout_s:
                    return None
        j = int(np.searchsorted(self._t, t1, side="left"))
        pkt = EventPacket(self._x[self._i:j], self._y[self._i:j], self._t[self._i:j], self._p[self._i:j], t0, t1)
        self._i, self._t_cur = j, t1
        if t0 > self._t_final:
            self.finished = True
            return None
        return pkt


class DavisLiveSource(EventSource):
    """Live event stream from an iniVation camera. [UNTESTED - written against the dv-processing
    Python API (CameraCapture / getNextEventBatch / EventStore.numpy()); check the version you
    have installed.]"""

    def __init__(self, packet_us: int = 10_000, camera_name: str = ""):
        import dv_processing as dv  # type: ignore

        self._cap = dv.io.CameraCapture(camera_name) if camera_name else dv.io.CameraCapture()
        res = self._cap.getEventResolution()
        self.width, self.height = int(res[0]), int(res[1])
        self.packet_us = int(packet_us)
        self._pending: Deque[EventPacket] = deque()
        self._anchor: Optional[tuple] = None  # (sensor_us, monotonic_s) from the last real event

    def _split(self, arr) -> None:
        t = arr["timestamp"].astype(np.int64)
        x, y, p = arr["x"].astype(np.int16), arr["y"].astype(np.int16), arr["polarity"].astype(np.int8)
        t0 = int(t[0])
        edge = t0 + self.packet_us
        i = 0
        while i < t.size:
            j = int(np.searchsorted(t, edge, side="left"))
            self._pending.append(EventPacket(x[i:j], y[i:j], t[i:j], p[i:j], edge - self.packet_us, edge))
            i, edge = j, edge + self.packet_us
        self._anchor = (int(t[-1]), time.monotonic())

    def read(self, timeout_s: float = 0.1) -> Optional[EventPacket]:
        if self._pending:
            return self._pending.popleft()
        batch = self._cap.getNextEventBatch()
        if batch is not None and len(batch) > 0:
            self._split(batch.numpy())
            return self._pending.popleft() if self._pending else None
        time.sleep(min(0.002, timeout_s))
        if self._anchor is None:
            return None
        # quiet scene: advance time using the host clock so the neurons keep leaking
        s_us, m_s = self._anchor
        now_us = s_us + int((time.monotonic() - m_s) * 1e6)
        return EventPacket.empty(now_us - self.packet_us, now_us)

    def close(self) -> None:
        self._cap = None
