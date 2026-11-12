"""Classical (non-spiking) event-based time-to-contact reflex: the baseline the spiking detector is
compared against.

Method (standard event-based vision, no learning, no spiking neurons):

1. Normal optical flow per event from a local plane fitted to the Surface of Active Events (SAE),
   built from the first event of each burst (refractory filter, `refractory_us`):
   for each event, the most recent timestamps of its 5x5 neighbourhood (same polarity, within
   `window_us`) are fitted with t = a*dx + b*dy + c by least squares; the normal flow is
   v = (a, b) / (a^2 + b^2) in px/s (Benosman et al. 2014, "Event-based visual flow", IEEE TNNLS).
2. Expansion rate from normal flow: for an approaching surface, the flow at image point p is
   v = (p - c) / TTC around the focus of expansion c (Clady et al. 2014, "Asynchronous visual
   event-based time-to-contact", Front. Neurosci.). Each normal flow only shows the component
   along its own direction n, so the inverse TTC s = 1/TTC is fitted to all events of a packet at
   once by least squares: s = sum(|v_i| (r_i . n_i)) / sum((r_i . n_i)^2), r_i = p_i - c.
3. The packet estimates are smoothed (exponential, `ema_s`) and the alarm fires when the smoothed
   TTC estimate stays at or below `tau_alarm_s` for 2 consecutive packets (the spiking detector's
   readout also needs 2 alarm spikes).

The focus of expansion is assumed at the image centre, as in the spiking detector, and the same
rows (roi_y) and minimum radius (r_min_frac) are used, so both detectors see the same pixels.
Returns the same ThreatReport as the spiking and hybrid detectors.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from .config import SNNConfig
from .types import EventPacket, ThreatReport


class ClassicalTTC:
    def __init__(self, width: int, height: int, cfg: Optional[SNNConfig] = None):
        self.cfg = cfg or SNNConfig(impl="classical")
        self.c = self.cfg.classical
        self.W, self.H = int(width), int(height)
        self.cx, self.cy = (self.W - 1) / 2.0, (self.H - 1) / 2.0
        R = int(self.c.radius)
        self.R = R
        # SAE per polarity, padded by R so neighbourhoods never leave the array; -inf = never fired
        self._sae = np.full((2, self.H + 2 * R, self.W + 2 * R), -np.inf)
        dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
        keep = (dx != 0) | (dy != 0)
        self._dx, self._dy = dx[keep].astype(float), dy[keep].astype(float)
        y0, y1 = self.cfg.roi_y
        self._ylo, self._yhi = int(y0 * self.H), int(math.ceil(y1 * self.H))
        self._rmin = self.cfg.r_min_frac * self.W
        self._rng = np.random.default_rng(0)
        self._s = 0.0          # smoothed inverse TTC (1/s)
        self._t_last: Optional[int] = None
        self._below = 0        # consecutive packets with TTC estimate <= tau_alarm
        self.last_inv_ttc = float("nan")  # this packet's raw estimate (nan = too few flow vectors)
        self.inv_ttc_smoothed = 0.0
        self.stats = dict(packets=0, events=0, flow_vectors=0)

    def reset(self) -> None:
        self.__init__(self.W, self.H, self.cfg)

    def _flow(self, x, y, t, p):
        """Normal flow at the given events from the (already updated) SAE. Returns (vx, vy, ok)."""
        R = self.R
        xs = x[:, None].astype(np.int64) + R + self._dx[None, :].astype(np.int64)
        ys = y[:, None].astype(np.int64) + R + self._dy[None, :].astype(np.int64)
        tn = self._sae[p[:, None].astype(np.int64), ys, xs]          # neighbour timestamps (us)
        dt = (tn - t[:, None].astype(float)) * 1e-6                  # seconds, <= 0 for older
        m = np.isfinite(dt) & (np.abs(dt) <= self.c.window_us * 1e-6)
        dx, dy = self._dx[None, :], self._dy[None, :]

        def fit(m):
            n = m.sum(1)
            Sxx = (m * dx * dx).sum(1); Syy = (m * dy * dy).sum(1); Sxy = (m * dx * dy).sum(1)
            dt0 = np.where(m, dt, 0.0)
            Sxt = (dt0 * dx).sum(1); Syt = (dt0 * dy).sum(1)
            det = Sxx * Syy - Sxy * Sxy
            ok = (n >= self.c.min_neighbours) & (det > 1e-9)
            det = np.where(ok, det, 1.0)
            return (Syy * Sxt - Sxy * Syt) / det, (Sxx * Syt - Sxy * Sxt) / det, ok  # dt/dx, dt/dy (s/px)

        a, b, ok = fit(m)
        # Robust refit (Benosman et al. 2014): in textured scenes a neighbour the edge has not reached
        # yet still holds the time of an EARLIER edge. Such stale timestamps are far off the local
        # plane and would flip the fitted direction; drop neighbours whose residual exceeds
        # `outlier_s` and fit again.
        for _ in range(int(self.c.refit_iters)):
            res = np.abs(dt - (a[:, None] * dx + b[:, None] * dy))
            m = m & (res <= self.c.outlier_s)
            a, b, ok2 = fit(m)
            ok &= ok2
        g2 = a * a + b * b
        ok &= g2 > 1e-12
        g2 = np.where(ok, g2, 1.0)
        # t = a dx + b dy: the edge moves along the gradient, speed 1/|grad| -> v = grad / |grad|^2.
        # Timestamps grow in the direction of motion, so v points the same way as (a, b).
        vx, vy = a / g2, b / g2
        speed = np.sqrt(vx * vx + vy * vy)
        ok &= speed <= self.c.v_max_pxs
        return vx, vy, ok

    def process(self, pkt: EventPacket) -> ThreatReport:
        t0 = time.perf_counter()
        n_ev = int(pkt.n)
        inv = float("nan")
        if n_ev:
            x = np.asarray(pkt.x, dtype=np.int64); y = np.asarray(pkt.y, dtype=np.int64)
            t = np.asarray(pkt.t_us, dtype=np.int64); p = (np.asarray(pkt.p) > 0).astype(np.int64)
            # Refractory filter: an edge crossing a pixel makes a burst of events; only the FIRST event
            # of a burst marks when the edge arrived. Later ones would flatten the time surface and
            # inflate the flow speed (a known bias of surface-of-active-events flow). An event counts
            # as fresh if its pixel (same polarity) has not fired for `refractory_us`.
            prev = self._sae[p, y + self.R, x + self.R]
            fresh = (t - prev) > self.c.refractory_us
            # within the packet, keep only the first event per pixel and polarity
            key = (p * self.H + y) * self.W + x
            _, first = np.unique(key, return_index=True)
            once = np.zeros(n_ev, dtype=bool)
            once[first] = True
            fresh &= once
            x, y, t, p = x[fresh], y[fresh], t[fresh], p[fresh]
            self._sae[p, y + self.R, x + self.R] = t.astype(float)
            sel = (y >= self._ylo) & (y < self._yhi)
            rx, ry = x - self.cx, y - self.cy
            sel &= (rx * rx + ry * ry) >= self._rmin ** 2
            idx = np.flatnonzero(sel)
            if idx.size > self.c.max_events:
                idx = np.sort(self._rng.choice(idx, self.c.max_events, replace=False))
            if idx.size:
                vx, vy, ok = self._flow(x[idx], y[idx], t[idx], p[idx])
                if ok.sum() >= self.c.min_vectors:
                    vx, vy = vx[ok], vy[ok]
                    sp = np.sqrt(vx * vx + vy * vy)
                    nx, ny = vx / sp, vy / sp
                    rn = rx[idx][ok] * nx + ry[idx][ok] * ny          # r . n  (px)
                    if self.c.aggregate == "median":  # robust: median of the per-event inverse TTCs
                        good = np.abs(rn) > 1e-6
                        inv = float(np.median(sp[good] / rn[good])) if good.any() else float("nan")
                    else:                             # least squares over all flow vectors
                        inv = float((sp * rn).sum() / max((rn * rn).sum(), 1e-9))  # 1/s, > 0 = expanding
                    self.stats["flow_vectors"] += int(ok.sum())
        # smoothing on the sensor clock
        t_end = int(pkt.t_end_us)
        dt_s = 0.0 if self._t_last is None else max(0.0, (t_end - self._t_last) * 1e-6)
        self._t_last = t_end
        if math.isfinite(inv):
            k = 1.0 - math.exp(-dt_s / self.c.ema_s) if dt_s > 0 else 1.0
            self._s += k * (inv - self._s)
        self.last_inv_ttc = inv
        self.inv_ttc_smoothed = self._s
        ttc_est = (1.0 / self._s) if self._s > 1e-9 else float("inf")
        self._below = self._below + 1 if ttc_est <= self.c.tau_alarm_s else 0
        alarm = self._below >= 2
        frontal = float(min(1.0, max(0.0, self._s * self.c.tau_alarm_s)))
        self.stats["packets"] += 1
        self.stats["events"] += n_ev
        return ThreatReport(t_us=t_end, urgency=(0.0, frontal, 0.0), frontal=frontal, alarm=alarm,
                            n_events=n_ev, proc_s=time.perf_counter() - t0)
