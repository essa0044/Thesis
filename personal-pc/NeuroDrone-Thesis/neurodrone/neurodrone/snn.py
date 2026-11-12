"""Event-driven spiking looming detector (the "fast loop").

How it works (hand-designed, no training needed):

1. Time surface. For every pixel and polarity we remember the timestamp of its last event.
2. Elementary motion detectors (EMDs). An event at pixel p is *outward-correlated* if the
   neighbour delta pixels closer to the image centre (the focus of expansion, FOE) fired
   between dt_min and dt_max earlier - i.e. an edge just moved outward through p. It is
   *inward-correlated* if the neighbour on the far side fired. This is a delay-and-coincidence
   detector, the classic insect motion-detection circuit, evaluated on event timestamps.
3. Divisive normalisation. Per region, z = (out - kappa*in) / sqrt(out + in + eps).
   Uniform image translation (yaw, wind, vibration) gives out ~ in, so z ~ N(0,1) whatever
   the scene activity. True expansion gives out >> in, so z is large and positive.
4. Advisory neurons: left / centre / right thirds. Their output is a rough "something close
   on that side" cue for sidestep direction. They ARE fooled by rotation; the coordinator
   only uses them while the measured angular rate is small.
5. Alarm neurons: one per radial band, driven by min(z_left_half, z_right_half) of that band.
   A static obstacle is on a collision course only if it straddles the direction of motion
   (the FOE), so it expands on BOTH sides of the FOE; translation gives mirror-image pixels
   opposite-sign flow, so the min is never positive. The min is taken BEFORE integration so
   the slowly decaying trace of one side can't overlap with the onset on the other side when
   a jitter reverses. (Earlier variants - a signed sum over the centre sector, or a min taken
   after the rate traces - false-alarmed under attitude jitter in testing.)
6. Leaky integrate-and-fire neurons, spike -> reset, plus a leaky spike-rate trace per neuron
   giving urgency in 0..1 and the alarm (alarm = any band neuron's rate >= rate_alarm).

Known limits (state them in the thesis):
  * Needs texture/contrast: blank walls, glass, thin wires, low light and flicker defeat it.
    An obstacle that is blank AND off-centre exposes only one edge and is invisible to the
    mirrored-pair test. Pair this with a range sensor (see coordinator `ClearanceReport`).
  * Detection happens at roughly constant time-to-contact for a given obstacle size, so
    small obstacles are detected late and thin ones (poles, wires) not at all.
  * Left/right urgencies are biased by rotation (no gyro derotation yet).
"""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np

from .config import SNNConfig
from .types import EventPacket, ThreatReport

_NEG = -(1 << 60)  # "never fired"


class LoomingSNN:
    def __init__(self, width: int, height: int, cfg: Optional[SNNConfig] = None):
        self.cfg = cfg or SNNConfig()
        self.W, self.H = int(width), int(height)
        c = self.cfg
        if c.deltas_px is None:
            self.deltas: Tuple[int, ...] = tuple(sorted({max(1, round(self.W * 0.008)),
                                                         max(2, round(self.W * 0.016))}))
        else:
            self.deltas = tuple(int(d) for d in c.deltas_px)

        hw = self.W * self.H
        ys, xs = np.mgrid[0:self.H, 0:self.W]
        cx, cy = (self.W - 1) / 2.0, (self.H - 1) / 2.0
        rx, ry = xs - cx, ys - cy
        r = np.hypot(rx, ry)
        rs = np.maximum(r, 1e-6)
        ux, uy = rx / rs, ry / rs
        y0, y1 = c.roi_y
        valid = (r >= c.r_min_frac * self.W) & (ys >= y0 * self.H) & (ys < y1 * self.H)
        self._valid = valid.reshape(-1)
        self._sector = np.minimum((xs * 3) // self.W, 2).reshape(-1).astype(np.int64)
        # radial bands of |x - cx|, split into left/right halves -> bins 3 + 2*band + side
        edges = np.asarray(c.band_edges_frac, dtype=float) * self.W
        self.n_bands = len(edges) - 1
        band = np.clip(np.digitize(np.abs(xs - cx), edges[1:-1]), 0, self.n_bands - 1)
        self._bandsec = (3 + 2 * band + (xs > cx).astype(np.int64)).reshape(-1).astype(np.int64)
        self._nbins = 3 + 2 * self.n_bands

        # For every pixel: flat index of the neighbour `delta` px towards the FOE (upstream)
        # and `delta` px away from it (downstream). Index hw is a permanent "never fired" slot.
        self._up, self._dn = [], []
        for d in self.deltas:
            for sign, store in ((-1.0, self._up), (+1.0, self._dn)):
                qx = np.rint(xs + sign * d * ux).astype(np.int64)
                qy = np.rint(ys + sign * d * uy).astype(np.int64)
                inb = (qx >= 0) & (qx < self.W) & (qy >= 0) & (qy < self.H)
                store.append(np.where(inb, qy * self.W + qx, hw).reshape(-1))

        self._T = np.full((2, hw + 1), _NEG, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._T.fill(_NEG)
        nn = 3 + self.n_bands  # neurons: left, centre, right (advisory) + one coincidence neuron per band
        self._V = np.zeros(nn)
        self._rate = np.zeros(nn)
        self._t_last: Optional[int] = None
        self.last_counts = (np.zeros(self._nbins), np.zeros(self._nbins))  # (outward, inward), for debugging
        self.last_drive = np.zeros(nn)

    @property
    def V(self) -> np.ndarray:
        return self._V

    @property
    def rate(self) -> np.ndarray:
        return self._rate

    def process(self, pkt: EventPacket) -> ThreatReport:
        t_cpu = time.perf_counter()
        c = self.cfg
        t_end = int(pkt.t_end_us)
        if self._t_last is None:
            dt = max((t_end - int(pkt.t_start_us)) / 1e6, 1e-3)
        else:
            dt = max((t_end - self._t_last) / 1e6, 1e-4)
        self._t_last = t_end

        out = np.zeros(self._nbins)
        inn = np.zeros(self._nbins)
        if pkt.n:
            pol_all = pkt.p.astype(np.int64)
            flat_all = pkt.y.astype(np.int64) * self.W + pkt.x.astype(np.int64)
            m = self._valid[flat_all]
            if m.any():
                flat, pol, t = flat_all[m], pol_all[m], pkt.t_us[m]
                ids = np.concatenate([self._sector[flat], self._bandsec[flat]])
                for up, dn in zip(self._up, self._dn):
                    dtu = t - self._T[pol, up[flat]]
                    dtd = t - self._T[pol, dn[flat]]
                    co = (dtu >= c.dt_min_us) & (dtu <= c.dt_max_us)
                    ci = (dtd >= c.dt_min_us) & (dtd <= c.dt_max_us)
                    out += np.bincount(ids, weights=np.concatenate([co, co]), minlength=self._nbins)
                    inn += np.bincount(ids, weights=np.concatenate([ci, ci]), minlength=self._nbins)
            # update the time surface with every event (neighbours may lie outside the ROI)
            np.maximum.at(self._T, (pol_all, flat_all), pkt.t_us)

        z = (out - c.kappa * inn) / np.sqrt(out + inn + c.z_eps)
        drive = np.concatenate([z[:3], [min(z[3 + 2 * k], z[4 + 2 * k]) for k in range(self.n_bands)]])
        self.last_counts = (out, inn)
        self.last_drive = drive

        # leaky integrate-and-fire
        self._V = self._V * math.exp(-dt / c.tau_m_s) + c.w_syn * drive
        np.maximum(self._V, c.v_floor * c.theta, out=self._V)
        spiked = self._V >= c.theta
        self._V[spiked] = 0.0
        self._rate = self._rate * math.exp(-dt / c.tau_rate_s) + spiked / c.tau_rate_s

        rate = self._rate
        urgency = np.clip(rate[:3] / c.rate_alarm, 0.0, 1.0)
        frontal_rate = float(rate[3:].max())
        frontal = min(1.0, frontal_rate / c.rate_alarm)
        return ThreatReport(
            t_us=t_end,
            urgency=(float(urgency[0]), float(urgency[1]), float(urgency[2])),
            frontal=float(frontal),
            alarm=bool(frontal_rate >= c.rate_alarm),
            n_events=int(pkt.n),
            proc_s=time.perf_counter() - t_cpu,
        )
