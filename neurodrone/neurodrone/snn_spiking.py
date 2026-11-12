"""Fully spiking looming detector: every stage is a population of spiking neurons.

Same hand-designed logic and the same image geometry as `snn.LoomingSNN` (the "hybrid" baseline,
kept unchanged). There, motion detection, normalisation and the both-sides test are ordinary array
maths and only the last 6 neurons spike. Here every stage is made of neurons:

  layer 0  input        one spike per DVS event (pixel, polarity).
  layer 1  coincidence  time-difference-encoder neurons ("TDE", spiking elementary motion detectors,
                        cf. Milde et al. 2018), two per pixel x polarity x baseline:
                          OUT  a spike from the neighbour on the image-centre side *facilitates* it
                               for dt_min..dt_max (a facilitation trace with a rise and a decay);
                               a spike from its own pixel then *triggers* it -> it fires when an
                               edge has just moved outward through the pixel.
                          IN   the mirror image (facilitated by the neighbour on the far side).
  layer 2  region       one leaky integrate-and-fire "expansion" neuron (E) per image region.
                        OUT spikes from its region excite it, IN spikes inhibit it. Its threshold
                        rises with the region's recent total activity (adaptive threshold): busy
                        texture or noise raises the bar, so only a clear outward majority fires it.
                        This replaces the hybrid's z-score normalisation.
                        Regions: left / centre / right thirds (advisory), and the left and the right
                        half of every radial band (alarm path).
  layer 3  alarm        one coincidence neuron per band: it fires when the band's left-half and
                        right-half E neurons spike within `coinc_us` of each other. This replaces
                        min(z_left, z_right): an obstacle on a collision course expands on BOTH sides
                        of the direction of motion, while turning, shaking and sideways translation
                        make one side contract, so that side's E neuron stays silent.
  readout  leaky spike-rate traces of the alarm and advisory neurons -> the same ThreatReport as
           the hybrid (urgency left/centre/right, frontal, alarm). The alarm needs >= 2 alarm
           spikes close together, never a single one.

(An inhibitory "veto" from contraction neurons was tried for layer 3; it changed nothing in any
test scenario because the coincidence already does that job, so it was removed.)

Simulation is packet-independent: layer 1 is evaluated exactly per event (each event sees every
earlier event, including earlier events of the same packet), and layers 2-3 run on a global 1 ms
clock (`step_us`) whose last, still incomplete step is carried over to the next packet. So the
spikes do not depend on how the event stream is cut into packets; only the report times do (by at
most one step). The facilitation state of the layer-1 synapses is stored as the time of the last
facilitating spike per pixel. Spike and synaptic-operation counts are kept in `stats`, for energy
estimates on neuromorphic hardware.

Known limits: as the hybrid (needs texture/contrast; blank, thin and off-centre blank obstacles are
missed; left/right urgencies are biased by rotation). Tuned on the synthetic simulator only.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np

from .config import SNNConfig
from .types import EventPacket, ThreatReport

_NEG = -(1 << 60)  # "never fired"
_SHIFT = 1 << 40  # composite sort key: key * _SHIFT + time-in-packet (us)


class SpikingLoomingSNN:
    def __init__(self, width: int, height: int, cfg: Optional[SNNConfig] = None):
        self.cfg = cfg or SNNConfig()
        self.sp = self.cfg.spiking
        self.W, self.H = int(width), int(height)
        c = self.cfg
        if c.deltas_px is None:
            self.deltas: Tuple[int, ...] = tuple(sorted({max(1, round(self.W * 0.008)),
                                                         max(2, round(self.W * 0.016))}))
        else:
            self.deltas = tuple(int(d) for d in c.deltas_px)

        # ---- geometry: identical to snn.LoomingSNN (a test checks this) ----
        hw = self.W * self.H
        self._hw1 = hw + 1  # one extra never-firing pixel per polarity stands in for off-image neighbours
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
        edges = np.asarray(c.band_edges_frac, dtype=float) * self.W
        self.n_bands = len(edges) - 1
        band = np.clip(np.digitize(np.abs(xs - cx), edges[1:-1]), 0, self.n_bands - 1)
        self._bandsec = (3 + 2 * band + (xs > cx).astype(np.int64)).reshape(-1).astype(np.int64)
        self._nbins = 3 + 2 * self.n_bands
        self._up, self._dn = [], []
        for d in self.deltas:
            for sign, store in ((-1.0, self._up), (+1.0, self._dn)):
                qx = np.rint(xs + sign * d * ux).astype(np.int64)
                qy = np.rint(ys + sign * d * uy).astype(np.int64)
                inb = (qx >= 0) & (qx < self.W) & (qy >= 0) & (qy < self.H)
                store.append(np.where(inb, qy * self.W + qx, hw).reshape(-1))
        self._L = 3 + 2 * np.arange(self.n_bands)  # region index of each band's left half
        self._R = self._L + 1  # ... and right half

        self._T = np.full(2 * self._hw1, _NEG, dtype=np.int64)  # layer-1 facilitation state
        self.reset()

    # ------------------------------------------------------------------ bookkeeping
    @property
    def neuron_counts(self) -> Dict[str, int]:
        n_px = int(self._valid.sum())
        return {"layer 1 coincidence (motion)": 2 * n_px * 2 * len(self.deltas),  # polarity x pixel x OUT/IN x baseline
                "layer 2 region (expansion)": self._nbins,
                "layer 3 alarm (both sides)": self.n_bands}

    def reset(self) -> None:
        self._T.fill(_NEG)
        nb = self._nbins
        self._VE = np.zeros(nb)  # membrane of each region's E neuron
        self._A = np.zeros(nb)  # recent activity trace (drives the adaptive threshold)
        self._lastE = np.full(nb, _NEG, dtype=np.int64)  # last spike time of each E neuron
        self._rate = np.zeros(3 + self.n_bands)  # readout: advisory L/C/R, then one per band
        self._g: Optional[int] = None  # next global clock step to simulate
        self._pend = (np.zeros(nb), np.zeros(nb))  # layer-1 spikes already in that (incomplete) step
        self._t_last: Optional[int] = None
        self.last_counts = (np.zeros(nb), np.zeros(nb))  # layer-1 OUT / IN spikes per region, last packet
        self.stats = {"packets": 0, "events": 0, "layer1_spikes": 0, "layer2_spikes": 0,
                      "layer3_spikes": 0, "synops": 0}

    @property
    def V(self) -> np.ndarray:
        return self._VE

    @property
    def rate(self) -> np.ndarray:
        return self._rate

    # ------------------------------------------------------------------ layer 1
    def _layer1(self, pkt: EventPacket):
        """Exact per-event coincidence neurons. Returns (flat pixel, time, n_OUT, n_IN) for the
        in-ROI events, or None. Every event also updates the facilitation state."""
        c = self.cfg
        hw1 = self._hw1
        flat_all = pkt.y.astype(np.int64) * self.W + pkt.x.astype(np.int64)
        pol_all = pkt.p.astype(np.int64)
        t_all = pkt.t_us.astype(np.int64)
        key_all = pol_all * hw1 + flat_all
        t0 = int(t_all.min())
        comp = np.sort(key_all * _SHIFT + (t_all - t0))
        m = self._valid[flat_all]
        res = None
        if m.any():
            flat, pol, t = flat_all[m], pol_all[m], t_all[m]
            trel = t - t0
            n_out = np.zeros(flat.size)
            n_in = np.zeros(flat.size)
            for up, dn in zip(self._up, self._dn):
                for nb_idx, acc in ((up, n_out), (dn, n_in)):
                    q = pol * hw1 + nb_idx[flat]  # the facilitating neighbour (same polarity)
                    i = np.searchsorted(comp, q * _SHIFT + trel, side="left") - 1  # its last spike before t
                    cand = comp[np.maximum(i, 0)]
                    same_pkt = (i >= 0) & (cand // _SHIFT == q)
                    t_fac = np.where(same_pkt, cand % _SHIFT + t0, self._T[q])
                    dt = t - t_fac
                    acc += (dt >= c.dt_min_us) & (dt <= c.dt_max_us)
            res = (flat, t, n_out, n_in)
        np.maximum.at(self._T, key_all, t_all)
        return res

    # ------------------------------------------------------------------ layers 2-3
    def _decay(self, dt_s: float) -> None:
        """Leak only (no input). Exact: without input no neuron can reach its threshold."""
        sp = self.sp
        self._VE *= math.exp(-dt_s / sp.tau_m_s)
        self._A *= math.exp(-dt_s / sp.tau_a_s)
        self._rate *= math.exp(-dt_s / sp.tau_rate_s)

    def _step(self, t_s: int, o: np.ndarray, i: np.ndarray, d) -> Tuple[int, int]:
        sp = self.sp
        d_m, d_a, d_r = d
        self._A = self._A * d_a + (o + i)
        thr = sp.theta0 + sp.beta * self._A
        self._VE = np.maximum(self._VE * d_m + o - sp.kappa * i, sp.v_floor * sp.theta0)
        sE = self._VE >= thr
        self._VE[sE] = 0.0
        self._lastE[sE] = t_s
        eL, eR = sE[self._L], sE[self._R]
        fire = (eL & (t_s - self._lastE[self._R] <= sp.coinc_us)) | \
               (eR & (t_s - self._lastE[self._L] <= sp.coinc_us))
        self._rate[:3] = self._rate[:3] * d_r + sE[:3] / sp.tau_rate_s
        self._rate[3:] = self._rate[3:] * d_r + fire / sp.tau_rate_s
        return int(sE.sum()), int(fire.sum())

    # ------------------------------------------------------------------ main
    def process(self, pkt: EventPacket) -> ThreatReport:
        t_cpu = time.perf_counter()
        sp = self.sp
        nb = self._nbins
        step = int(sp.step_us)
        t_start, t_end = int(pkt.t_start_us), int(pkt.t_end_us)
        if self._g is None:
            self._g = t_start // step
        g0 = self._g
        g_end = max(t_end // step, g0)  # the step containing t_end is not complete yet
        n = g_end - g0 + 1
        cnt_out = np.zeros((n, nb))
        cnt_in = np.zeros((n, nb))
        cnt_out[0] += self._pend[0]
        cnt_in[0] += self._pend[1]
        n_l1 = 0
        out_tot, in_tot = np.zeros(nb), np.zeros(nb)
        if pkt.n:
            l1 = self._layer1(pkt)
            if l1 is not None:
                flat, t, n_out, n_in = l1
                k = np.clip(t // step - g0, 0, n - 1)
                for ids in (self._sector[flat], self._bandsec[flat]):
                    bo = np.bincount(k * nb + ids, weights=n_out, minlength=n * nb).reshape(n, nb)
                    bi = np.bincount(k * nb + ids, weights=n_in, minlength=n * nb).reshape(n, nb)
                    cnt_out += bo
                    cnt_in += bi
                    out_tot += bo.sum(0)
                    in_tot += bi.sum(0)
                n_l1 = int(n_out.sum() + n_in.sum())
        self.last_counts = (out_tot, in_tot)

        dt1 = step / 1e6
        d = (math.exp(-dt1 / sp.tau_m_s), math.exp(-dt1 / sp.tau_a_s), math.exp(-dt1 / sp.tau_rate_s))
        l2 = l3 = 0
        peak = float(self._rate[3:].max())
        busy = np.flatnonzero((cnt_out[:-1] + cnt_in[:-1]).sum(1) > 0)  # complete steps with input
        done = 0  # steps simulated so far, relative to g0
        for s in busy:
            if s > done:
                self._decay((s - done) * dt1)  # empty steps in between: leak only
            a, b = self._step((g0 + s + 1) * step, cnt_out[s], cnt_in[s], d)
            l2 += a
            l3 += b
            done = s + 1
            peak = max(peak, float(self._rate[3:].max()))
        if n - 1 > done:
            self._decay((n - 1 - done) * dt1)
        self._g = g_end
        self._pend = (cnt_out[-1].copy(), cnt_in[-1].copy())
        self._t_last = t_end

        n_ev = int(pkt.n)
        st = self.stats
        st["packets"] += 1
        st["events"] += n_ev
        st["layer1_spikes"] += n_l1
        st["layer2_spikes"] += l2
        st["layer3_spikes"] += l3
        # Synaptic operations, as a neuromorphic chip would count them: every event is the trigger
        # input of its own OUT/IN neurons and the facilitating input of its neighbours' (4 per
        # baseline); every layer-1 spike reaches its sector's and its band half's E neuron (2);
        # every layer-2/3 spike reaches one alarm neuron or the readout (1).
        st["synops"] += 4 * len(self.deltas) * n_ev + 2 * n_l1 + l2 + l3

        urgency = np.clip(self._rate[:3] / sp.adv_rate_full, 0.0, 1.0)
        return ThreatReport(
            t_us=t_end,
            urgency=(float(urgency[0]), float(urgency[1]), float(urgency[2])),
            frontal=float(min(1.0, peak / sp.rate_alarm)),
            alarm=bool(peak >= sp.rate_alarm),
            n_events=n_ev,
            proc_s=time.perf_counter() - t_cpu,
        )
