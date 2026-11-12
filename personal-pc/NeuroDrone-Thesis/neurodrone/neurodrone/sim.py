"""Minimal synthetic world used to test and tune the SNN + coordinator without hardware.

It is deliberately simple: textured billboards in a 2.5D world, an analytic distant
background, an ideal DVS model (log-intensity threshold crossings + shot noise) and a
point-mass drone. It cannot tell you how the system performs on a real sensor in a real
scene; it can tell you whether the *logic* works and catch regressions.

World frame: x forward (north), y right (east), z up. Heading psi (rad) rotates x -> y.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .types import EventPacket, VelocityCmd


# --------------------------------------------------------------------------- DVS model
class DVSSimulator:
    """Ideal event camera: an event whenever log-intensity moves by `contrast_thresh`."""

    def __init__(self, width: int, height: int, contrast_thresh: float = 0.2,
                 noise_hz: float = 0.3, seed: int = 0):
        self.W, self.H = width, height
        self.C = contrast_thresh
        self.noise_hz = noise_hz
        self.rng = np.random.default_rng(seed)
        self._ref: Optional[np.ndarray] = None

    def step(self, img: np.ndarray, t0_us: int, t1_us: int) -> EventPacket:
        L = np.log(img)
        if self._ref is None:
            self._ref = L.copy()
            return EventPacket.empty(t0_us, t1_us)
        diff = L - self._ref
        n = np.floor(np.abs(diff) / self.C).astype(np.int64)
        ys, xs = np.nonzero(n)
        cnt = n[ys, xs]
        xe = np.repeat(xs, cnt)
        ye = np.repeat(ys, cnt)
        pe = np.repeat((diff[ys, xs] > 0).astype(np.int8), cnt)
        tot = int(cnt.sum())
        if tot:
            k = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)  # 0..cnt-1 within pixel
            frac = (k + 0.5) / np.repeat(cnt, cnt)
            te = t0_us + (frac * (t1_us - t0_us)).astype(np.int64)
            self._ref[ys, xs] += np.sign(diff[ys, xs]) * cnt * self.C
        else:
            te = np.zeros(0, dtype=np.int64)
        # shot noise
        dt_s = (t1_us - t0_us) / 1e6
        nn = int(self.rng.poisson(self.noise_hz * self.W * self.H * dt_s))
        if nn:
            xe = np.concatenate([xe, self.rng.integers(0, self.W, nn)])
            ye = np.concatenate([ye, self.rng.integers(0, self.H, nn)])
            pe = np.concatenate([pe, self.rng.integers(0, 2, nn).astype(np.int8)])
            te = np.concatenate([te, self.rng.integers(t0_us, max(t1_us, t0_us + 1), nn)])
        order = np.argsort(te, kind="stable")
        return EventPacket(xe[order].astype(np.int16), ye[order].astype(np.int16),
                           te[order].astype(np.int64), pe[order], t0_us, t1_us)


# --------------------------------------------------------------------------- world
@dataclass
class Box:
    """A vertical billboard facing the drone. `texture`: 'checker' or 'blank' (dark, featureless)."""

    x: float
    y: float
    w: float = 2.0
    h: float = 2.0
    z: float = 1.5  # centre height above ground (same units as drone z)
    texture: str = "checker"
    k: float = 3.0  # checker cells across the box


@dataclass
class Camera:
    width: int = 128
    height: int = 96
    f_px: float = 100.0

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / (2 * self.f_px)))


class World:
    def __init__(self, boxes: List[Box], cam: Optional[Camera] = None):
        self.boxes = list(boxes)
        self.cam = cam or Camera()
        self._ys, self._xs = np.mgrid[0:self.cam.height, 0:self.cam.width]
        self._bg_cache: Tuple[float, Optional[np.ndarray]] = (float("nan"), None)

    # distant background: depends on yaw only (rotation shifts it, translation does not)
    def _background(self, yaw: float) -> np.ndarray:
        shift = self.cam.f_px * yaw
        if self._bg_cache[1] is not None and self._bg_cache[0] == shift:
            return self._bg_cache[1]
        X = self._xs + shift
        Y = self._ys
        bg = (0.5 + 0.16 * np.sin(0.55 * X + 0.21 * Y + 1.3)
              + 0.12 * np.sin(0.31 * X - 0.47 * Y + 0.4) + 0.10 * np.sin(0.85 * X + 0.1))
        self._bg_cache = (shift, bg)
        return bg

    def to_body(self, pose, b: Box) -> Tuple[float, float, float]:
        px, py, pz, yaw = pose
        dx, dy = b.x - px, b.y - py
        fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
        right = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return fwd, right, b.z - pz

    def project(self, pose, b: Box):
        """(depth, u_px, v_px, half_w_px, half_h_px) or None when behind the camera."""
        fwd, right, up = self.to_body(pose, b)
        if fwd < 0.3:
            return None
        f = self.cam.f_px
        return (fwd, self.cam.width / 2 + f * right / fwd, self.cam.height / 2 - f * up / fwd,
                f * b.w / (2 * fwd), f * b.h / (2 * fwd))

    def render(self, pose) -> np.ndarray:
        img = self._background(pose[3]).copy()
        projs = [(self.project(pose, b), b) for b in self.boxes]
        projs = [pb for pb in projs if pb[0] is not None]
        for (depth, bx, by, hx, hy), b in sorted(projs, key=lambda pb: -pb[0][0]):
            x0, x1 = max(0, int(bx - hx) - 2), min(self.cam.width, int(bx + hx) + 3)
            y0, y1 = max(0, int(by - hy) - 2), min(self.cam.height, int(by + hy) + 3)
            if x0 >= x1 or y0 >= y1:
                continue
            xs = self._xs[y0:y1, x0:x1]
            ys = self._ys[y0:y1, x0:x1]
            u = (xs - bx) / max(hx, 1e-3)
            v = (ys - by) / max(hy, 1e-3)
            m = (np.clip((1 - np.abs(u)) * hx + 0.5, 0, 1) * np.clip((1 - np.abs(v)) * hy + 0.5, 0, 1))
            if b.texture == "blank":
                tex = 0.12
            else:
                tex = 0.5 + 0.4 * np.sin(math.pi * b.k * u) * np.sin(math.pi * b.k * v)
            img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - m) + tex * m
        return np.clip(img, 0.05, 0.95)

    def collided(self, pose, radius: float = 0.35) -> bool:
        for b in self.boxes:
            fwd, right, up = self.to_body(pose, b)
            if -radius <= fwd <= radius and abs(right) <= b.w / 2 + radius and abs(up) <= b.h / 2 + radius:
                return True
        return False

    def range_sensor(self, pose, max_range: float = 4.0, corridor: float = 0.5, vert: float = 0.3) -> float:
        """Ideal forward range sensor (stand-in for a depth camera): nearest billboard in a corridor
        ahead, inf beyond max_range. Unlike the DVS it sees textureless surfaces."""
        best = float("inf")
        for b in self.boxes:
            fwd, right, up = self.to_body(pose, b)
            if fwd > 0.0 and abs(right) <= b.w / 2 + corridor and abs(up) <= b.h / 2 + vert:
                best = min(best, fwd)
        return best if best <= max_range else float("inf")

    def clearance(self, pose) -> float:
        """Distance (m) to the nearest billboard plane that is ahead and laterally overlapping."""
        best = float("inf")
        for b in self.boxes:
            fwd, right, up = self.to_body(pose, b)
            if fwd > 0 and abs(right) <= b.w / 2 + 0.35:
                best = min(best, fwd)
        return best


# --------------------------------------------------------------------------- dynamics
class PointMassDrone:
    """Velocity-tracking point mass: first-order lag with an acceleration limit (like PX4's
    velocity controller, roughly). State is kept in the world frame, commands in body FRD."""

    def __init__(self, x=0.0, y=0.0, z=1.5, yaw_deg=0.0, a_max=2.0, tau_v=0.3, tau_yaw=0.2):
        self.x, self.y, self.z = x, y, z
        self.yaw = math.radians(yaw_deg)
        self.v = np.zeros(3)  # body FRD velocity
        self.yaw_rate = 0.0  # rad/s
        self.a_max, self.tau_v, self.tau_yaw = a_max, tau_v, tau_yaw

    @property
    def pose(self):
        return (self.x, self.y, self.z, self.yaw)

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw)

    def step(self, cmd: VelocityCmd, dt: float) -> None:
        tgt = np.array([cmd.vx, cmd.vy, cmd.vz])
        a = np.clip((tgt - self.v) / self.tau_v, -self.a_max, self.a_max)
        self.v = self.v + a * dt
        yr_t = math.radians(cmd.yaw_rate)
        self.yaw_rate += (yr_t - self.yaw_rate) * min(1.0, dt / self.tau_yaw)
        self.yaw += self.yaw_rate * dt
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (self.v[0] * c - self.v[1] * s) * dt
        self.y += (self.v[0] * s + self.v[1] * c) * dt
        self.z -= self.v[2] * dt
