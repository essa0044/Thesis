"""VLM-goal mission with a classical depth guard (the thesis stack, no SNN).

    RGB  -> VLM box -> 3-D goal (box centre ray x depth)      every ~1-2 s, consensus over a few answers
    depth -> corridor check along the commanded direction     every control step (10-30 Hz)
    goal + guard -> velocity setpoint for PX4 (OFFBOARD)

Everything here is pure (no I/O, no clock): unit-tested, identical in Gazebo and on the drone.

Frames. Camera optical: x right, y down, z forward. Body FRD: x forward, y right, z down (the camera looks
along body x; cam_offset_frd is the camera position in the body frame). World: PX4 local NED.
A pinhole camera is described by its image size and horizontal field of view (square pixels).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------- geometry
@dataclass
class Pinhole:
    width: int
    height: int
    hfov_rad: float

    @property
    def f(self) -> float:
        return (self.width / 2.0) / math.tan(self.hfov_rad / 2.0)

    def ray(self, u: float, v: float) -> np.ndarray:
        """Unit-depth ray (body FRD, x = 1) through pixel (u, v) (pixel centres at +0.5)."""
        return np.array([1.0, (u - self.width / 2.0) / self.f, (v - self.height / 2.0) / self.f])

    def pixel(self, p_frd: np.ndarray) -> Optional[Tuple[float, float]]:
        if p_frd[0] <= 1e-6:
            return None
        return (p_frd[1] / p_frd[0] * self.f + self.width / 2.0, p_frd[2] / p_frd[0] * self.f + self.height / 2.0)


def R_ned_from_frd(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Body FRD -> local NED (ZYX Euler, as PX4/MAVSDK report it)."""
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                     [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                     [-sp, sr * cp, cr * cp]])


def box_depth(depth_m: np.ndarray, depth_cam: Pinhole, rgb_cam: Pinhole, box: Sequence[float],
              shrink: float = 0.5, min_valid: int = 5, dmin: float = 0.2, dmax: float = 20.0) -> Optional[float]:
    """Distance (along the camera axis) of the object in a normalised RGB box [x1, y1, x2, y2] (0..1): the median
    depth of the central part of the box (shrink = fraction of the box kept), mapped into the depth image by ray
    angle (the two cameras share an origin and axis but can differ in size and field of view)."""
    x1, y1, x2, y2 = box
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * shrink / 2, (y2 - y1) * shrink / 2
    corners = []
    for ux, vy in ((cx - hw, cy - hh), (cx + hw, cy + hh)):
        q = depth_cam.pixel(rgb_cam.ray(ux * rgb_cam.width, vy * rgb_cam.height))
        corners.append(q)
    (u0, v0), (u1, v1) = corners
    h, w = depth_m.shape[:2]
    a0, a1 = int(max(0, math.floor(min(u0, u1)))), int(min(w, math.ceil(max(u0, u1)) + 1))
    b0, b1 = int(max(0, math.floor(min(v0, v1)))), int(min(h, math.ceil(max(v0, v1)) + 1))
    if a0 >= a1 or b0 >= b1:
        return None
    win = depth_m[b0:b1, a0:a1].astype(np.float64)
    ok = np.isfinite(win) & (win >= dmin) & (win <= dmax)
    if ok.sum() < min_valid:
        return None
    return float(np.median(win[ok]))


def box_at_edge(box: Sequence[float], margin: float = 0.02) -> bool:
    """The box touches the left or right image edge: the object is cut off, its centre is not the object's centre, and
    the depth inside it can belong to something in front (14 Nov: the green box at the image edge behind a pole gave the
    pole's distance, and the drone stopped 1.5 m before the pole, 6 m short). No goal from such a box: turn first."""
    return box[0] <= margin or box[2] >= 1.0 - margin


def goal_from_box(box: Sequence[float], dist_m: float, rgb_cam: Pinhole, state, cam_offset_frd=(0.12, -0.03, -0.24),
                  standoff_m: float = 1.5, keep_alt: bool = True) -> np.ndarray:
    """NED point to fly to: along the ray through the box centre, `standoff_m` short of the object; at the current
    altitude if keep_alt (the VLM picks WHERE, not how high - as in VLN on the Fly)."""
    cx, cy = (box[0] + box[2]) / 2 * rgb_cam.width, (box[1] + box[3]) / 2 * rgb_cam.height
    ray = rgb_cam.ray(cx, cy)
    p_obj = np.asarray(cam_offset_frd) + ray * dist_m  # body FRD
    R = R_ned_from_frd(state.roll_deg, state.pitch_deg, state.yaw_deg)
    pos = np.array([state.north_m, state.east_m, state.down_m])
    obj = pos + R @ p_obj
    d = obj - pos
    d[2] = 0.0 if keep_alt else d[2]
    n = float(np.linalg.norm(d))
    goal = obj - d / n * min(standoff_m, n) if n > 1e-6 else pos.copy()
    if keep_alt:
        goal[2] = pos[2]
    return goal


# ----------------------------------------------------------------------------- depth guard
@dataclass
class GuardResult:
    dist_ahead_m: float  # closest obstacle in the corridor (inf = none in range)
    ttc_s: float  # dist / speed (inf if not moving towards anything)
    blocked: bool
    reason: str = ""


def level_rotation(att_deg: Optional[Tuple[float, float]]) -> Optional[np.ndarray]:
    """Camera/body FRD -> heading-aligned level frame (roll and pitch removed), from (roll_deg, pitch_deg)."""
    if att_deg is None:
        return None
    return R_ned_from_frd(float(att_deg[0]), float(att_deg[1]), 0.0)


def corridor_distance(depth_m: np.ndarray, cam: Pinhole, dir_frd: np.ndarray, radius_m: float = 0.6,
                      dmin: float = 0.2, dmax: float = 15.0, step: int = 2, k_robust: int = 8,
                      R_level: Optional[np.ndarray] = None) -> float:
    """Distance to the closest obstacle inside a corridor of radius_m around the direction dir_frd (body FRD, from
    the camera): the k_robust-th closest along-track distance, so single noisy pixels do not count. inf = clear
    (as far as the camera sees: only directions inside its field of view are covered)."""
    n = float(np.linalg.norm(dir_frd))
    if n < 1e-6:
        return math.inf
    dvec = np.asarray(dir_frd, float) / n
    d = depth_m[::step, ::step].astype(np.float64)
    h, w = depth_m.shape[:2]
    vs, us = np.mgrid[0:h:step, 0:w:step]
    ok = np.isfinite(d) & (d >= dmin) & (d <= dmax)
    Z = d[ok]
    P = np.stack([Z, (us[ok] + 0.5 - w / 2) / cam.f * Z, (vs[ok] + 0.5 - h / 2) / cam.f * Z], -1)  # FRD
    if R_level is not None:  # points into the level frame, so the corridor stays horizontal when the drone tilts
        P = P @ R_level.T
    s = P @ dvec
    miss = np.linalg.norm(P - s[:, None] * dvec, axis=-1)
    sel = np.sort(s[(s > 0) & (miss < radius_m)])
    return float(sel[k_robust - 1]) if sel.size >= k_robust else math.inf


def guard(depth_m: Optional[np.ndarray], cam: Pinhole, v_frd: np.ndarray, depth_age_s: float,
          radius_m: float = 0.8, stop_dist_m: float = 1.2, ttc_s: float = 2.0, max_age_s: float = 0.5,
          att_deg: Optional[Tuple[float, float]] = None) -> GuardResult:
    """Block the commanded motion if an obstacle in its corridor is closer than stop_dist_m or would be reached in
    less than ttc_s. No (fresh) depth image -> blocked: never fly blind.
    att_deg = (roll, pitch) of the drone: the corridor is then horizontal along the commanded (horizontal) velocity
    instead of tilting with the camera. Without it, a drone pitched forward 10-15 deg while accelerating at 1.2-1.4 m
    height had the floor inside its corridor and stopped in an empty world (14 Nov)."""
    if depth_m is None or depth_age_s > max_age_s:
        return GuardResult(math.inf, math.inf, True, "no fresh depth")
    sp = float(np.linalg.norm(v_frd))
    if sp < 0.05:
        return GuardResult(math.inf, math.inf, False, "")
    dist = corridor_distance(depth_m, cam, v_frd, radius_m, R_level=level_rotation(att_deg))
    t = dist / sp if math.isfinite(dist) else math.inf
    if dist < stop_dist_m:
        return GuardResult(dist, t, True, f"obstacle {dist:.1f} m ahead")
    if t < ttc_s:
        return GuardResult(dist, t, True, f"contact in {t:.1f} s")
    return GuardResult(dist, t, False, "")


# ----------------------------------------------------------------------------- goal consensus
@dataclass
class Consensus:
    """Accept a VLM goal only when `need` of the last `window` answers agree within `tol_m` (VLN on the Fly uses 3
    consistent answers; a single answer is wrong too often to fly on). 'Not found' answers count as votes too."""
    need: int = 2
    window: int = 3
    tol_m: float = 1.5
    hist: List[Optional[np.ndarray]] = field(default_factory=list)

    def add(self, goal: Optional[np.ndarray]) -> Optional[np.ndarray]:
        self.hist = (self.hist + [None if goal is None else np.asarray(goal, float)])[-self.window:]
        pts = [g for g in self.hist if g is not None]
        best = None
        for g in pts:
            close = [h for h in pts if np.linalg.norm((h - g)[:2]) <= self.tol_m]
            if len(close) >= self.need and (best is None or len(close) > len(best)):
                best = close
        return np.mean(best, axis=0) if best else None


# ----------------------------------------------------------------------------- velocity command
@dataclass
class Command:
    vx: float  # body forward, m/s
    vy: float  # body right
    vz: float  # down
    yaw_rate: float  # deg/s
    phase: str
    note: str = ""


def steer(goal_ned: Optional[np.ndarray], state, vmax: float = 1.0, k_pos: float = 0.6, k_yaw: float = 1.5,
          max_yaw_rate: float = 30.0, arrive_m: float = 0.5, yaw_first_deg: float = 30.0) -> Command:
    """P-control to the goal: turn to face it (the camera, and so the guard, look forward), then fly forward with
    speed proportional to the distance, capped at vmax. Altitude held to the goal's."""
    if goal_ned is None:
        return Command(0, 0, 0, 0, "search", "no goal")
    pos = np.array([state.north_m, state.east_m, state.down_m])
    d = np.asarray(goal_ned) - pos
    dh = math.hypot(d[0], d[1])
    vz = max(-0.5, min(0.5, k_pos * d[2]))
    if dh < arrive_m:
        return Command(0, 0, vz, 0, "arrived")
    bearing = math.degrees(math.atan2(d[1], d[0]))
    err = (bearing - state.yaw_deg + 180) % 360 - 180
    yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, k_yaw * err))
    if abs(err) > yaw_first_deg:
        return Command(0, 0, vz, yaw_rate, "turn")
    v = min(vmax, k_pos * dh) * math.cos(math.radians(err))
    return Command(v, 0, vz, yaw_rate, "fly")


def apply_guard(cmd: Command, g: GuardResult) -> Command:
    if g.blocked and (abs(cmd.vx) > 0 or abs(cmd.vy) > 0):
        return Command(0, 0, cmd.vz, cmd.yaw_rate, "blocked", g.reason)
    return cmd


@dataclass
class Avoider:
    """Reactive go-around, no map: when the guard blocks the path to the goal, try directions turned by
    step_deg, 2 * step_deg ... up to max_angle_deg to either side (body frame, heading kept towards the goal, so all
    of them stay inside the depth camera's view), and fly the first one whose corridor is free for clear_m at a
    reduced speed. A direction is only flown if the camera can see it: nothing sideways or backwards. After the
    straight path is free again, the go-around direction is kept for commit_s so the drone passes the obstacle with
    margin instead of turning back into it as soon as it leaves the camera's view. If no direction is free the drone
    stops and hovers (as without the avoider)."""
    max_angle_deg: float = 30.0
    step_deg: float = 10.0
    speed: float = 0.5
    clear_m: float = 2.5
    commit_s: float = 2.0
    angle: Optional[float] = None
    until: float = -1.0

    def _candidates(self, blocked: bool):
        if not blocked:
            return [self.angle]  # inside the commit time: only the direction already chosen
        first = 1.0 if (self.angle or 0.0) >= 0 else -1.0  # keep to the side chosen before
        out = []
        k = 1
        while k * self.step_deg <= self.max_angle_deg + 1e-9:
            out += [first * k * self.step_deg, -first * k * self.step_deg]
            k += 1
        return out

    def step(self, cmd: Command, depth_m: Optional[np.ndarray], cam: Pinhole, depth_age_s: float, now: float,
             radius_m: float = 0.8, stop_dist_m: float = 1.2, ttc_s: float = 2.0,
             att_deg: Optional[Tuple[float, float]] = None):
        """-> (command to send, guard result for the straight path). att_deg = (roll, pitch): see guard()."""
        v = np.array([cmd.vx, cmd.vy, 0.0])
        g = guard(depth_m, cam, v, depth_age_s, radius_m, stop_dist_m, ttc_s, att_deg=att_deg)
        if cmd.phase != "fly" or g.reason == "no fresh depth":
            self.angle = None
            return apply_guard(cmd, g), g
        if not g.blocked and (self.angle is None or now >= self.until):
            self.angle = None
            return cmd, g
        best = None
        for a in self._candidates(g.blocked):
            d = np.array([math.cos(math.radians(a)), math.sin(math.radians(a)), 0.0])
            if corridor_distance(depth_m, cam, d, radius_m, R_level=level_rotation(att_deg)) >= self.clear_m:
                best = a
                break
        if best is None:
            self.angle = None
            if g.blocked:
                return Command(0, 0, cmd.vz, cmd.yaw_rate, "blocked", g.reason + "; no free way around in view"), g
            return cmd, g  # committed direction no longer free but the straight one is
        if g.blocked:
            self.until = now + self.commit_s
        self.angle = best
        sp = min(self.speed, max(math.hypot(cmd.vx, cmd.vy), 0.2))
        r = math.radians(best)
        return Command(sp * math.cos(r), sp * math.sin(r), cmd.vz, cmd.yaw_rate, "avoid",
                       f"going around {'right' if best > 0 else 'left'} {abs(best):.0f} deg"), g


@dataclass
class Searcher:
    """Turn-stop-look search while no goal is accepted. A VLM answer takes several seconds, so turning continuously
    means the object has left the picture before the next answer and two answers never agree (12 Nov: 0 of 6 runs).
    Instead: turn to the next heading of a sweep around the start heading (0, +30, -30, +60, -60 ... 180 deg), hold
    still until a VLM answer arrives for a picture taken after the turn ended, stay while the answers find the object
    (the goal consensus then accepts it), and turn on as soon as an answer finds nothing."""
    offsets: Tuple[float, ...] = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)
    max_rate: float = 20.0  # deg/s
    k_yaw: float = 1.5
    tol_deg: float = 3.0
    yaw0: Optional[float] = None
    i: int = 0
    settled_at: Optional[float] = None

    def step(self, yaw_deg: float, now: float, last: Optional[dict]):
        """-> (yaw_rate deg/s, note). last: the newest VLM answer, dict with found and t_capture (monotonic s)."""
        if self.yaw0 is None:
            self.yaw0 = yaw_deg
        off = self.offsets[self.i % len(self.offsets)]
        err = (self.yaw0 + off - yaw_deg + 180.0) % 360.0 - 180.0
        if abs(err) > self.tol_deg:
            self.settled_at = None
            return max(-self.max_rate, min(self.max_rate, self.k_yaw * err)), f"turning to {off:+.0f} deg"
        if self.settled_at is None:
            self.settled_at = now
        if last and last.get("t_capture") is not None and last["t_capture"] >= self.settled_at:
            if not last.get("usable", last.get("found")):  # found only at the image edge counts as not found here
                self.i += 1
                self.settled_at = None
                return 0.0, f"nothing at {off:+.0f} deg, next view"
            return 0.0, f"found at {off:+.0f} deg, waiting for agreement"
        return 0.0, f"looking at {off:+.0f} deg"


@dataclass
class ActiveSearcher:
    """Active-sensing search: the VLM answers decide where to look next (the baseline Searcher sweeps a fixed list).

    Belief: probability that the target lies in each of `bins` directions (world bearing, uniform at the start).
    Each answer, for a picture taken at heading psi:
      * nothing found -> the directions inside the camera's view are multiplied by p_miss (the VLM sometimes misses,
        so they are not ruled out), then the belief is normalised;
      * found -> the direction of the box centre gets the belief mass (`p_hit` of it; the rest stays spread, as the
        VLM can name the wrong object), and the next look is turned so that the box is in the middle of the picture.
        The confirming answer then comes from a different picture, not a copy of the first one.
    Next view when nothing is found: the heading whose view holds the most belief, minus turn_cost per 180 deg of turn.
    Holds still until an answer for a picture taken after the last turn has arrived, like Searcher."""
    hfov_deg: float = 69.0  # RGB camera (IMX214 in Gazebo: 1.204 rad); the RealSense D435i colour camera is also ~69 deg
    bins: int = 36
    p_miss: float = 0.3
    p_hit: float = 0.8
    turn_cost: float = 0.15
    max_rate: float = 20.0
    k_yaw: float = 1.5
    tol_deg: float = 3.0
    center_tol_deg: float = 8.0
    belief: Optional[np.ndarray] = None
    target: Optional[float] = None
    settled_at: Optional[float] = None
    seen: Optional[float] = None  # t_capture of the last answer used

    def _centers(self) -> np.ndarray:
        return (np.arange(self.bins) + 0.5) * 360.0 / self.bins

    @staticmethod
    def _wrap(a):
        return (np.asarray(a) + 180.0) % 360.0 - 180.0

    def _in_view(self, heading: float) -> np.ndarray:
        return np.abs(self._wrap(self._centers() - heading)) <= self.hfov_deg / 2

    def box_bearing(self, box, heading: float) -> float:
        """World bearing (deg) of the centre of a normalised box [x1, y1, x2, y2] seen at `heading`."""
        u = (box[0] + box[2]) / 2.0
        half = math.radians(self.hfov_deg / 2)
        return heading + math.degrees(math.atan((2 * u - 1) * math.tan(half)))

    def update(self, found: bool, heading: float, box=None) -> None:
        if self.belief is None:
            self.belief = np.full(self.bins, 1.0 / self.bins)
        if found and box is not None:
            b = self.box_bearing(box, heading)
            near = np.abs(self._wrap(self._centers() - b)) <= 360.0 / self.bins
            self.belief = (1 - self.p_hit) * self.belief + self.p_hit * near / near.sum()
            self.target = float(self._wrap(b) % 360.0)
        else:
            self.belief = np.where(self._in_view(heading), self.belief * self.p_miss, self.belief)
            self.belief /= self.belief.sum()
            self.target = self.next_view(heading)

    def next_view(self, heading: float) -> float:
        best, best_score = heading, -1e9
        for h in self._centers():
            turn = abs(float(self._wrap(h - heading)))
            score = float(self.belief[self._in_view(h)].sum()) - self.turn_cost * turn / 180.0
            if turn < self.hfov_deg / 4:  # (almost) the view just seen: no new information
                score -= 1.0
            if score > best_score:
                best, best_score = float(h), score
        return best

    def step(self, yaw_deg: float, now: float, last: Optional[dict]):
        """-> (yaw_rate deg/s, note). last: newest VLM answer with found, box, t_capture and yaw_capture."""
        if self.belief is None:
            self.belief = np.full(self.bins, 1.0 / self.bins)
        if self.target is None:
            self.target = yaw_deg % 360.0
        fresh = (last is not None and last.get("t_capture") is not None and last.get("yaw_capture") is not None
                 and self.settled_at is not None and last["t_capture"] >= self.settled_at
                 and (self.seen is None or last["t_capture"] > self.seen))
        if fresh:
            self.seen = last["t_capture"]
            self.update(bool(last.get("found")), float(last["yaw_capture"]), last.get("box"))
            if last.get("found") and abs(float(self._wrap(self.target - yaw_deg))) <= self.center_tol_deg:
                self.target = yaw_deg % 360.0  # already centred: stay and let the consensus confirm
        err = float(self._wrap(self.target - yaw_deg))
        if abs(err) > self.tol_deg:
            self.settled_at = None
            return max(-self.max_rate, min(self.max_rate, self.k_yaw * err)), f"turning to {self.target:.0f} deg"
        if self.settled_at is None:
            self.settled_at = now
        return 0.0, f"looking at {self.target:.0f} deg (best view {100 * float(self.belief[self._in_view(yaw_deg)].sum()):.0f} %)"
