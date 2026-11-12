"""Closed-loop simulation: synthetic world -> ideal DVS -> LoomingSNN -> Brain/Coordinator ->
point-mass drone, all in simulated time. The VLM is replaced by an oracle that projects the
true target into the image (with noise and latency), so only the *fusion logic* is tested.

Treat results as logic checks and regression tests, not as performance predictions.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import CoordinatorConfig, SNNConfig
from .coordinator import ARRIVED, Coordinator
from .runtime import Brain
from .sim import Box, Camera, DVSSimulator, PointMassDrone, World
from .reflex import make_snn
from .types import EventPacket, SemanticGoal, VehicleState, VelocityCmd


class PacketCutter:
    """Cuts a continuous simulated event stream into packets on a grid that is deliberately NOT
    aligned with the simulator's frame grid and whose length jitters, like a real USB event stream.
    (Aligned packets flatter the detector: the alarm came ~0.8 s earlier than for any other phase.)
    phase_us=0, jitter_us=0 reproduces the aligned best case."""

    def __init__(self, packet_us: int = 10_000, phase_us: int = 3300, jitter_us: int = 2000, seed: int = 0):
        self.packet_us, self.jitter_us = packet_us, jitter_us
        self.rng = np.random.default_rng(seed + 12345)
        self._last_cut = 0
        self._next_cut = phase_us if phase_us > 0 else packet_us
        self._parts: List[EventPacket] = []

    def push(self, ev: EventPacket, t_now_us: int) -> List[EventPacket]:
        """Add one frame's events; return the packets that are now complete (all events < t_now_us are in)."""
        self._parts.append(ev)
        out: List[EventPacket] = []
        while t_now_us >= self._next_cut:
            x = np.concatenate([e.x for e in self._parts]); y = np.concatenate([e.y for e in self._parts])
            t = np.concatenate([e.t_us for e in self._parts]); p = np.concatenate([e.p for e in self._parts])
            o = np.argsort(t, kind="stable")
            x, y, t, p = x[o], y[o], t[o], p[o]
            k = int(np.searchsorted(t, self._next_cut, side="left"))
            out.append(EventPacket(x[:k], y[:k], t[:k], p[:k], self._last_cut, self._next_cut))
            self._parts = [EventPacket(x[k:], y[k:], t[k:], p[k:], 0, 0)] if k < t.size else []
            self._last_cut = self._next_cut
            jit = int(self.rng.integers(-self.jitter_us, self.jitter_us + 1)) if self.jitter_us else 0
            self._next_cut += self.packet_us + jit
        return out


@dataclass
class Scenario:
    name: str
    boxes: List[Box]
    target_idx: Optional[int]  # index in `boxes` the oracle-VLM reports; None -> always "not found"
    duration_s: float = 30.0
    start_yaw_deg: float = 0.0
    start_y: float = 0.0
    jitter_deg: float = 0.0  # camera-only attitude jitter amplitude
    jitter_hz: float = 1.0
    vlm_period_s: float = 2.0
    vlm_latency_s: float = 1.0
    vlm_noise: float = 0.02
    vlm_enabled: bool = True
    snn_blackout_at_s: Optional[float] = None  # event stream stops here (simulated SNN failure)
    range_sensor: bool = False  # feed the coordinator an ideal forward range sensor (4 m, 2% noise)
    coord: CoordinatorConfig = field(default_factory=lambda: CoordinatorConfig(cruise_speed=1.5))


@dataclass
class SimResult:
    name: str
    collided: bool
    min_clearance: float
    arrived: bool
    ended: str  # reason / final request
    t_end: float
    modes: Counter
    first_alarm_clearance: Optional[float]
    final_pose: Tuple[float, float, float]
    pose_at_blackout: Optional[Tuple[float, float]] = None
    path: List[Tuple[float, float, float, float, str]] = field(default_factory=list)


def default_scenarios() -> List[Scenario]:
    wide = CoordinatorConfig(cruise_speed=1.5, max_episodes=2)
    wide_rng = CoordinatorConfig(cruise_speed=1.5, max_episodes=2, use_clearance=True)
    rng = CoordinatorConfig(cruise_speed=1.5, use_clearance=True)
    wall = [Box(x=10, y=0, w=9, h=30, texture="blank"), Box(x=30, y=0, w=2, h=2)]
    return [
        Scenario("approach_target", [Box(x=18, y=0, w=2, h=2)], 0, duration_s=34),
        Scenario("dodge_obstacle", [Box(x=9, y=0, w=1.5, h=3), Box(x=26, y=0, w=2, h=2)], 1, duration_s=50),
        Scenario("bearing_offset", [Box(x=16, y=4.5, w=2, h=2)], 0, duration_s=34),
        Scenario("jitter_approach", [Box(x=18, y=0, w=2, h=2)], 0, duration_s=34, jitter_deg=3.0, jitter_hz=1.0),
        Scenario("vlm_silent", [Box(x=18, y=0, w=2, h=2)], 0, duration_s=8, vlm_enabled=False),
        Scenario("snn_blackout", [Box(x=40, y=0, w=2, h=2)], 0, duration_s=14, snn_blackout_at_s=5.0),
        # blank wall, 9 m wide and taller than the field of view: no visible top/bottom edges, and off-centre only one side edge is in view
        Scenario("blank_wall_snn_only", wall, 1, duration_s=30, start_y=2.5, coord=wide),  # KNOWN LIMITATION
        Scenario("blank_wall_with_range", wall, 1, duration_s=60, start_y=2.5, coord=wide_rng, range_sensor=True),
        Scenario("approach_with_range", [Box(x=18, y=0, w=2, h=2)], 0, duration_s=34, coord=rng, range_sensor=True),
    ]


class ClosedLoopSim:
    FRAME_US = 2000  # simulated sensor/physics step
    PACKET_US = 10_000  # SNN packet length
    CONTROL_HZ = 25.0

    def __init__(self, sc: Scenario, snn_cfg: Optional[SNNConfig] = None, seed: int = 0):
        self.sc = sc
        self.seed = seed
        cam = Camera()
        self.cam = cam
        self.world = World(sc.boxes, cam)
        self.dvs = DVSSimulator(cam.width, cam.height, seed=seed)
        self.snn = make_snn(cam.width, cam.height, snn_cfg)
        self.coord = Coordinator(sc.coord, hfov_deg=cam.hfov_deg)
        self.brain = Brain(self.snn, self.coord, lost_patience=2)
        self.drone = PointMassDrone(y=sc.start_y, yaw_deg=sc.start_yaw_deg)
        self.rng = np.random.default_rng(seed + 1)

    def _oracle_goal(self, t: float, pose) -> SemanticGoal:
        sc, cam = self.sc, self.cam
        found = False
        x = y = w = None
        if sc.target_idx is not None:
            pr = self.world.project(pose, sc.boxes[sc.target_idx])
            if pr is not None and 0.0 <= pr[1] < cam.width:
                found = True
                x = min(1.0, max(0.0, pr[1] / cam.width + self.rng.normal(0, sc.vlm_noise)))
                y = min(1.0, max(0.0, pr[2] / cam.height))
                w = min(1.0, 2 * pr[3] / cam.width)
        return SemanticGoal(kind="target" if found else "search", x=x, y=y, w=w, captured_at=t,
                            yaw_at_capture_deg=self.drone.yaw_deg, latency_s=sc.vlm_latency_s, raw="oracle")

    def run(self) -> SimResult:
        sc, dr = self.sc, self.drone
        frame_s = self.FRAME_US / 1e6
        cutter = PacketCutter(self.PACKET_US, seed=self.seed)
        ctrl_every = int(round(1.0 / (self.CONTROL_HZ * frame_s)))
        t_us, frames = 0, 0
        pending: List[Tuple[float, SemanticGoal]] = []
        next_vlm = 0.0
        cmd = VelocityCmd()
        modes: Counter = Counter()
        first_alarm = None
        min_clr = float("inf")
        collided = arrived = False
        ended = "duration elapsed"
        blackout_pose = None
        path = []
        jit = math.radians(sc.jitter_deg)

        while t_us < sc.duration_s * 1e6:
            t = t_us / 1e6
            x, y, z, yaw = dr.pose
            wj = 2 * math.pi * sc.jitter_hz
            yaw_cam = yaw + jit * math.sin(wj * t)
            ev = self.dvs.step(self.world.render((x, y, z, yaw_cam)), t_us, t_us + self.FRAME_US)
            t_us += self.FRAME_US
            frames += 1
            t = t_us / 1e6
            blackout = sc.snn_blackout_at_s is not None and t >= sc.snn_blackout_at_s
            if blackout and blackout_pose is None:
                blackout_pose = (dr.x, dr.y)
            if not blackout:
                for pkt in cutter.push(ev, t_us):
                    rep = self.brain.on_events(pkt, t)
                    if rep.alarm and first_alarm is None:
                        first_alarm = self.world.clearance(dr.pose)

            if sc.vlm_enabled and t >= next_vlm:
                pending.append((t + sc.vlm_latency_s, self._oracle_goal(t, (x, y, z, yaw_cam))))
                next_vlm = t + sc.vlm_period_s
            while pending and pending[0][0] <= t:
                self.brain.on_goal(pending.pop(0)[1])

            if frames % ctrl_every == 0:
                rot = abs(math.degrees(dr.yaw_rate)) + abs(sc.jitter_deg * wj * math.cos(wj * t))
                veh = VehicleState(armed=True, in_air=True, engaged=True, yaw_deg=dr.yaw_deg, rot_rate_deg_s=rot,
                                   battery_frac=1.0, dist_home_m=math.hypot(dr.x, dr.y), alt_m=dr.z)
                if sc.range_sensor:
                    r = self.world.range_sensor((x, y, z, yaw))
                    self.brain.on_clearance(r * (1 + self.rng.normal(0, 0.02)) if math.isfinite(r) else r, t)
                dec = self.brain.control_step(t, veh)
                modes[dec.mode] += 1
                cmd = dec.cmd
                if dec.mode == ARRIVED:
                    arrived = True
                if frames % (ctrl_every * 2) == 0:
                    path.append((t, dr.x, dr.y, dr.yaw_deg, dec.mode))
                if dec.request:
                    ended = f"{dec.request}: {dec.note}"
                    break

            dr.step(cmd, frame_s)
            min_clr = min(min_clr, self.world.clearance(dr.pose))
            if self.world.collided(dr.pose):
                collided = True
                ended = "COLLISION"
                break

        return SimResult(sc.name, collided, min_clr, arrived, ended, t_us / 1e6, modes, first_alarm,
                         (dr.x, dr.y, dr.yaw_deg), blackout_pose, path)


def open_loop_trial(boxes: List[Box], snn_cfg: Optional[SNNConfig] = None, speed: float = 1.5,
                    dur: float = 8.0, jitter_deg: float = 0.0, jitter_hz: float = 1.0,
                    yaw_rate_deg_s: float = 0.0, noise_hz: float = 0.3, seed: int = 0,
                    start_yaw_deg: float = 0.0, record: Optional[list] = None,
                    aligned_packets: bool = False) -> Tuple[Optional[float], float]:
    """Fly at constant speed (no coordinator) and report what the SNN does.
    Returns (clearance in metres at the first alarm or None, max frontal urgency seen).
    Used for detector tuning and for tests; independent of the Brain/Coordinator.
    If `record` is a list, every simulated sensor frame's EventPacket is appended to it.
    Packets are cut off the frame grid with jittered length unless aligned_packets=True (best case)."""
    cam = Camera()
    world = World(boxes, cam)
    dvs = DVSSimulator(cam.width, cam.height, noise_hz=noise_hz, seed=seed)
    snn = make_snn(cam.width, cam.height, snn_cfg)
    drone = PointMassDrone(yaw_deg=start_yaw_deg)
    drone.v[:] = [speed, 0.0, 0.0]
    frame_us = ClosedLoopSim.FRAME_US
    cutter = (PacketCutter(ClosedLoopSim.PACKET_US, phase_us=0, jitter_us=0, seed=seed) if aligned_packets
              else PacketCutter(ClosedLoopSim.PACKET_US, seed=seed))
    t_us = 0
    first, mx = None, 0.0
    while t_us < dur * 1e6:
        t = t_us / 1e6
        x, y, z, yaw = drone.pose
        yaw_cam = yaw + math.radians(jitter_deg) * math.sin(2 * math.pi * jitter_hz * t)
        ev = dvs.step(world.render((x, y, z, yaw_cam)), t_us, t_us + frame_us)
        if record is not None:
            record.append(ev)
        t_us += frame_us
        drone.step(VelocityCmd(vx=speed, yaw_rate=yaw_rate_deg_s), frame_us / 1e6)
        for pkt in cutter.push(ev, t_us):
            rep = snn.process(pkt)
            mx = max(mx, rep.frontal)
            if rep.alarm and first is None:
                first = world.clearance(drone.pose)
    return first, mx
