"""Plain data types passed between the SNN, VLM, coordinator and vehicle layers.

Frame conventions (used everywhere, matches PX4 body frame FRD):
    x = forward, y = right, z = down.  Yaw is degrees, positive = clockwise seen from above.
Time conventions:
    * Event timestamps (`t_us`) are in the *sensor* clock, microseconds.
    * Everything else uses one monotonic clock in seconds (`time.monotonic()` on the real
      system, simulated time in the simulator).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class EventPacket:
    """A time-ordered batch of DVS events (sensor clock, microseconds)."""

    x: np.ndarray  # pixel column
    y: np.ndarray  # pixel row
    t_us: np.ndarray  # int64 timestamps, non-decreasing
    p: np.ndarray  # polarity: 1 = ON (brighter), 0 = OFF (darker)
    t_start_us: int
    t_end_us: int

    @property
    def n(self) -> int:
        return int(self.t_us.size)

    @staticmethod
    def empty(t_start_us: int, t_end_us: int) -> "EventPacket":
        z = np.zeros(0, dtype=np.int64)
        return EventPacket(z.astype(np.int16), z.astype(np.int16), z, z.astype(np.int8),
                           int(t_start_us), int(t_end_us))


@dataclass
class ThreatReport:
    """Output of the spiking looming detector (fast loop, ~100 Hz)."""

    t_us: int  # sensor time of the packet that produced this report
    urgency: Tuple[float, float, float]  # (left, centre, right), each 0..1
    frontal: float  # 0..1, translation-robust "something is coming at me"
    alarm: bool  # frontal urgency reached 1.0 -> brake/dodge
    n_events: int
    proc_s: float  # CPU time spent processing the packet
    stamp: float = 0.0  # monotonic time at which the report was produced (set by the caller)


@dataclass
class ClearanceReport:
    """Range to the nearest return in the forward flight corridor (depth camera / ToF / lidar).
    The independent second reflex: it sees textureless surfaces the looming detector cannot."""

    stamp: float  # monotonic time at which the report was produced
    min_range_m: float  # inf = nothing within sensor range; 0.0 = unreliable -> treated as "too close"


@dataclass
class SemanticGoal:
    """Output of the VLM grounding step (slow loop, every 1..10 s)."""

    kind: str  # "target" (found) | "search" (not visible)
    x: Optional[float]  # normalised image x of target centre: 0 = left edge, 1 = right edge
    y: Optional[float]  # normalised image y of target centre
    w: Optional[float]  # normalised box width (None if the model only gave a point)
    captured_at: float  # monotonic time at which the image was grabbed
    yaw_at_capture_deg: Optional[float]  # vehicle yaw when the image was grabbed
    latency_s: float = 0.0  # VLM inference latency
    raw: str = ""  # raw model text, for debugging
    ok: bool = True


@dataclass
class VelocityCmd:
    vx: float = 0.0  # m/s forward
    vy: float = 0.0  # m/s right
    vz: float = 0.0  # m/s down
    yaw_rate: float = 0.0  # deg/s, + = clockwise


@dataclass
class VehicleState:
    armed: bool = False
    in_air: bool = False
    engaged: bool = False  # True when the autopilot is in OFFBOARD, i.e. obeying our setpoints
    yaw_deg: float = 0.0
    rot_rate_deg_s: float = 0.0  # measured angular rate magnitude (gyro): yaw+pitch+roll
    battery_frac: float = 1.0  # 0..1
    dist_home_m: float = 0.0
    alt_m: float = 0.0
    # local position / velocity (PX4 NED, origin = where PX4 started) and attitude; used by the VLM-goal mission
    north_m: float = 0.0
    east_m: float = 0.0
    down_m: float = 0.0
    vn: float = 0.0
    ve: float = 0.0
    vd: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
