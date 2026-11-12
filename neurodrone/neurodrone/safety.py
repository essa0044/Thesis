"""Back-of-envelope safety maths. Use the *measured* detection range/TTC from your own
recorded flights (scripts/replay_snn.py) - not the synthetic numbers - to set cruise speed."""
from __future__ import annotations

import math


def stopping_distance(v: float, reaction_s: float, decel: float) -> float:
    """Distance covered between the obstacle becoming detectable and the drone being stopped."""
    return v * reaction_s + v * v / (2.0 * decel)


def max_safe_speed(detect_range_m: float, reaction_s: float, decel: float) -> float:
    """Largest v with v*t_r + v^2/(2a) <= detect_range  (fixed-range detector, e.g. a lidar)."""
    a, tr, d = decel, reaction_s, detect_range_m
    return a * (-tr + math.sqrt(tr * tr + 2.0 * d / a))


def max_safe_speed_ttc(ttc_detect_s: float, reaction_s: float, decel: float) -> float:
    """Largest v if detection happens at a fixed time-to-contact (what flow-based looming
    detection roughly gives): v*TTC >= v*t_r + v^2/(2a)  ->  v <= 2a (TTC - t_r)."""
    return max(0.0, 2.0 * decel * (ttc_detect_s - reaction_s))
