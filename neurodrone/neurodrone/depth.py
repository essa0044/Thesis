"""Range-sensor reflex: the independent second line of defence.

The event-camera looming detector cannot see untextured surfaces (blank walls, glass); a depth
camera can (active-stereo cameras project their own texture). Their failure modes differ -
depth is weak on glass/black surfaces, in direct sunlight and beyond a few metres - so the
coordinator uses both: an alarm from EITHER one stops the vehicle.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


def corridor_range(depth_m: np.ndarray, col_frac: Tuple[float, float] = (0.35, 0.65),
                   row_frac: Tuple[float, float] = (0.30, 0.70), min_valid_m: float = 0.15,
                   max_valid_m: float = 8.0, min_valid_pixels: float = 0.20, percentile: float = 5.0,
                   invalid_policy: str = "blocked") -> float:
    """Robust 'nearest thing ahead' from a depth image in metres (0 = no return).

    Takes a low percentile (not the minimum) of the valid pixels in a central window, so a few
    speckle pixels don't stop the drone. If fewer than `min_valid_pixels` of the window have a
    return, the answer is ambiguous - open sky, or a glass door / black wall? - so:
        invalid_policy="blocked" -> 0.0  (fail closed; safe indoors, but it won't fly in open space
                                          with nothing in range)
        invalid_policy="free"    -> inf  (accept the risk, e.g. outdoors)
    """
    h, w = depth_m.shape[:2]
    win = depth_m[int(row_frac[0] * h):int(row_frac[1] * h), int(col_frac[0] * w):int(col_frac[1] * w)]
    ok = np.isfinite(win) & (win >= min_valid_m) & (win <= max_valid_m)
    if ok.mean() < min_valid_pixels:
        return 0.0 if invalid_policy == "blocked" else float("inf")
    return float(np.percentile(win[ok], percentile))


class ClearanceSource(ABC):
    @abstractmethod
    def read(self, timeout_s: float = 0.2) -> Optional[float]:
        """Latest forward range in metres (inf = clear within sensor range), or None on timeout."""

    def close(self) -> None:
        pass


class ConstantClearance(ClearanceSource):
    """Fixed value (tests / bench)."""

    def __init__(self, value: float = float("inf"), period_s: float = 0.05):
        self.value, self.period_s = value, period_s

    def read(self, timeout_s: float = 0.2) -> Optional[float]:
        time.sleep(min(self.period_s, timeout_s))
        return self.value


class RealSenseClearance(ClearanceSource):
    """Intel RealSense depth stream -> corridor_range().  [UNTESTED: needs hardware and
    pyrealsense2; written against its documented pipeline/config/get_depth_scale API.]"""

    def __init__(self, width: int = 424, height: int = 240, fps: int = 30, invalid_policy: str = "blocked"):
        import pyrealsense2 as rs  # type: ignore

        self._rs = rs
        self.invalid_policy = invalid_policy
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self._pipe.start(cfg)
        self._scale = profile.get_device().first_depth_sensor().get_depth_scale()

    def read(self, timeout_s: float = 0.2) -> Optional[float]:
        ok, frames = self._pipe.try_wait_for_frames(int(timeout_s * 1000))
        if not ok:
            return None
        d = frames.get_depth_frame()
        if not d:
            return None
        depth_m = np.asanyarray(d.get_data()).astype(np.float32) * self._scale
        return corridor_range(depth_m, invalid_policy=self.invalid_policy)

    def close(self) -> None:
        self._pipe.stop()
