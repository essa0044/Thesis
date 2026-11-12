"""Gazebo (gz-sim Harmonic) cameras via gz-transport's Python bindings: the latest RGB and depth frames.

The PX4 model x500_depth carries an OAK-D Lite model: RGB "IMX214" 1920x1080, hfov 1.204 rad, and a depth camera
640x480 float32 metres, hfov 1.274 rad, on topic /depth_camera (both 30 Hz, same position and axis).

Needs the system packages python3-gz-transport13 and python3-gz-msgs10 (Harmonic), i.e. run with the system
Python or a venv created with --system-site-packages.
"""
from __future__ import annotations

import subprocess
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .mission import Pinhole

RGB_CAM = Pinhole(1920, 1080, 1.204)
DEPTH_CAM = Pinhole(640, 480, 1.274)


def decode_image(msg) -> np.ndarray:
    """gz.msgs Image -> HxWx3 uint8 (RGB) or HxW float32 (depth), decided by the data size."""
    w, h, data = int(msg.width), int(msg.height), msg.data
    if len(data) == w * h * 3:
        return np.frombuffer(data, np.uint8).reshape(h, w, 3)
    if len(data) == w * h * 4:
        fmt = int(getattr(msg, "pixel_format_type", 0))
        if fmt in (4, 5):  # RGBA_INT8 / BGRA_INT8
            a = np.frombuffer(data, np.uint8).reshape(h, w, 4)
            return a[..., :3] if fmt == 4 else a[..., 2::-1]
        return np.frombuffer(data, np.float32).reshape(h, w)
    raise ValueError(f"unexpected image: {w}x{h}, {len(data)} bytes, format {getattr(msg, 'pixel_format_type', '?')}")


def find_topic(contains: str, suffix: str = "/image") -> Optional[str]:
    """First gz topic containing `contains` and ending with `suffix` (e.g. the RGB camera of x500_depth_0)."""
    try:
        out = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if contains in line and line.endswith(suffix):
            return line.strip()
    return None


class GzCameras:
    def __init__(self, rgb_topic: Optional[str] = None, depth_topic: str = "/depth_camera"):
        from gz.msgs10.image_pb2 import Image  # noqa: WPS433 (system package, imported lazily)
        from gz.transport13 import Node
        self.rgb_topic = rgb_topic or find_topic("IMX214")
        if not self.rgb_topic:
            raise RuntimeError("RGB camera topic not found (is `make px4_sitl gz_x500_depth` running?)  gz topic -l")
        self.depth_topic = depth_topic
        self._lock = threading.Lock()
        self._rgb: Tuple[Optional[np.ndarray], float] = (None, 0.0)
        self._depth: Tuple[Optional[np.ndarray], float] = (None, 0.0)
        self.node = Node()
        if not self.node.subscribe(Image, self.rgb_topic, self._on_rgb):
            raise RuntimeError(f"cannot subscribe to {self.rgb_topic}")
        if not self.node.subscribe(Image, self.depth_topic, self._on_depth):
            raise RuntimeError(f"cannot subscribe to {self.depth_topic}")

    def _on_rgb(self, msg):
        img = decode_image(msg)
        with self._lock:
            self._rgb = (img.copy(), time.monotonic())

    def _on_depth(self, msg):
        img = decode_image(msg)
        with self._lock:
            self._depth = (img.copy(), time.monotonic())

    def rgb(self) -> Tuple[Optional[np.ndarray], float]:
        """(frame, age in s)."""
        with self._lock:
            f, t = self._rgb
        return f, (time.monotonic() - t) if f is not None else float("inf")

    def depth(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            f, t = self._depth
        return f, (time.monotonic() - t) if f is not None else float("inf")

    def wait(self, timeout_s: float = 20.0) -> None:
        t0 = time.monotonic()
        while self.rgb()[0] is None or self.depth()[0] is None:
            if time.monotonic() - t0 > timeout_s:
                raise RuntimeError(f"no images after {timeout_s:.0f} s: rgb {self.rgb_topic}, depth {self.depth_topic}")
            time.sleep(0.1)
