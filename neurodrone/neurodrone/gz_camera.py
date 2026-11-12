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
    """Latest RGB and depth frames from Gazebo.

    The callbacks only keep a reference to the newest message; it is decoded when rgb() / depth() is called.
    Decoding (and copying 6 MB) every 1920 x 1080 frame at 30 Hz in the callback held the Python GIL long enough
    to starve the MAVSDK telemetry tasks ('User callback queue slow'). The subscriptions are also throttled
    (rgb_hz, depth_hz) where the gz-transport Python bindings support it."""

    def __init__(self, rgb_topic: Optional[str] = None, depth_topic: str = "/depth_camera",
                 rgb_hz: float = 5.0, depth_hz: float = 20.0):
        from gz.msgs10.image_pb2 import Image  # noqa: WPS433 (system package, imported lazily)
        from gz.transport13 import Node
        self.rgb_topic = rgb_topic or find_topic("IMX214")
        if not self.rgb_topic:
            raise RuntimeError("RGB camera topic not found (is `make px4_sitl gz_x500_depth` running?)  gz topic -l")
        self.depth_topic = depth_topic
        self._lock = threading.Lock()
        self._rgb_msg, self._rgb_t = None, 0.0
        self._depth_msg, self._depth_t = None, 0.0
        self._rgb_cache = (None, None)  # (msg it was decoded from, array)
        self._depth_cache = (None, None)
        self.throttled = False
        self.node = Node()
        for topic, cb, hz in ((self.rgb_topic, self._on_rgb, rgb_hz), (self.depth_topic, self._on_depth, depth_hz)):
            if not self._subscribe(Image, topic, cb, hz):
                raise RuntimeError(f"cannot subscribe to {topic}")

    def _subscribe(self, Image, topic, cb, hz) -> bool:
        try:
            from gz.transport13 import SubscribeOptions
            opts = SubscribeOptions()
            opts.msgs_per_sec = int(max(1, round(hz)))
            ok = self.node.subscribe(Image, topic, cb, opts)
            if ok:
                self.throttled = True
                return True
        except Exception:  # noqa: BLE001  (older bindings: no SubscribeOptions in Python)
            pass
        return self.node.subscribe(Image, topic, cb)

    def _on_rgb(self, msg):
        with self._lock:
            self._rgb_msg, self._rgb_t = msg, time.monotonic()

    def _on_depth(self, msg):
        with self._lock:
            self._depth_msg, self._depth_t = msg, time.monotonic()

    def _get(self, which: str):
        with self._lock:
            msg, t = (self._rgb_msg, self._rgb_t) if which == "rgb" else (self._depth_msg, self._depth_t)
            cache = self._rgb_cache if which == "rgb" else self._depth_cache
        if msg is None:
            return None, float("inf")
        if cache[0] is not msg:
            cache = (msg, decode_image(msg))
            with self._lock:
                if which == "rgb":
                    self._rgb_cache = cache
                else:
                    self._depth_cache = cache
        return cache[1], time.monotonic() - t

    def rgb(self) -> Tuple[Optional[np.ndarray], float]:
        """(frame, age in s). The array is read-only and shared: copy it before changing it."""
        return self._get("rgb")

    def depth(self) -> Tuple[Optional[np.ndarray], float]:
        return self._get("depth")

    def wait(self, timeout_s: float = 20.0) -> None:
        t0 = time.monotonic()
        while self.rgb()[0] is None or self.depth()[0] is None:
            if time.monotonic() - t0 > timeout_s:
                raise RuntimeError(f"no images after {timeout_s:.0f} s: rgb {self.rgb_topic}, depth {self.depth_topic}")
            time.sleep(0.1)
