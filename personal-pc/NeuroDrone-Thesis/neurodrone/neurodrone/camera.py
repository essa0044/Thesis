"""RGB camera for the VLM path (the event camera is separate: see events.py)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Camera(ABC):
    @abstractmethod
    def read(self) -> np.ndarray:
        """Latest frame as HxWx3 uint8 RGB."""

    def close(self) -> None:
        pass


class StaticCamera(Camera):
    """Always returns the same frame (tests / bench)."""

    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def read(self) -> np.ndarray:
        return self.frame


class OpenCVCamera(Camera):
    """Any UVC/V4L2 camera (a RealSense D435's RGB stream appears as one), a video file, or a
    GStreamer pipeline string (e.g. for CSI cameras on Jetson; needs OpenCV built with GStreamer).
    [UNTESTED: needs hardware]"""

    def __init__(self, device: str = "0", width: int = 640, height: int = 480):
        import cv2  # type: ignore

        self._cv2 = cv2
        if str(device).isdigit():
            self._cap = cv2.VideoCapture(int(device))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        elif "!" in str(device):
            self._cap = cv2.VideoCapture(device, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {device!r}")

    def read(self) -> np.ndarray:
        ok, bgr = self._cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._cap.release()
