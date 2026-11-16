"""The Intel RealSense D435i on the Jetson: live frames for the bench tests, a recorder, and a reader that replays a
recording through the same mission code (stages 3 and 4 of the deployment plan, thesis Chapter 6).

Live frames (pyrealsense2). Colour and depth, with the depth ALIGNED to the colour image (librealsense's rs.align), so
one pinhole model - the colour camera's calibration - serves both: a VLM box in the colour picture reads its depth from
the same pixels. The class has the interface of neurodrone.gz_camera.GzCameras (rgb(), depth(), wait()), so the
mission code does not change. Frame times use the camera's hardware time stamps in the host's clock
(global_time_enabled), so the age check of the guard works as in simulation.

Note on sharing the camera. Only one process can own the D435i. In stages 3-4 that is this module. From stage 6 on,
the visual odometry (Isaac ROS, realsense2_camera node in Docker) owns it and the mission must read colour and depth
from that node's ROS 2 topics instead (see jetson/README.md).

Recording format (a folder; readable without librealsense, so a recording can be replayed on the laptop):
    meta.json       camera serial / firmware, colour intrinsics (width, height, fx, fy, ppx, ppy), depth scale,
                    stream settings, emitter, notes
    frames.csv      idx, t_wall (s), t_mono (s), color_ts_ms, depth_ts_ms, color_no, depth_no
    rgb/NNNNNN.jpg  colour picture (JPEG q 92)
    depth/NNNNNN.png depth aligned to colour, uint16 millimetres (0 = no depth)
    imu.csv         t_wall, ts_ms, kind (a = accelerometer m/s^2, g = gyro rad/s), x, y, z   (camera axes)
    marks.csv       t_wall, label    (typed by the operator while recording; "d=4.0" = target 4.0 m, "o=1.2" = obstacle)
"""
from __future__ import annotations

import csv
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .mission import Pinhole

G = 9.80665


# ----------------------------------------------------------------------------- small helpers (pure)
def attitude_from_accel(a_cam: np.ndarray) -> Tuple[float, float]:
    """(roll, pitch) in degrees, body FRD, from the accelerometer at rest, given in the camera's optical axes
    (x right, y down, z forward - the D435i IMU axes as librealsense reports them; level and still it reads about
    (0, -9.8, 0)). Only valid while the camera is not accelerating; average over a short window."""
    fx_c, fy_c, fz_c = (float(v) for v in a_cam)
    f_fwd, f_right, f_down = fz_c, fx_c, fy_c
    pitch = math.degrees(math.atan2(f_fwd, math.hypot(f_right, f_down)))
    roll = math.degrees(math.atan2(-f_right, -f_down))
    return roll, pitch


def depth_to_png16(depth_m: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.round(d * 1000.0), 0, 65535).astype(np.uint16)


def png16_to_depth(mm: np.ndarray) -> np.ndarray:
    d = mm.astype(np.float32) / 1000.0
    d[mm == 0] = np.nan  # no depth
    return d


def _imwrite(path: str, img: np.ndarray, quality: int = 92) -> None:
    try:
        import cv2  # type: ignore
        if img.ndim == 3:
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(path, img)
        return
    except ImportError:
        pass
    from PIL import Image
    if img.ndim == 3:
        Image.fromarray(img).save(path, quality=quality)
    else:
        Image.fromarray(img.astype(np.uint16)).save(path)


def _imread(path: str, depth: bool = False) -> np.ndarray:
    try:
        import cv2  # type: ignore
        if depth:
            return cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image
        im = Image.open(path)
        return np.asarray(im) if depth else np.asarray(im.convert("RGB"))


@dataclass
class Frame:
    idx: int
    t_wall: float  # host clock (time.time()) when the frame arrived
    t_mono: float  # host monotonic clock when it arrived
    color_ts_ms: float  # camera time stamp (host clock domain with global time on)
    depth_ts_ms: float
    rgb: Optional[np.ndarray] = None
    depth_m: Optional[np.ndarray] = None  # aligned to rgb; NaN = no depth
    color_no: int = -1
    depth_no: int = -1


# ----------------------------------------------------------------------------- live camera
class RealSenseCameras:
    """D435i colour + aligned depth (+ IMU) in a background thread; rgb() / depth() return the newest frame and
    its age in seconds, like GzCameras. Needs pyrealsense2 (librealsense with Python bindings) on the Jetson."""

    def __init__(self, color=(1280, 720, 15), depth=(848, 480, 15), emitter: bool = True, imu: bool = False,
                 serial: Optional[str] = None, on_frame=None, on_imu=None):
        import pyrealsense2 as rs  # type: ignore
        self.rs = rs
        self.pipeline, cfg = rs.pipeline(), rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, color[0], color[1], rs.format.rgb8, color[2])
        cfg.enable_stream(rs.stream.depth, depth[0], depth[1], rs.format.z16, depth[2])
        if imu:
            cfg.enable_stream(rs.stream.accel)
            cfg.enable_stream(rs.stream.gyro)
        self.on_frame, self.on_imu = on_frame, on_imu
        self._q: "queue.Queue" = queue.Queue(maxsize=2)
        self.lock = threading.Lock()
        self.latest: Optional[Frame] = None
        self.n_frames, self.n_dropped, self.idx = 0, 0, 0
        self.imu: List[tuple] = []  # (t_wall, ts_ms, kind, x, y, z), newest last (kept for 60 s)
        self.profile = self.pipeline.start(cfg, self._callback)
        dev = self.profile.get_device()
        self.depth_sensor = dev.first_depth_sensor()
        self.depth_scale = float(self.depth_sensor.get_depth_scale())
        for s in dev.query_sensors():  # camera time stamps in the host clock domain
            if s.supports(rs.option.global_time_enabled):
                s.set_option(rs.option.global_time_enabled, 1)
        self.set_emitter(emitter)
        cs = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        it = cs.get_intrinsics()
        self.intrinsics = dict(width=it.width, height=it.height, fx=it.fx, fy=it.fy, ppx=it.ppx, ppy=it.ppy,
                               model=str(it.model), coeffs=list(it.coeffs))
        self.cam = Pinhole.from_intrinsics(it.width, it.height, it.fx, it.ppx, it.ppy)
        self.device = dict(name=dev.get_info(rs.camera_info.name), serial=dev.get_info(rs.camera_info.serial_number),
                           firmware=dev.get_info(rs.camera_info.firmware_version))
        self.settings = dict(color=list(color), depth=list(depth), imu=imu)
        self.align = rs.align(rs.stream.color)
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="realsense", daemon=True)
        self._thread.start()

    def set_emitter(self, on: bool) -> None:
        if self.depth_sensor.supports(self.rs.option.emitter_enabled):
            self.depth_sensor.set_option(self.rs.option.emitter_enabled, 1.0 if on else 0.0)
        self.emitter = bool(on)

    def _callback(self, frame) -> None:  # librealsense thread: keep it short
        rs = self.rs
        if frame.is_frameset():
            fs = frame.as_frameset()
            fs.keep()
            try:
                self._q.put_nowait((time.time(), time.monotonic(), fs))
            except queue.Full:
                self.n_dropped += 1
        elif frame.is_motion_frame():
            m = frame.as_motion_frame()
            d = m.get_motion_data()
            kind = "a" if m.get_profile().stream_type() == rs.stream.accel else "g"
            rec = (time.time(), m.get_timestamp(), kind, d.x, d.y, d.z)
            with self.lock:
                self.imu.append(rec)
                if len(self.imu) > 30000:
                    del self.imu[:10000]
            if self.on_imu:
                self.on_imu(rec)

    def _run(self) -> None:
        while not self._stop:
            try:
                t_wall, t_mono, fs = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            al = self.align.process(fs)
            c, d = al.get_color_frame(), al.get_depth_frame()
            if not c or not d:
                continue
            rgb = np.asanyarray(c.get_data()).copy()
            dm = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
            dm[dm <= 0] = np.nan
            self.idx += 1
            f = Frame(self.idx, t_wall, t_mono, c.get_timestamp(), d.get_timestamp(), rgb, dm,
                      c.get_frame_number(), d.get_frame_number())
            with self.lock:
                self.latest = f
                self.n_frames += 1
            if self.on_frame:
                self.on_frame(f)

    def _age(self, f: Frame, ts_ms: float) -> float:
        now_mono, now_wall = time.monotonic(), time.time()
        return max(now_mono - f.t_mono, now_wall - ts_ms / 1000.0)

    def rgb(self) -> Tuple[Optional[np.ndarray], float]:
        with self.lock:
            f = self.latest
        return (None, float("inf")) if f is None else (f.rgb, self._age(f, f.color_ts_ms))

    def depth(self) -> Tuple[Optional[np.ndarray], float]:
        with self.lock:
            f = self.latest
        return (None, float("inf")) if f is None else (f.depth_m, self._age(f, f.depth_ts_ms))

    def frame(self) -> Optional[Frame]:
        with self.lock:
            return self.latest

    def wait_new(self, after_idx: int, timeout_s: float = 3.0) -> Optional[Frame]:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            f = self.frame()
            if f is not None and f.idx > after_idx:
                return f
            time.sleep(0.005)
        return None

    def accel_mean(self, window_s: float = 0.5) -> Optional[np.ndarray]:
        with self.lock:
            t_end = time.time()
            a = [r[3:] for r in self.imu if r[2] == "a" and r[0] >= t_end - window_s]
        return np.mean(np.array(a), axis=0) if a else None

    def wait(self, timeout_s: float = 10.0) -> None:
        if self.wait_new(0, timeout_s) is None:
            raise TimeoutError("no frames from the RealSense")

    def meta(self) -> dict:
        return dict(device=self.device, intrinsics=self.intrinsics, depth_scale=self.depth_scale, emitter=self.emitter,
                    aligned_to="color", settings=self.settings)

    def stop(self) -> None:
        self._stop = True
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001
            pass
        self._thread.join(timeout=2)


# ----------------------------------------------------------------------------- recording
class Recorder:
    """Writes frames, IMU samples and marks to a recording folder. Frames go to disk in a writer thread; if the disk
    falls behind, frames are dropped and counted (never the IMU)."""

    def __init__(self, path: str, meta: dict, every: int = 1):
        self.path, self.every = path, max(1, int(every))
        os.makedirs(os.path.join(path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(path, "depth"), exist_ok=True)
        meta = dict(meta, started=time.strftime("%Y-%m-%d %H:%M:%S"), format="neurodrone-recording-1")
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        self._frames = open(os.path.join(path, "frames.csv"), "w", newline="")
        self._fw = csv.writer(self._frames)
        self._fw.writerow(["idx", "t_wall", "t_mono", "color_ts_ms", "depth_ts_ms", "color_no", "depth_no"])
        self._imu = open(os.path.join(path, "imu.csv"), "w", newline="")
        self._iw = csv.writer(self._imu)
        self._iw.writerow(["t_wall", "ts_ms", "kind", "x", "y", "z"])
        self._marks = open(os.path.join(path, "marks.csv"), "w", newline="")
        self._mw = csv.writer(self._marks)
        self._mw.writerow(["t_wall", "label"])
        self.n_written, self.n_dropped, self._seen = 0, 0, 0
        self._q: "queue.Queue" = queue.Queue(maxsize=64)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def add_frame(self, f: Frame) -> None:
        self._seen += 1
        if (self._seen - 1) % self.every:
            return
        try:
            self._q.put_nowait(f)
        except queue.Full:
            self.n_dropped += 1

    def add_imu(self, rec: tuple) -> None:
        with self._lock:
            self._iw.writerow([f"{rec[0]:.6f}", f"{rec[1]:.3f}", rec[2], f"{rec[3]:.5f}", f"{rec[4]:.5f}", f"{rec[5]:.5f}"])

    def mark(self, label: str, t_wall: Optional[float] = None) -> None:
        with self._lock:
            self._mw.writerow([f"{(t_wall or time.time()):.6f}", label])
            self._marks.flush()

    def _run(self) -> None:
        while True:
            f = self._q.get()
            if f is None:
                break
            name = f"{f.idx:06d}"
            _imwrite(os.path.join(self.path, "rgb", name + ".jpg"), f.rgb)
            _imwrite(os.path.join(self.path, "depth", name + ".png"), depth_to_png16(f.depth_m))
            with self._lock:
                self._fw.writerow([f.idx, f"{f.t_wall:.6f}", f"{f.t_mono:.6f}", f"{f.color_ts_ms:.3f}",
                                   f"{f.depth_ts_ms:.3f}", f.color_no, f.depth_no])
                self.n_written += 1

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=30)
        for fh in (self._frames, self._imu, self._marks):
            fh.close()


class Recording:
    """A recording folder: frames (images loaded on demand), IMU, marks and the colour camera model."""

    def __init__(self, path: str):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        it = self.meta["intrinsics"]
        self.cam = Pinhole.from_intrinsics(it["width"], it["height"], it["fx"], it["ppx"], it["ppy"])
        with open(os.path.join(path, "frames.csv")) as f:
            self.frames = [dict(idx=int(r["idx"]), t_wall=float(r["t_wall"]), t_mono=float(r["t_mono"]),
                                color_ts_ms=float(r["color_ts_ms"]), depth_ts_ms=float(r["depth_ts_ms"]),
                                color_no=int(r["color_no"]), depth_no=int(r["depth_no"])) for r in csv.DictReader(f)]
        self.frames.sort(key=lambda r: r["t_wall"])
        self.imu = np.zeros((0, 5))
        self.imu_kind = np.zeros(0, dtype="<U1")
        p = os.path.join(path, "imu.csv")
        if os.path.exists(p):
            rows = list(csv.DictReader(open(p)))
            if rows:
                self.imu = np.array([[float(r["t_wall"]), float(r["ts_ms"]), float(r["x"]), float(r["y"]), float(r["z"])]
                                     for r in rows])
                self.imu_kind = np.array([r["kind"] for r in rows])
        self.marks = []
        p = os.path.join(path, "marks.csv")
        if os.path.exists(p):
            self.marks = [(float(r["t_wall"]), r["label"]) for r in csv.DictReader(open(p))]

    def __len__(self) -> int:
        return len(self.frames)

    def load(self, i: int) -> Frame:
        r = self.frames[i]
        name = f"{r['idx']:06d}"
        rgb = _imread(os.path.join(self.path, "rgb", name + ".jpg"))
        dep = png16_to_depth(_imread(os.path.join(self.path, "depth", name + ".png"), depth=True))
        return Frame(r["idx"], r["t_wall"], r["t_mono"], r["color_ts_ms"], r["depth_ts_ms"], rgb, dep,
                     r["color_no"], r["depth_no"])

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self.frames)):
            yield self.load(i)

    def accel_mean(self, t_wall: float, window_s: float = 0.5) -> Optional[np.ndarray]:
        if not len(self.imu):
            return None
        sel = (self.imu_kind == "a") & (self.imu[:, 0] >= t_wall - window_s) & (self.imu[:, 0] <= t_wall)
        return self.imu[sel, 2:5].mean(axis=0) if sel.any() else None

    def summary(self) -> dict:
        t = np.array([r["t_wall"] for r in self.frames])
        out = dict(path=self.path, frames=len(self.frames), marks=len(self.marks), device=self.meta.get("device"),
                   emitter=self.meta.get("emitter"), notes=self.meta.get("notes"))
        if len(t) > 1:
            dt = np.diff(t)
            out.update(duration_s=float(t[-1] - t[0]), fps=float((len(t) - 1) / (t[-1] - t[0])),
                       gap_max_s=float(dt.max()))
            nos = np.array([r["color_no"] for r in self.frames])
            if (nos >= 0).all():
                step = np.median(np.diff(nos)) if len(nos) > 1 else 1
                out["camera_frames_missed"] = int(np.sum(np.maximum(0, np.diff(nos) - step)) // max(1, step))
        if len(self.imu):
            for k, name in (("a", "accel_hz"), ("g", "gyro_hz")):
                tt = self.imu[self.imu_kind == k, 0]
                if len(tt) > 1:
                    out[name] = float((len(tt) - 1) / (tt[-1] - tt[0]))
        return out


def parse_mark(label: str) -> dict:
    """Truth typed into a mark: 'd=4.0' (target distance, m), 'o=1.2' (obstacle distance, m), 'absent' (target not
    in view). Everything else is kept as text."""
    out: dict = {"label": label}
    for tok in label.replace(",", " ").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k.strip().lower()] = float(v)
            except ValueError:
                out[k.strip().lower()] = v
        elif tok.lower() == "absent":
            out["absent"] = True
    return out
