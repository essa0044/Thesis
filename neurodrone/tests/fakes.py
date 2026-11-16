"""Stand-ins for the hardware and servers the Jetson tools talk to: an Ollama server (HTTP, in a thread) and a
`tegrastats` command. Shared by the tests of the stage-1, stage-3, stage-4 and stage-6 tools."""
import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

TEGRA_LINE = ("11-20-2026 10:00:{s:02d} RAM {ram}/7620MB (lfb 2x4MB) SWAP 12/3810MB (cached 0MB) "
              "CPU [35%@1510,20%@1510,off,10%@1510,5%@1510,0%@1510] EMC_FREQ 31%@3199 GR3D_FREQ {gpu}%@[{f}] "
              "NVENC off NVDEC off VIC off APE 200 cpu@55.4C soc2@52.1C soc0@53.0C gpu@{t}C tj@{t}C soc1@51.8C "
              "VDD_IN 14520mW/12010mW VDD_CPU_GPU_CV 6120mW/5000mW VDD_SOC 3010mW/2800mW")


def tegra_line(s=0, ram=5123, gpu=97, f=1020, t=57.3):
    return TEGRA_LINE.format(s=s % 60, ram=ram, gpu=gpu, f=f, t=t)


def fake_tegrastats_cmd(tmpdir, n=1000, interval=0.02, ram=5123, f_start=1020, f_end=1020):
    """A command that prints tegrastats-like lines (the GPU clock moving from f_start to f_end)."""
    path = os.path.join(str(tmpdir), "fake_tegrastats.py")
    with open(path, "w") as fh:
        fh.write("import sys, time\n"
                 f"L = {TEGRA_LINE!r}\n"
                 f"for i in range({n}):\n"
                 f"    f = int({f_start} + ({f_end} - {f_start}) * i / max(1, {n} - 1))\n"
                 f"    print(L.format(s=i % 60, ram={ram}, gpu=97, f=f, t=57.3), flush=True)\n"
                 f"    time.sleep({interval})\n")
    return [sys.executable, path]


class FakeOllama:
    """Minimal Ollama: /api/chat answers with `answer(instruction, image_bytes)` and Ollama-like timing fields;
    /api/generate (load/unload), /api/ps, /api/tags, /api/version; /v1/chat/completions (OpenAI-style)."""

    def __init__(self, answer=None, model="qwen3-vl:4b-instruct"):
        self.answer = answer or (lambda instr, img: '{"found": true, "box": [450, 450, 550, 550]}')
        self.model, self.calls, self.loaded_ctx = model, [], None
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                b = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/api/version":
                    self._send({"version": "0.40.99-fake"})
                elif self.path == "/api/ps":
                    self._send({"models": [{"name": srv.model, "size": 4_000_000_000, "size_vram": 4_000_000_000,
                                            "digest": "84f7ed8042fa" + "0" * 52, "context_length": srv.loaded_ctx}]})
                elif self.path == "/api/tags":
                    self._send({"models": [{"name": srv.model, "digest": "84f7ed8042fa" + "0" * 52, "size": 3_300_000_000,
                                            "details": {"quantization_level": "Q4_K_M", "parameter_size": "4.4B"}}]})
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                if self.path == "/api/generate":
                    srv.loaded_ctx = (body.get("options") or {}).get("num_ctx", srv.loaded_ctx)
                    self._send({"done": True})
                elif self.path == "/api/chat":
                    msg = body["messages"][-1]
                    img = base64.b64decode(msg["images"][0]) if msg.get("images") else b""
                    srv.calls.append(body)
                    text = srv.answer(msg["content"], img)
                    self._send({"message": {"role": "assistant", "content": text}, "done": True,
                                "total_duration": 900_000_000, "load_duration": 1_000_000,
                                "prompt_eval_count": 1210, "prompt_eval_duration": 700_000_000,
                                "eval_count": 25, "eval_duration": 180_000_000})
                elif self.path == "/v1/chat/completions":
                    srv.calls.append(body)
                    text = srv.answer(json.dumps(body["messages"][-1]["content"]), b"")
                    self._send({"choices": [{"message": {"content": text}}],
                                "usage": {"prompt_tokens": 1210, "completion_tokens": 25}})
                else:
                    self.send_error(404)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def jpeg_bytes(img: np.ndarray) -> bytes:
    from PIL import Image
    bio = io.BytesIO()
    Image.fromarray(img).save(bio, format="JPEG", quality=90)
    return bio.getvalue()


class FakeRealSense:
    """Stand-in for neurodrone.realsense.RealSenseCameras: 160 x 120 colour picture with a red square in the middle
    and an aligned depth picture of a flat wall at `wall_m` (plus noise; the square is `target_m` away), a level
    accelerometer, frames produced in a thread at `fps`."""

    def __init__(self, wall_m=5.0, target_m=None, noise_m=0.01, fps=100.0, on_frame=None, on_imu=None, hole_frac=0.0):
        import threading as _th
        import time as _t
        from neurodrone.mission import Pinhole
        from neurodrone.realsense import Frame
        self._Frame, self._t = Frame, _t
        self.w, self.h = 160, 120
        self.intrinsics = dict(width=160, height=120, fx=110.0, fy=110.0, ppx=80.0, ppy=60.0, model="fake", coeffs=[0] * 5)
        self.cam = Pinhole.from_intrinsics(160, 120, 110.0, 80.0, 60.0)
        self.wall_m, self.target_m, self.noise_m, self.hole_frac = wall_m, target_m, noise_m, hole_frac
        self.emitter, self.depth_scale, self.n_dropped = True, 0.001, 0
        self.device = dict(name="Fake D435I", serial="0000", firmware="5.13.0.50")
        self.on_frame, self.on_imu = on_frame, on_imu
        self.latest, self.idx, self.lock = None, 0, _th.Lock()
        self._rng = np.random.default_rng(1)
        self._stop = False
        self.period = 1.0 / fps
        self._th = _th.Thread(target=self._run, daemon=True)
        self._th.start()

    def _make(self):
        rgb = np.full((self.h, self.w, 3), 120, np.uint8)
        rgb[50:70, 70:90] = (200, 20, 20)
        d = np.full((self.h, self.w), self.wall_m, np.float32) + self._rng.normal(0, self.noise_m, (self.h, self.w)).astype(np.float32)
        if self.target_m is not None:
            d[50:70, 70:90] = self.target_m
        if self.hole_frac and not self.emitter:
            d[self._rng.random((self.h, self.w)) < self.hole_frac] = np.nan
        return rgb, d

    def _run(self):
        while not self._stop:
            rgb, d = self._make()
            now = self._t.time()
            with self.lock:
                self.idx += 1
                f = self._Frame(self.idx, now, self._t.monotonic(), now * 1000 - 20, now * 1000 - 20, rgb, d, self.idx, self.idx)
                self.latest = f
            if self.on_imu:
                self.on_imu((now, now * 1000, "a", 0.0, -9.81, 0.0))
                self.on_imu((now, now * 1000, "g", 0.0, 0.0, 0.0))
            if self.on_frame:
                self.on_frame(f)
            self._t.sleep(self.period)

    def set_emitter(self, on):
        self.emitter = bool(on)

    def frame(self):
        with self.lock:
            return self.latest

    def wait_new(self, after_idx, timeout_s=3.0):
        t0 = self._t.monotonic()
        while self._t.monotonic() - t0 < timeout_s:
            f = self.frame()
            if f is not None and f.idx > after_idx:
                return f
            self._t.sleep(0.002)
        return None

    def rgb(self):
        f = self.frame()
        return (f.rgb, 0.02) if f else (None, float("inf"))

    def depth(self):
        f = self.frame()
        return (f.depth_m, 0.02) if f else (None, float("inf"))

    def accel_mean(self, window_s=0.5):
        return np.array([0.0, -9.81, 0.0])

    def wait(self, timeout_s=10.0):
        self.wait_new(0, timeout_s)

    def meta(self):
        return dict(device=self.device, intrinsics=self.intrinsics, depth_scale=self.depth_scale, emitter=self.emitter,
                    aligned_to="color", settings={})

    def stop(self):
        self._stop = True
        self._th.join(timeout=2)
