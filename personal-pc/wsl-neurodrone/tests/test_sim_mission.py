"""scripts/sim_mission.py and neurodrone/gz_camera.py without Gazebo / PX4: image decoding and one VLM step."""
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import sim_mission as sm  # noqa: E402

from neurodrone import gz_camera as gc  # noqa: E402
from neurodrone.telemetry import NullLogger  # noqa: E402
from neurodrone.types import VehicleState  # noqa: E402


def test_decode_image():
    rgb = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(3, 4, 3)
    m = types.SimpleNamespace(width=4, height=3, data=rgb.tobytes(), pixel_format_type=3)
    assert (gc.decode_image(m) == rgb).all()
    dep = np.full((3, 4), 2.5, np.float32)
    m = types.SimpleNamespace(width=4, height=3, data=dep.tobytes(), pixel_format_type=13)
    assert np.allclose(gc.decode_image(m), 2.5)


class FakeCams:
    def __init__(self):
        self.r = np.zeros((1080, 1920, 3), np.uint8)
        self.d = np.full((480, 640), 12.0, np.float32)
        u, v = gc.DEPTH_CAM.pixel(gc.RGB_CAM.ray(960, 540))
        self.d[int(v) - 20:int(v) + 20, int(u) - 20:int(u) + 20] = 6.0  # the target, straight ahead, 6 m

    def rgb(self):
        return self.r, 0.01

    def depth(self):
        return self.d, 0.01


def test_vlm_worker_one_step(monkeypatch):
    calls = []

    def fake_ask(url, model, jpeg, instr, timeout=60):
        calls.append(instr)
        return '{"found": true, "box": [450, 450, 550, 550]}', 0.3  # norm1000, centred

    monkeypatch.setattr(sm, "ask_ollama", fake_ask)
    a = types.SimpleNamespace(consensus_need=2, consensus_window=3, consensus_tol=1.5, max_side=896, ollama="x", model="m",
                              instruction="the red box", standoff=1.5, save_frames=None, vlm_period=0.01)
    st = VehicleState(north_m=0, east_m=0, down_m=-2, yaw_deg=0)
    w = sm.VlmWorker(a, FakeCams(), lambda: st, NullLogger())
    w.start()
    t0 = time.time()
    while w.goal is None and time.time() - t0 < 10:
        time.sleep(0.02)
    w.stop = True
    w.join(timeout=5)
    assert w.goal is not None and len(calls) >= 2  # two agreeing answers needed
    g = w.goal
    assert 4.4 < g[0] < 4.8 and abs(g[1]) < 0.3 and abs(g[2] + 2) < 1e-6  # 6 m ahead (+ camera offset) - 1.5 m standoff
