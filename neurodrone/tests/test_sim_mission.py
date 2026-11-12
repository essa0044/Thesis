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


def _fake_gz(monkeypatch, with_options):
    """Stand-in gz.transport13 / gz.msgs10 modules: records subscriptions, lets the test deliver messages."""
    import sys as _sys
    subs = {}

    class Node:
        def subscribe(self, msg_type, topic, cb, *opts):
            subs[topic] = (cb, opts)
            return True

    class SubscribeOptions:
        msgs_per_sec = 0

    tr = types.ModuleType("gz.transport13")
    tr.Node = Node
    if with_options:
        tr.SubscribeOptions = SubscribeOptions
    pb = types.ModuleType("gz.msgs10.image_pb2")
    pb.Image = object
    for name, mod in (("gz", types.ModuleType("gz")), ("gz.transport13", tr),
                      ("gz.msgs10", types.ModuleType("gz.msgs10")), ("gz.msgs10.image_pb2", pb)):
        monkeypatch.setitem(_sys.modules, name, mod)
    return subs


def test_gz_cameras_throttle_and_decode_lazily(monkeypatch):
    for with_options in (True, False):
        subs = _fake_gz(monkeypatch, with_options)
        cams = gc.GzCameras(rgb_topic="/rgb", depth_topic="/depth", rgb_hz=5, depth_hz=20)
        assert cams.throttled is with_options
        if with_options:
            assert subs["/rgb"][1][0].msgs_per_sec == 5 and subs["/depth"][1][0].msgs_per_sec == 20
        assert cams.rgb() == (None, float("inf"))
        calls = []
        real = gc.decode_image
        monkeypatch.setattr(gc, "decode_image", lambda m: calls.append(m) or real(m))
        rgb = np.full((3, 4, 3), 7, np.uint8)
        m = types.SimpleNamespace(width=4, height=3, data=rgb.tobytes(), pixel_format_type=3)
        for _ in range(10):  # many frames arrive, none decoded yet
            subs["/rgb"][0](m)
        assert calls == []
        a1, _ = cams.rgb()
        a2, _ = cams.rgb()  # same message: decoded once
        assert (a1 == 7).all() and a1 is a2 and len(calls) == 1
        monkeypatch.setattr(gc, "decode_image", real)
