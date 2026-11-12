"""Real-time Pipeline (threads + asyncio) with fake sensors, a mock VLM and a simulated vehicle."""
import asyncio
import time

import numpy as np
import pytest

from neurodrone.camera import StaticCamera
from neurodrone.config import Config
from neurodrone.depth import ConstantClearance
from neurodrone.events import EventSource
from neurodrone.runtime import Pipeline
from neurodrone.telemetry import JsonlLogger, summarize_log
from neurodrone.types import EventPacket
from neurodrone.vehicle import SimVehicle
from neurodrone.vlm import MockVLM

pytest.importorskip("PIL")  # encode_jpeg needs it (or cv2)

GOOD = '{"found": true, "box": [0.45, 0.4, 0.55, 0.6]}'  # small, centred: a far target


class NoiseSource(EventSource):
    """~10 ms real-time packets of sparse noise events; optionally dies after `die_after_s`."""

    width, height = 64, 48

    def __init__(self, die_after_s=None):
        self.t0, self.die_after_s, self.rng = time.monotonic(), die_after_s, np.random.default_rng(0)

    def read(self, timeout_s=0.1):
        time.sleep(0.01)
        now = time.monotonic()
        if self.die_after_s is not None and now - self.t0 > self.die_after_s:
            raise RuntimeError("sensor unplugged")
        t1 = int((now - self.t0) * 1e6)
        n = 12
        t = np.sort(self.rng.integers(t1 - 10_000, t1, n)).astype(np.int64)
        return EventPacket(self.rng.integers(0, 64, n).astype(np.int16), self.rng.integers(0, 48, n).astype(np.int16),
                           t, self.rng.integers(0, 2, n).astype(np.int8), t1 - 10_000, t1)


def make(cfg=None, source=None, vlm=None, vehicle=None, logger=None, clearance=None):
    cfg = cfg or Config()
    cfg.runtime.control_hz = 50.0
    cfg.vlm.period_s = 0.15
    cfg.camera.hfov_deg = 65.0
    return Pipeline(cfg, source or NoiseSource(), StaticCamera(np.full((96, 128, 3), 120, np.uint8)),
                    vlm or MockVLM([GOOD], latency_s=0.02), vehicle or SimVehicle(), "the red door",
                    logger=logger, clearance=clearance)


def test_flies_towards_a_far_target_when_nothing_is_wrong(tmp_path):
    veh = SimVehicle()
    log = JsonlLogger(str(tmp_path / "run.jsonl"))
    pl = make(vehicle=veh, logger=log)
    reason = asyncio.run(pl.run(duration_s=2.5))
    log.close()
    assert reason == "duration elapsed"
    assert veh.drone.x > 0.4, f"only moved {veh.drone.x:.2f} m forward"
    s = summarize_log(str(tmp_path / "run.jsonl"))
    assert s["vlm_parse_success"] == 1.0 and s["vlm_latency_s"]["n"] >= 5
    assert s["mode_share"].get("TRACK", 0) > 0.5 and s["snn_proc_ms"]["p95"] < 50


def test_losing_the_event_sensor_mid_flight_stops_and_lands():
    cfg = Config()
    cfg.coordinator.snn_timeout_s, cfg.coordinator.blind_timeout_s = 0.15, 0.4
    veh = SimVehicle()
    reason = asyncio.run(make(cfg, NoiseSource(die_after_s=1.0), vehicle=veh).run(duration_s=6.0))
    assert reason.startswith("land") and "reflex layer lost" in reason and veh.mode == "LAND"


def test_a_vlm_that_returns_garbage_never_moves_the_drone():
    veh = SimVehicle()
    reason = asyncio.run(make(vlm=MockVLM(["I see a lovely door!"]), vehicle=veh).run(duration_s=1.5))
    assert reason == "duration elapsed" and abs(veh.drone.x) < 1e-6


def test_a_vlm_that_raises_never_moves_the_drone():
    def boom(jpeg, ins):
        raise TimeoutError("server hung")
    veh = SimVehicle()
    asyncio.run(make(vlm=MockVLM(boom), vehicle=veh).run(duration_s=1.2))
    assert abs(veh.drone.x) < 1e-6


def test_pilot_not_in_offboard_means_no_commands_are_acted_on():
    veh = SimVehicle(engaged=False)
    asyncio.run(make(vehicle=veh).run(duration_s=1.2))
    assert abs(veh.drone.x) < 1e-6 and all(c.vx == 0 for c in veh.commands)


def test_refuses_to_start_without_the_range_sensor_once_it_is_required():
    cfg = Config()
    cfg.coordinator.use_clearance = True
    with pytest.raises(ValueError, match="range reflex"):
        make(cfg)


def test_range_sensor_close_ahead_stops_the_drone():
    cfg = Config()
    cfg.coordinator.use_clearance = True
    veh = SimVehicle()
    asyncio.run(make(cfg, vehicle=veh, clearance=ConstantClearance(0.5)).run(duration_s=1.5))
    assert veh.drone.x < 0.05  # never got going: 0.5 m ahead is inside the margin


def test_stop_request_ends_the_run_cleanly_and_leaves_a_zero_setpoint():
    veh = SimVehicle()
    pl = make(vehicle=veh)

    async def go():
        task = asyncio.create_task(pl.run())
        await asyncio.sleep(0.8)
        pl.stop()
        return await task
    assert asyncio.run(go()) == "stopped"
    assert veh.commands[-1].vx == 0 and veh.commands[-1].yaw_rate == 0
