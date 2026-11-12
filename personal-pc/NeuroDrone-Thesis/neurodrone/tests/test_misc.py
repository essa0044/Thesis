"""Safety maths, power parsing, config, depth corridor, replay source, logging."""
import json
import math

import numpy as np
import pytest

from neurodrone.config import Config, load_config
from neurodrone.depth import corridor_range
from neurodrone.events import ReplayEventSource, save_events_npz
from neurodrone.safety import max_safe_speed, max_safe_speed_ttc, stopping_distance
from neurodrone.telemetry import JsonlLogger, parse_tegrastats_line, summarize_log


# ----------------------------------------------------------------------------- safety maths
def test_max_safe_speed_inverts_stopping_distance():
    for d, tr, a in ((3.0, 0.3, 2.0), (6.0, 0.5, 3.0), (1.0, 0.2, 1.5)):
        v = max_safe_speed(d, tr, a)
        assert math.isclose(stopping_distance(v, tr, a), d, rel_tol=1e-9)


def test_ttc_based_limit():
    # detection at 3 s TTC, 0.4 s reaction, 2 m/s^2 braking -> 2*2*(3-0.4) = 10.4 m/s
    assert math.isclose(max_safe_speed_ttc(3.0, 0.4, 2.0), 10.4)
    assert max_safe_speed_ttc(0.3, 0.4, 2.0) == 0.0  # reaction slower than the warning: no safe speed
    v = max_safe_speed_ttc(3.0, 0.4, 2.0)
    assert stopping_distance(v, 0.4, 2.0) <= v * 3.0 + 1e-9


# ----------------------------------------------------------------------------- power parsing
# NOTE: the sample lines are synthetic, written from memory of typical tegrastats output. These
# tests prove the regexes do what they say, NOT that they match your board's real output.
def test_tegrastats_orin_nx_style():
    line = "RAM 2371/7620MB CPU [12%@1510] GR3D_FREQ 0% VDD_IN 4210mW/4210mW VDD_CPU_GPU_CV 1248mW/1248mW"
    assert parse_tegrastats_line(line) == pytest.approx(4.21)


def test_tegrastats_older_pom_style_and_agx_sum_and_unknown():
    assert parse_tegrastats_line("POM_5V_IN 2000/2100 POM_5V_GPU 100/100") == pytest.approx(2.0)
    agx = "VDD_GPU_SOC 4000mW/4000mW VDD_CPU_CV 1500mW/1500mW VIN_SYS_5V0 3000mW/3000mW"
    assert parse_tegrastats_line(agx) == pytest.approx(8.5)
    assert parse_tegrastats_line("RAM 1/2MB nothing useful here") is None


# ----------------------------------------------------------------------------- config
def test_yaml_overrides_and_tuple_conversion(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("snn:\n  theta: 5.5\n  band_edges_frac: [0.0, 0.2, 0.5]\ncoordinator:\n  cruise_speed: 0.7\n")
    cfg = load_config(str(p))
    assert cfg.snn.theta == 5.5 and cfg.snn.band_edges_frac == (0.0, 0.2, 0.5)
    assert cfg.coordinator.cruise_speed == 0.7 and cfg.coordinator.max_speed == Config().coordinator.max_speed


def test_unknown_config_key_is_an_error_not_silently_ignored(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("snn:\n  thetaa: 5\n")  # typo
    with pytest.raises(KeyError):
        load_config(str(p))


def test_shipped_default_yaml_matches_dataclass_defaults():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")
    assert load_config(path) == Config()


# ----------------------------------------------------------------------------- depth corridor
def test_corridor_range_basic_and_robust_to_speckle():
    d = np.full((100, 100), 3.0, np.float32)
    assert corridor_range(d) == pytest.approx(3.0)
    d[40:60, 40:60] = 1.2  # a wall patch in the corridor
    assert corridor_range(d) == pytest.approx(1.2)
    d = np.full((100, 100), 3.0, np.float32)
    d.flat[np.random.default_rng(0).choice(d.size, 60, replace=False)] = 0.2  # 0.6% speckle
    assert corridor_range(d) == pytest.approx(3.0), "a few speckle pixels must not stop the drone"


def test_corridor_ignores_things_outside_the_flight_corridor():
    d = np.full((100, 100), 3.0, np.float32)
    d[:, :20] = 0.5
    d[:, 80:] = 0.5
    assert corridor_range(d) == pytest.approx(3.0)


def test_no_returns_is_ambiguous_so_policy_decides():
    d = np.zeros((100, 100), np.float32)
    assert corridor_range(d) == 0.0  # fail closed by default
    assert math.isinf(corridor_range(d, invalid_policy="free"))
    d[:] = np.nan
    assert corridor_range(d) == 0.0


# ----------------------------------------------------------------------------- replay source
def test_replay_source_returns_every_event_exactly_once_in_time_windows(tmp_path):
    rng = np.random.default_rng(0)
    n = 1000
    t = np.sort(rng.integers(0, 100_000, n)).astype(np.int64)
    p = tmp_path / "e.npz"
    save_events_npz(str(p), rng.integers(0, 64, n), rng.integers(0, 48, n), t, rng.integers(0, 2, n), 64, 48)
    src = ReplayEventSource(str(p), packet_us=10_000)
    assert (src.width, src.height) == (64, 48)
    total, last_end = 0, None
    while True:
        pk = src.read()
        if pk is None:
            break
        total += pk.n
        if pk.n:
            assert pk.t_us.min() >= pk.t_start_us and pk.t_us.max() < pk.t_end_us
        if last_end is not None:
            assert pk.t_start_us == last_end
        last_end = pk.t_end_us
    assert total == n and src.finished


# ----------------------------------------------------------------------------- logging
def test_logger_roundtrip_numpy_values_and_field_named_kind_does_not_clobber(tmp_path):
    path = str(tmp_path / "run.jsonl")
    lg = JsonlLogger(path)
    lg.log("goal", goal_kind="target", x=np.float32(0.5), arr=np.arange(3))
    lg.close()
    rec = json.loads(open(path).readline())
    assert rec["kind"] == "goal" and rec["goal_kind"] == "target" and rec["arr"] == [0, 1, 2]


def test_summarize_log_reports_latency_snn_time_modes_and_power(tmp_path):
    path = str(tmp_path / "run.jsonl")
    lg = JsonlLogger(path)
    for lat, ok in ((1.0, True), (2.0, True), (3.0, True), (None, False)):
        lg.log("vlm_call", ok=ok, latency_s=lat)
    for i in range(10):
        lg.log("threat", proc_ms=1.0 + i * 0.1, alarm=(i == 9))
    for m in ["TRACK"] * 6 + ["ADJUDICATE"] * 4:
        lg.log("decision", mode=m)
    for w in (6.0, 8.0):
        lg.log("power", watts=w)
    lg.close()
    s = summarize_log(path)
    assert s["vlm_latency_s"]["p50"] == pytest.approx(2.0) and s["vlm_parse_success"] == pytest.approx(0.75)
    assert s["snn_proc_ms"]["max"] == pytest.approx(1.9) and s["alarm_packets"] == 1
    assert s["mode_share"]["TRACK"] == pytest.approx(0.6) and s["power_w"]["mean"] == pytest.approx(7.0)


def test_summarize_log_survives_a_log_with_only_failed_vlm_calls(tmp_path):
    path = str(tmp_path / "run.jsonl")
    lg = JsonlLogger(path)
    lg.log("vlm_call", ok=False, latency_s=None, error="timeout")
    lg.close()
    s = summarize_log(path)
    assert s["vlm_parse_success"] == 0.0 and "vlm_latency_s" not in s
