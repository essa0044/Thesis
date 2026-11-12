"""HYBRID looming detector (snn.LoomingSNN, the baseline): direction logic, translation robustness,
detection, false alarms. The fully spiking detector has its own tests in test_snn_spiking.py.
These run on the synthetic simulator: they check logic and guard against regressions; they say
nothing about performance on a real sensor."""
import numpy as np

from neurodrone.closedloop import open_loop_trial
from neurodrone.config import SNNConfig
from neurodrone.sim import Box
from neurodrone.snn import LoomingSNN
from neurodrone.types import EventPacket

HYB = SNNConfig(impl="hybrid")


def _moving_edge(x_positions, y=24, step_us=20_000, t0=0):
    """One ON event per pixel, edge advancing one position per step_us."""
    packets = []
    for k, x in enumerate(x_positions):
        t = t0 + k * step_us
        packets.append(EventPacket(np.array([x], np.int16), np.array([y], np.int16),
                                   np.array([t], np.int64), np.array([1], np.int8), t, t + step_us))
    return packets


def _totals(packets, W=64, H=48):
    snn = LoomingSNN(W, H)
    out = np.zeros(snn._nbins)
    inn = np.zeros(snn._nbins)
    for p in packets:
        snn.process(p)
        out += snn.last_counts[0]
        inn += snn.last_counts[1]
    return out, inn


def test_edge_moving_away_from_centre_counts_as_outward():
    out, inn = _totals(_moving_edge(range(40, 52)))  # right half, moving right = away from the centre
    assert out.sum() > 0
    assert inn.sum() == 0


def test_edge_moving_towards_centre_counts_as_inward():
    out, inn = _totals(_moving_edge(range(52, 40, -1)))
    assert inn.sum() > 0
    assert out.sum() == 0


def test_left_half_mirror_image():
    out, inn = _totals(_moving_edge(range(23, 11, -1)))  # left half, moving left = outward
    assert out.sum() > 0 and inn.sum() == 0


def test_reset_clears_state():
    snn = LoomingSNN(64, 48)
    for p in _moving_edge(range(40, 52)):
        snn.process(p)
    snn.reset()
    assert snn.V.sum() == 0 and snn.rate.sum() == 0


def test_detects_frontal_wall_with_room_to_stop():
    first, mx = open_loop_trial([Box(x=14, y=0, w=3, h=3)], HYB, speed=1.5, dur=8)
    assert first is not None, "textured wall dead ahead must raise the alarm"
    assert first > 3.0, f"alarm only at {first:.1f} m: not enough room to stop from 1.5 m/s"


def test_blank_wall_dead_centre_is_detected_through_its_edges():
    first, _ = open_loop_trial([Box(x=14, y=0, w=3, h=3, texture="blank")], HYB, speed=1.5, dur=8)
    assert first is not None and first > 3.0


def test_pure_rotation_does_not_alarm():
    _, mx = open_loop_trial([], HYB, speed=0.0, yaw_rate_deg_s=30.0, dur=5)
    assert mx == 0.0


def test_attitude_jitter_with_textured_object_does_not_alarm():
    # regression: a signed sum over the centre sector reached 0.94 of the alarm threshold here
    for y in (0.0, 2.0):
        _, mx = open_loop_trial([Box(x=14, y=y, w=2, h=2)], HYB, speed=0.0, jitter_deg=3.0, dur=5)
        assert mx == 0.0, f"frontal urgency {mx:.2f} with object at y={y}"


def test_noise_and_jitter_on_empty_scene_do_not_alarm():
    _, mx = open_loop_trial([], HYB, speed=1.5, jitter_deg=5.0, noise_hz=3.0, dur=6)
    assert mx == 0.0


def test_object_passing_to_the_side_gives_side_urgency_but_no_alarm():
    first, mx = open_loop_trial([Box(x=14, y=-3.0, w=1.5, h=3)], HYB, dur=8)
    assert first is None and mx == 0.0


def test_defaults_keep_margin_above_false_alarm_cliff():
    # In sweeps the false-alarm cliff was between theta=1.5 and 2.5 (3 Hz/px noise scene);
    # the shipped default must stay at least ~1.5x above the lower end.
    assert SNNConfig().theta >= 1.5 * 1.5
    assert SNNConfig().rate_alarm > 1.0 / SNNConfig().tau_rate_s  # one spike alone must not alarm
