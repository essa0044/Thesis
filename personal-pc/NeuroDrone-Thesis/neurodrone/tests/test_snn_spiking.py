"""FULLY SPIKING looming detector (snn_spiking.SpikingLoomingSNN): same geometry as the hybrid,
direction logic, exact per-event motion detection, independence from packet cutting, detection,
false alarms. Synthetic simulator only: logic checks and regression guards, not performance claims."""
import numpy as np
import pytest

from neurodrone.closedloop import ClosedLoopSim, PacketCutter, open_loop_trial
from neurodrone.config import SNNConfig, SpikingConfig, load_config
from neurodrone.reflex import make_snn
from neurodrone.sim import Box
from neurodrone.snn import LoomingSNN
from neurodrone.snn_spiking import SpikingLoomingSNN
from neurodrone.types import EventPacket

SPK = SNNConfig(impl="spiking")


def _moving_edge(x_positions, y=24, step_us=20_000, t0=0):
    """One ON event per pixel, edge advancing one position per step_us, one packet per position."""
    packets = []
    for k, x in enumerate(x_positions):
        t = t0 + k * step_us
        packets.append(EventPacket(np.array([x], np.int16), np.array([y], np.int16),
                                   np.array([t], np.int64), np.array([1], np.int8), t, t + step_us))
    return packets


def _totals(det, packets):
    out = np.zeros(det._nbins)
    inn = np.zeros(det._nbins)
    for p in packets:
        det.process(p)
        out += det.last_counts[0]
        inn += det.last_counts[1]
    return out, inn


@pytest.mark.parametrize("W,H", [(64, 48), (128, 96), (346, 260)])
def test_same_image_geometry_as_the_hybrid(W, H):
    a, b = LoomingSNN(W, H, SPK), SpikingLoomingSNN(W, H, SPK)
    assert a.deltas == b.deltas and a.n_bands == b.n_bands
    for name in ("_valid", "_sector", "_bandsec"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    for ua, ub in zip(a._up + a._dn, b._up + b._dn):
        assert np.array_equal(ua, ub)


def test_edge_moving_away_from_centre_fires_the_outward_neurons():
    out, inn = _totals(SpikingLoomingSNN(64, 48, SPK), _moving_edge(range(40, 52)))
    assert out.sum() > 0 and inn.sum() == 0


def test_edge_moving_towards_centre_fires_the_inward_neurons():
    out, inn = _totals(SpikingLoomingSNN(64, 48, SPK), _moving_edge(range(52, 40, -1)))
    assert inn.sum() > 0 and out.sum() == 0


def test_left_half_mirror_image():
    out, inn = _totals(SpikingLoomingSNN(64, 48, SPK), _moving_edge(range(23, 11, -1)))
    assert out.sum() > 0 and inn.sum() == 0


def test_motion_inside_a_single_packet_is_detected():
    # an edge crossing 12 pixels 3 ms apart, all in ONE 40 ms packet
    xs = np.arange(40, 52)
    pkt = EventPacket(xs.astype(np.int16), np.full(xs.size, 24, np.int16),
                      (np.arange(xs.size) * 3000).astype(np.int64), np.ones(xs.size, np.int8), 0, 40_000)
    det = SpikingLoomingSNN(64, 48, SPK)
    det.process(pkt)
    assert det.last_counts[0].sum() > 0 and det.last_counts[1].sum() == 0
    hyb = LoomingSNN(64, 48, SNNConfig(impl="hybrid"))
    hyb.process(pkt)
    assert hyb.last_counts[0].sum() == 0  # the hybrid only compares against earlier packets


def test_spikes_do_not_depend_on_how_the_stream_is_cut():
    frames = []
    open_loop_trial([Box(x=12, y=0, w=3, h=3)], SPK, speed=1.5, dur=6, record=frames)
    runs = {}
    for name, cutter in (("frames", None), ("aligned", PacketCutter(10_000, phase_us=0, jitter_us=0)),
                         ("jittered", PacketCutter(10_000, seed=3)), ("long", PacketCutter(33_000, phase_us=7000))):
        det = SpikingLoomingSNN(128, 96, SPK)
        onset = None
        t_last = frames[-1].t_end_us  # then 50 ms of empty frames, so the cutter releases everything
        stream = frames + [EventPacket.empty(t, t + 2000) for t in range(t_last, t_last + 50_000, 2000)]
        for ev in stream:
            for pkt in ([ev] if cutter is None else cutter.push(ev, ev.t_end_us)):
                rep = det.process(pkt)
                if rep.alarm and onset is None:
                    onset = rep.t_us
        runs[name] = (det.stats["layer1_spikes"], det.stats["layer2_spikes"], det.stats["layer3_spikes"], onset)
    ref = runs["frames"]
    assert ref[2] > 0 and ref[3] is not None
    for name, r in runs.items():
        # identical spike counts, except possibly the very last (still incomplete) clock step
        assert r[0] == ref[0], (name, r, ref)
        assert abs(r[1] - ref[1]) <= 2 and abs(r[2] - ref[2]) <= 1, (name, r, ref)
        assert abs(r[3] - ref[3]) <= 34_000, (name, r, ref)  # onset differs only by the packet length


def test_reset_clears_state():
    det = SpikingLoomingSNN(64, 48, SPK)
    for p in _moving_edge(range(40, 52)):
        det.process(p)
    det.reset()
    assert det.V.sum() == 0 and det.rate.sum() == 0 and det.stats["events"] == 0


def test_detects_frontal_wall_with_room_to_stop():
    first, _ = open_loop_trial([Box(x=14, y=0, w=3, h=3)], SPK, speed=1.5, dur=8)
    assert first is not None and first > 3.0, f"alarm at {first} m"


def test_blank_wall_dead_centre_is_detected_through_its_edges():
    first, _ = open_loop_trial([Box(x=14, y=0, w=3, h=3, texture="blank")], SPK, speed=1.5, dur=8)
    assert first is not None and first > 3.0


def test_pure_rotation_does_not_alarm():
    _, mx = open_loop_trial([], SPK, speed=0.0, yaw_rate_deg_s=30.0, dur=5)
    assert mx == 0.0


@pytest.mark.parametrize("y", [0.0, 2.0])
def test_attitude_jitter_with_textured_object_does_not_alarm(y):
    _, mx = open_loop_trial([Box(x=14, y=y, w=2, h=2)], SPK, speed=0.0, jitter_deg=3.0, dur=5)
    assert mx == 0.0


def test_noise_and_jitter_on_empty_scene_do_not_alarm():
    _, mx = open_loop_trial([], SPK, speed=1.5, jitter_deg=5.0, noise_hz=3.0, dur=6)
    assert mx < 1.0


def test_object_passing_to_the_side_does_not_alarm():
    first, mx = open_loop_trial([Box(x=14, y=-3.0, w=1.5, h=3)], SPK, dur=8)
    assert first is None and mx < 1.0


def test_defaults_keep_margin_above_false_alarm_cliff():
    # In sweeps on the synthetic scenes, false alarms appeared at theta0 = 3 (none at 4) and at
    # beta = 0.04 (one stray spike at 0.08). The defaults must stay >= 1.5x above those.
    sp = SpikingConfig()
    assert sp.theta0 >= 1.5 * 4.0 and sp.beta >= 1.5 * 0.08
    assert sp.rate_alarm > 1.0 / sp.tau_rate_s  # one alarm spike alone must not raise the alarm


def test_counts_spikes_and_synaptic_operations():
    frames = []
    open_loop_trial([Box(x=12, y=0, w=3, h=3)], SPK, speed=1.5, dur=6, record=frames)
    det = SpikingLoomingSNN(128, 96, SPK)
    for ev in frames:
        det.process(ev)
    st = det.stats
    assert st["layer1_spikes"] > st["layer2_spikes"] > st["layer3_spikes"] > 0 and st["events"] > 0
    assert st["synops"] >= 4 * len(det.deltas) * st["events"]
    nc = det.neuron_counts
    assert nc["layer 1 coincidence (motion)"] > 10_000 and nc["layer 3 alarm (both sides)"] == det.n_bands


def test_factory_picks_the_implementation_and_rejects_unknown_ones():
    assert isinstance(make_snn(64, 48, SNNConfig(impl="spiking")), SpikingLoomingSNN)
    assert isinstance(make_snn(64, 48, SNNConfig(impl="hybrid")), LoomingSNN)
    assert isinstance(make_snn(64, 48), SpikingLoomingSNN)  # the default
    with pytest.raises(ValueError):
        make_snn(64, 48, SNNConfig(impl="ann"))


def test_yaml_selects_the_detector_and_its_settings(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("snn:\n  impl: hybrid\n  spiking:\n    theta0: 7.5\n")
    cfg = load_config(str(p))
    assert cfg.snn.impl == "hybrid" and cfg.snn.spiking.theta0 == 7.5 and cfg.snn.spiking.beta == SpikingConfig().beta


@pytest.mark.slow
def test_closed_loop_dodges_obstacle_with_the_spiking_reflex():
    from neurodrone.closedloop import default_scenarios
    sc = next(s for s in default_scenarios() if s.name == "dodge_obstacle")
    r = ClosedLoopSim(sc, SPK).run()
    assert r.arrived and not r.collided and r.modes["AVOID"] > 0
