"""PacketCutter: the realistic packetisation used by the simulator (and why it exists)."""
import numpy as np

from neurodrone.closedloop import PacketCutter
from neurodrone.types import EventPacket


def frame_events(t0, t1, n, rng):
    t = np.sort(rng.integers(t0, t1, n)).astype(np.int64)
    return EventPacket(rng.integers(0, 64, n).astype(np.int16), rng.integers(0, 48, n).astype(np.int16),
                       t, rng.integers(0, 2, n).astype(np.int8), t0, t1)


def run(cutter, frames=500, frame_us=2000, n=7, seed=0):
    rng = np.random.default_rng(seed)
    pushed, pkts = 0, []
    for k in range(frames):
        ev = frame_events(k * frame_us, (k + 1) * frame_us, n, rng)
        pushed += ev.n
        pkts += cutter.push(ev, (k + 1) * frame_us)
    return pushed, pkts


def test_no_event_lost_or_duplicated_and_each_inside_its_window():
    pushed, pkts = run(PacketCutter(10_000, phase_us=3300, jitter_us=2000))
    for p in pkts:
        assert p.n == 0 or (p.t_us.min() >= p.t_start_us and p.t_us.max() < p.t_end_us)
    # packets tile time with no gaps
    assert all(a.t_end_us == b.t_start_us for a, b in zip(pkts, pkts[1:]))
    leftover = pushed - sum(p.n for p in pkts)
    assert 0 <= leftover <= 7 * 7, "only the not-yet-complete tail may be held back"


def test_realistic_grid_is_off_the_frame_grid_and_jitters_but_aligned_mode_does_not():
    _, real = run(PacketCutter(10_000, phase_us=3300, jitter_us=2000))
    lens = np.array([p.t_end_us - p.t_start_us for p in real[1:]])  # first packet is the short phase offset
    assert real[0].t_end_us % 2000 != 0 and real[0].t_end_us == 3300
    assert lens.min() >= 8000 and lens.max() <= 12000 and len(set(lens.tolist())) > 3
    _, aligned = run(PacketCutter(10_000, phase_us=0, jitter_us=0))
    assert {p.t_end_us - p.t_start_us for p in aligned} == {10_000}
    assert all(p.t_end_us % 2000 == 0 for p in aligned)
