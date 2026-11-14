"""neurodrone.mission: geometry, depth guard, consensus and steering of the VLM-goal mission (no I/O)."""
import math

import numpy as np

from neurodrone import mission as m
from neurodrone.types import VehicleState

RGB = m.Pinhole(1920, 1080, 1.204)  # the Gazebo x500_depth cameras (OakD-Lite model)
DEP = m.Pinhole(640, 480, 1.274)


def st(n=0.0, e=0.0, d=-2.5, yaw=0.0, roll=0.0, pitch=0.0):
    return VehicleState(north_m=n, east_m=e, down_m=d, yaw_deg=yaw, roll_deg=roll, pitch_deg=pitch)


def test_pinhole_roundtrip_and_rotation():
    p = RGB.pixel(RGB.ray(1500.0, 300.0) * 7.0)
    assert abs(p[0] - 1500) < 1e-6 and abs(p[1] - 300) < 1e-6
    R = m.R_ned_from_frd(0, 0, 90)  # facing east: body forward = east
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-9) and np.allclose(R @ [0, 1, 0], [-1, 0, 0], atol=1e-9)


def test_box_depth_maps_between_cameras():
    depth = np.full((480, 640), 9.0, np.float32)
    # an object 4 m away, right of centre: where does the RGB box land in the depth image?
    ray = RGB.ray(0.7 * 1920, 0.5 * 1080)
    u, v = DEP.pixel(ray)
    depth[int(v) - 15:int(v) + 15, int(u) - 15:int(u) + 15] = 4.0
    assert abs(m.box_depth(depth, DEP, RGB, [0.68, 0.47, 0.72, 0.53]) - 4.0) < 1e-6
    assert m.box_depth(np.full((480, 640), np.nan, np.float32), DEP, RGB, [0.4, 0.4, 0.6, 0.6]) is None


def test_goal_from_box_standoff_and_heading():
    g = m.goal_from_box([0.45, 0.45, 0.55, 0.55], 6.0, RGB, st(yaw=90), cam_offset_frd=(0, 0, 0), standoff_m=1.5)
    assert abs(g[0]) < 1e-6 and abs(g[1] - 4.5) < 1e-6 and abs(g[2] + 2.5) < 1e-9  # 6 m east, 1.5 m short, same altitude
    g2 = m.goal_from_box([0.0, 0.45, 0.1, 0.55], 6.0, RGB, st(), cam_offset_frd=(0, 0, 0))
    assert g2[1] < -1.0  # a box on the left of the image -> a goal to the west of north


def test_corridor_and_guard():
    wall = np.full((480, 640), 3.0, np.float32)
    assert abs(m.corridor_distance(wall, DEP, np.array([1.0, 0, 0])) - 3.0) < 0.01
    g = m.guard(wall, DEP, np.array([1.0, 0, 0]), 0.05)
    assert not g.blocked and abs(g.ttc_s - 3.0) < 0.05
    g = m.guard(wall, DEP, np.array([2.0, 0, 0]), 0.05)  # 1.5 s to contact
    assert g.blocked and "contact" in g.reason
    g = m.guard(np.full((480, 640), 1.0, np.float32), DEP, np.array([0.3, 0, 0]), 0.05)
    assert g.blocked and "obstacle" in g.reason
    assert m.guard(None, DEP, np.array([1.0, 0, 0]), 0.0).blocked  # no depth: never fly blind
    assert m.guard(wall, DEP, np.array([1.0, 0, 0]), 2.0).blocked  # stale depth
    side = np.full((480, 640), np.inf, np.float32)
    side[:, 600:] = 2.0  # something off to the right, not in the corridor
    assert not m.guard(side, DEP, np.array([1.0, 0, 0]), 0.05).blocked
    assert m.corridor_distance(np.where(np.random.default_rng(0).random((480, 640)) < 2e-5, 0.5, 9.0).astype(np.float32),
                               DEP, np.array([1.0, 0, 0])) > 8.0  # a few speckle pixels are ignored


def test_consensus_needs_agreement():
    c = m.Consensus(need=2, window=3, tol_m=1.0)
    assert c.add(np.array([5.0, 0, -2])) is None
    assert c.add(np.array([0.0, 9, -2])) is None  # disagrees
    got = c.add(np.array([5.4, 0.2, -2]))
    assert got is not None and abs(got[0] - 5.2) < 1e-9
    assert c.add(None) is None  # the first answer dropped out of the window: no agreement left
    assert c.add(np.array([5.2, 0.1, -2])) is not None  # agrees with the 3rd again


def test_steer_turn_then_fly_and_guard():
    s = st(yaw=0)
    cmd = m.steer(np.array([0.0, 5.0, -2.5]), s)  # goal due east: turn first
    assert cmd.phase == "turn" and cmd.yaw_rate > 0 and cmd.vx == 0
    cmd = m.steer(np.array([5.0, 0.2, -2.5]), s, vmax=1.0)
    assert cmd.phase == "fly" and 0.9 < cmd.vx <= 1.0
    assert m.steer(np.array([0.2, 0.1, -2.5]), s).phase == "arrived"
    assert m.steer(None, s).phase == "search"
    blocked = m.apply_guard(cmd, m.GuardResult(1.0, 1.0, True, "obstacle 1.0 m ahead"))
    assert blocked.vx == 0 and blocked.phase == "blocked"
    assert math.isclose(m.apply_guard(cmd, m.GuardResult(9, 9, False)).vx, cmd.vx)


# ----------------------------------------------------------------------------- go-around
def _pole_depth(cam, z=1.8, y0=-0.35, y1=-0.05):
    """Depth image with only a pole (lateral y0..y1 m, body FRD, + = right) at z metres, everything else no return."""
    d = np.full((cam.height, cam.width), np.nan, np.float32)
    u0, u1 = (int(y / z * cam.f + cam.width / 2) for y in (y0, y1))
    d[:, max(u0, 0):min(u1 + 1, cam.width)] = z
    return d


def test_avoider_goes_around_the_free_side_and_commits():
    from neurodrone.mission import Avoider, Command, Pinhole
    cam = Pinhole(160, 120, 1.274)
    av = Avoider()
    fly = Command(1.0, 0.0, 0.0, 0.0, "fly")
    out, g = av.step(fly, _pole_depth(cam), cam, 0.0, now=0.0)
    assert g.blocked and out.phase == "avoid"
    assert out.vy > 0 and abs(out.vx) > 0  # pole slightly left -> goes right, still forwards
    assert abs(math.hypot(out.vx, out.vy) - 0.5) < 1e-6
    # straight path free again (pole out of view): keeps the go-around direction for commit_s ...
    empty = np.full((cam.height, cam.width), np.nan, np.float32)
    out2, g2 = av.step(fly, empty, cam, 0.0, now=1.0)
    assert not g2.blocked and out2.phase == "avoid" and out2.vy > 0
    # ... then back to the plain command
    out3, _ = av.step(fly, empty, cam, 0.0, now=2.5)
    assert out3.phase == "fly" and out3.vy == 0


def test_avoider_stops_when_no_direction_is_free():
    from neurodrone.mission import Avoider, Command, Pinhole
    cam = Pinhole(160, 120, 1.274)
    wall = np.full((cam.height, cam.width), 1.5, np.float32)
    out, g = Avoider().step(Command(1.0, 0.0, 0.0, 0.0, "fly"), wall, cam, 0.0, now=0.0)
    assert g.blocked and out.phase == "blocked" and out.vx == 0 and out.vy == 0


def test_avoider_stale_depth_stops_and_turning_untouched():
    from neurodrone.mission import Avoider, Command, Pinhole
    cam = Pinhole(160, 120, 1.274)
    av = Avoider()
    out, g = av.step(Command(1.0, 0.0, 0.0, 0.0, "fly"), None, cam, 9.9, now=0.0)
    assert out.phase == "blocked" and out.vx == 0
    turn = Command(0.0, 0.0, 0.0, 20.0, "turn")
    out, _ = av.step(turn, _pole_depth(cam), cam, 0.0, now=0.0)
    assert out == turn


# ----------------------------------------------------------------------------- search
def test_searcher_turns_stops_and_looks():
    from neurodrone.mission import Searcher
    s = Searcher(max_rate=20.0)
    yr, note = s.step(90.0, now=0.0, last=None)  # start heading = first view
    assert yr == 0 and "looking at +0" in note
    yr, _ = s.step(90.0, now=1.0, last=dict(found=False, t_capture=-1.0))  # old picture (before settling): ignored
    assert yr == 0 and s.i == 0
    yr, note = s.step(90.0, now=6.0, last=dict(found=False, t_capture=0.5))  # fresh picture, nothing -> next view
    assert s.i == 1 and "next view" in note
    yr, note = s.step(90.0, now=6.1, last=dict(found=False, t_capture=0.5))
    assert yr == 20.0 and "turning to +30" in note  # clockwise towards +30, rate capped
    yr, _ = s.step(119.0, now=8.0, last=dict(found=False, t_capture=0.5))  # within tolerance: settle
    assert yr == 0 and s.settled_at == 8.0
    yr, note = s.step(119.0, now=14.0, last=dict(found=True, t_capture=8.5))  # found: stay
    assert yr == 0 and "waiting for agreement" in note and s.i == 1
    s.step(119.0, now=20.0, last=dict(found=False, t_capture=14.5))  # lost it: next view is -30
    yr, note = s.step(119.0, now=20.1, last=None)
    assert yr < 0 and "turning to -30" in note


def test_searcher_wraps_around_north():
    from neurodrone.mission import Searcher
    s = Searcher(max_rate=20.0)
    s.step(170.0, 0.0, None)
    s.step(170.0, 5.0, dict(found=False, t_capture=1.0))  # -> +30 = 200 = -160
    yr, note = s.step(170.0, 5.1, None)
    assert yr > 0 and "+30" in note  # short way (clockwise through 180), not 330 deg back


def test_active_searcher_skips_seen_views_and_centres_a_found_box():
    from neurodrone.mission import ActiveSearcher
    s = ActiveSearcher(max_rate=20.0)
    yr, _ = s.step(90.0, now=0.0, last=None)  # settles on the start heading
    assert yr == 0 and s.settled_at == 0.0
    # nothing at 90 deg -> the next view is a neighbouring, unseen direction, not the one just seen
    yr, note = s.step(90.0, now=6.0, last=dict(found=False, box=None, t_capture=0.5, yaw_capture=90.0))
    assert abs(((s.target - 90.0 + 180) % 360) - 180) >= 34.5 and yr != 0
    assert s.belief[s._in_view(90.0)].sum() < s.belief[s._in_view(270.0)].sum()
    # turn there, settle, then the target is found near the right edge -> turn to centre it
    h = s.target
    s.step(h, now=8.0, last=None)
    yr, _ = s.step(h, now=14.0, last=dict(found=True, box=[0.85, 0.4, 0.95, 0.6], t_capture=8.5, yaw_capture=h))
    assert yr > 0 and 20.0 < ((s.target - h) % 360) < 34.5  # box right of centre -> turn clockwise towards it
    # centred and found again -> stay
    c = s.target
    s.step(c, now=16.0, last=None)
    yr, note = s.step(c, now=22.0, last=dict(found=True, box=[0.45, 0.4, 0.55, 0.6], t_capture=16.5, yaw_capture=c))
    assert yr == 0 and "looking" in note


def test_active_searcher_answer_used_once():
    from neurodrone.mission import ActiveSearcher
    s = ActiveSearcher()
    s.step(0.0, 0.0, None)
    ans = dict(found=False, box=None, t_capture=1.0, yaw_capture=0.0)
    s.step(0.0, 5.0, ans)
    b = s.belief.copy()
    s.step(s.target, 6.0, ans)  # same answer again: no second update
    assert (s.belief == b).all()


def _floor_depth(cam, height_m, pitch_deg):
    """Depth image of a flat floor `height_m` below a camera pitched by pitch_deg (nose down < 0)."""
    from neurodrone.mission import R_ned_from_frd
    R = R_ned_from_frd(0.0, pitch_deg, 0.0)
    v, u = np.mgrid[0:cam.height, 0:cam.width]
    rays = np.stack([np.ones(u.shape), (u + 0.5 - cam.width / 2) / cam.f, (v + 0.5 - cam.height / 2) / cam.f], -1)
    down = rays @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(down[..., 2] > 1e-6, height_m / down[..., 2], np.nan)
    return t.astype(np.float32)  # x-component of the ray is 1: t is the depth


def test_guard_ignores_the_floor_when_pitched():
    from neurodrone.mission import Pinhole, guard
    cam = Pinhole(160, 120, 1.274)
    d = _floor_depth(cam, height_m=1.2, pitch_deg=-20.0)
    v = np.array([1.0, 0.0, 0.0])
    g0 = guard(d, cam, v, 0.0)
    assert g0.blocked and g0.dist_ahead_m < 2.0  # corridor tilted with the camera: the floor counts as an obstacle
    g = guard(d, cam, v, 0.0, att_deg=(0.0, -20.0))
    assert not g.blocked and g.dist_ahead_m == float("inf")  # level corridor at 1.2 m height, radius 0.8
    level = _floor_depth(cam, height_m=1.3, pitch_deg=0.0)
    assert not guard(level, cam, v, 0.0, att_deg=(0.0, 0.0)).blocked


def test_box_at_edge_and_sweep_moves_on():
    from neurodrone.mission import Searcher, box_at_edge
    assert box_at_edge([0.91, 0.6, 1.0, 0.8]) and box_at_edge([0.0, 0.6, 0.1, 0.8])
    assert not box_at_edge([0.4, 0.4, 0.6, 0.6])
    s = Searcher(max_rate=20.0)
    s.step(90.0, 0.0, None)
    s.step(90.0, 6.0, dict(found=True, usable=False, t_capture=1.0))  # cut off at the edge: next view
    assert s.i == 1
