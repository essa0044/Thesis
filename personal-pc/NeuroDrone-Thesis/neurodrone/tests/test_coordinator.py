"""Coordinator logic with an injected clock: no sensors, no threads, fully deterministic."""
import math

from neurodrone.config import CoordinatorConfig, SNNConfig
from neurodrone.coordinator import (ADJUDICATE, ARRIVED, AVOID, BLIND_HOLD, FAILSAFE, IDLE, NO_GOAL,
                                    SEARCH, SETTLE, TRACK, Coordinator)
from neurodrone.runtime import Brain
from neurodrone.safety import max_safe_speed
from neurodrone.snn import LoomingSNN
from neurodrone.types import ClearanceReport, SemanticGoal, ThreatReport, VehicleState

HFOV = 65.0
DT = 0.04


def threat(now, alarm=False, frontal=0.0, urg=(0.0, 0.0, 0.0)):
    return ThreatReport(t_us=0, urgency=urg, frontal=frontal, alarm=alarm, n_events=0, proc_s=0.0, stamp=now)


def veh(**kw):
    return VehicleState(armed=True, in_air=True, engaged=True, **kw)


def goal(captured_at, x=0.5, w=0.05, yaw=0.0, kind="target"):
    return SemanticGoal(kind=kind, x=x if kind == "target" else None, y=0.5,
                        w=w if kind == "target" else None, captured_at=captured_at,
                        yaw_at_capture_deg=yaw)


def drive(c, t0, dur, threat_fn, goal_fn, veh_fn=lambda t: veh(), clr_fn=lambda t: None):
    """Step the coordinator at 25 Hz; returns [(t, Decision)]."""
    out, t = [], t0
    while t < t0 + dur - 1e-9:
        out.append((t, c.step(t, threat_fn(t), goal_fn(t), veh_fn(t), clr_fn(t))))
        t += DT
    return out


def modes(trace):
    seen = []
    for _, d in trace:
        if not seen or seen[-1] != d.mode:
            seen.append(d.mode)
    return seen


def fresh(x=0.5, w=0.05, kind="target"):
    """Always-fresh VLM goal (period 0, latency 0)."""
    return lambda t: goal(t, x=x, w=w, kind=kind)


def test_stale_snn_holds_still_then_failsafe():
    c = Coordinator(CoordinatorConfig(), HFOV)
    tr = drive(c, 0.0, 3.3, lambda t: None, fresh())
    assert tr[0][1].mode == BLIND_HOLD and tr[0][1].cmd.vx == 0.0
    last = tr[-1][1]
    assert last.mode == FAILSAFE and last.request == "land" and "reflex" in last.note


def test_snn_that_stops_updating_is_treated_as_dead():
    c = Coordinator(CoordinatorConfig(snn_timeout_s=0.25), HFOV)
    stamp = 0.0
    tr = drive(c, 0.0, 1.0, lambda t: threat(stamp), fresh(w=0.05))  # report never refreshed
    assert tr[0][1].mode == TRACK or tr[0][1].mode == NO_GOAL
    assert tr[-1][1].mode == BLIND_HOLD


def test_not_engaged_is_idle_and_commands_nothing():
    c = Coordinator(CoordinatorConfig(), HFOV)
    d = c.step(0.0, threat(0.0), goal(0.0), VehicleState(armed=True, in_air=True, engaged=False))
    assert d.mode == IDLE and d.cmd.vx == 0 and d.request is None


def test_tracks_target_and_compensates_for_yaw_during_vlm_latency():
    cfg = CoordinatorConfig(cruise_speed=1.0)
    # target at x=0.75 -> +16.25 deg; too large to drive forward, so it turns in place first
    c = Coordinator(cfg, HFOV)
    d = c.step(0.0, threat(0.0), goal(0.0, x=0.75), veh(yaw_deg=0.0))
    assert d.mode == TRACK and d.cmd.yaw_rate > 0 and d.cmd.vx == 0.0
    # the drone has since yawed +10 deg: remaining bearing 6.25 deg -> yaw 5 deg/s and now it flies
    c = Coordinator(cfg, HFOV)
    tr = drive(c, 0.0, 1.5, lambda t: threat(t), lambda t: goal(0.0, x=0.75, yaw=0.0), lambda t: veh(yaw_deg=10.0))
    d = tr[-1][1]
    assert math.isclose(d.cmd.yaw_rate, 0.8 * (16.25 - 10.0), abs_tol=0.01)
    assert d.cmd.vx > 0.5
    # yaw wrap-around: heading 359 -> 1 deg is +2 deg, not -358
    c = Coordinator(cfg, HFOV)
    d = c.step(0.0, threat(0.0), goal(0.0, x=0.5, yaw=359.0), veh(yaw_deg=1.0))
    assert math.isclose(d.cmd.yaw_rate, 0.8 * (0.0 - 2.0), abs_tol=0.01)


def test_forward_speed_ramps_up_but_braking_is_immediate():
    c = Coordinator(CoordinatorConfig(cruise_speed=1.0, accel_limit=1.0), HFOV)
    first = c.step(0.0, threat(0.0), goal(0.0), veh())
    assert first.cmd.vx <= 1.0 * DT + 1e-9
    tr = drive(c, DT, 2.0, lambda t: threat(t), lambda t: goal(t))
    assert tr[-1][1].cmd.vx > 0.95
    d = c.step(2.1, threat(2.1, alarm=True), goal(2.1), veh())
    assert d.mode == ADJUDICATE and d.cmd.vx == 0.0


def test_alarm_brakes_and_waits_for_a_FRESH_goal_then_arrives_if_target():
    c = Coordinator(CoordinatorConfig(), HFOV)
    big_centred = lambda cap: goal(cap, x=0.5, w=0.3)
    d = c.step(10.0, threat(10.0, alarm=True), big_centred(5.0), veh())  # goal is old (captured at t=5)
    assert d.mode == ADJUDICATE and d.cmd.vx == 0
    d = c.step(10.5, threat(10.5, alarm=True), big_centred(5.0), veh())  # still the pre-alarm goal
    assert d.mode == ADJUDICATE, "a goal captured before the alarm must not decide anything"
    d = c.step(11.0, threat(11.0), big_centred(10.2), veh())  # captured after the alarm began
    assert d.mode == ARRIVED and d.request == "hold"
    d = c.step(11.1, threat(11.1), None, veh())
    assert d.mode == ARRIVED, "arrival is latched"


def test_alarm_on_something_that_is_not_the_target_dodges_away_from_the_busier_side():
    c = Coordinator(CoordinatorConfig(dodge_distance_m=2.0, dodge_speed=1.0), HFOV)
    urg = (1.0, 0.0, 0.2)  # left side busier -> go right
    c.step(0.0, threat(0.0, alarm=True, urg=urg), goal(-5.0), veh())
    d = c.step(0.2, threat(0.2, urg=urg), goal(0.1, x=0.5, w=0.05), veh())  # fresh: small = far = not the target
    assert d.mode == AVOID and d.cmd.vy > 0 and d.cmd.vx == 0
    tr = drive(c, 0.24, 3.5, lambda t: threat(t), fresh(w=0.05))
    assert modes(tr)[:3] == [AVOID, SETTLE, NO_GOAL] or modes(tr)[:2] == [AVOID, SETTLE]
    assert tr[-1][1].mode in (TRACK, NO_GOAL)


def test_dodge_is_open_loop_and_lasts_the_full_duration_even_if_alarm_clears():
    c = Coordinator(CoordinatorConfig(dodge_distance_m=2.0, dodge_speed=1.0), HFOV)
    c.step(0.0, threat(0.0, alarm=True), goal(-5.0), veh())
    c.step(0.1, threat(0.1), goal(0.05, w=0.05), veh())
    tr = drive(c, 0.14, 1.8, lambda t: threat(t, alarm=False), fresh(w=0.05))
    assert all(d.mode == AVOID for _, d in tr), "dodge must not end just because the looming cue vanished"


def test_goals_captured_before_a_sidestep_are_not_trusted_afterwards():
    c = Coordinator(CoordinatorConfig(dodge_distance_m=1.0, settle_s=0.2), HFOV)
    c.step(0.0, threat(0.0, alarm=True), goal(-5.0), veh())
    c.step(0.1, threat(0.1), goal(0.05, w=0.05), veh())  # dodge starts at ~0.1 for 1 s
    stale = goal(0.5, x=0.9, w=0.05)  # taken mid-dodge
    tr = drive(c, 0.14, 2.0, lambda t: threat(t), lambda t: stale)
    assert tr[-1][1].mode == NO_GOAL and tr[-1][1].cmd.vx == 0.0


def test_no_fresh_goal_within_timeout_assumes_obstacle_and_dodges():
    c = Coordinator(CoordinatorConfig(adjudicate_timeout_s=2.0, adj_arm_s=100.0), HFOV)
    tr = drive(c, 0.0, 3.0, lambda t: threat(t, alarm=(t < 0.05)), lambda t: None)
    ms = modes(tr)
    assert ADJUDICATE in ms and AVOID in ms
    assert [d.note for _, d in tr if d.mode == AVOID][0].startswith("no fresh VLM goal")


def test_alarm_that_persists_while_stopped_means_it_is_coming_at_us():
    c = Coordinator(CoordinatorConfig(adj_arm_s=1.0, adjudicate_timeout_s=100.0), HFOV)
    tr = drive(c, 0.0, 2.0, lambda t: threat(t, alarm=True), lambda t: None)
    first_avoid = next(t for t, d in tr if d.mode == AVOID)
    assert 0.9 < first_avoid < 1.2


def test_gives_up_and_holds_after_too_many_dodges():
    c = Coordinator(CoordinatorConfig(max_episodes=1), HFOV)
    tr = drive(c, 0.0, 12.0, lambda t: threat(t, alarm=True), fresh(w=0.05))
    last = [d for _, d in tr if d.request][-1]
    assert last.mode == FAILSAFE and last.request == "hold" and "stuck" in last.note


def test_repeat_dodges_keep_direction_and_get_longer():
    c = Coordinator(CoordinatorConfig(max_episodes=3, dodge_distance_m=1.0, dodge_speed=1.0, escalation=0.5), HFOV)
    tr = drive(c, 0.0, 12.0, lambda t: threat(t, alarm=True, urg=(0.0, 0.0, 1.0)), fresh(w=0.05))
    starts = [(t, d) for (t, d), (_, p) in zip(tr, [(0, None)] + tr[:-1]) if d.mode == AVOID and (p is None or p.mode != AVOID)]
    assert len(starts) >= 2
    assert all(d.cmd.vy < 0 for _, d in starts), "right side busy -> always dodge left"
    durs = [float(d.note.split(",")[-1].strip().rstrip("s")) for _, d in starts]
    assert durs[1] > durs[0]


def test_search_rotates_then_gives_up_after_a_full_turn():
    c = Coordinator(CoordinatorConfig(search_yaw_rate=90.0, max_yaw_rate=90.0, search_max_deg=360.0), HFOV)
    tr = drive(c, 0.0, 5.0, lambda t: threat(t), fresh(kind="search"))
    assert tr[0][1].mode == SEARCH and tr[0][1].cmd.yaw_rate == 90.0 and tr[0][1].cmd.vx == 0
    gave_up = next(t for t, d in tr if d.request == "hold")
    assert 3.9 < gave_up < 4.2, "360 deg at 90 deg/s takes 4 s"


def test_search_accounting_uses_the_clamped_rate_not_the_configured_one():
    # configured 90 deg/s but max_yaw_rate=40: the drone really turns at 40, so a full turn takes 9 s
    c = Coordinator(CoordinatorConfig(search_yaw_rate=90.0, max_yaw_rate=40.0, search_max_deg=360.0), HFOV)
    tr = drive(c, 0.0, 10.0, lambda t: threat(t), fresh(kind="search"))
    assert tr[0][1].cmd.yaw_rate == 40.0
    gave_up = next(t for t, d in tr if d.request == "hold")
    assert 8.8 < gave_up < 9.2


def test_expired_goal_is_ignored():
    c = Coordinator(CoordinatorConfig(goal_ttl_s=2.0), HFOV)
    d = c.step(10.0, threat(10.0), goal(5.0), veh())
    assert d.mode == NO_GOAL and d.cmd.vx == 0


def test_battery_and_geofence_hand_over():
    c = Coordinator(CoordinatorConfig(min_battery=0.25, max_dist_home_m=40, max_alt_m=30), HFOV)
    assert c.step(0.0, threat(0.0), goal(0.0), veh(battery_frac=0.2)).request == "land"
    assert c.step(0.1, threat(0.1), goal(0.1), veh(dist_home_m=41)).request == "hold"
    assert c.step(0.2, threat(0.2), goal(0.2), veh(alt_m=31)).request == "hold"
    # on the ground a low battery reading must not trigger a "land" request
    assert c.step(0.3, threat(0.3), goal(0.3), VehicleState(engaged=True, battery_frac=0.1)).request is None


def test_urgency_slowdown_is_off_by_default_and_side_push_is_off_by_default():
    c = Coordinator(CoordinatorConfig(cruise_speed=1.0), HFOV)
    tr = drive(c, 0.0, 2.0, lambda t: threat(t, frontal=0.7, urg=(1.0, 0.0, 0.0)), lambda t: goal(t))
    d = tr[-1][1]
    assert d.cmd.vx > 0.95 and d.cmd.vy == 0.0


def test_range_sensor_caps_speed_to_what_can_be_stopped():
    cfg = CoordinatorConfig(use_clearance=True, cruise_speed=1.0)
    c = Coordinator(cfg, HFOV)
    tr = drive(c, 0.0, 3.0, lambda t: threat(t), lambda t: goal(t), clr_fn=lambda t: ClearanceReport(t, 1.2))
    d = tr[-1][1]
    expected = max_safe_speed(0.75 * (1.2 - cfg.clearance_margin_m), cfg.reaction_s, cfg.brake_decel)
    assert d.mode == TRACK and math.isclose(d.cmd.vx, expected, rel_tol=0.02) and d.cmd.vx < cfg.cruise_speed


def test_range_sensor_alarm_brakes_and_is_named_in_the_log():
    c = Coordinator(CoordinatorConfig(use_clearance=True), HFOV)
    d = c.step(0.0, threat(0.0), goal(0.0), veh(), ClearanceReport(0.0, 0.6))
    assert d.mode == ADJUDICATE and "range alarm" in d.note


def test_range_sensor_is_mandatory_once_enabled():
    cfg = CoordinatorConfig(use_clearance=True, clearance_timeout_s=0.5)
    c = Coordinator(cfg, HFOV)
    d = c.step(0.0, threat(0.0), goal(0.0), veh(), None)
    assert d.mode == BLIND_HOLD and "range sensor" in d.note
    d = c.step(1.0, threat(1.0), goal(1.0), veh(), ClearanceReport(0.0, 5.0))  # 1 s old
    assert d.mode == BLIND_HOLD


def test_range_sensor_ignored_when_not_enabled():
    c = Coordinator(CoordinatorConfig(use_clearance=False), HFOV)
    d = c.step(0.0, threat(0.0), goal(0.0), veh(), ClearanceReport(0.0, 0.1))
    assert d.mode == TRACK


def test_brain_ignores_a_single_lost_frame_but_not_two():
    brain = Brain(LoomingSNN(32, 24, SNNConfig()), Coordinator(CoordinatorConfig(), HFOV), lost_patience=2)
    brain.on_goal(goal(0.0, x=0.4, w=0.1))
    brain.on_goal(goal(1.0, kind="search"))
    assert brain._goal.kind == "target"
    brain.on_goal(goal(2.0, kind="search"))
    assert brain._goal.kind == "search"
    brain.on_goal(goal(3.0, x=0.6))
    brain.on_goal(goal(4.0, kind="search"))
    assert brain._goal.kind == "target", "miss counter must reset after a successful detection"
