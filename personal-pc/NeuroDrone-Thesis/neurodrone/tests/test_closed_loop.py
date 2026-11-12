"""End-to-end behaviour in the synthetic closed loop (same Brain/Coordinator as the real pipeline).
Uses the default detector (the fully spiking one); the hybrid baseline is checked at the end.

These are LOGIC checks on an idealised simulator. They do not predict real-world performance."""
import math

import pytest

from neurodrone.closedloop import ClosedLoopSim, default_scenarios

SC = {s.name: s for s in default_scenarios()}


def run(name):
    return ClosedLoopSim(SC[name]).run()


def test_flies_to_target_and_stops_when_vlm_confirms_it():
    r = run("approach_target")
    assert r.arrived and not r.collided and "VLM confirms" in r.ended
    assert r.final_pose[0] > 8.0 and abs(r.final_pose[1]) < 1.0  # got most of the way there, on the line


def test_target_off_to_the_side_is_reached_by_turning_towards_it():
    r = run("bearing_offset")
    assert r.arrived and not r.collided
    assert r.final_pose[1] > 1.5 and 5.0 < r.final_pose[2] < 40.0  # moved right and is pointing right


def test_attitude_jitter_does_not_stop_the_mission():
    r = run("jitter_approach")
    assert r.arrived and not r.collided


@pytest.mark.slow
def test_obstacle_in_the_way_is_dodged_and_mission_completes():
    r = run("dodge_obstacle")
    assert not r.collided and r.arrived
    assert r.modes["AVOID"] > 0 and r.modes["ADJUDICATE"] > 0


def test_without_any_vlm_goal_the_drone_does_not_move():
    r = run("vlm_silent")
    assert r.modes["NO_GOAL"] == r.modes.total() and math.hypot(*r.final_pose[:2]) < 1e-6


def test_reflex_layer_dying_mid_flight_stops_the_vehicle_and_requests_landing():
    r = run("snn_blackout")
    assert r.ended.startswith("land") and "reflex layer lost" in r.ended
    drift = r.final_pose[0] - r.pose_at_blackout[0]
    assert 0 <= drift < 1.6, f"kept flying {drift:.2f} m after losing its reflexes"
    assert r.modes["BLIND_HOLD"] > 0


@pytest.mark.slow
def test_range_sensor_prevents_collision_with_a_blank_wall_and_gives_up_gracefully():
    r = run("blank_wall_with_range")
    assert not r.collided and r.min_clearance > 0.6


def test_KNOWN_LIMITATION_blank_wall_is_invisible_to_the_snn_alone():
    """Documents a real failure mode: a textureless wall taller than the field of view, approached
    off-centre, produces no looming alarm at all. If this ever STOPS colliding, something changed
    in the detector: update the README's limitations section before celebrating."""
    r = run("blank_wall_snn_only")
    assert r.collided and r.first_alarm_clearance is None


@pytest.mark.slow
@pytest.mark.parametrize("name", ["approach_target", "bearing_offset", "jitter_approach", "dodge_obstacle"])
def test_hybrid_baseline_detector_still_flies_the_core_scenarios(name):
    from neurodrone.config import SNNConfig
    r = ClosedLoopSim(SC[name], SNNConfig(impl="hybrid")).run()
    assert r.arrived and not r.collided
