"""Stages 3 and 4 without a RealSense: the recording format, the IMU attitude, the bench tools and the replay through
the mission code, on a fake camera (tests/fakes.py)."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)
import fakes  # noqa: E402
import realsense_bench as rb  # noqa: E402
import record_replay as rr  # noqa: E402

from neurodrone import mission as M  # noqa: E402
from neurodrone import realsense as R  # noqa: E402


def test_attitude_from_accel_signs():
    assert R.attitude_from_accel(np.array([0.0, -9.81, 0.0])) == pytest.approx((0.0, 0.0), abs=1e-6)
    # nose up 10 deg: gravity's reaction gets a forward (camera +z) component
    roll, pitch = R.attitude_from_accel(np.array([0.0, -9.81 * math.cos(math.radians(10)), 9.81 * math.sin(math.radians(10))]))
    assert pitch == pytest.approx(10.0, abs=1e-6) and roll == pytest.approx(0.0, abs=1e-6)
    # right side down 15 deg (FRD roll +15): specific force gets a -right (camera -x) component
    roll, pitch = R.attitude_from_accel(np.array([-9.81 * math.sin(math.radians(15)), -9.81 * math.cos(math.radians(15)), 0.0]))
    assert roll == pytest.approx(15.0, abs=1e-6) and pitch == pytest.approx(0.0, abs=1e-6)
    # consistent with the mission's own rotation (R_ned_from_frd): f_body = R^T (0, 0, -g)
    Rm = M.R_ned_from_frd(7.0, -4.0, 30.0)
    f_frd = Rm.T @ np.array([0, 0, -9.81])
    r2, p2 = R.attitude_from_accel(np.array([f_frd[1], f_frd[2], f_frd[0]]))  # FRD -> camera optical (x=R, y=D, z=F)
    assert (r2, p2) == pytest.approx((7.0, -4.0), abs=1e-6)


def test_png16_roundtrip_and_principal_point():
    d = np.array([[0.0, 1.2345, np.nan], [70.0, 3.0, 0.5]], np.float32)
    back = R.png16_to_depth(R.depth_to_png16(d))
    assert np.isnan(back[0, 0]) and np.isnan(back[0, 2]) and back[0, 1] == pytest.approx(1.2345, abs=1e-3)
    assert back[1, 0] == pytest.approx(65.535)  # clipped at the uint16 range
    cam = M.Pinhole.from_intrinsics(640, 480, 600.0, 330.0, 250.0)
    assert cam.f == pytest.approx(600.0) and cam.ray(330.0, 250.0)[1:] == pytest.approx([0.0, 0.0])
    u, v = cam.pixel(cam.ray(100.0, 400.0) * 3.0)
    assert (u, v) == pytest.approx((100.0, 400.0))


def test_parse_mark():
    m = R.parse_mark("pole d=4.0 o=1.5")
    assert m["d"] == 4.0 and m["o"] == 1.5 and m["label"] == "pole d=4.0 o=1.5"
    assert R.parse_mark("absent")["absent"] is True


def test_depth_stats_and_noise_fit():
    rng = np.random.default_rng(0)
    frames = [np.full((120, 160), 2.0) + rng.normal(0, 0.02, (120, 160)) for _ in range(20)]
    s = rb.depth_stats(frames, 0.2)
    assert s["measured_m"] == pytest.approx(2.0, abs=0.01) and s["noise_m"] == pytest.approx(0.02, abs=0.005)
    assert s["fill"] == 1.0
    rows = [dict(tape_m=z, measured_m=z + 0.03, noise_m=0.005 * z * z) for z in (1.0, 2.0, 3.0, 4.0)]
    fit = rb.noise_fit(rows)
    assert fit["a_sigma_z2"] == pytest.approx(0.005) and fit["bias_m"] == pytest.approx(0.03)
    holes = [f.copy() for f in frames]
    for h in holes:
        h[::2] = np.nan
    assert rb.depth_stats(holes, 0.2)["fill"] == pytest.approx(0.5, abs=0.05)


def _args(**kw):
    a = rb.build_parser().parse_args(["depth"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_bench_depth_with_emitter_both(tmp_path):
    cams = fakes.FakeRealSense(wall_m=3.0, noise_m=0.01, hole_frac=0.3)
    try:
        answers = iter(["3.0", "1,5", ""])
        rb.main(["depth", "--out", str(tmp_path), "--emitter", "both", "--frames", "5", "--settle", "0"],
                cams=cams, ask=lambda _p: next(answers))
    finally:
        cams.stop()
    rows = list(csv.DictReader(open(tmp_path / "depth.csv")))
    assert [(float(r["tape_m"]), r["emitter"]) for r in rows] == [(3.0, "on"), (3.0, "off"), (1.5, "on"), (1.5, "off")]
    on = [r for r in rows if r["emitter"] == "on"]
    assert float(on[0]["measured_m"]) == pytest.approx(3.0, abs=0.02) and float(on[0]["fill"]) == 1.0
    off = [r for r in rows if r["emitter"] == "off"]
    assert float(off[0]["fill"]) < 0.8  # the fake loses depth with the emitter off
    assert json.load(open(tmp_path / "depth.json"))["fit"]["on"]["rms_error_m"] > 1.0  # 1.5 m tape vs 3 m wall


def test_bench_vlm_and_score(tmp_path):
    cams = fakes.FakeRealSense(wall_m=6.0, target_m=2.5, noise_m=0.0)
    try:
        with fakes.FakeOllama(lambda instr, img: '{"found": true, "box": [440, 420, 560, 580]}') as srv:
            answers = iter(["2.5", "absent", ""])
            rb.main(["vlm", "--out", str(tmp_path), "--ollama", srv.url, "--shots", "2", "--skip", "1",
                     "--instructions", "the red box"], cams=cams, ask=lambda _p: next(answers))
    finally:
        cams.stop()
    review = tmp_path / "review.csv"
    rows = list(csv.DictReader(open(review)))
    assert len(rows) == 4 and (tmp_path / "index.html").exists()
    assert float(rows[0]["dist_m"]) == pytest.approx(2.5, abs=0.01)  # depth in the box = the target, not the wall
    for r in rows:
        r["correct"] = "y" if r["absent"] == "False" else ""
    with open(review, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    s = rb.cmd_vlm_score(str(review))
    assert s["found_rate"] == 1.0 and s["correct_rate"] == 1.0 and s["not_found_when_absent"] == 0.0
    assert s["dist_rms_m"] == pytest.approx(0.0, abs=0.01)


def test_bench_guard_blocks_at_close_wall(tmp_path):
    cams = fakes.FakeRealSense(wall_m=1.0, noise_m=0.0)
    try:
        rb.main(["guard", "--out", str(tmp_path), "--duration", "0.5", "--no-ask"], cams=cams)
    finally:
        cams.stop()
    s = json.load(open(tmp_path / "guard_summary.json"))
    assert s["blocked_share"] == 1.0 and s["block_events"] == 1 and s["first_block_dist_m"][0] == pytest.approx(1.0, abs=0.05)
    cams = fakes.FakeRealSense(wall_m=8.0, noise_m=0.0)
    try:
        rb.main(["guard", "--out", str(tmp_path / "free"), "--duration", "0.5", "--no-ask"], cams=cams)
    finally:
        cams.stop()
    assert json.load(open(tmp_path / "free" / "guard_summary.json"))["blocked_share"] == 0.0


def test_bench_timing(tmp_path):
    cams = fakes.FakeRealSense()
    try:
        rb.main(["timing", "--out", str(tmp_path), "--duration", "0.5"], cams=cams)
    finally:
        cams.stop()
    s = json.load(open(tmp_path / "timing.json"))
    assert s["frames"] > 10 and s["age_median_s"] == pytest.approx(0.02, abs=0.02) and s["imu_roll_pitch_deg"] == [0.0, 0.0]


def _record(tmp_path, wall_m=1.0, target_m=3.0, seconds=0.6):
    out = str(tmp_path / "rec")
    holder = {}

    def open_cams(on_frame, on_imu):
        holder["c"] = fakes.FakeRealSense(wall_m=wall_m, target_m=target_m, noise_m=0.0, fps=40, on_frame=on_frame, on_imu=on_imu)
        return holder["c"]

    import threading
    import time as _t

    def lines():
        _t.sleep(seconds / 2)
        yield "red box d=3.0 o=1.0"
        _t.sleep(seconds / 2)
        yield "q"
    rr.main(["record", "--out", out, "--notes", "test"], open_cams=open_cams, lines=lines())
    return out


def test_record_and_replay(tmp_path):
    out = _record(tmp_path)
    rec = R.Recording(out)
    s = rec.summary()
    assert s["frames"] >= 10 and s["marks"] == 1 and s["accel_hz"] > 10
    f = rec.load(0)
    assert f.rgb.shape == (120, 160, 3) and f.depth_m[60, 80] == pytest.approx(3.0, abs=1e-3)
    assert rr.main(["info", out]) == 0

    def ask(rgb):  # the VLM answers the red square in the middle every time
        return '{"found": true, "box": [440, 420, 560, 580]}', (160, 120), 0.5
    summ = rr.replay(rec, rr.build_parser().parse_args(["replay", out, "--vlm-period", "0.05"]), ask=ask,
                     out_dir=str(tmp_path / "replay"))
    assert summ["vlm_calls"] >= 2 and summ["vlm_found_share"] == 1.0 and summ["goal_accepted"] is not None
    goal = summ["goal_accepted"]
    assert goal[0] == pytest.approx(3.0 - 1.5, abs=0.1) and abs(goal[1]) < 0.1  # 1.5 m short of the target, ahead
    assert summ["guard_blocked_share"] == 1.0  # wall at 1.0 m around the target, inside the corridor
    m = list(csv.DictReader(open(tmp_path / "replay" / "marks.csv")))[0]
    assert float(m["d"]) == 3.0 and float(m["dist_error"]) == pytest.approx(0.0, abs=0.01)
    assert m["guard_blocked"] == "True"


def test_replay_with_pose_file(tmp_path):
    out = _record(tmp_path, wall_m=9.0, target_m=4.0)
    rec = R.Recording(out)
    t0, t1 = rec.frames[0]["t_wall"] - 1, rec.frames[-1]["t_wall"] + 1
    pose = tmp_path / "pose.csv"
    pose.write_text("t_wall,north,east,down,roll,pitch,yaw\n"
                    f"{t0},10,20,-2,0,0,90\n{t1},10,20,-2,0,0,90\n")

    def ask(rgb):
        return '{"found": true, "box": [440, 420, 560, 580]}', (160, 120), 0.5
    a = rr.build_parser().parse_args(["replay", out, "--vlm-period", "0.05", "--pose", str(pose)])
    summ = rr.replay(rec, a, ask=ask, out_dir=str(tmp_path / "rp"))
    g = summ["goal_accepted"]
    assert g[0] == pytest.approx(10.0, abs=0.1) and g[1] == pytest.approx(20.0 + 2.5, abs=0.1)  # facing east
    assert summ["guard_blocked_share"] == 0.0
