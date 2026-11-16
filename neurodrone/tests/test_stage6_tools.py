"""Stage 6 tools without a Pixhawk: the parameter profile logic and the VIO checks on a fake PX4 position."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)
import fakes  # noqa: E402
import px4_params as pp  # noqa: E402
import vio_check as vc  # noqa: E402


def test_px4_profile_compare_and_nsh():
    cur = {"MAV_1_CONFIG": ("int", 0), "SER_TEL2_BAUD": ("int", 921600), "EKF2_EV_DELAY": ("float", 50.0),
           "EKF2_HGT_REF": ("int", 0)}
    rows = {r[0]: r[3] for r in pp.compare("vio", cur)}
    assert rows["MAV_1_CONFIG"] == "differs" and rows["SER_TEL2_BAUD"] == "ok" and rows["EKF2_EV_DELAY"] == "ok"
    assert rows["EKF2_HGT_REF"] == "differs" and rows["EKF2_MAG_TYPE"] == "missing in this PX4"
    lines = pp.nsh_lines("vio")
    assert "param set EKF2_EV_CTRL 15" in lines and "param set EKF2_HGT_REF 3" in lines and lines[-1] == "reboot"
    assert all(v is None for v, _ in pp.PROFILES["safety"].values())  # read-only profile
    assert pp.main(["nsh"]) == 0


def test_drift_and_log_summary():
    start = [dict(n=1.0, e=2.0, d=0.0, yaw=179.0)] * 5 + [dict(n=1.0, e=2.0, d=0.0, yaw=-179.0)] * 5
    end = [dict(n=1.3, e=2.4, d=-0.1, yaw=-175.0)] * 10
    d = vc.drift(start, end)
    assert d["horizontal_m"] == pytest.approx(0.5) and d["vertical_m"] == pytest.approx(-0.1)
    assert d["yaw_deg"] == pytest.approx(5.0, abs=0.1)  # across the +-180 wrap
    recs = [dict(t=i * 0.05, n=0.1 * (i % 2), e=0.0, d=-1.0, local_ok=True) for i in range(200)]
    s = vc.log_summary(recs)
    assert s["horizontal_spread_rms_m"] == pytest.approx(0.05) and s["position_update_hz"] == pytest.approx(20, rel=0.05)
    assert s["local_position_ok_share"] == 1.0
    assert vc.parse_ros_hz("average rate: 29.871\n\tmin: 0.03s\naverage rate: 30.012\n") == [29.871, 30.012]


class FakePose:
    def __init__(self):
        self.k = 0

    async def sample(self):
        import time
        self.k += 1
        return dict(t=time.time(), n=0.01 * (self.k % 3), e=0.0, d=-1.0, yaw=10.0, roll=0, pitch=0, local_ok=True)


def test_vio_check_commands(tmp_path):
    answers = iter(["", "", "", ""])
    vc.main(["drift", "--laps", "2", "--average-s", "0.1", "--out", str(tmp_path / "d")], pose=FakePose(),
            ask=lambda _p: next(answers))
    s = json.load(open(tmp_path / "d" / "drift.json"))
    assert len(s["laps"]) == 2 and s["horizontal_m_median"] < 0.05
    with fakes.FakeOllama() as srv:
        vc.main(["log", "--duration", "0.5", "--vlm-load", "--ollama", srv.url, "--out", str(tmp_path / "l")], pose=FakePose())
    s = json.load(open(tmp_path / "l" / "summary.json"))
    assert s["n"] > 3 and s["vlm_load"] is True and s["vlm_answers"] >= 1
