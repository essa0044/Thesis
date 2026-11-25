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


def test_gps_summary_standing_still():
    import math
    lat0, lon0 = 48.766, 11.432  # Ingolstadt
    recs = []
    for i in range(100):
        dn = 2.0 if i == 50 else 0.3 * math.sin(i / 5)  # wandering, one 2 m jump
        recs.append(dict(t=i, lat=lat0 + dn / 111_320.0, lon=lon0, alt=400 + 0.01 * i, sats=7 + i % 3, fix="FIX_3D",
                         hdop=2.5, h_unc=4.0, n=0.1 * math.sin(i / 7), e=0.0, d=0.0, local_ok=True))
    recs += [dict(t=100 + i, lat=None, lon=None, sats=2, fix="NO_FIX", n=None) for i in range(25)]
    s = vc.gps_summary(recs)
    assert s["fix_share"] == pytest.approx(0.8) and s["fix_types"]["NO_FIX"] == pytest.approx(0.2)
    assert s["sats_min"] == 2 and s["hdop_median"] == 2.5 and s["receiver_horizontal_uncertainty_median_m"] == 4.0
    assert s["gps_wander_max_m"] == pytest.approx(2.0, abs=0.1) and s["gps_largest_jump_m"] > 1.5
    assert s["gps_altitude_range_m"] == pytest.approx(0.99, abs=0.01) and s["estimate_wander_max_m"] > 0


class FakeGpsPose(FakePose):
    async def sample(self):
        r = await super().sample()
        r.update(sats=6, fix="FIX_3D", lat=48.766 + 1e-6 * (self.k % 5), lon=11.432, alt=400.0, hdop=3.0, h_unc=5.0)
        return r


def test_vio_check_gps_command(tmp_path):
    vc.main(["gps", "--duration", "0.5", "--note", "test", "--out", str(tmp_path)], pose=FakeGpsPose())
    s = json.load(open(tmp_path / "gps_summary.json"))
    assert s["n"] >= 2 and s["sats_median"] == 6 and s["note"] == "test" and s["gps_wander_max_m"] < 0.5
