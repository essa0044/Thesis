"""scripts/analyze_series.py on a synthetic mission log in the real nd_obstacles world."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import analyze_series as A  # noqa: E402

WORLDS = os.path.join(HERE, "..", "sim", "worlds")


def _write_log(path, final_e=10.0, lateral=-0.8, outcome="arrived", instruction="the red box"):
    rows = [dict(t=0.0, kind="start", instruction=instruction, world="nd_obstacles", settings={}),
            dict(t=10.0, kind="offboard", alt_m=2.0),
            dict(t=12.0, kind="vlm", box=[0.4, 0.4, 0.6, 0.6], latency_s=25.0, accepted=None),
            dict(t=18.0, kind="vlm", box=[0.4, 0.4, 0.6, 0.6], latency_s=6.0, accepted=[0.0, 10.0, -2.0]),
            dict(t=24.0, kind="vlm", box=None, latency_s=5.0, accepted=[0.0, 10.0, -2.0])]
    t, e = 18.0, 0.0
    while e < final_e:
        n = lateral if 4.0 < e < 8.0 else 0.0
        rows.append(dict(t=round(t, 2), kind="ctl", phase="avoid" if 4.0 < e < 8.0 else "fly", n=n, e=e, d=-2.0))
        t += 0.05
        e = round(e + 0.05, 3)
    rows.append(dict(t=round(t, 2), kind="ctl", phase="arrived", n=0.0, e=final_e, d=-2.0))
    rows.append(dict(t=round(t + 3, 2), kind="end", outcome=outcome, duration_s=round(t + 3 - 10, 1), n=0.0,
                     e=final_e, d=-2.0))
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")


def test_world_parsing():
    w = A.load_world(os.path.join(WORLDS, "nd_obstacles.sdf"))
    assert {"red_box", "blue_barrel", "green_box", "white_box", "yellow_pole"} <= set(w)
    assert w["red_box"]["n"] == 0 and w["red_box"]["e"] == 12 and w["red_box"]["he"] == 0.5
    assert abs(A.surface_dist(w["yellow_pole"], 0.2, 7.0) - 0.85) < 1e-9


def test_run_metrics(tmp_path):
    p = tmp_path / "run.jsonl"
    _write_log(p)
    r = A.analyze_run(str(p), WORLDS)
    assert r["outcome"] == "arrived" and r["right_target"] and r["success"]
    assert r["standoff_m"] == 1.5  # red box face at east 11.5
    assert r["pole_surface_m"] == 0.85 and r["prop_clear_m"] == round(0.85 - A.PROP_RADIUS_M, 2)
    assert r["t_goal_s"] == 8.0 and r["vlm_n"] == 3 and r["vlm_first_s"] == 25.0 and r["vlm_median_s"] == 5.5
    assert 3.5 < r["avoid_s"] < 4.5


def test_collision_and_timeout_fail(tmp_path):
    p = tmp_path / "hit.jsonl"
    _write_log(p, lateral=0.0)  # straight through the pole
    r = A.analyze_run(str(p), WORLDS)
    assert r["prop_clear_m"] < 0 and not r["success"]
    p2 = tmp_path / "late.jsonl"
    _write_log(p2, outcome="timeout")
    assert not A.analyze_run(str(p2), WORLDS)["success"]


def test_summary_table(tmp_path, capsys):
    for i in range(3):
        _write_log(tmp_path / f"r{i}.jsonl")
    assert A.main([str(tmp_path / "*.jsonl"), "--worlds", WORLDS, "--out", str(tmp_path / "sum")]) == 0
    out = capsys.readouterr().out
    assert "| nd_obstacles | the red box | sweep | 3 | 3/3 |" in out
    assert (tmp_path / "sum.csv").exists() and (tmp_path / "sum.json").exists()
