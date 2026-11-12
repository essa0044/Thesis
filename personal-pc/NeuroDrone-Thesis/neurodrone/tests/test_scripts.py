"""Smoke tests: the scripts run end-to-end (fake VLM server, replayed events, simulated vehicle)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neurodrone.closedloop import open_loop_trial
from neurodrone.events import save_events_npz
from neurodrone.sim import Box, Camera
from test_vlm import _Fake

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def sh(script, *args, cwd, timeout=120):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, timeout=timeout)


def make_recording(path, wall_x, dur):
    rec = []
    open_loop_trial([Box(x=wall_x, y=0, w=3, h=3)], speed=1.5, dur=dur, record=rec)
    x = np.concatenate([e.x for e in rec]); y = np.concatenate([e.y for e in rec])
    t = np.concatenate([e.t_us for e in rec]); p = np.concatenate([e.p for e in rec])
    o = np.argsort(t, kind="stable")
    cam = Camera()
    save_events_npz(str(path), x[o], y[o], t[o], p[o], cam.width, cam.height)


def test_bench_vlm_reports_latency_parse_rate_and_accuracy(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(3):
        Image.fromarray(np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)).save(imgs / f"f{i}.png")
    (tmp_path / "labels.json").write_text(json.dumps({f"f{i}.png": {"found": True, "x": 0.5} for i in range(3)}))
    fake = _Fake()
    try:
        r = sh("bench_vlm.py", "--base-url", fake.url, "--model", "m", "--images", imgs, "--instruction", "the door",
               "-n", 4, "--warmup", 1, "--labels", tmp_path / "labels.json", "--out", tmp_path / "o.json", cwd=tmp_path)
    finally:
        fake.close()
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["queries"] == 4 and out["errors"] == 0 and out["parse_success"] == 1.0
    assert out["latency_s"]["n"] == 4 and out["found_accuracy"] == 1.0 and out["x_mae_norm"] < 1e-6
    assert out["prompt_tokens_mean"] == 123
    assert len(fake.reqs) == 5, "1 warm-up + 4 measured"


def test_replay_snn_finds_the_alarm_in_a_recording(tmp_path):
    make_recording(tmp_path / "rec.npz", wall_x=10, dur=7)
    r = sh("replay_snn.py", tmp_path / "rec.npz", "--out", tmp_path / "t.csv", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "alarm onsets at t = none" not in r.stdout
    rows = (tmp_path / "t.csv").read_text().splitlines()
    assert rows[0].startswith("t_s,") and len(rows) > 600


def test_run_px4_dry_run_end_to_end_with_replayed_events_static_camera_and_mock_vlm(tmp_path):
    make_recording(tmp_path / "rec.npz", wall_x=60, dur=4)  # far wall: no alarm during the run
    Image.fromarray(np.full((96, 128, 3), 120, np.uint8)).save(tmp_path / "view.png")
    r = sh("run_px4.py", "--dry-run", "--instruction", "the door", "--source", "replay", "--replay", tmp_path / "rec.npz",
           "--camera", f"static:{tmp_path / 'view.png'}", "--vlm-url", "mock", "--vlm-period", "0.2", "--hfov", "65",
           "--duration", "2.5", "--log", tmp_path / "run.jsonl", cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "run ended: duration elapsed" in r.stdout and "vlm_parse_success: 1.0" in r.stdout
    modes = json.loads(json.dumps([json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()
                                   if '"decision"' in l]))
    assert any(m["mode"] == "TRACK" for m in modes)


def test_run_px4_refuses_to_fly_for_real_without_the_range_reflex(tmp_path):
    r = sh("run_px4.py", "--instruction", "x", "--source", "replay", "--replay", tmp_path / "none.npz",
           "--camera", "static:none.png", "--vlm-url", "mock", cwd=tmp_path)
    assert r.returncode == 2 and "REFUSING" in r.stdout


def test_trace_scenario_prints_mode_transitions_and_result(tmp_path):
    r = sh("trace_scenario.py", "vlm_silent", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "NO_GOAL" in r.stdout and "result:" in r.stdout
