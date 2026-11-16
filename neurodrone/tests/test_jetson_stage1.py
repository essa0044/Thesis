"""Stage 1 (VLM on the Jetson) without a Jetson: tegrastats parsing, the Ollama timing split, and every phase of
scripts/jetson_vlm_bench.py against a fake Ollama and a fake tegrastats."""
import base64
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)
import fakes  # noqa: E402
import jetson_vlm_bench as jb  # noqa: E402

from neurodrone import jetson as J  # noqa: E402
from neurodrone import ollama as O  # noqa: E402


def test_parse_tegrastats_orin_line():
    r = J.parse_tegrastats(fakes.tegra_line(ram=5123, gpu=97, f=1020, t=57.3))
    assert r["ram_used_mb"] == 5123 and r["ram_total_mb"] == 7620 and r["swap_used_mb"] == 12
    assert r["gpu_load_pct"] == 97 and r["gpu_freq_mhz"] == 1020
    assert r["cpu_load_pct"] == [35, 20, 10, 5, 0] and r["cpu_freq_mhz"][0] == 1510
    assert r["temp_c"]["tj"] == 57.3 and r["temp_c"]["cpu"] == 55.4
    assert J.total_power_w(r) == pytest.approx(14.52) and J.max_temp(r) == 57.3
    assert r["emc_load_pct"] == 31 and r["emc_freq_mhz"] == 3199


def test_parse_tegrastats_variants_and_garbage():
    r = J.parse_tegrastats("RAM 2000/7620MB GR3D_FREQ 0%@305 GPU@40C CPU@41.5C VDD_IN 5000/5000")
    assert r["gpu_freq_mhz"] == 305 and r["temp_c"]["gpu"] == 40.0 and J.total_power_w(r) == 5.0
    assert J.parse_tegrastats("hello world") is None


def test_summarize_detects_throttling():
    recs = [dict(J.parse_tegrastats(fakes.tegra_line(f=1020 - 3 * i, ram=5000 + i)), t=float(i)) for i in range(120)]
    s = J.summarize(recs, window_s=10)
    assert s["ram_used_mb_max"] == 5119 and s["ram_free_mb_min"] == 7620 - 5119
    assert s["gpu_freq_first_mhz"] > s["gpu_freq_last_mhz"] and J.throttled(s) is True
    steady = [dict(J.parse_tegrastats(fakes.tegra_line()), t=float(i)) for i in range(120)]
    assert J.throttled(J.summarize(steady, window_s=10)) is False
    assert J.throttled(J.summarize([])) is None


def test_tegra_logger_with_fake_command(tmp_path):
    tl = J.TegraLogger(path=str(tmp_path / "t.jsonl"), cmd=fakes.fake_tegrastats_cmd(tmp_path, n=20, interval=0.01)).start()
    import time
    for _ in range(200):
        if len(tl.records) >= 20:
            break
        time.sleep(0.02)
    tl.stop()
    assert tl.available and len(tl.records) == 20
    assert len(open(tmp_path / "t.jsonl").read().splitlines()) == 20
    missing = J.TegraLogger(cmd=["/nonexistent/tegrastats"]).start()
    assert missing.available is False


def test_ollama_chat_timing_split():
    with fakes.FakeOllama() as srv:
        r = O.chat(srv.url, "qwen3-vl:4b-instruct", b"\xff\xd8jpeg", "the red box", num_ctx=2048)
    assert r["text"].startswith('{"found": true')
    assert r["prefill_s"] == pytest.approx(0.7) and r["decode_s"] == pytest.approx(0.18)
    assert r["prefill_tokens"] == 1210 and r["decode_tok_per_s"] == pytest.approx(25 / 0.18)
    body = srv.calls[0]
    assert body["options"]["num_ctx"] == 2048 and body["think"] is False and body["messages"][-1]["images"]


@pytest.fixture
def fake_tegra(monkeypatch, tmp_path):
    cmd = fakes.fake_tegrastats_cmd(tmp_path, n=100000, interval=0.01)
    orig = J.TegraLogger.__init__

    def init(self, interval_ms=1000, path=None, cmd_=None):
        orig(self, interval_ms, path, cmd)
    monkeypatch.setattr(J.TegraLogger, "__init__", init)


def test_stage1_phases_and_report(tmp_path, fake_tegra):
    seen = []

    def answer(instr, img):
        seen.append(img)
        return '{"found": true, "box": [400, 400, 600, 600]}'
    with fakes.FakeOllama(answer) as srv:
        out = str(tmp_path / "s1")
        common = ["--out", out, "--ollama", srv.url]
        assert jb.main(["info"] + common) == 0
        assert jb.main(["cold"] + common) == 0
        assert srv.loaded_ctx == 2048  # loaded with the missions' context
        assert jb.main(["latency", "--calls", "6"] + common) == 0
        assert jb.main(["soak", "--minutes", "0.02"] + common) == 0
        rc = jb.main(["report"] + common)
    assert len(set(seen)) == len(seen)  # a new picture every call
    lat = json.load(open(os.path.join(out, "latency.json")))
    assert lat["calls"]["n"] == 6 and lat["calls"]["prefill_s_median"] == pytest.approx(0.7)
    assert lat["tegrastats"]["n"] > 0 and lat["throttled"] is False
    rep = open(os.path.join(out, "report.md")).read()
    assert "Pass checks" in rep and "median answer time" in rep
    summ = json.load(open(os.path.join(out, "summary.json")))
    assert {c["name"]: c["ok"] for c in summ["checks"]}["median answer time (s)"] is True
    assert rc == 0  # accuracy not measured: no failed check


def _fake_refdrone(root, n_target=60, n_empty=20):
    d = os.path.join(root, "refdrone")
    os.makedirs(d)
    images, anns, rows = [], [], ["index\timage"]
    rng = np.random.default_rng(0)
    for i in range(n_target + n_empty):
        img = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
        rows.append(f"{i}\t{base64.b64encode(fakes.jpeg_bytes(img)).decode()}")
        images.append(dict(id=i, width=800, height=600, caption=f"the car number {i}"))
        if i < n_target:
            anns.append(dict(image_id=i, bbox=[320, 240, 160, 120], empty=0))  # box 0.4-0.6 x 0.4-0.6
    json.dump(dict(images=images, annotations=anns), open(os.path.join(d, "RefDrone_test_mdetr.json"), "w"))
    open(os.path.join(d, "refdrone_test_base64.tsv"), "w").write("\n".join(rows) + "\n")


def test_stage1_accuracy_uses_bench_grounding(tmp_path, fake_tegra):
    _fake_refdrone(str(tmp_path / "data"))
    with fakes.FakeOllama(lambda instr, img: '{"found": true, "box": [400, 400, 600, 600]}') as srv:
        out = str(tmp_path / "s1")
        assert jb.main(["accuracy", "--out", out, "--ollama", srv.url, "--data", str(tmp_path / "data"),
                        "--tasks", "refdrone:test"]) == 0
        assert all(c["options"]["num_ctx"] == 2048 for c in srv.calls)  # native API, no reload
        jb.main(["report", "--out", out, "--ollama", srv.url])
    s = json.load(open(os.path.join(out, "accuracy", "summary.json")))[0]
    assert s["task"] == "refdrone:test" and s["max_side"] == 896
    assert s["acc_iou50"] == 1.0 and s["prefill_s_median"] == pytest.approx(0.7)
    checks = {c["name"]: c for c in json.load(open(os.path.join(out, "summary.json")))["checks"]}
    assert checks["accuracy refdrone:test vs workstation 0.683"]["ok"] is True
