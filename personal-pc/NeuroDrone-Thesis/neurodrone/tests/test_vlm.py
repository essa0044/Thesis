"""VLM layer: tolerant parsing of sloppy model output, the HTTP client against a local fake
OpenAI-compatible server, JPEG encoding."""
import json
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from neurodrone.vlm import MockVLM, OpenAICompatVLM, build_prompt, encode_jpeg, parse_goal

SZ = (448, 336)


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_plain_box():
    p = parse_goal('{"found": true, "box": [0.4, 0.3, 0.6, 0.5]}', SZ)
    assert p.found and close(p.x, 0.5) and close(p.y, 0.4) and close(p.w, 0.2) and close(p.h, 0.2)


def test_code_fence_and_chatter_around_json():
    txt = 'Sure! Here you go:\n```json\n{"found": true, "box": [0.1, 0.1, 0.3, 0.3]}\n```\nHope that helps.'
    assert parse_goal(txt, SZ).found


def test_not_found():
    p = parse_goal('{"found": false}', SZ)
    assert p is not None and p.found is False
    assert parse_goal('{"found": "false"}', SZ).found is False
    assert parse_goal('{"found": "True", "box": [0.1, 0.1, 0.3, 0.3]}', SZ).found is True


def test_qwen_style_list_with_pixel_coordinates():
    txt = '```json\n[{"bbox_2d": [100, 50, 300, 250], "label": "door"}]\n```'
    p = parse_goal(txt, SZ)
    assert close(p.x, 200 / 448) and close(p.y, 150 / 336) and close(p.w, 200 / 448)


def test_thousand_scale_coordinates():
    p = parse_goal('{"box": [400, 300, 600, 500]}', SZ)  # 600 > image size -> 0..1000 scale
    assert close(p.x, 0.5) and close(p.y, 0.4)


def test_point_only_has_no_size():
    p = parse_goal('{"point": [0.25, 0.75]}', SZ)
    assert p.found and close(p.x, 0.25) and p.w is None
    p = parse_goal('{"found": true, "x": 0.3, "y": 0.6}', SZ)
    assert close(p.x, 0.3) and close(p.y, 0.6)


def test_reversed_box_is_normalised_and_overflow_clamped():
    p = parse_goal('{"box": [0.6, 0.5, 0.4, 0.3]}', SZ)
    assert close(p.x, 0.5) and p.w > 0
    p = parse_goal('{"box": [0.9, 0.1, 1.2, 0.3]}', SZ)
    assert p.x <= 1.0 and close(p.w, 0.1)


def test_garbage_and_degenerate_outputs_return_none():
    assert parse_goal("I cannot see anything in this image.", SZ) is None
    assert parse_goal("", SZ) is None
    assert parse_goal('{"found": true}', SZ) is None  # claims found but gives no location
    assert parse_goal('{"box": [0.5, 0.5, 0.5, 0.7]}', SZ) is None  # zero width
    assert parse_goal('{"box": ["a", "b", "c", "d"]}', SZ) is None
    assert parse_goal("{broken json", SZ) is None


def test_skips_invalid_braces_before_the_real_json():
    assert parse_goal('Note {not json} then {"found": false}', SZ).found is False


def test_prompt_contains_instruction_and_format_contract():
    p = build_prompt("the red door")
    assert "the red door" in p and "found" in p and "box" in p


def test_mock_backend_cycles_and_supports_callables():
    m = MockVLM(["a", "b"])
    assert [m.query(b"", "x").text for _ in range(3)] == ["a", "b", "a"]
    assert MockVLM(lambda jpeg, ins: ins.upper()).query(b"", "go").text == "GO"


def test_encode_jpeg_dims_are_patch_multiples():
    pytest.importorskip("PIL")
    jpeg, (w, h) = encode_jpeg(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), max_side=448)
    assert jpeg[:2] == b"\xff\xd8" and (w, h) == (448, 336) and w % 28 == 0 and h % 28 == 0
    jpeg, (w, h) = encode_jpeg(np.zeros((10, 10, 3), np.uint8), max_side=448)  # never upscaled below 1 patch
    assert w >= 28 and h >= 28


# ----------------------------------------------------------------------------- HTTP client
class _Fake:
    def __init__(self, reply=None, delay=0.0, status=200):
        self.reqs, outer = [], self
        reply = reply or '{"found": true, "box": [0.4, 0.4, 0.6, 0.6]}'

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.reqs.append((self.path, dict(self.headers), body))
                time.sleep(delay)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                payload = {"choices": [{"message": {"content": reply}}],
                           "usage": {"prompt_tokens": 123, "completion_tokens": 17}}
                self.wfile.write(json.dumps(payload).encode())

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def test_client_sends_an_openai_style_vision_request_and_parses_the_reply():
    fake = _Fake()
    try:
        c = OpenAICompatVLM(fake.url, "qwen-test", api_key="k", timeout_s=5)
        res = c.query(b"\xff\xd8fakejpeg", "the red door")
        path, headers, body = fake.reqs[0]
        assert path == "/v1/chat/completions" and headers["Authorization"] == "Bearer k"
        assert body["model"] == "qwen-test" and body["temperature"] == 0
        parts = body["messages"][1]["content"]
        assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert "the red door" in parts[1]["text"]
        assert res.prompt_tokens == 123 and res.completion_tokens == 17 and res.latency_s >= 0
        assert parse_goal(res.text, SZ).found
    finally:
        fake.close()


def test_client_raises_on_timeout_and_on_http_error():
    slow = _Fake(delay=0.6)
    try:
        with pytest.raises(Exception):
            OpenAICompatVLM(slow.url, "m", timeout_s=0.1).query(b"x", "t")
    finally:
        slow.close()
    bad = _Fake(status=500)
    try:
        with pytest.raises(urllib.error.HTTPError):
            OpenAICompatVLM(bad.url, "m", timeout_s=2).query(b"x", "t")
    finally:
        bad.close()


def test_coords_mode_fixes_low_range_norm1000_box():
    # real Qwen3-VL answer (Ollama, run 2): a 0-1000 box whose values are all below the image size
    text = '{"found": true, "box": [0, 0, 364, 414]}'
    size = (448, 336)
    auto = parse_goal(text, size)
    fixed = parse_goal(text, size, "norm1000")
    assert abs(fixed.x - 0.182) < 1e-6 and abs(fixed.w - 0.364) < 1e-6
    assert abs(auto.x - 0.182) > 0.2  # "auto" reads it as pixels: the reason the mode exists


def test_coords_mode_rejects_unknown_value():
    import pytest
    with pytest.raises(ValueError):
        parse_goal('{"found": false}', (448, 448), "percent")
