"""VLM layer (the "slow loop").

The VLM is used as a *visual grounding* module, not a pilot: given a target description it
returns where the target is in the image. The coordinator turns that into a bearing.
This is deliberately the most reliable thing small VLMs can do; asking them for headings
or velocities directly is much less reliable.

The model runs in a separate server process (llama.cpp `llama-server`, Ollama, vLLM, ... -
anything exposing an OpenAI-compatible /v1/chat/completions with image input). That keeps
the flight loop isolated from a VLM that is slow, hung or crashed, and lets you swap models
for the thesis comparison without touching code.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

SYSTEM_PROMPT = "You are the visual grounding module of a small drone. You reply with ONE JSON object and nothing else."


def build_prompt(instruction: str) -> str:
    return (
        f"Mission target: {instruction}\n"
        "Look at the image. If the target is visible, give its bounding box in normalised image "
        "coordinates (0,0 = top-left, 1,1 = bottom-right) as [x1, y1, x2, y2]. "
        "If it is not visible, say found=false.\n"
        'Reply with ONLY JSON, either {"found": true, "box": [x1, y1, x2, y2]} or {"found": false}.'
    )


# ----------------------------------------------------------------------------- parsing
@dataclass
class Parsed:
    found: bool
    x: Optional[float] = None  # box/point centre, 0..1
    y: Optional[float] = None
    w: Optional[float] = None  # box width, 0..1 (None for a bare point)
    h: Optional[float] = None
    unit_answer: bool = False  # the model's numbers were already 0..1 (whatever convention was expected)


def _first_wins(pairs):
    """JSON object hook: for a repeated key keep the FIRST value. Small VLMs sometimes list many
    boxes as repeated "box" keys; the first one is their top answer (json.loads would keep the last)."""
    d = {}
    for k, v in pairs:
        d.setdefault(k, v)
    return d


_NUM = r"(-?\d+(?:\.\d+)?)"
_BOX_RE = re.compile(r'"(?:box|bbox|bbox_2d|box_2d|bounding_box)"\s*:\s*\[\s*\[?\s*' + r"\s*,\s*".join([_NUM] * 4) + r"\s*\]")
_FOUND_RE = re.compile(r'"found"\s*:\s*"?(true|false)"?', re.I)


def _balanced(text: str, opener: str, closer: str):
    i = text.find(opener)
    while i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == opener:
                depth += 1
            elif text[j] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1], object_pairs_hook=_first_wins)
                    except json.JSONDecodeError:
                        break
        i = text.find(opener, i + 1)
    return None


def _first_json(text: str):
    """First balanced {...} that parses as JSON; else, if the answer was cut off (token limit hit
    while the model kept listing boxes), "found" and the first box salvaged from the text; else the
    first balanced [...] (e.g. a bare [x1, y1, x2, y2]); else None."""
    text = re.sub(r"<think>.*?(?:</think>|$)", " ", text, flags=re.S)  # a reasoning block, if the server left it in
    text = re.sub(r"```(?:json)?", " ", text)
    obj = _balanced(text, "{", "}")
    if obj is not None:
        return obj
    m = _BOX_RE.search(text)
    f = _FOUND_RE.search(text)
    if m or f:
        out = {}
        if f:
            out["found"] = f.group(1).lower() == "true"
        if m:
            out["box"] = [float(v) for v in m.groups()]
        return out
    return _balanced(text, "[", "]")


def _num_list(v, n) -> Optional[List[float]]:
    if isinstance(v, (list, tuple)) and len(v) == n:
        try:
            return [float(a) for a in v]
        except (TypeError, ValueError):
            return None
    return None


COORD_MODES = ("auto", "norm1000", "pixels", "unit")


def parse_goal(text: str, image_size: Tuple[int, int], coords: str = "auto") -> Optional[Parsed]:
    """Tolerant parser. Handles code fences, trailing chatter, a JSON list (Qwen-style
    [{"bbox_2d": ...}]), box/bbox/bbox_2d/point/x,y keys, and coordinates given as 0-1,
    pixels, or 0-1000. Returns None if nothing usable is found (caller keeps the old goal).

    `coords` fixes the model's coordinate convention. "auto" guesses from the numbers, which
    misreads a 0-1000 box whose values all happen to be below the image size (seen with
    Qwen2.5-VL and Qwen3-VL in Ollama, which both answer in 0-1000), so set it per model."""
    if coords not in COORD_MODES:
        raise ValueError(f"coords must be one of {COORD_MODES}, not {coords!r}")
    obj = _first_json(text)
    if obj == []:  # "[]" = nothing found (Qwen's own grounding format)
        return Parsed(found=False)
    if isinstance(obj, list):
        bare = _num_list(obj, 4)  # a bare [x1, y1, x2, y2] (Qwen3-VL's answer to a plain grounding prompt)
        obj = {"box": bare} if bare else next((o for o in obj if isinstance(o, dict)), None)
    if not isinstance(obj, dict):
        return None

    found = obj.get("found", None)
    if isinstance(found, str):
        found = found.strip().lower() in ("true", "yes", "1")
    box = None
    for k in ("box", "bbox", "bbox_2d", "box_2d", "bounding_box"):
        v = obj.get(k)
        box = _num_list(v, 4)
        if box is None and isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
            box = _num_list(v[0], 4)  # a list of boxes, e.g. Qwen3.5's "box": [[x1, y1, x2, y2]]: the first one
        if box:
            break
    pt = _num_list(obj.get("point"), 2) or _num_list(obj.get("center"), 2)
    if pt is None and isinstance(obj.get("x"), (int, float)) and isinstance(obj.get("y"), (int, float)):
        pt = [float(obj["x"]), float(obj["y"])]

    if found is False or (found is None and box is None and pt is None):
        return Parsed(found=False) if found is False else None
    if box is None and pt is None:
        return None

    W, H = image_size
    vals = list(box or pt)
    peak = max(abs(v) for v in vals)
    unit_answer = peak <= 1.5
    if unit_answer and coords in ("norm1000", "pixels"):
        # numbers this small cannot be 0-1000 or pixel coordinates (that box would be a dot in the corner):
        # the model followed the prompt's 0-1 wording this time (Qwen3.5 / Gemma 4 mix both)
        scale_x = scale_y = 1.0
    elif coords == "norm1000":
        scale_x = scale_y = 1000.0
    elif coords == "pixels":
        scale_x, scale_y = float(W), float(H)
    elif coords == "unit":
        scale_x = scale_y = 1.0
    elif peak > 1.5:  # auto: not normalised
        scale_x, scale_y = (1000.0, 1000.0) if peak > 1.05 * max(W, H) else (float(W), float(H))
    else:
        scale_x = scale_y = 1.0

    def nx(v): return min(1.0, max(0.0, v / scale_x))
    def ny(v): return min(1.0, max(0.0, v / scale_y))

    if box:
        x1, y1, x2, y2 = nx(box[0]), ny(box[1]), nx(box[2]), ny(box[3])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
            return None
        return Parsed(True, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, unit_answer)
    return Parsed(True, nx(pt[0]), ny(pt[1]), None, None, unit_answer)


# ----------------------------------------------------------------------------- images
def encode_jpeg(frame_rgb: np.ndarray, max_side: int = 448, quality: int = 85) -> Tuple[bytes, Tuple[int, int]]:
    """Resize (keeping aspect, sides rounded to multiples of 28 = Qwen2.5-VL patch grid) and
    JPEG-encode an HxWx3 uint8 RGB frame. Returns (jpeg_bytes, (width, height))."""
    h, w = frame_rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    nw = max(28, int(round(w * s / 28)) * 28)
    nh = max(28, int(round(h * s / 28)) * 28)
    try:
        import cv2  # type: ignore

        img = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes(), (nw, nh)
    except ImportError:
        pass
    try:
        from PIL import Image  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("encode_jpeg needs opencv-python or pillow") from e
    bio = io.BytesIO()
    Image.fromarray(frame_rgb).resize((nw, nh), Image.BILINEAR).save(bio, format="JPEG", quality=quality)
    return bio.getvalue(), (nw, nh)


# ----------------------------------------------------------------------------- backends
@dataclass
class VLMResult:
    text: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class VLMBackend(ABC):
    @abstractmethod
    def query(self, jpeg: bytes, instruction: str) -> VLMResult: ...


class OpenAICompatVLM(VLMBackend):
    """Talks to any OpenAI-compatible chat-completions server with image support."""

    def __init__(self, base_url: str, model: str, api_key: str = "none",
                 timeout_s: float = 20.0, max_tokens: int = 80):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key = model, api_key
        self.timeout_s, self.max_tokens = timeout_s, max_tokens

    def query(self, jpeg: bytes, instruction: str) -> VLMResult:
        b64 = base64.b64encode(jpeg).decode("ascii")
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                    {"type": "text", "text": build_prompt(instruction)},
                ]},
            ],
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            data = json.loads(r.read().decode("utf-8"))
        lat = time.monotonic() - t0
        usage = data.get("usage") or {}
        return VLMResult(text=data["choices"][0]["message"]["content"] or "", latency_s=lat,
                         prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))


class MockVLM(VLMBackend):
    """Scripted backend for tests / dry runs. `responses` is a list (cycled) or a callable
    (jpeg, instruction) -> text."""

    def __init__(self, responses: Union[Sequence[str], Callable[[bytes, str], str]], latency_s: float = 0.0):
        self.responses, self.latency_s, self._i = responses, latency_s, 0

    def query(self, jpeg: bytes, instruction: str) -> VLMResult:
        t0 = time.monotonic()
        if self.latency_s:
            time.sleep(self.latency_s)
        if callable(self.responses):
            text = self.responses(jpeg, instruction)
        else:
            text = self.responses[self._i % len(self.responses)]
            self._i += 1
        return VLMResult(text=text, latency_s=time.monotonic() - t0)
