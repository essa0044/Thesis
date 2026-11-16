"""One VLM call to an Ollama server (native /api/chat), with Ollama's own timing split.

Ollama's non-streamed answer carries durations in nanoseconds:
    load_duration         loading the model (0 when it is already loaded)
    prompt_eval_duration  reading the prompt - for a VLM this includes encoding the image ("prefill")
    eval_duration         writing the answer, eval_count tokens ("decode")
    total_duration        all of it, inside the server
On an edge GPU the prefill (image tokens) dominates (LiteVLA-H: ~94 % on an AGX Orin), so the two are reported
separately. Ollama reuses its work on a picture it has just seen, so timing the same picture twice is not honest:
send a new picture every call.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from typing import Optional

from .vlm import SYSTEM_PROMPT, build_prompt

NS = 1e-9


def timing_from_response(data: dict) -> dict:
    """Seconds and token counts from Ollama's answer (missing fields -> None)."""
    def s(k):
        v = data.get(k)
        return None if v is None else v * NS
    pe, ec, ed = data.get("prompt_eval_count"), data.get("eval_count"), data.get("eval_duration")
    pd = data.get("prompt_eval_duration")
    return dict(load_s=s("load_duration"), prefill_s=s("prompt_eval_duration"), decode_s=s("eval_duration"),
                server_total_s=s("total_duration"), prefill_tokens=pe, decode_tokens=ec,
                prefill_tok_per_s=(pe / (pd * NS)) if pe and pd else None,
                decode_tok_per_s=(ec / (ed * NS)) if ec and ed else None)


def chat(url: str, model: str, jpeg: bytes, instruction: str, timeout: float = 240.0, num_ctx: int = 2048,
         num_gpu: Optional[int] = None, num_predict: int = 128, system: bool = False, keep_alive: str = "60m") -> dict:
    """One grounding call with the drone's prompt. Returns dict(text, latency_s, **timing_from_response).

    num_ctx must be the same in every call (and when the model is loaded): a different value makes Ollama reload the
    model. system=True adds the drone's system prompt (as the workstation benchmark did); the missions send none."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": build_prompt(instruction), "images": [base64.b64encode(jpeg).decode()]})
    body = {"model": model, "stream": False, "think": False, "keep_alive": keep_alive,
            "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": num_ctx}, "messages": msgs}
    if num_gpu is not None:  # model layers on the GPU (0 = CPU only); must match the load, or Ollama reloads
        body["options"]["num_gpu"] = int(num_gpu)
    full = url.rstrip("/") + "/api/chat"
    t0 = time.monotonic()
    try:
        data = _post(full, body, timeout)
    except urllib.error.HTTPError as e:  # models without a thinking switch reject "think"
        if e.code == 400 and "think" in e.read().decode(errors="replace").lower():
            body.pop("think")
            data = _post(full, body, timeout)
        else:
            raise
    out = dict(text=(data.get("message") or {}).get("content") or "", latency_s=time.monotonic() - t0)
    out.update(timing_from_response(data))
    return out


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def post(base: str, path: str, body: dict, timeout: float = 300.0) -> dict:
    return _post(base.rstrip("/") + path, body, timeout)


def get(base: str, path: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def unload(base: str, model: str) -> None:
    post(base, "/api/generate", {"model": model, "keep_alive": 0})


def load(base: str, model: str, num_ctx: int = 2048, num_gpu: Optional[int] = None, keep_alive: str = "60m") -> float:
    """Load the model with the missions' settings; returns the seconds it took."""
    opts = {"num_ctx": num_ctx}
    if num_gpu is not None:
        opts["num_gpu"] = num_gpu
    t0 = time.monotonic()
    post(base, "/api/generate", {"model": model, "keep_alive": keep_alive, "prompt": "", "options": opts})
    return time.monotonic() - t0


def loaded_info(base: str, model: str) -> dict:
    """What /api/ps says about the loaded model: size, size on the GPU, context, digest."""
    for m in get(base, "/api/ps").get("models", []):
        if m.get("name") == model or m.get("model") == model:
            size, vram = m.get("size") or 0, m.get("size_vram") or 0
            return dict(size_bytes=size, vram_bytes=vram, gpu_share=(vram / size) if size else None,
                        digest=m.get("digest"), context_length=m.get("context_length"))
    return {}


def server_info(base: str, model: str) -> dict:
    """Ollama version and the model's digest and quantisation (to prove it is the same file as on the workstation)."""
    out = {}
    try:
        out["ollama_version"] = get(base, "/api/version").get("version")
    except Exception as e:  # noqa: BLE001
        out["ollama_version_error"] = repr(e)
    try:
        for m in get(base, "/api/tags").get("models", []):
            if m.get("name") == model or m.get("model") == model:
                out.update(digest=m.get("digest"), size_bytes=m.get("size"),
                           quantization=(m.get("details") or {}).get("quantization_level"),
                           parameter_size=(m.get("details") or {}).get("parameter_size"))
    except Exception as e:  # noqa: BLE001
        out["tags_error"] = repr(e)
    return out
