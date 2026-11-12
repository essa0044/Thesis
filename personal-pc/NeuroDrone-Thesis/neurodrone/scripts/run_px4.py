#!/usr/bin/env python3
"""Run the neuromorphic-reflex + VLM stack against a PX4 vehicle.

    # 1. Bench, props OFF, simulated vehicle, real sensors/VLM:        --dry-run
    # 2. PX4 SITL (no hardware risk):                                  --connect udpin://0.0.0.0:14540
    # 3. Real vehicle on a Jetson UART:                                --connect serial:///dev/ttyTHS1:921600

    python scripts/run_px4.py --dry-run --instruction "the red door" --source davis --camera 0
    python scripts/run_px4.py --connect udpin://0.0.0.0:14540 --instruction "the red door" \
        --depth realsense --source davis --vlm-url http://127.0.0.1:8080/v1 --vlm-model qwen2.5-vl-3b

Default engagement is "pilot": the script streams ZERO setpoints and does nothing until YOU switch
the autopilot to OFFBOARD from the RC. Flip the switch back to take over instantly.

Hardware-facing parts (Px4Vehicle, DavisLiveSource, OpenCVCamera, RealSenseClearance) are written
against documented APIs but have NOT been run on hardware: go through SITL and the bench first.
"""
import _path  # noqa: F401
import argparse
import asyncio
import os
import sys
import time

from neurodrone.camera import OpenCVCamera, StaticCamera
from neurodrone.config import load_config
from neurodrone.depth import ConstantClearance
from neurodrone.runtime import Pipeline
from neurodrone.telemetry import JsonlLogger, PowerSampler, summarize_log
from neurodrone.vehicle import Px4Vehicle, SimVehicle
from neurodrone.vlm import MockVLM, OpenAICompatVLM

BANNER = """
=========================== PRE-FLIGHT (read it, every time) ===========================
 * Tested in PX4 SITL first? Props OFF on the bench first?
 * RC transmitter in hand, a switch mapped to OFFBOARD and another to POSITION/HOLD
   (that second one is your instant escape). Verify COM_RC_OVERRIDE lets sticks win.
 * PX4 failsafes configured: offboard-loss action, geofence (GF_*), battery (BAT_*).
 * Open area, netted/tethered for first flights, nobody within reach of the props.
 * Local drone regulations and your lab's flight-safety sign-off.
 * The event-camera reflex CANNOT see blank walls, glass, thin wires or poles, and may miss
   small obstacles. The range sensor is the backstop: do not fly without it.
=========================================================================================
"""


def build_source(args):
    if args.source == "replay":
        from neurodrone.events import ReplayEventSource
        return ReplayEventSource(args.replay, realtime=True)
    from neurodrone.events import DavisLiveSource
    return DavisLiveSource()


def build_camera(spec, cfg):
    if spec.startswith("static:"):
        import numpy as np
        from PIL import Image
        return StaticCamera(np.asarray(Image.open(spec[7:]).convert("RGB")))
    return OpenCVCamera(spec, cfg.camera.width, cfg.camera.height)


def build_vlm(cfg):
    if cfg.vlm.base_url == "mock":
        return MockVLM(['{"found": true, "box": [0.45, 0.4, 0.55, 0.6]}'], latency_s=0.05)
    return OpenAICompatVLM(cfg.vlm.base_url, cfg.vlm.model, cfg.vlm.api_key, cfg.vlm.timeout_s, cfg.vlm.max_tokens)


def build_clearance(args):
    if args.depth == "realsense":
        from neurodrone.depth import RealSenseClearance
        return RealSenseClearance(invalid_policy=args.depth_invalid)
    if args.depth == "const":
        return ConstantClearance(args.depth_const)
    return None


async def main(args) -> int:
    cfg = load_config(args.config)
    if args.vlm_url:
        cfg.vlm.base_url = args.vlm_url
    if args.vlm_model:
        cfg.vlm.model = args.vlm_model
    if args.vlm_period is not None:
        cfg.vlm.period_s = args.vlm_period
    if args.hfov:
        cfg.camera.hfov_deg = args.hfov
    if args.depth != "none":
        cfg.coordinator.use_clearance = True
    camera_spec = args.camera or cfg.camera.device

    if not args.dry_run:
        print(BANNER)
        if not cfg.coordinator.use_clearance and not args.allow_no_range:
            print("REFUSING to fly without the range-sensor reflex (--depth realsense). In simulation a textureless\n"
                  "wall approached off-centre was invisible to the event-camera reflex and the drone flew into it.\n"
                  "Override only if you accept that risk: --allow-no-range")
            return 2

    os.makedirs("runs", exist_ok=True)
    log_path = args.log or os.path.join("runs", time.strftime("run_%Y%m%d_%H%M%S.jsonl"))
    log = JsonlLogger(log_path)
    source = build_source(args)
    camera = build_camera(camera_spec, cfg)
    vlm = build_vlm(cfg)
    clearance = build_clearance(args)
    vehicle = SimVehicle(verbose=args.verbose) if args.dry_run else Px4Vehicle(args.connect)
    pipeline = Pipeline(cfg, source, camera, vlm, vehicle, args.instruction, logger=log, clearance=clearance)
    power = PowerSampler(logger=log) if args.power else None
    reason = "not started"
    try:
        await vehicle.connect()
        st = await vehicle.get_state()
        print(f"connected. armed={st.armed} in_air={st.in_air} engaged={st.engaged} battery={st.battery_frac:.0%}")
        if power:
            power.start()
        if args.engage == "script" and not args.dry_run:
            if input("Type FLY to arm, take off and engage OFFBOARD (RC in hand): ").strip() != "FLY":
                print("aborted")
                return 1
            await vehicle.arm_and_takeoff(args.takeoff_alt)
            await vehicle.start_offboard()
        elif not args.dry_run:
            print("pilot mode: streaming zero setpoints. Take off yourself, then switch to OFFBOARD when ready.")
        print(f"running. instruction={args.instruction!r}  log={log_path}  (Ctrl-C hands over to HOLD)")
        reason = await pipeline.run(args.duration)
    finally:
        if power:
            power.stop()
        for closer in (camera, source, clearance):
            try:
                closer and closer.close()
            except Exception:
                pass
        await vehicle.close()
        log.close()
    print(f"run ended: {reason}")
    summary = summarize_log(log_path)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instruction", required=True, help="target description for the VLM, e.g. 'the red door'")
    ap.add_argument("--config", help="YAML overriding config/default.yaml")
    ap.add_argument("--connect", default="udpin://0.0.0.0:14540")
    ap.add_argument("--dry-run", action="store_true", help="simulated vehicle; real sensors/VLM; safe on the bench")
    ap.add_argument("--engage", choices=["pilot", "script"], default="pilot")
    ap.add_argument("--takeoff-alt", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=None, help="seconds; default: until arrival/handover")
    ap.add_argument("--source", choices=["davis", "replay"], default="davis")
    ap.add_argument("--replay", help=".npz recording for --source replay")
    ap.add_argument("--camera", help="index | video file | gstreamer pipeline | static:IMAGE_PATH")
    ap.add_argument("--hfov", type=float, help="RGB camera horizontal FOV in degrees")
    ap.add_argument("--vlm-url", help="OpenAI-compatible base URL, or 'mock'")
    ap.add_argument("--vlm-model")
    ap.add_argument("--vlm-period", type=float, help="seconds between VLM calls; 0 = back-to-back")
    ap.add_argument("--depth", choices=["none", "realsense", "const"], default="none")
    ap.add_argument("--depth-invalid", choices=["blocked", "free"], default="blocked",
                    help="what 'no depth return' means: blocked (safe indoors) or free (outdoors, riskier)")
    ap.add_argument("--depth-const", type=float, default=float("inf"), help="fixed range for --depth const (bench)")
    ap.add_argument("--allow-no-range", action="store_true", help="fly without the range reflex (NOT recommended)")
    ap.add_argument("--power", action="store_true", help="log Jetson power (jtop/tegrastats)")
    ap.add_argument("--log", help="JSONL log path (default runs/run_<time>.jsonl)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.source == "replay" and not a.replay:
        ap.error("--source replay needs --replay FILE.npz")
    try:
        sys.exit(asyncio.run(main(a)))
    except KeyboardInterrupt:
        print("\ninterrupted: vehicle handed to HOLD")
