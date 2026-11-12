"""Runtime wiring.

Brain     synchronous, clock-injected core: SNN -> mailbox <- VLM goals -> Coordinator.
          Used verbatim by the closed-loop simulator AND the real pipeline, so the logic that
          was tested is the logic that flies.
Pipeline  real-time I/O around the Brain:
              thread 1  event source -> SNN                (~100 Hz, independent of everything)
              thread 2  camera -> VLM server -> goal       (every `vlm.period_s`, may take seconds)
              asyncio   control loop @ control_hz: state -> Brain -> setpoint -> autopilot
          The three never block each other. If the SNN thread dies or stalls, the Coordinator
          sees a stale threat report and stops the vehicle: it never flies without its reflexes.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from .camera import Camera
from .config import Config
from .depth import ClearanceSource
from .coordinator import Coordinator, Decision
from .events import EventSource
from .reflex import Detector, make_snn
from .telemetry import NullLogger
from .types import ClearanceReport, EventPacket, SemanticGoal, ThreatReport, VehicleState, VelocityCmd
from .vehicle import Vehicle
from .vlm import VLMBackend, encode_jpeg, parse_goal


class Brain:
    def __init__(self, snn: Detector, coord: Coordinator, lost_patience: int = 2,
                 logger=None, log_threat_every: int = 5):
        self.snn, self.coord = snn, coord
        self.lost_patience = lost_patience
        self.log = logger or NullLogger()
        self.log_threat_every = max(1, log_threat_every)
        self._lock = threading.Lock()
        self._threat: Optional[ThreatReport] = None
        self._goal: Optional[SemanticGoal] = None
        self._clearance: Optional[ClearanceReport] = None
        self._misses = 0
        self._n_pkts = 0

    # -- fast loop ------------------------------------------------------------
    def on_events(self, pkt: EventPacket, now: float) -> ThreatReport:
        rep = self.snn.process(pkt)
        rep.stamp = now
        with self._lock:
            self._threat = rep
        self._n_pkts += 1
        if rep.alarm or self._n_pkts % self.log_threat_every == 0:
            self.log.log("threat", frontal=rep.frontal, uL=rep.urgency[0], uC=rep.urgency[1],
                         uR=rep.urgency[2], alarm=rep.alarm, n_events=rep.n_events,
                         proc_ms=rep.proc_s * 1e3)
        return rep

    # -- slow loop ------------------------------------------------------------
    def on_goal(self, goal: SemanticGoal) -> None:
        """A single 'not found' while tracking is ignored (VLMs flicker); `lost_patience`
        consecutive misses switch to search."""
        if goal.kind == "search":
            self._misses += 1
            if self._misses < self.lost_patience and self._goal is not None and self._goal.kind == "target":
                self.log.log("goal", goal_kind="search", ignored=True)
                return
        else:
            self._misses = 0
        with self._lock:
            self._goal = goal
        self.log.log("goal", goal_kind=goal.kind, x=goal.x, y=goal.y, w=goal.w,
                     latency_s=goal.latency_s, raw=goal.raw[:200])

    # -- range sensor (second reflex) -----------------------------------------
    def on_clearance(self, range_m: float, now: float) -> None:
        with self._lock:
            self._clearance = ClearanceReport(now, float(range_m))

    # -- control loop ---------------------------------------------------------
    def control_step(self, now: float, veh: Optional[VehicleState]) -> Decision:
        with self._lock:
            threat, goal, clearance = self._threat, self._goal, self._clearance
        dec = self.coord.step(now, threat, goal, veh, clearance)
        c = dec.cmd
        self.log.log("decision", mode=dec.mode, vx=c.vx, vy=c.vy, vz=c.vz, yaw_rate=c.yaw_rate,
                     request=dec.request, note=dec.note,
                     range_m=(clearance.min_range_m if clearance else None))
        return dec

    def reset_mission(self) -> None:
        self.coord.reset_mission()
        with self._lock:
            self._goal = None
        self._misses = 0


class Pipeline:
    def __init__(self, cfg: Config, source: EventSource, camera: Camera, vlm: VLMBackend,
                 vehicle: Vehicle, instruction: str, logger=None,
                 clearance: Optional[ClearanceSource] = None):
        if cfg.coordinator.use_clearance and clearance is None:
            raise ValueError("coordinator.use_clearance is on but no ClearanceSource was given "
                             "(the range reflex is required once enabled: refusing to fly without it)")
        self.cfg, self.source, self.camera, self.vlm, self.vehicle = cfg, source, camera, vlm, vehicle
        self.clearance = clearance
        self.instruction = instruction
        self.log = logger or NullLogger()
        self.snn = make_snn(source.width, source.height, cfg.snn)
        self.coord = Coordinator(cfg.coordinator, cfg.camera.hfov_deg)
        self.brain = Brain(self.snn, self.coord, cfg.vlm.lost_patience, self.log)
        self._stop = threading.Event()
        self._yaw = 0.0

    def set_instruction(self, text: str) -> None:
        self.instruction = text
        self.brain.reset_mission()
        self.log.log("event", msg=f"instruction: {text}")

    def stop(self) -> None:
        self._stop.set()

    # -- threads --------------------------------------------------------------
    def _event_loop(self) -> None:
        try:
            while not self._stop.is_set():
                pkt = self.source.read(0.05)
                if pkt is not None:
                    self.brain.on_events(pkt, time.monotonic())
        except Exception as e:
            # Not re-raised on purpose: the control loop will see the stale report and stop.
            self.log.log("event", msg=f"event thread died: {e!r}")

    def _clearance_loop(self) -> None:
        try:
            while not self._stop.is_set():
                r = self.clearance.read(0.2)
                if r is not None:
                    self.brain.on_clearance(r, time.monotonic())
        except Exception as e:  # stale range -> coordinator holds, then failsafe
            self.log.log("event", msg=f"clearance thread died: {e!r}")

    def _vlm_loop(self) -> None:
        c = self.cfg.vlm
        next_t = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_t:
                self._stop.wait(min(0.05, next_t - now))
                continue
            next_t = now + c.period_s  # start-to-start; a slow call makes the next one immediate
            try:
                frame = self.camera.read()
                cap_t, yaw = time.monotonic(), self._yaw
                jpeg, size = encode_jpeg(frame, c.image_max_side, c.jpeg_quality)
                res = self.vlm.query(jpeg, self.instruction)
                parsed = parse_goal(res.text, size, c.coords)
            except Exception as e:
                self.log.log("vlm_call", ok=False, latency_s=None, error=repr(e))
                continue
            self.log.log("vlm_call", ok=parsed is not None, latency_s=res.latency_s,
                         prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens)
            if parsed is None:
                continue  # keep the previous goal; it expires on its own if this persists
            self.brain.on_goal(SemanticGoal(
                kind="target" if parsed.found else "search", x=parsed.x, y=parsed.y, w=parsed.w,
                captured_at=cap_t, yaw_at_capture_deg=yaw, latency_s=res.latency_s, raw=res.text))

    # -- main -----------------------------------------------------------------
    async def _handover(self, request: str) -> None:
        self.log.log("event", msg=f"handover: {request}")
        if request == "land":
            await self.vehicle.land()
        else:
            await self.vehicle.hold()

    async def run(self, duration_s: Optional[float] = None) -> str:
        """Run until a handover request, `duration_s`, or stop()/cancel. Returns the reason."""
        threads = [threading.Thread(target=self._event_loop, name="events", daemon=True),
                   threading.Thread(target=self._vlm_loop, name="vlm", daemon=True)]
        if self.clearance is not None:
            threads.append(threading.Thread(target=self._clearance_loop, name="range", daemon=True))
        for t in threads:
            t.start()
        period = 1.0 / self.cfg.runtime.control_hz
        t_start = next_tick = time.monotonic()
        reason = "stopped"
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if duration_s is not None and now - t_start >= duration_s:
                    reason = "duration elapsed"
                    break
                st = await self.vehicle.get_state()
                self._yaw = st.yaw_deg
                dec = self.brain.control_step(now, st)
                if dec.request:
                    reason = f"{dec.request} ({dec.note})"
                    await self._handover(dec.request)
                    break
                await self.vehicle.send_velocity(dec.cmd)
                next_tick += period
                if next_tick < time.monotonic() - 1.0:  # fell far behind: do not burst
                    next_tick = time.monotonic()
                await asyncio.sleep(max(0.0, next_tick - time.monotonic()))
        except (KeyboardInterrupt, asyncio.CancelledError):
            reason = "interrupted"
            await self._handover("hold")
        finally:
            self._stop.set()
            for t in threads:
                t.join(timeout=2.0)
            try:
                await self.vehicle.send_velocity(VelocityCmd())
            except Exception:
                pass
        self.log.log("event", msg=f"run ended: {reason}")
        return reason
