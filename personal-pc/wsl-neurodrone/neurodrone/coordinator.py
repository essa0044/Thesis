"""Coordinator: turns (SNN threat, VLM goal, range clearance, vehicle state) into a body-frame
velocity setpoint.

It is a pure function of its inputs plus a little state, with the clock injected, so it is
fully unit-testable and runs identically in the simulator and on the real drone.

Priority order (first match wins):
  1. Vehicle guards       battery / geofence / not engaged
  2. Reflex liveness      SNN (and the range sensor, if enabled) stale -> hold still, never fly
                          blind; after `blind_timeout_s` -> failsafe action
  3. Alarm handling       phases: ADJUDICATE -> AVOID(dodge) -> SETTLE   (see below)
  4. Arrival latch        the VLM confirmed the thing we stopped for IS the target
  5. Mission following    steer to the VLM's target; speed capped by SNN urgency and by what
                          can be stopped within the measured free distance

Alarm handling. An alarm (SNN looming OR range sensor too close) first BRAKES. The VLM then
looks again (an image captured *after* the alarm) and decides: the alarm is the target itself
(centred and large -> ARRIVED) or an obstacle (-> dodge). Reflex stops; semantics decide. If
the alarm persists while we are stopped, something is coming at us: dodge immediately. If no
fresh VLM goal arrives in `adjudicate_timeout_s`, assume obstacle and dodge.

Why the dodge is open loop: looming is optic *expansion*, which vanishes the moment we stop,
so "dodge until the alarm clears" would end after ~0.5 s and re-approach the obstacle. We commit
to a timed sidestep, repeat (longer, same direction) if the obstacle is still there, and give up
(hold) after `max_episodes`. The sidestep is blind to the sides (forward-facing sensors).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from .config import CoordinatorConfig
from .safety import max_safe_speed, stopping_distance
from .types import ClearanceReport, SemanticGoal, ThreatReport, VehicleState, VelocityCmd

# modes (strings so they log cleanly)
IDLE, BLIND_HOLD, NO_GOAL, TRACK, SEARCH = "IDLE", "BLIND_HOLD", "NO_GOAL", "TRACK", "SEARCH"
ADJUDICATE, AVOID, SETTLE, ARRIVED, FAILSAFE = "ADJUDICATE", "AVOID", "SETTLE", "ARRIVED", "FAILSAFE"


@dataclass
class Decision:
    mode: str
    cmd: VelocityCmd
    request: Optional[str] = None  # "land" | "hold": ask the vehicle layer to take over; ends the run
    note: str = ""


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _smoothstep(a: float, b: float, x: float) -> float:
    t = _clamp((x - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Coordinator:
    def __init__(self, cfg: Optional[CoordinatorConfig] = None, hfov_deg: float = 69.0):
        self.cfg = cfg or CoordinatorConfig()
        self.hfov = hfov_deg
        self.arrived = False
        self.reset()

    # -- state -----------------------------------------------------------------
    def reset(self) -> None:
        """Forget motion state (called when the autopilot is not in our control)."""
        self._last_cmd = VelocityCmd()
        self._last_t: Optional[float] = None
        self._phase: Optional[str] = None  # None | "adjudicate" | "dodge" | "settle"
        self._phase_end = 0.0
        self._adj_t0 = 0.0
        self._adj_urg = (0.0, 0.0, 0.0)
        self._dir = 1
        self._last_dir: Optional[int] = None
        self._episodes: Deque[float] = deque()
        self._blind_since: Optional[float] = None
        self._search_deg = 0.0
        self._goal_valid_after = float("-inf")  # goals captured before this are geometrically stale

    def reset_mission(self) -> None:
        """New instruction: clear arrival latch and avoidance history."""
        self.arrived = False
        self.reset()

    # -- helpers ---------------------------------------------------------------
    def _emit(self, mode: str, cmd: VelocityCmd, dt: float, note: str = "",
              request: Optional[str] = None) -> Decision:
        c = self.cfg
        vx = _clamp(cmd.vx, -c.max_speed, c.max_speed)
        if vx > self._last_cmd.vx:  # only speeding up is rate limited; braking is immediate
            vx = min(vx, self._last_cmd.vx + c.accel_limit * dt)
        out = VelocityCmd(vx=vx, vy=_clamp(cmd.vy, -c.max_speed, c.max_speed),
                          vz=_clamp(cmd.vz, -c.max_speed, c.max_speed),
                          yaw_rate=_clamp(cmd.yaw_rate, -c.max_yaw_rate, c.max_yaw_rate))
        self._last_cmd = out
        return Decision(mode, out, request, note)

    def _bearing(self, goal: Optional[SemanticGoal], now: float, veh: Optional[VehicleState]):
        """(usable, bearing_deg). Bearing is relative to the *current* heading, corrected for
        yaw that happened while the (slow) VLM was thinking."""
        if goal is None or now - goal.captured_at > self.cfg.goal_ttl_s:
            return False, 0.0
        if goal.captured_at < self._goal_valid_after:
            return False, 0.0  # taken before we sidestepped: its bearing no longer applies
        if goal.kind != "target" or goal.x is None:
            return True, 0.0
        b = (goal.x - 0.5) * self.hfov
        if veh is not None and goal.yaw_at_capture_deg is not None:
            b -= _wrap180(veh.yaw_deg - goal.yaw_at_capture_deg)
        return True, b

    def _is_target(self, goal: Optional[SemanticGoal], usable: bool, bearing: float) -> bool:
        c = self.cfg
        return bool(usable and goal is not None and goal.kind == "target" and abs(bearing) <= c.arrive_bearing_deg
                    and goal.w is not None and goal.w >= c.arrive_w_frac)

    def _start_dodge(self, now: float, dt: float, usable: bool, bearing: float, why: str) -> Decision:
        c = self.cfg
        zero = VelocityCmd()
        while self._episodes and now - self._episodes[0] > c.episode_window_s:
            self._episodes.popleft()
        self._episodes.append(now)
        n = len(self._episodes)
        if n > c.max_episodes:
            self._phase = None
            return Decision(FAILSAFE, zero, "hold", f"{n} avoid episodes in {c.episode_window_s:.0f}s: stuck")
        uL, _, uR = self._adj_urg
        if n > 1 and self._last_dir is not None:
            d = self._last_dir  # keep going the same way
        elif abs(uL - uR) > 0.1:
            d = 1 if uL > uR else -1  # away from the busier side
        elif usable and abs(bearing) > 2.0:
            d = 1 if bearing > 0 else -1  # towards the target's side
        else:
            d = 1
        self._dir = self._last_dir = d
        dur = c.dodge_distance_m / max(c.dodge_speed, 1e-3) * (1 + c.escalation * (n - 1))
        self._phase, self._phase_end = "dodge", now + dur
        return self._emit(AVOID, VelocityCmd(vx=0.0, vy=d * c.dodge_speed), dt,
                          f"{why}; episode {n}, {'right' if d > 0 else 'left'}, {dur:.1f}s")

    # -- main ------------------------------------------------------------------
    def step(self, now: float, threat: Optional[ThreatReport], goal: Optional[SemanticGoal],
             veh: Optional[VehicleState], clearance: Optional[ClearanceReport] = None) -> Decision:
        c = self.cfg
        dt = 0.04 if self._last_t is None else _clamp(now - self._last_t, 1e-3, 0.25)
        self._last_t = now
        zero = VelocityCmd()

        # 1. vehicle guards ----------------------------------------------------
        if veh is not None:
            if veh.armed and veh.in_air:
                if veh.battery_frac < c.min_battery:
                    return Decision(FAILSAFE, zero, "land", f"battery {veh.battery_frac:.0%}")
                if veh.dist_home_m > c.max_dist_home_m or veh.alt_m > c.max_alt_m:
                    return Decision(FAILSAFE, zero, "hold",
                                    f"geofence dist={veh.dist_home_m:.0f}m alt={veh.alt_m:.0f}m")
            if not veh.engaged:
                self.reset()
                return Decision(IDLE, zero, None, "autopilot not in offboard")

        # 2. reflex-layer liveness ----------------------------------------------
        snn_age = (now - threat.stamp) if threat is not None else float("inf")
        cl_age = (now - clearance.stamp) if clearance is not None else float("inf")
        dead = []
        if snn_age > c.snn_timeout_s:
            dead.append(f"snn stale ({snn_age:.2f}s)")
        if c.use_clearance and cl_age > c.clearance_timeout_s:
            dead.append(f"range sensor stale ({cl_age:.2f}s)")
        if dead:
            if self._blind_since is None:
                self._blind_since = now
            self._phase = None
            if now - self._blind_since > c.blind_timeout_s:
                return Decision(FAILSAFE, zero, c.failsafe_action, "reflex layer lost: " + ", ".join(dead))
            return self._emit(BLIND_HOLD, zero, dt, ", ".join(dead))
        self._blind_since = None

        if self.arrived:  # latched: nothing else matters once the VLM has confirmed the target
            self._phase = None
            return Decision(ARRIVED, zero, "hold", "target reached")

        rot = veh.rot_rate_deg_s if veh is not None else 0.0
        usable, bearing = self._bearing(goal, now, veh)
        rng = clearance.min_range_m if (c.use_clearance and clearance is not None) else float("inf")
        stop_d = stopping_distance(abs(self._last_cmd.vx), c.reaction_s, c.brake_decel)
        alarm = threat.alarm or (rng <= c.clearance_margin_m + stop_d)
        src = "snn" if threat.alarm else "range"

        # 3. alarm handling phases -----------------------------------------------
        if self._phase == "adjudicate":
            if goal is not None and goal.captured_at >= self._adj_t0 and usable:
                if self._is_target(goal, usable, bearing):
                    self.arrived = True
                    return Decision(ARRIVED, zero, "hold", "VLM confirms the target is what we stopped for")
                return self._start_dodge(now, dt, usable, bearing, "VLM: obstacle, not target")
            if alarm and now - self._adj_t0 >= c.adj_arm_s:
                return self._start_dodge(now, dt, usable, bearing, "alarm persists while stopped")
            if now - self._adj_t0 > c.adjudicate_timeout_s:
                return self._start_dodge(now, dt, usable, bearing, "no fresh VLM goal")
            return self._emit(ADJUDICATE, zero, dt, f"waiting for a fresh VLM goal ({now - self._adj_t0:.1f}s)")
        if self._phase == "dodge":
            if now < self._phase_end:
                return self._emit(AVOID, VelocityCmd(vx=0.0, vy=self._dir * c.dodge_speed), dt, "sidestep")
            self._phase, self._phase_end = "settle", now + c.settle_s
        if self._phase == "settle":
            if now < self._phase_end:
                return self._emit(SETTLE, zero, dt)
            self._phase = None
            self._goal_valid_after = now  # wait for an image taken AFTER the sidestep
            usable, bearing = self._bearing(goal, now, veh)

        # 4. new alarm ---------------------------------------------------------------
        if alarm:
            self._phase, self._adj_t0, self._adj_urg = "adjudicate", now, threat.urgency
            return self._emit(ADJUDICATE, zero, dt, f"{src} alarm: braking, asking the VLM")

        # 5. mission following -----------------------------------------------------
        if not usable:
            return self._emit(NO_GOAL, zero, dt, "no fresh VLM goal")
        if goal.kind == "search":
            rate = _clamp(c.search_yaw_rate, -c.max_yaw_rate, c.max_yaw_rate)  # what will actually be commanded
            self._search_deg += abs(rate) * dt
            if self._search_deg >= c.search_max_deg:
                return Decision(FAILSAFE, zero, "hold", "target not found after a full rotation")
            return self._emit(SEARCH, VelocityCmd(yaw_rate=rate), dt)
        self._search_deg = 0.0
        yaw_cmd = _clamp(c.yaw_gain * bearing, -c.max_yaw_rate, c.max_yaw_rate)
        fwd = c.cruise_speed if abs(yaw_cmd) < c.yaw_gate_deg_s else 0.0  # turn first, then go
        if c.urgency_slowdown:
            fwd *= 1.0 - _smoothstep(c.slow_start, c.slow_end, threat.frontal)
        note = f"bearing {bearing:+.1f} deg"
        if c.use_clearance:  # never go faster than we could stop within the free distance
            # 25% buffer: the uncapped limit sits exactly on the range-alarm boundary and would trip it
            cap = max_safe_speed(0.75 * max(0.0, rng - c.clearance_margin_m), c.reaction_s, c.brake_decel)
            if cap < fwd:
                fwd, note = cap, note + f", range-capped {cap:.2f} m/s (free {rng:.1f} m)"
        uL, _, uR = threat.urgency
        lateral = c.lateral_gain * (uL - uR) if (abs(yaw_cmd) < c.lateral_gate_yaw and rot < c.lateral_gate_yaw) else 0.0
        return self._emit(TRACK, VelocityCmd(vx=fwd, vy=lateral, yaw_rate=yaw_cmd), dt, note)
