"""Vehicle layer: the only place that talks to the autopilot.

The autopilot (PX4 on a Pixhawk) does ALL the stabilisation and runs its own failsafes.
This layer only streams velocity setpoints to it in OFFBOARD mode and reads telemetry back.

* SimVehicle - kinematic stand-in (dry runs, tests).
* Px4Vehicle - MAVSDK-Python.  [Written against the verified MAVSDK-Python 3.17 gRPC API;
  NOT yet run against PX4 SITL or a real Pixhawk - do SITL first.]
"""
from __future__ import annotations

import asyncio
import importlib
import math
import sys
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from .sim import PointMassDrone
from .types import VehicleState, VelocityCmd


class Vehicle(ABC):
    @abstractmethod
    async def connect(self) -> None: ...
    @abstractmethod
    async def get_state(self) -> VehicleState: ...
    @abstractmethod
    async def send_velocity(self, cmd: VelocityCmd) -> None: ...
    @abstractmethod
    async def start_offboard(self) -> None: ...
    @abstractmethod
    async def hold(self) -> None: ...
    @abstractmethod
    async def land(self) -> None: ...
    @abstractmethod
    async def arm_and_takeoff(self, alt_m: float) -> None: ...

    async def close(self) -> None:
        pass


# ----------------------------------------------------------------------------- simulated
class SimVehicle(Vehicle):
    """Point-mass vehicle integrated on the wall clock. `engaged` mimics OFFBOARD being active."""

    def __init__(self, drone: Optional[PointMassDrone] = None, engaged: bool = True, verbose: bool = False):
        self.drone = drone or PointMassDrone()
        self.engaged = engaged
        self.verbose = verbose
        self.mode = "OFFBOARD" if engaged else "POSCTL"
        self.commands: List[VelocityCmd] = []
        self._cmd = VelocityCmd()
        self._t: Optional[float] = None

    def _advance(self) -> None:
        now = time.monotonic()
        if self._t is not None:
            dt = min(now - self._t, 0.25)
            n = max(1, int(dt / 0.01))
            for _ in range(n):
                self.drone.step(self._cmd, dt / n)
        self._t = now

    async def connect(self) -> None:
        self._t = time.monotonic()

    async def get_state(self) -> VehicleState:
        self._advance()
        d = self.drone
        return VehicleState(armed=True, in_air=True, engaged=self.engaged, yaw_deg=d.yaw_deg,
                            rot_rate_deg_s=abs(math.degrees(d.yaw_rate)), battery_frac=1.0,
                            dist_home_m=math.hypot(d.x, d.y), alt_m=d.z)

    async def send_velocity(self, cmd: VelocityCmd) -> None:
        self._advance()
        self._cmd = cmd
        self.commands.append(cmd)
        if self.verbose:
            print(f"[sim] vx={cmd.vx:+.2f} vy={cmd.vy:+.2f} vz={cmd.vz:+.2f} yaw_rate={cmd.yaw_rate:+.1f}")

    async def start_offboard(self) -> None:
        self.engaged, self.mode = True, "OFFBOARD"

    async def hold(self) -> None:
        self._advance()
        self._cmd, self.engaged, self.mode = VelocityCmd(), False, "HOLD"
        if self.verbose:
            print("[sim] HOLD")

    async def land(self) -> None:
        self._advance()
        self._cmd, self.engaged, self.mode = VelocityCmd(), False, "LAND"
        if self.verbose:
            print("[sim] LAND")

    async def arm_and_takeoff(self, alt_m: float) -> None:
        self.drone.z = alt_m


# ----------------------------------------------------------------------------- PX4
def _mavsdk():
    """Import the gRPC-based MAVSDK-Python under either of its names. (mavsdk>=4 is a
    different, native API and is NOT supported here.)"""
    for pkg in ("mavsdk_grpc", "mavsdk"):
        try:
            mod = importlib.import_module(pkg)
            off = importlib.import_module(pkg + ".offboard")
            tel = importlib.import_module(pkg + ".telemetry")
            return mod.System, off.VelocityBodyYawspeed, off.OffboardError, tel.FlightMode
        except Exception:
            continue
    raise ImportError("Px4Vehicle needs the gRPC MAVSDK-Python wrapper:  pip install mavsdk-grpc   "
                      "(or: pip install 'mavsdk<4').  Plain mavsdk>=4 has a different API.")


class Px4Vehicle(Vehicle):
    """address examples:  SITL  udpin://0.0.0.0:14540   |   Jetson UART  serial:///dev/ttyTHS1:921600"""

    TELEMETRY_TIMEOUT_S = 1.0  # no telemetry for this long -> report engaged=False (fail closed)

    def __init__(self, address: str = "udpin://0.0.0.0:14540"):
        self.address = address
        self._sys = None
        self._st = VehicleState()
        self._tasks: List[asyncio.Task] = []
        self._last_msg = 0.0
        self._origin = None
        self._mods = None

    async def connect(self, timeout_s: float = 30.0) -> None:
        System, self._Vel, self._OffErr, self._FlightMode = _mavsdk()
        self._sys = System()
        await self._sys.connect(system_address=self.address)

        async def _wait():
            async for s in self._sys.core.connection_state():
                if s.is_connected:
                    return
        await asyncio.wait_for(_wait(), timeout_s)

        t = self._sys.telemetry
        for stream, handler in ((t.attitude_euler, self._on_att),
                                (t.attitude_angular_velocity_body, self._on_rate),
                                (t.armed, self._on_armed), (t.in_air, self._on_in_air),
                                (t.flight_mode, self._on_mode), (t.battery, self._on_batt),
                                (t.position_velocity_ned, self._on_pos)):
            self._tasks.append(asyncio.create_task(self._pump(stream, handler)))

    async def _pump(self, stream, handler) -> None:
        try:
            async for item in stream():
                self._last_msg = time.monotonic()
                handler(item)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # a dead stream makes telemetry go stale -> engaged=False
            print(f"[px4] telemetry stream died: {e!r}", file=sys.stderr)

    # telemetry handlers -------------------------------------------------------
    def _on_att(self, a): self._st.yaw_deg = float(a.yaw_deg)
    def _on_armed(self, v): self._st.armed = bool(v)
    def _on_in_air(self, v): self._st.in_air = bool(v)
    def _on_mode(self, m): self._st.engaged = (m == self._FlightMode.OFFBOARD)

    def _on_rate(self, w):
        self._st.rot_rate_deg_s = math.degrees(math.sqrt(w.roll_rad_s ** 2 + w.pitch_rad_s ** 2 + w.yaw_rad_s ** 2))

    def _on_batt(self, b):
        r = b.remaining_percent
        if r is None or (isinstance(r, float) and math.isnan(r)):
            return  # unknown: leave the previous value (default 1.0)
        self._st.battery_frac = r / 100.0 if r > 1.5 else float(r)  # API versions differ: 0..1 or 0..100

    def _on_pos(self, pv):
        n, e, d = pv.position.north_m, pv.position.east_m, pv.position.down_m
        if self._origin is None:
            self._origin = (n, e)  # first sample ~ takeoff point if you connect on the ground
        self._st.dist_home_m = math.hypot(n - self._origin[0], e - self._origin[1])
        self._st.alt_m = -float(d)

    # interface ----------------------------------------------------------------
    async def get_state(self) -> VehicleState:
        s = self._st
        fresh = (time.monotonic() - self._last_msg) < self.TELEMETRY_TIMEOUT_S
        return VehicleState(armed=s.armed, in_air=s.in_air, engaged=s.engaged and fresh,
                            yaw_deg=s.yaw_deg, rot_rate_deg_s=s.rot_rate_deg_s,
                            battery_frac=s.battery_frac, dist_home_m=s.dist_home_m, alt_m=s.alt_m)

    async def send_velocity(self, cmd: VelocityCmd) -> None:
        await self._sys.offboard.set_velocity_body(self._Vel(cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate))

    async def start_offboard(self) -> None:
        await self.send_velocity(VelocityCmd())  # PX4 refuses OFFBOARD without a prior setpoint
        try:
            await self._sys.offboard.start()
        except self._OffErr as e:
            raise RuntimeError(f"offboard start refused: {e}") from e

    async def hold(self) -> None:
        await self._sys.action.hold()

    async def land(self) -> None:
        await self._sys.action.land()

    async def arm_and_takeoff(self, alt_m: float, timeout_s: float = 30.0) -> None:
        a = self._sys.action
        await a.set_takeoff_altitude(float(alt_m))
        await a.arm()
        await a.takeoff()
        t0 = time.monotonic()
        while self._st.alt_m < 0.9 * alt_m:
            if time.monotonic() - t0 > timeout_s:
                raise RuntimeError("takeoff timed out")
            await asyncio.sleep(0.2)

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
