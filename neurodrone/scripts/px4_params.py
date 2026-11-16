#!/usr/bin/env python3
"""PX4 parameters for indoor flight on visual-inertial odometry (stage 6, thesis Chapter 6), read and set over MAVLink.

    python scripts/px4_params.py show  --profile vio      # read and compare, change nothing
    python scripts/px4_params.py apply --profile vio --yes  # set what differs, read back, then REBOOT the Pixhawk
    python scripts/px4_params.py show  --profile safety     # values the safety pilot should know (read only)
    python scripts/px4_params.py nsh   --profile vio        # the same as `param set` lines for QGC's MAVLink console

The vio profile is the set Bernas used on the same hardware (Orin Nano + D435i + Isaac ROS VSLAM, PX4 v1.15.4; the
VIO pose reaches TELEM2 through MAVROS), plus EKF2_EV_CTRL, which the PX4 VIO guide requires and his list does not
show. Names and values are those of PX4 v1.15; a parameter the installed PX4 does not have is reported, not set.
The mission program itself stays on MAVSDK over USB (default address below), a separate link.

Connection: USB to the Pixhawk 6C = serial:///dev/ttyACM0:57600 (the baud rate is ignored on USB).
"""
import _path  # noqa: F401
import argparse
import asyncio
import sys

# name: (value, why). Integers are written as int, floats as float; the type PX4 reports wins when setting.
PROFILES = {
    "vio": {
        "MAV_1_CONFIG": (102, "MAVLink instance 1 on TELEM2 (the VIO pose from MAVROS on the Jetson)"),
        "SER_TEL2_BAUD": (921600, "TELEM2 at 921600 baud (USB-to-serial adapter on the Jetson)"),
        "UXRCE_DDS_CFG": (0, "uXRCE-DDS off: it would otherwise claim TELEM2"),
        "EKF2_EV_CTRL": (15, "fuse vision: horizontal position + vertical position + velocity + yaw (bits 0-3)"),
        "EKF2_HGT_REF": (3, "height reference = vision"),
        "EKF2_EV_DELAY": (50.0, "vision delay (ms); tune later by minimising the innovations"),
        "EKF2_GPS_CTRL": (0, "no GPS indoors"),
        "EKF2_BARO_CTRL": (0, "barometer off (indoors: doors, air flow, prop wash)"),
        "EKF2_RNG_CTRL": (0, "no range sensor"),
        "EKF2_MAG_TYPE": (5, "no magnetometer (steel and cables indoors); heading from vision"),
        "EKF2_REQ_NSATS": (5, "as in Bernas's set (irrelevant with GPS off)"),
        "MAV_USEHILGPS": (1, "as in Bernas's set"),
    },
    # read-only: what the safety pilot must know before the first flight (never set by this script)
    "safety": {
        "COM_OF_LOSS_T": (None, "s without OFFBOARD setpoints before the failsafe"),
        "COM_OBL_RC_ACT": (None, "what PX4 does when OFFBOARD setpoints stop (0 Position, 1 Altitude, 2 Manual, 3 Return, 4 Land, 5 Hold)"),
        "COM_RC_IN_MODE": (None, "RC required? (the safety pilot's transmitter)"),
        "NAV_RCL_ACT": (None, "action on RC loss"),
        "NAV_DLL_ACT": (None, "action on ground-station link loss"),
        "COM_ARM_WO_GPS": (None, "arming without GPS allowed?"),
        "MPC_XY_VEL_MAX": (None, "max horizontal speed (m/s)"),
        "COM_DISARM_LAND": (None, "auto-disarm after landing (s)"),
    },
}
REBOOT_NEEDED = {"MAV_1_CONFIG", "SER_TEL2_BAUD", "UXRCE_DDS_CFG"}


def nsh_lines(profile):
    out = [f"param set {k} {v}" for k, (v, _why) in PROFILES[profile].items() if v is not None]
    if profile == "vio":
        out.append("reboot")
    return out


def compare(profile, current):
    """current: {name: (type 'int'|'float', value)} -> rows (name, wanted, current, status, why)."""
    rows = []
    for name, (want, why) in PROFILES[profile].items():
        if name not in current:
            rows.append((name, want, None, "missing in this PX4", why))
            continue
        typ, val = current[name]
        if want is None:
            rows.append((name, None, val, "info", why))
        elif (typ == "float" and abs(float(val) - float(want)) < 1e-6) or (typ == "int" and int(val) == int(want)):
            rows.append((name, want, val, "ok", why))
        else:
            rows.append((name, want, val, "differs", why))
    return rows


def print_rows(rows):
    for name, want, val, status, why in rows:
        print(f"{name:16s} {str(val):>10s} -> {str(want):>10s}  {status:20s} {why}", flush=True)


async def read_all(param):
    allp = await param.get_all_params()
    cur = {p.name: ("int", p.value) for p in allp.int_params}
    cur.update({p.name: ("float", p.value) for p in allp.float_params})
    return cur


async def run(a):
    from neurodrone.vehicle import _mavsdk
    System = _mavsdk()[0]
    drone = System()
    await drone.connect(system_address=a.connect)
    async for s in drone.core.connection_state():
        if s.is_connected:
            break
    cur = await read_all(drone.param)
    rows = compare(a.profile, cur)
    print_rows(rows)
    if a.cmd != "apply":
        return 0
    if a.profile == "safety":
        print("the safety profile is read-only", flush=True)
        return 1
    todo = [r for r in rows if r[3] == "differs"]
    if not todo:
        print("nothing to change", flush=True)
        return 0
    if not a.yes:
        print(f"{len(todo)} parameters differ; run again with --yes to set them", flush=True)
        return 1
    for name, want, _val, _st, _why in todo:
        if cur[name][0] == "int":
            await drone.param.set_param_int(name, int(want))
        else:
            await drone.param.set_param_float(name, float(want))
    rows = compare(a.profile, await read_all(drone.param))
    print("\nafter setting:", flush=True)
    print_rows(rows)
    if any(r[0] in REBOOT_NEEDED for r in todo):
        print("\nREBOOT the Pixhawk now (serial-port parameters take effect only after a reboot).", flush=True)
    return 0 if all(r[3] in ("ok", "missing in this PX4") for r in rows) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "apply", "nsh"])
    ap.add_argument("--profile", choices=list(PROFILES), default="vio")
    ap.add_argument("--connect", default="serial:///dev/ttyACM0:57600")
    ap.add_argument("--yes", action="store_true", help="really set the parameters (apply)")
    a = ap.parse_args(argv)
    if a.cmd == "nsh":
        print("\n".join(nsh_lines(a.profile)), flush=True)
        return 0
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
