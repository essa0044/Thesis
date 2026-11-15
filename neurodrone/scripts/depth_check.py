#!/usr/bin/env python3
"""One-minute check that Gazebo's depth camera follows the drone (no VLM).

    # terminal 1:  cd ~/PX4-Autopilot && GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
    #              HEADLESS=1 PX4_GZ_WORLD=nd_obstacles make px4_sitl gz_x500_depth
    # terminal 2:  ~/nd-venv/bin/python scripts/depth_check.py

Takes off to 2 m, flies 3 m straight ahead (east, towards the yellow pole 6 m away) at 0.5 m/s, lands. Five times a
second it prints the drone's east position, the nearest depth in the middle of the picture (5th percentile), how many depth pixels
changed since the previous picture, the picture's simulation-time stamp and the simulation clock. Healthy: the depth
falls from about 5.7 m to about 2.7 m as the drone moves 3 m. Stale rendering: the depth stays near 5.7 m.
Writes depth_check.csv.
"""
import _path  # noqa: F401
import asyncio
import csv
import sys
import time

import numpy as np

from neurodrone.types import VelocityCmd


async def main():
    from neurodrone.gz_camera import GzCameras
    from neurodrone.vehicle import Px4Vehicle
    cams = GzCameras()
    cams.wait()
    print(f"rgb {cams.rgb_topic}, depth {cams.depth_topic}, clock {cams.clock_topic}, throttled {cams.throttled}", flush=True)
    veh = Px4Vehicle("udpin://0.0.0.0:14540")
    await veh.connect()
    await veh.arm_and_takeoff(2.0)
    await veh.start_offboard()
    rows, prev = [], None
    t0 = time.monotonic()
    print(" t(s)  east(m)  depth_centre(m)  changed_px  stamp(s)  sim(s)  lag(s)", flush=True)
    while time.monotonic() - t0 < 12.0:
        s = await veh.get_state()
        moving = 2.0 < time.monotonic() - t0 < 8.0  # hover 2 s, fly 6 s at 0.5 m/s = 3 m, hover
        await veh.send_velocity(VelocityCmd(0.5 if moving else 0.0, 0.0, 0.0, 0.0))
        d, age = cams.depth()
        h, w = d.shape
        c = d[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20]
        cc = np.where(np.isfinite(c) & (c > 0.2), c, np.nan)
        med = float(np.nanpercentile(cc, 5)) if np.isfinite(cc).any() else float("nan")  # nearest thing ahead
        changed = None if prev is None else int(np.sum(np.abs(np.nan_to_num(d) - np.nan_to_num(prev)) > 1e-4))
        prev = np.array(d, copy=True)
        with cams._lock:
            msg = cams._depth_msg
        stamp = cams._stamp(msg)
        row = dict(t=round(time.monotonic() - t0, 2), east=round(s.east_m, 2), north=round(s.north_m, 2),
                   yaw=round(s.yaw_deg), depth_centre=round(med, 2), changed_px=changed,
                   stamp=None if stamp is None else round(stamp, 2),
                   sim=None if cams._sim_now is None else round(cams._sim_now, 2), age=round(age, 3))
        rows.append(row)
        print(f"{row['t']:5.1f}  {row['east']:6.2f}  {row['depth_centre']:8.2f}  {str(changed):>10}  "
              f"{str(row['stamp']):>8}  {str(row['sim']):>7}  {row['age']:.3f}", flush=True)
        await asyncio.sleep(0.2)
    await veh.land()
    await veh.close()
    with open("depth_check.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    moved = rows[-1]["east"] - rows[0]["east"]
    dd = rows[0]["depth_centre"] - rows[-1]["depth_centre"]
    print(f"\nmoved {moved:.2f} m east; centre depth fell by {dd:.2f} m "
          f"({'OK' if abs(dd - moved) < 0.5 else 'NOT FOLLOWING THE DRONE'})", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(1)
