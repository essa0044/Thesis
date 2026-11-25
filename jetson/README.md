# Jetson day one: stages 1, 3, 4 and 6

Tools to start the deployment plan (thesis Chapter 6, Table 6.2) the day the hardware arrives. Everything is tested
on fakes (camera, Ollama, PX4) in `neurodrone/tests/`; nothing here has run on the real Jetson yet, so check every
first result for plausibility.

## 0. Once

1. Flash JetPack 6.2 onto the NVMe SSD (SDK Manager), boot from it, connect to the network.
2. `git clone https://github.com/essa0044/Thesis.git && cd Thesis`
3. `bash jetson/setup_tools.sh --check`, then `bash jetson/setup_tools.sh --ollama --model --data`
   (venv `~/nd_venv`, Python packages, Ollama, `qwen3-vl:4b-instruct`, 2.2 GB of benchmark data, unit tests).
4. Power mode MAXN SUPER: `bash jetson/setup_vio.sh` sets it (step 2), or `sudo nvpmodel -m <id of MAXN_SUPER>`
   and `sudo jetson_clocks`. Check: `nvpmodel -q`.
5. Check the tegrastats parser on the real board once:
   `tegrastats --interval 1000 | head -1 | ~/nd_venv/bin/python -c "import sys; sys.path.insert(0,'neurodrone'); from neurodrone.jetson import parse_tegrastats as p; print(p(sys.stdin.readline()))"`
   (RAM, GPU clock, temperatures and VDD_IN must all appear).

Below, `PY=~/nd_venv/bin/python` and `ND=neurodrone/scripts`.

## Stage 1: the VLM on the Jetson (no camera, no drone)

```
$PY $ND/jetson_vlm_bench.py all --data ~/nd_data --out ~/nd_results/stage1
```

About 4-5 hours; nothing else may run on the board. Phases: `info` (JetPack, power mode, Ollama version, model
digest - must be 84f7ed8042fa as on the workstation), `cold` (load time, first answer), `latency` (50 answers, a new
picture each), `soak` (30 min non-stop with tegrastats), `accuracy` (RefDrone test, all 1,304 queries, and gRefCOCO val
seed 0, at 896 px through `bench_grounding.py --api ollama`), `report`. Each phase can be run alone.

Result: `~/nd_results/stage1/report.md` with the pass checks - median answer <= 10 s, accuracy within 2 points of the
workstation (RefDrone 0.683 at 896 px), >= 1 GB memory free, no throttling - and the split of every answer into
prefill (reading the prompt and the image) and decode (writing the answer). If it fails: LFM3-VL 2B or Qwen3-VL-2B q8
(`--model ...`), or smaller pictures (`--max-side 640`).

`vlm_warmup.py` (used by the simulation series) now prints the same prefill/decode split.

## Stage 3: the RealSense on a tripod (no drone)

```
$PY $ND/realsense_bench.py depth  --emitter both --out ~/nd_results/stage3/depth
$PY $ND/realsense_bench.py vlm    --instructions "the red box" "the blue barrel" --out ~/nd_results/stage3/vlm
$PY $ND/realsense_bench.py vlm-score ~/nd_results/stage3/vlm/review.csv
$PY $ND/realsense_bench.py guard  --duration 60 --out ~/nd_results/stage3/guard
$PY $ND/realsense_bench.py timing --duration 60 --out ~/nd_results/stage3/timing [--vlm-load]
```

- `depth`: camera square on to a wall or board; type the tape distance (0.5, 1, 1.5, 2, 3, 4, 5, 6 m) at each step.
  Measures error, noise, spread and fill with the dot projector on AND off: the visual odometry wants it off, so this
  decides what the guard loses (fill on plain walls!). Ends with the noise model sigma = a Z^2.
- `vlm`: the lab's objects; per position type the tape distance or `absent`; 5 answers per position; then open
  `index.html`, fill `correct` (y/n) in `review.csv` and run `vlm-score` (found rate, wrong-object rate, not-found
  when absent, distance from box + depth vs tape).
- `guard`: move a box towards the camera; it beeps at the first stop; compare with the tape (should be 1.2 m or
  the 2 s time to contact at `--speed`). Pointed at an empty floor or wall it must not block (false stops).
- `timing`: frame age, frames missed, with and without the VLM running; also prints roll/pitch from the IMU -
  **on a level tripod both must be near 0**. If not, the IMU axis convention in `neurodrone/realsense.py`
  (`attitude_from_accel`) is wrong for this camera and must be fixed before the guard is used with the IMU
  (meanwhile `--no-imu`).

## Stage 4: record and replay (camera + Jetson carried by hand)

```
$PY $ND/record_replay.py record --out ~/nd_rec/run01 --notes "red box ahead, pole at 3 m"
    # walk the planned path; at each measured point stand still and type e.g.  d=4.0  or  d=2.5 o=1.5  or  absent
    # q + Enter stops
$PY $ND/record_replay.py info   ~/nd_rec/run01
$PY $ND/record_replay.py replay ~/nd_rec/run01 --instruction "the red box"
```

The replay runs the mission's own functions on the recording: guard on every frame, VLM -> box -> edge rule -> depth ->
goal -> 2-of-3 consensus every `--vlm-period` s. `replay/marks.csv` compares, at each mark, the target distance and the
guard's obstacle distance with the tape. Recordings are plain files (JPEG, 16-bit PNG, CSV), so they also replay on the
laptop and are the real-world data set for the sim-to-real comparison. Without odometry the camera position is
unknown: compare at the marks (standing still). With stage 6 running, `--pose` takes the VIO track.

## Stage 5: Pixhawk on the bench, propellers off

`$PY $ND/px4_params.py show --profile safety` lists the failsafe settings to check with the safety pilot
(OFFBOARD loss, RC loss, link loss). The failsafe tests themselves are by hand (kill the program, unplug USB, stale
depth) as in the plan.

## Stage 6: indoor position by visual-inertial odometry

The set-up of Andrew Bernas (Hackster.io, June 2025; github.com/bandofpv/VSLAM-UAV) on the same Orin Nano + D435i:
Isaac ROS Visual SLAM 3.2 in Docker on the infrared images + IMU; the pose goes through MAVROS and a USB-to-serial
adapter into the Pixhawk's TELEM2 (921600 baud). The mission program keeps its own MAVSDK link over USB.

1. Host and workspace: `bash jetson/setup_vio.sh --check`, then `bash jetson/setup_vio.sh` and
   `bash jetson/setup_vio.sh --build-mavros`. It sets MAXN SUPER, the Docker NVIDIA runtime and GPU CDI spec, the
   RealSense udev rules, clones isaac_ros_common (release-3.2) and VSLAM-UAV (pinned commit 037198f), writes
   `CONFIG_IMAGE_KEY=ros2_humble.realsense`, builds Bernas's MAVROS image. By hand: D435i firmware 5.13.0.50
   (`rs-fw-update -l`).
2. PX4 (Pixhawk on USB): `$PY $ND/px4_params.py show --profile vio`, then `apply --yes`, then reboot. Bernas's values
   plus `EKF2_EV_CTRL = 15`; `nsh` prints them as `param set` lines for QGC's MAVLink console instead.
3. Start the VIO: `cd $ISAAC_ROS_WS/src/isaac_ros_common && ./scripts/run_dev.sh -d $ISAAC_ROS_WS`, inside
   `cd /workspaces/isaac_ros-dev/VSLAM-UAV/vslam && ./vslam_launch.sh` (Bernas's: infrared + IMU only).
4. Start MAVROS + the pose relay **only**: `bash jetson/vio/run_mavros_vio.sh`. Do not use Bernas's
   `mavrospy.launch.py` here: it also starts a node that flies OFFBOARD patterns.
5. Optional, better: IMU noise by Allan variance (>= 3 h of the D435i lying still, `allan_ros2`), values into the
   VSLAM launch file.
6. Before step 2 (still on PX4's default GPS parameters): **GPS indoors, measured.** Drone standing still on a
   table, GPS connected, 5 minutes, once near a window and once in the middle of the lab:
   `$PY $ND/vio_check.py gps --note "middle of the lab" --out ~/nd_results/stage6/gps_middle`.
   Nothing moves, so every metre the position wanders (and every jump) is GPS error: the evidence in Chapter 6 for
   why the drone flies on visual odometry. QGroundControl shows the satellites and HDOP live meanwhile.
7. Checks (propellers off first): `$PY $ND/vio_check.py drift --laps 3` (carry the drone around the room and back onto
   the floor mark); then with the safety pilot hovering in Position mode `$PY $ND/vio_check.py log --duration 120`;
   then the same with `--vlm-load --ros-hz-container isaac_ros_dev-aarch64-container` - the pass check is
   VIO >= 30 Hz and stable hold while the VLM answers non-stop.

**Camera sharing (for stage 6b/7).** Only one process can own the D435i. Once the VIO runs, the mission cannot open
the camera with pyrealsense2 any more; it must take colour and depth from the RealSense ROS 2 node. 
`jetson/vio/nd_vslam_realsense.py` is Bernas's launch file with colour and depth switched on for that (untested:
check that VSLAM stays at >= 30 Hz). The mission side - a camera class reading those ROS 2 topics with the interface
of `GzCameras` / `RealSenseCameras` - is the next piece of code to write.
