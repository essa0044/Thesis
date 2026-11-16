# System Design {#sec:design}

This chapter derives the requirements of the perception module (objective 1), describes the hardware and the software architecture, and explains each module and the data flow between them. The same code runs in simulation (Chapter 5) and on the drone (Chapter 6); only the source of the camera images and the connection to the flight controller change.

## Requirements

**Mission.** The drone takes off, receives a target in plain language ("the red box"), finds the target, flies to a point about 1.5 m in front of it at a fixed altitude and lands. It must not touch obstacles on the way, and it must stop rather than fly blind when it is unsure.

**Functional requirements.**

- F1: Open-vocabulary goal selection: the target is named in free text, not chosen from a fixed list of classes.
- F2: A 3-D goal from the camera: the image position of the target and its distance must give a point in the drone's local frame.
- F3: Robustness to wrong answers: a single wrong or invented detection must not send the drone anywhere.
- F4: Search: if the target is not in view, the drone must look for it.
- F5: Obstacle safety independent of the VLM: obstacles in the flight direction must stop or divert the drone within one control step, whatever the VLM is doing.
- F6: Fail safe: missing, frozen or late depth images must stop the drone; a lost link or crashed program must leave the drone to PX4's failsafes.

**Non-functional requirements.**

- N1: Everything runs on board an 8 GB embedded computer (no network needed in flight).
- N2: The VLM may be slow (one answer every few seconds); the control loop must not wait for it (20 Hz).
- N3: Standard, replaceable parts: PX4 flight controller, MAVLink, an off-the-shelf RGB-D camera.
- N4: Indoor operation: no GPS; the position comes from the camera (visual-inertial odometry, Chapter 6).
- N5: Reproducibility: the same code in simulation and on the drone; every run logged.

## Hardware

Table: Hardware of the quadrotor (ordered November 2026). {#tbl:hardware}

| Part | Model | Role |
|---|---|---|
| Airframe and flight controller | Holybro PX4 Development Kit X500 V2 with Pixhawk 6C, M10 GPS | 500 mm quadrotor; PX4 keeps the drone stable and executes velocity setpoints |
| Companion computer | NVIDIA Jetson Orin Nano Super Developer Kit, 8 GB (67 TOPS) | runs the VLM, the perception and safety code and the visual-inertial odometry |
| Storage | Kingston NV3 500 GB NVMe SSD (M.2 2280) | operating system, models, logs |
| Camera | Intel RealSense D435i | RGB image for the VLM, depth image for the 3-D goal and the guard, IMU for odometry |
| Camera mount | Holybro depth-camera mount for X500 V2 | forward-looking camera |
| Battery | Gens Ace 4S 5000 mAh LiPo (XT90, with XT90→XT60 adapter) | flight power; the Jetson is fed from the power distribution board (XT30 lead, 5.5 × 2.5 mm barrel plug) |
| Link | USB-A to USB-C cable | MAVLink between Jetson and Pixhawk 6C |

The Jetson Orin Nano was chosen because it is the smallest NVIDIA module with a GPU that runs the CUDA-based VLM runtimes and the NVIDIA visual-odometry library, at a mass and power a 500 mm quadrotor can carry; Section @sec:edge compares it with other edge computers. The RealSense D435i gives aligned RGB and depth and an IMU from one USB connection.

## Software architecture {#sec:arch}

The stack is split by speed (Figure @fig:arch). PX4 on the Pixhawk runs the attitude and velocity controllers at hundreds of hertz. The companion computer runs the mission code: a **control loop at 20 Hz** that reads the vehicle state and the newest depth image, steers towards the current goal, checks the path with the depth guard and sends a velocity setpoint; and a **VLM worker** in its own thread that takes a picture, asks the VLM, and turns the answer into a goal candidate. The VLM never commands the drone directly: it only proposes goals, and the control loop decides what to send.

![Software architecture and the two deployment pipelines. Pipeline A runs everything on the companion computer; in pipeline B the RGB stream goes to a ground station that runs the VLM and returns goal candidates, while the depth guard and the control loop stay on board. In simulation, Gazebo provides the camera images and the vehicle physics in place of the real hardware.](figures/fig_architecture.png){#fig:arch width=100%}

**Pipeline A — everything on board.** Camera → companion computer (VLM, 3-D goal, search, guard, control) → MAVLink → PX4. No network is needed; the VLM's speed is limited by the companion computer.

**Pipeline B — VLM on a ground station.** The RGB stream is sent to a ground station (for example as an RTSP video stream), which runs a larger or faster VLM and returns goal candidates over the network. The depth guard and the control loop stay on board, so a slow or broken link only delays the goal, not the safety layer. Pipeline B allows larger models but depends on a radio link and adds its latency.

The thesis implements pipeline A as the target design. The simulation set-up is a form of pipeline B: the VLM runs in Ollama on the Windows host and the mission code in the Linux environment of the same laptop, connected over a local network socket (Chapter 5).

## Perception: from a sentence to a 3-D goal {#sec:perception}

**2-D grounding.** The VLM receives the RGB image, scaled so that the longest side is 896 px, and the prompt:

> Mission target: *the red box*. Look at the image. If the target is visible, give its bounding box in normalised image coordinates (0,0 = top-left, 1,1 = bottom-right) as [x1, y1, x2, y2]. If it is not visible, say found=false. Reply with ONLY JSON, either {"found": true, "box": [x1, y1, x2, y2]} or {"found": false}.

Requests use temperature 0 and at most 128 output tokens; "thinking" modes are switched off (Chapter 4 shows they make grounding worse and many times slower). A tolerant parser reads the first JSON object and accepts the coordinate conventions the models actually use (Qwen3-VL answers in 0–1000 rather than 0–1).

**3-D goal.** The centre of the box gives a viewing ray of the RGB camera; the median depth inside the central half of the box, read from the depth image at the same ray angles, gives the distance. With the camera's mounting offset and the drone's position and attitude at the moment the picture was taken, this gives the object's position in the local NED frame. The goal is the point 1.5 m short of the object on the horizontal line from the drone, at the mission altitude.

**Edge rule.** A box that touches the left or right image edge is not turned into a goal: the object is cut off, the box centre is not the object's centre, and the depth inside the box may belong to something in front of it (Chapter 5 shows such a case). The search treats it as a hint to turn instead.

**Consensus.** A goal is accepted only when two of the last three answers give goals within 1.5 m of each other; "not found" answers count as votes. A single invented detection — which the best small model produces for about half of the absent targets (Chapter 4) — therefore cannot move the drone. VLN on the Fly [@tayar2026vln] uses a similar rule with three consecutive agreeing predictions. Once accepted, the goal is a fixed point in the room; later answers can replace it only by agreeing on a new point.

## Search {#sec:search}

While no goal is accepted, the drone searches by **turn, stop and look**: it turns to a heading, holds still until the VLM has answered for a picture taken after the turn ended, and moves on only if that answer finds nothing. Holding still matters because one answer takes seconds: a drone that keeps turning has looked elsewhere by the time the answer arrives, and two answers never agree. During the search the drone also holds its mission altitude actively.

- **Sweep (baseline).** A fixed list of headings around the start heading: 0°, +30°, −30°, +60°, −60° … 180°.
- **Active search.** A belief over 36 directions (10° each, uniform at the start) is updated by every answer. "Not found" multiplies the directions inside the camera's 69° view by 0.3 — not zero, because the VLM sometimes misses targets. "Found" moves 80 % of the belief to the direction of the box and turns the drone to centre the box, so the confirming answer comes from a different picture. The next view, when nothing is found, is the heading whose view holds the most belief, minus 0.15 per 180° of turning; the view just seen is skipped. This is a simple form of active perception [@bajcsy1988active; @aloimonos1988active]: the sensor is moved to where the next answer is expected to be most informative.

## Safety layer: depth guard and go-around {#sec:safety}

The safety layer uses only the depth image and runs in every control step.

**Depth guard.** Each depth pixel (every second row and column) becomes a 3-D point. The points are rotated by the drone's roll and pitch into a level frame, so that a drone tilting forward does not see the floor in its path. The guard takes the points inside a horizontal corridor of radius 0.8 m around the commanded flight direction (the X500's propeller tips reach about 0.42 m from its centre) and uses the 8th-closest of them as the obstacle distance, so that single noisy pixels do not count. The motion is blocked if the obstacle is closer than 1.2 m or would be reached in less than 2 s at the commanded speed.

**Go-around.** If the straight path is blocked, the guard tests corridors turned by 10°, 20° and 30° to either side — all inside the depth camera's view — and the drone flies the first one that is free for 2.5 m at 0.5 m/s, keeping its nose towards the goal. It keeps that direction for 2 s after the straight path is free again, then flies to the goal. If no direction is free, the drone stays stopped. The go-around has no map: it knows only what the camera sees in the current image.

**Stale and frozen depth.** A depth image older than 0.5 s counts as missing, and missing depth blocks all motion. The age is the larger of the time since the image arrived and, in simulation, the simulation time since it was taken (image time stamps against the world clock). A depth image that has not changed while the drone moved more than 0.5 m counts as frozen and is treated as missing too. Both checks were added after they were needed in simulation (Chapter 5); the RealSense's frame time stamps allow the same checks on the drone.

## Control and the PX4 interface

The mission code talks to PX4 [@meier2015px4] through MAVSDK [@mavsdk] (MAVLink). It arms the drone, takes off to 2 m, switches to OFFBOARD mode and then sends a body-frame velocity setpoint (forward, right, down, yaw rate) every 50 ms. The steering law turns the drone to face the goal (at most 30°/s), flies forward at a speed proportional to the remaining distance (at most 1 m/s) once the heading error is below 30°, holds the altitude with a proportional term, and declares arrival within 0.5 m of the goal; after 3 s of hovering at the goal the drone lands. PX4 keeps its own failsafes: if the setpoints stop, it leaves OFFBOARD mode, and the safety pilot can take over at any time. The telemetry streams the code uses (position, velocity, attitude) are requested at 25 Hz.

## Data flow

Figure @fig:seq shows one cycle of a mission as a sequence diagram: which module sends which data to which other module. The control loop never waits for the VLM; the VLM worker publishes goal candidates and the latest answer, which the search and the control loop read in their next step.

![Sequence of one mission: data relayed between the camera, the VLM worker, the VLM server, the control loop (search, steering, guard) and PX4.](figures/fig_sequence.png){#fig:seq width=100%}
