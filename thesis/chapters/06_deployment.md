# Deployment on the Quadrotor {#sec:deploy}

This chapter answers RQ3: how the software is deployed on a standard quadrotor platform. It compares the two pipeline architectures, describes how the parts are connected and what runs where, how the drone knows its position indoors, and the staged plan by which the system goes from the simulation to flights in the lab. The hardware was ordered in November 2026; the results of the stages will complete this chapter.

## Two pipelines {#sec:pipelines}

Section @sec:arch introduced the two architectures. Table @tbl:pipelines compares them.

Table: Pipeline A (VLM on board) and pipeline B (VLM on a ground station). {#tbl:pipelines}

| | Pipeline A: on board | Pipeline B: ground station |
|---|---|---|
| Where the VLM runs | Jetson Orin Nano (8 GB, shared with everything else) | desktop GPU on the ground |
| Data sent over the radio | none needed (telemetry only) | the RGB stream (e.g. RTSP, H.264) down, goal candidates up |
| Model size | ≤ 4 B parameters at 4 bit (Chapter 4) | any; e.g. a 30 B model |
| Expected VLM delay | about 1–10 s per answer (to be measured, stage 1) | about 0.6 s on the GPU plus network and video encoding |
| Behaviour if the link fails | unaffected | no new goals; the guard and the control loop stay on board, so the drone stops or holds |
| Depth guard | on board | on board (never over the network) |
| Suits | independent operation, no infrastructure | large models, development, debugging |

In both pipelines the safety-relevant part — the depth guard and the control loop — stays on board, so the network can only delay the goal, never the reaction to an obstacle. Chapter 4 showed that for drone views a larger model gives no better boxes (69 % for the 4B, 69 % for the 30B), so pipeline B's main advantage, model size, buys little for this task; it would help mainly with the correct "not found" answers. The thesis therefore deploys **pipeline A**. Pipeline B is supported by the code (the VLM is reached over HTTP, so the same program can use a model on another machine) and is used in stage 2 of the plan below.

## Hardware integration

Figure @fig:wiring shows how the parts are connected.

- **Flight controller.** The Pixhawk 6C runs PX4 and keeps the drone stable. It is connected to the Jetson by USB (MAVLink); the mission code uses MAVSDK as in simulation.
- **Companion computer.** The Jetson Orin Nano Super developer kit with a 500 GB NVMe SSD sits on the X500's payload platform. It is fed from the battery through an XT30 output of the power board and a 5.5 × 2.5 mm barrel plug. The voltage at the carrier board's DC input is checked with a bench supply before the first connection (stage 5).
- **Camera.** The RealSense D435i is mounted forward-looking on the depth-camera mount and connected to the Jetson by USB 3. It provides the RGB image (VLM), the depth image (3-D goal and guard), the two infrared images and the IMU (visual-inertial odometry).
- **Battery.** A 4S 5000 mAh LiPo powers motors, flight controller and Jetson.

![Connections on the quadrotor: power (red), data (blue) and what runs on each computer.](figures/fig_wiring.png){#fig:wiring width=95%}

## Software on the Jetson

The same Python code as in simulation runs on the Jetson; only the camera source changes (the RealSense SDK instead of Gazebo's transport). The software stack is:

- JetPack (Ubuntu with CUDA), headless, on the NVMe SSD;
- the VLM server: Ollama or llama.cpp [@ggml_llamacpp] with CUDA, Qwen3-VL-4B-Instruct q4_K_M (3.3 GB), kept loaded;
- the RealSense SDK (librealsense) for RGB, depth, infrared and IMU streams with hardware time stamps;
- visual-inertial odometry (below);
- the mission program: VLM worker, control loop at 20 Hz, logs.

**Memory budget.** The 8 GB are shared between CPU and GPU. The model file is 3.3 GB; with the vision encoder's work memory and a 2,048-token context, the VLM is expected to need 4–5 GB. The operating system, the visual odometry and the mission program have to fit in the rest. This is tight: NVIDIA's own TensorRT benchmarks list Qwen3-VL-2B, but not the 4B, for the Orin Nano [@nvidia2026trtedgellm], and community guides give about 4 B at 4 bit as the board's limit [@adsahu2026orinnanoguide]. It is therefore measured first (stage 1). If it does not fit, the fallbacks are LFM3-VL 2B (1.1 GB, Chapter 4) or the 2B Qwen3-VL at 8 bit (2.7 GB).

## Indoor positioning by visual-inertial odometry {#sec:vio}

The control loop commands velocities, and PX4 needs a position estimate to hold them; indoors there is no GPS, and the Pixhawk's IMUs alone drift within seconds. VLN on the Fly [@tayar2026vln], with the same flight controller and camera and a larger Jetson Orin NX, used an OptiTrack motion-capture system fused by EKF2. The lab has none, and nothing is to be bought, so the drone estimates its own motion from its camera: **visual-inertial odometry (VIO)** from the D435i's two infrared images and its IMU, running on the Jetson. The first choice is NVIDIA's Isaac ROS Visual SLAM (cuVSLAM), which supports the RealSense on Jetson; OpenVINS [@geneva2020openvins] is the alternative. The VIO pose and velocity are sent to PX4 as MAVLink odometry messages, and PX4's estimator (EKF2) fuses them as external vision, with GPS and compass switched off, as described in PX4's visual-inertial-odometry guide [@px4_vio].

This choice has four consequences for the thesis:

1. The D435i's infrared dot projector improves depth but disturbs VIO feature tracking; it is either switched on every other frame or off, at some cost to the depth image.
2. VIO and the VLM share the Jetson's GPU and memory. VIO must never stall; the VLM can wait.
3. Without motion capture there is no external ground truth. Distances (stopping distance, clearance, stand-off) are measured with floor markings, a tape measure and video; VIO drift with return-to-start tests.
4. VIO can lose track in plain or dark rooms; the safety pilot and PX4's failsafes must be ready to take over.

## Staged test plan

The step from a laptop simulation to flights in the lab is split into stages (Table @tbl:stages). Each stage changes one or two things compared with the one before and has a pass check; the next stage starts only when it passes, and every stage keeps its logs as thesis data. Stages 1–5 need no flying and can start the day the hardware arrives.

Table: Staged test plan from simulation to lab flights. {#tbl:stages}

| # | Stage | What changes | Measured | Pass check |
|---|---|---|---|---|
| 0 | Laptop simulation | – | Chapter 5 | done |
| 1 | VLM bench on the Jetson | the computer | latency per answer on new pictures (50+ calls), first-call time, memory, temperature and clock throttling over 30 min, power; accuracy on the Chapter 4 samples at 896 px | median ≤ 10 s; accuracy within about 2 points of the workstation; ≥ 1 GB free; no throttling |
| 2 | Jetson in the loop with Gazebo | mission code and VLM on the Jetson; PX4 and Gazebo on the laptop, over Ethernet | a subset of the Chapter 5 series with the real VLM delay | success comparable to stage 0; 20 Hz held on the Jetson |
| 3 | RealSense bench | real RGB and depth, camera on a tripod | depth error against tape at 0.5–6 m on the lab's objects; RGB–depth alignment; VLM found rate and wrong-object rate on real pictures; goal distance against tape; guard stop distance with an object moved towards the camera; frame time stamps; dot projector on/off | errors and found rate known; guard stops at the set distance |
| 4 | Record and replay | camera and Jetson carried by hand along the planned paths | recorded sequences replayed through the mission code: goals, consensus and guard decisions against measured truth | goals within about 0.5 m; guard triggers only where it should |
| 5 | Flight controller on the bench, propellers off | real Pixhawk 6C and Jetson | link and message rates; OFFBOARD accepted; failsafes when the program is killed, the USB cable unplugged, the depth stale | every failsafe behaves as set |
| 6 | Indoor position without the VLM | VIO into PX4 | VIO drift by hand and on a cart; hover and position hold with a safety pilot; drift over 2 min | stable hold; drift known |
| 7 | First flights, step by step | the full drone, safety pilot, low speed | (a) scripted moves; (b) guard only, towards a foam obstacle; (c) go-around a soft pole; (d) VLM goal from hover with turn-stop-look search; (e) the full test list | each sub-step safe before the next |
| 8 | Lab test series | – | the Chapter 5 test list (targets, with and without an obstacle, sweep and active), same metrics | simulation-to-reality comparison |

Stage 3 also provides the real-depth data for the sim-to-real comparison even if flights are delayed.

## Status

At the time of this draft the hardware has been ordered and stages 1–8 have not started. The results of the stages will be added here: the Jetson's VLM latency and memory (stage 1), the closed-loop result with the real delay (stage 2), the real depth and VLM found rates (stage 3), and the lab series compared with Chapter 5 (stage 8).
