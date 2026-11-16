# Related Work {#sec:related}

This chapter reviews what already exists: language-driven drone navigation (Section @sec:rw-vln), the perception approaches that run on drones today (Section @sec:rw-perception), the edge computers they run on (Section @sec:edge), the grounding benchmarks and models (Section @sec:rw-grounding), depth-based obstacle avoidance (Section @sec:rw-avoid), active perception and object search (Section @sec:rw-search) and indoor state estimation (Section @sec:rw-state). Section @sec:rw-gap compares the closest works with this thesis and states the gap it addresses.

## Language-driven navigation for drones {#sec:rw-vln}

Aerial vision-and-language navigation began in simulation. AerialVLN [@liu2023aerialvln] provided city-scale scenes with natural-language routes, UAV-ON [@xiao2025uavon] open-world object-goal navigation with 1,270 targets in 14 Unreal Engine scenes, and FlySearch [@pardyl2025flysearch] tested whether VLMs can explore to find an object. In FlySearch the best model (Gemini 2.0 Flash) succeeded in about 42 % of the episodes against about 67 % for untrained humans, and small open-weight models scored near zero; typical failures were losing sight of the target, hallucinated objects and the lack of a consistent search strategy. Recent surveys name onboard deployment as an open problem [@xia2026vlnsurvey; @javaid2024llmuav].

Systems that have flown fall into three groups.

**VLM or LLM off board.** Most real-flight systems send the camera image to a ground computer or a cloud service. TypeFly [@chen2023typefly] lets an LLM write drone programs in a compact language to cut response time. See-Point-Fly [@hu2025spf] asks large commercial VLMs for a 2-D point in the image and turns it into a 3-D step for a DJI Tello, at about 1–3 s per query. VLFly [@zhang2025vlfly] runs CLIP and a LLaMA instruction encoder on a laptop GPU. AirHunt [@chen2026airhunt] queries a cloud VLM over 4G (more than 2 s, up to about 10 s per answer) asynchronously with an onboard frontier planner. Taking Flight with Dialogue [@lim2025takingflight] uses a Pixhawk 6C Mini and a Jetson Orin Nano on the drone, but its Ollama models ran on an RTX 3080 Ti ground workstation.

**Modular VLM on board.** VLN on the Fly [@tayar2026vln] runs a 4-bit Qwen-3.5-2B on a Jetson Orin NX with a Pixhawk 6C and a RealSense D435i — the same flight controller and camera as this thesis. The VLM picks one cell of a 3 × 3 image grid, the median depth in that cell gives a 3-D goal, three agreeing predictions are required, and EGO-Planner plans the path. OnFly [@zheng2026onfly] runs Qwen3-VL-4B (4-bit AWQ, TensorRT) on an Orin NX 16 GB with a depth camera and a LiDAR.

**End-to-end vision-language-action models on board.** OpenFlyV2 [@liu2026openflyv2] runs one model from image to waypoints on an Orin NX at 10 Hz; LiteVLA-H [@williams2026litevlah] runs a 256 M-parameter model on an AGX Orin at 19.7 Hz; GRaD-Nav++ [@chen2025gradnav] also flies with an onboard model. These systems are fast but must be trained for the task, and their perception and safety cannot be inspected separately.

## Onboard perception: detectors and VLMs {#sec:rw-perception}

Table @tbl:rw-perception compares how drone systems perceive their goal. Three families can be distinguished.

**Closed-vocabulary detectors (YOLO).** YOLO is the standard onboard detector: YOLOv8n runs at 52 frames per second on an Orin NX and 65 with 8-bit quantisation, while a Raspberry Pi 5 does not reach real time [@rey2025yolo]. Its limit is the fixed list of classes it was trained on.

**Open-vocabulary detectors.** YOLO-World [@cheng2024yoloworld] and Grounding DINO [@liu2023groundingdino] detect objects named in text. On an AGX Orin, NanoOWL takes about 10 ms and YOLO-World 26–46 ms per image; NanoOWL suits plain noun phrases, while YOLO-World copes better with longer referring expressions [@park2025openvocabedge]. Grounding DINO runs at 6.2 frames per second on the Orin Nano Super [@maheshwari2025jetpacksuper]. Follow Anything [@maalouf2023followanything] used open-set detection and tracking for a drone on a laptop GPU.

**Generative VLMs.** VLMs answer free-form instructions and can reason about relations ("the box next to the door"), but run at about one answer per second or slower. See-Point-Fly found it faster and more accurate to let the VLM also report the obstacles than to add a YOLO detector (1.08 s against 1.73 s per step, 88.8 % against 72.2 % success) [@hu2025spf].

Table: Perception approaches of drone systems. "Where" = where the model runs. {#tbl:rw-perception}

| Work | Computer | Perception | Where | Latency reported |
|---|---|---|---|---|
| VLN on the Fly [@tayar2026vln] | Jetson Orin NX | VLM (Qwen-3.5-2B, 4 bit): 3 × 3 grid cell + depth | on board | 0.79 s median |
| OnFly [@zheng2026onfly] | Jetson Orin NX 16 GB | VLM (Qwen3-VL-4B AWQ, TensorRT): pixel + depth | on board | 0.81 s per decision |
| OpenFlyV2 [@liu2026openflyv2] | Jetson Orin NX | VLA (one model) | on board | 10 Hz |
| LiteVLA-H [@williams2026litevlah] | Jetson AGX Orin | VLA, 256 M parameters | on board | 51 ms per action |
| Taking Flight [@lim2025takingflight] | Orin Nano (ground: RTX 3080 Ti) | LLM + VLM via Ollama (Gemma 3) | off board | not measured |
| See-Point-Fly [@hu2025spf] | DJI Tello + cloud | commercial VLMs: 2-D point | off board | 1–3 s |
| VLFly [@zhang2025vlfly] | laptop RTX 4070 | CLIP + LLaMA encoder | off board | 7–10 Hz control |
| AirHunt [@chen2026airhunt] | Intel NUC + cloud | cloud VLM + Grounding DINO + SAM 2 | VLM off board | > 2 s per VLM answer |
| Follow Anything [@maalouf2023followanything] | laptop GPU | open-set detection and tracking | off board | 6–20 frames/s |
| Rey et al. [@rey2025yolo] | Orin Nano / NX, RPi 5 | YOLOv8 | on board | 52–65 frames/s (Orin NX) |
| Park et al. [@park2025openvocabedge] | Jetson AGX Orin | NanoOWL, YOLO-World | on board (not flown) | 10–46 ms |
| This thesis | Jetson Orin Nano Super 8 GB | VLM (Qwen3-VL-4B, 4 bit): box + depth; depth guard | on board | Chapter 4 (GPU), Chapter 6 (Jetson) |

## Edge computers: why the Jetson Orin Nano 8 GB {#sec:edge}

Table @tbl:rw-edge compares the computers used in the works above.

Table: Edge computers for onboard perception. AI performance in INT8 TOPS (sparse). {#tbl:rw-edge}

| Device | Brand | AI perf. | Memory | Power | Used for (in the literature) |
|---|---|---|---|---|---|
| Jetson Orin Nano 8 GB, Super mode | NVIDIA | 67 TOPS | 8 GB LPDDR5, shared CPU/GPU, 102 GB/s | 7–25 W | VLMs up to about 4 B at 4 bit, LLMs, Grounding DINO, YOLO [@maheshwari2025jetpacksuper; @nvidia2026trtedgellm; @rey2025yolo] |
| Jetson Orin NX 16 GB, Super mode | NVIDIA | 157 TOPS | 16 GB, 102 GB/s | 10–40 W | onboard VLM drones: VLN on the Fly, OnFly, OpenFlyV2 [@tayar2026vln; @zheng2026onfly; @liu2026openflyv2] |
| Jetson AGX Orin 32/64 GB | NVIDIA | 200–275 TOPS | 32/64 GB, 205 GB/s | 15–60 W | VLA models, open-vocabulary detectors [@williams2026litevlah; @park2025openvocabedge] |
| Raspberry Pi 5 | Raspberry Pi | CPU only | 2–16 GB | 5 V / 5 A supply | YOLO not in real time [@rey2025yolo; @raspberrypi2025pi5brief] |
| Raspberry Pi 5 + AI Kit (Hailo-8L) | Raspberry Pi / Hailo | 13 TOPS | Pi memory | – | CNN detection and segmentation only [@patuck2024aikit] |
| Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H) | Raspberry Pi / Hailo | 40 TOPS (INT4) | 8 GB on the board | – | LLMs of 1–1.5 B [@patuck2026aihat2] |
| ModalAI VOXL 2 | ModalAI (Qualcomm) | 15 TOPS | 8 GB | – | PX4 companion computer; no VLM use found [@px4_voxl2] |

The Orin Nano 8 GB in Super mode is the smallest and cheapest of these boards that runs a VLM of a few billion parameters on a GPU with CUDA and the llama.cpp, Ollama and TensorRT runtimes. The Raspberry Pi accelerators run CNNs or LLMs of about 1.5 B at most [@patuck2024aikit; @patuck2026aihat2]. JetPack 6.2's Super mode raised the Orin Nano from 40 to 67 TOPS and its memory bandwidth from 68 to 102 GB/s — the same bandwidth as the Orin NX — and bandwidth largely limits how fast a language model generates tokens [@maheshwari2025jetpacksuper; @shin2026smallvlmquant]. Its 25 W budget suits the X500's payload, and the Orin Nano was the most energy-efficient of the boards in a small-language-model study [@islam2025slmenergy].

The price is memory: the 8 GB are shared between CPU and GPU. NVIDIA's TensorRT Edge-LLM benchmarks list Qwen3-VL-2B on the Orin Nano 8 GB (190.9 ms prefill, 36.9 tokens/s), but Qwen3-VL-4B and 8B only on the Orin NX 16 GB (4B: 244.6 ms prefill, 31.2 tokens/s) [@nvidia2026trtedgellm], and community guides put about 4 B parameters at 4 bit as the upper limit of the Orin Nano [@adsahu2026orinnanoguide]. A 4-bit 4B model is therefore at the edge of what the board supports — which is why the thesis measures it there before flying (Chapter 6) and keeps a 2B-class fallback. Every published onboard VLM drone stack uses an Orin NX or AGX Orin; none runs on the Orin Nano.

Two further findings from the edge literature matter for the design. On an AGX Orin, about 94 % of a small VLA's latency is the processing of the input (prefill), not the generation of the answer [@williams2026litevlah], so the number of image tokens drives the latency. And 4-bit quantisation saves memory but does not always speed up generation, because the weights must be de-quantised [@shin2026smallvlmquant]; both are measured on the Jetson in stage 1.

## Grounding benchmarks and models {#sec:rw-grounding}

Referring-expression grounding — finding the object a phrase describes — is measured on RefCOCO, RefCOCO+ [@kazemzadeh2014referitgame; @yu2016refcoco] and RefCOCOg [@mao2016refcocog], built on COCO photographs [@lin2014coco]. gRefCOCO [@liu2023gres] adds expressions that refer to several objects or to none, which matters for a drone that must be able to say "not here" during a search. RefDrone [@sun2025refdrone], built on VisDrone [@zhu2020visdrone], is the drone-view counterpart: 17.9 k expressions on 8.5 k images, including expressions with no target. In its zero-shot tests, large multimodal models did poorly, especially on small objects, which the authors attribute to limited input resolution.

Among open models, Qwen2.5-VL [@bai2025qwen25vl] and Qwen3-VL [@bai2025qwen3vl] are trained to localise objects with boxes and come in sizes from 2 B parameters; the Qwen3-VL-4B model card claims stronger 2-D grounding than its predecessor [@qwen2025qwen3vl4bcard]. Liquid AI's LFM2-VL family [@amini2025lfm2; @liquidai2025lfm2vl] targets edge devices and processes large images as tiles. Gemma 3 [@gemmateam2025gemma3] also accepts images. Published grounding scores are usually measured at full precision with each model's own prompt; Chapter 4 measures them with the drone's prompt, 4-bit quantisation and on drone views.

## Depth-based reactive obstacle avoidance {#sec:rw-avoid}

The Vector Field Histogram [@borenstein1991vfh] reduces range data to a polar histogram of obstacle density and steers through the free opening closest to the goal direction; 3DVFH+ [@vanneste2014vfh3d] extends it to three dimensions. The go-around of this thesis is a minimal form of the same idea, limited to directions inside the camera's view. Time to contact goes back to Lee's τ [@lee1976tau]. Recent drone work uses model-predictive control with control-barrier functions on RGB-D points [@saviolo2024reactive], dynamic-window planning on a LiDAR map [@bes2024dwa3d], or trajectory optimisation on an occupancy map, as EGO-Planner [@zhou2020egoplanner] in VLN on the Fly.

PX4 itself offers Collision Prevention [@px4_collisionprevention], which limits velocity setpoints from distance sensors — but only in Position mode, not in the OFFBOARD mode a companion computer flies in. The ROS-based PX4-Avoidance planner, which supported the RealSense D435, was archived in August 2024 [@px4_avoidance]. A guard on the companion computer, as in this thesis, is therefore the practical way to keep an OFFBOARD drone away from obstacles. The D435i's depth works best between about 0.3 and 3 m [@intel_d435i], which bounds the guard's horizon on the real drone.

## Active perception and object search {#sec:rw-search}

Active perception means moving the sensor to gain information [@bajcsy1988active; @aloimonos1988active]; Tsotsos [@tsotsos1992relative] showed that active search can make visual search tractable. Recent work combines this with VLMs. VLFM [@yokoyama2023vlfm] builds a map of how promising each frontier is according to a vision-language model; AirHunt [@chen2026airhunt] ranks frontiers by semantic value and distance and avoids redundant VLM queries; Srivastava et al. [@srivastava2026spatialsemantic] keep separate beliefs over where to look and which candidate is the target, choose views by expected information gain, and reduce the number of VLM queries this way. FlySearch's finding that VLMs explore poorly on their own [@pardyl2025flysearch] supports keeping the search policy outside the VLM, as this thesis does with its sweep and its belief over viewing directions.

## Indoor state estimation {#sec:rw-state}

PX4 fuses external position and velocity estimates (`ODOMETRY` messages) in its EKF2 estimator, at 30–50 Hz with a tuned delay [@px4_extpos; @px4_vio]. Visual-inertial odometry on Jetson computers is available as NVIDIA's CUDA-accelerated cuVSLAM [@korovko2025cuvslam], packaged as Isaac ROS Visual SLAM [@nvidia_isaacrosvslam], and as the open-source OpenVINS [@geneva2020openvins] and VINS-Mono [@qin2018vinsmono]; Jeon et al. [@jeon2021vio] compared such methods on Jetson boards for drones. The closest VLM drone work localised with motion capture [@tayar2026vln] or LiDAR [@zheng2026onfly]; Taking Flight used a ZED Mini's VIO but ran its models off board [@lim2025takingflight].

## Simulation and flight-controller interface {#sec:rw-sim}

PX4 [@meier2015px4] with Gazebo [@gazebo] provides an X500 with a forward depth camera (`gz_x500_depth`) in lockstep simulation [@px4_gazebo]. OFFBOARD mode needs a setpoint stream of more than 2 Hz before and during the mode and a configured failsafe for its loss [@px4_offboard]; MAVSDK's offboard plugin sends body-frame velocity and yaw-rate setpoints [@mavsdk_offboard]. No study was found that measures the sim-to-real gap of a PX4–Gazebo pipeline with a VLM in the loop; Taking Flight names missing motion blur and imaging artefacts as a limit of its simulation [@lim2025takingflight].

## Closest works and the gap {#sec:rw-gap}

Table @tbl:rw-closest compares the three works closest to this thesis.

Table: The closest works compared with this thesis. {#tbl:rw-closest}

| | VLN on the Fly [@tayar2026vln] | OnFly [@zheng2026onfly] | Taking Flight [@lim2025takingflight] | This thesis |
|---|---|---|---|---|
| Computer | Jetson Orin NX | Jetson Orin NX 16 GB | Jetson Orin Nano (models on ground GPU) | Jetson Orin Nano Super 8 GB |
| Flight controller | Pixhawk 6C | PX4 | Pixhawk 6C Mini | Pixhawk 6C |
| Sensors | RealSense D435i | D435 + Mid360 LiDAR | ZED Mini | RealSense D435i |
| Model | Qwen-3.5-2B, 4 bit | Qwen3-VL-4B AWQ | Gemma 3 via Ollama | Qwen3-VL-4B q4 via Ollama |
| VLM output | 1 of 9 grid cells | pixel | forward and yaw-rate commands | bounding box |
| Goal confirmation | 3 agreeing predictions | semantic-geometric check | – | 2 of 3 agreeing answers |
| Target out of view | not handled | – | – | sweep or active search |
| Obstacle handling | EGO-Planner on a depth map | Fast-Planner | – | depth guard + reactive go-around |
| Positioning | OptiTrack motion capture | LiDAR | ZED VIO | VIO from the D435i |
| Results | 13/15 flights reached the goal | 67.8 % success in simulation | 40 % mission success in simulation | Chapter 5; lab flights pending |

VLN on the Fly also reports that its VLM said a visible object was not visible in 44–53 % of the frames, and that failures happened when the target left the view near the goal; OnFly found that replacing its 4B model by the 2B lowered success in simulation from 67.8 % to 43.4 %, an independent argument for the 4B size.

**The gap.** Every published onboard VLM drone stack uses an Orin NX or AGX Orin; none runs on the 8 GB Orin Nano. None combines VLM grounding with an explicit search for targets out of view and a reactive depth guard that works in OFFBOARD mode, and none positions itself by visual-inertial odometry from the same camera instead of motion capture or LiDAR. No study compares several small VLMs on both everyday and drone-view grounding with the prompt, quantisation and image size used on the drone. This thesis addresses these points.
