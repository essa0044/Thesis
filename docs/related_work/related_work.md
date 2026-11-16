# Related Work: Onboard VLM Goal Selection for a PX4 Quadrotor

Collected 2026-11-16. Each source cited here was opened, either the arXiv abstract or HTML page, the publisher page, or the official documentation. BibTeX entries are in `references.bib`. When a source gave only an abstract, this document says so. Items that could not be verified are listed at the end.

Thesis system, for reference: Holybro X500 V2 with a Pixhawk 6C (PX4), a Jetson Orin Nano Super 8 GB, and a RealSense D435i. Qwen3-VL-4B-Instruct (4-bit, Ollama/llama.cpp) returns a 2-D box. The box and the depth image give a 3-D goal, which is accepted on 2-of-3 agreement. The drone flies there with OFFBOARD velocity setpoints sent through MAVSDK. A depth-camera corridor and time-to-contact (TTC) guard runs alongside, with a reactive go-around. The target is searched for with either a fixed sweep or an active-sensing (belief plus next-best-view) strategy.

---

## 1. Overview by topic

### A. VLM/LLM-based navigation and object-goal search for UAVs

**What exists.** Aerial vision-language navigation (VLN) began with simulator benchmarks. AerialVLN [liu2023aerialvln] provided 25 city-scale scenes. UAV-ON [xiao2025uavon] added open-world object-goal navigation with 1,270 targets in 14 Unreal Engine scenes. FlySearch [pardyl2025flysearch] tested whether VLMs can explore. In FlySearch, the best model (Gemini 2.0 Flash) reached about 42 % success against about 67 % for untrained humans on FS-1, and small open-weight models scored near zero. Common failures were losing sight of the target, hallucinating objects and having no consistent search strategy. Surveys [xia2026vlnsurvey, javaid2024llmuav] name *onboard deployment* as an open problem.

Real-flight systems fall into three groups:

1. **Offboard or cloud VLM.**
   - TypeFly [chen2023typefly] cuts LLM planning latency by up to 62 % with a token-efficient DSL called MiniSpec.
   - See-Point-Fly (SPF) [hu2025spf] treats action prediction as 2-D point grounding on a DJI Tello. It uses frontier APIs (Gemini, GPT-4.1 and others) with about 1–3 s per inference, and reports 92.7 % real-world success.
   - VLFly [zhang2025vlfly] combines CLIP retrieval with a LLaMA instruction encoder on an RTX 4070 laptop and a Tello, reaching 83 % success on direct instructions.
   - AirHunt [chen2026airhunt] queries Qwen-VL-Max over 4G, which takes more than 2 s and up to about 10 s, and runs it asynchronously with a frontier planner.
   - Taking Flight with Dialogue [lim2025takingflight] uses a Pixhawk 6C Mini, a Jetson Orin Nano and a ZED Mini, but the Ollama models run on an RTX 3080 Ti ground workstation.
2. **Onboard modular VLM.**
   - VLN on the Fly [tayar2026vln] runs Qwen-3.5-2B INT4 on a Jetson Orin NX with a D435i and a Pixhawk 6C, and localizes with OptiTrack.
   - OnFly [zheng2026onfly] runs Qwen3-VL-4B-AWQ through TensorRT on an Orin NX 16 GB, with a D435, a Mid360 LiDAR and PX4.
3. **Onboard end-to-end VLA.**
   - OpenFlyV2 [liu2026openflyv2] runs on an Orin NX at 10 Hz and reaches 48 % real-world success.
   - LiteVLA-H [williams2026litevlah] runs a 256M model on an AGX Orin at 19.7 Hz.
   - GRaD-Nav++ [chen2025gradnav] runs onboard with an unnamed compute platform.

**What is missing.** All onboard modular systems use an **Orin NX (16 GB) or AGX Orin**. None reports VLM grounding on the cheaper, lower-power **Orin Nano 8 GB**. The two closest systems depend on **motion capture** [tayar2026vln] or **LiDAR** [zheng2026onfly] for state estimation. Neither combines VLM grounding with an explicit **search strategy over viewing directions** for when the target is not in view, and VLN on the Fly says this directly ("a referent outside the field of view must be brought into view first"). No study compares several small VLMs (2B–8B) on both generic (RefCOCO/gRefCOCO) and drone-view (RefDrone) grounding *and* on the same onboard hardware. This thesis targets that gap.

### B. Running VLMs/LLMs on edge hardware

**Vendor numbers.** Vendor benchmarks give the clearest per-device figures.

- **JetPack 6.2 "Super" blog [maheshwari2025jetpacksuper].** The Orin Nano 8 GB rises from 40 to 67 sparse INT8 TOPS, and its memory bandwidth from 68 to 102 GB/s. The blog gives Orin Nano 8 GB decode rates (tokens/s, original vs. Super) of Qwen2-VL-2B 2.8→4.4, InternVL2.5-4B 2.5→5.1, PaliGemma2-3B 13.7→21.6 and SmolVLM-2B 8.1→12.9. Its text-LLM figures (INT4, MLC) include Qwen2.5-7B 14.2→21.8, Llama-3.2-3B 27.7→43.1 and Grounding DINO 4.1→6.2 FPS. The VLM table header contains a labelling inconsistency, noted in the source.
- **TensorRT Edge-LLM benchmarks [nvidia2026trtedgellm].** These are batch 1, from the v0.9.0 dashboard.
  - Orin Nano 8 GB: Qwen3-VL-2B (INT4 AWQ LLM, FP16 ViT) has a 190.9 ms prefill and decodes at 36.9 tok/s.
  - Orin NX 16 GB: Qwen3-VL-2B has a 103.1 ms prefill and 60.1 tok/s; Qwen3-VL-4B has 244.6 ms and 31.2 tok/s; Qwen3-VL-8B has 462.1 ms and 18.9 tok/s.
  - Qwen3-VL-4B and -8B are **not listed for Orin Nano** in that dashboard. This suggests that 4B is at the upper end of what NVIDIA's own stack supports on 8 GB.
- **Jetson AI Lab [jetsonailab_ollama].** The tutorial notes that Ollama uses llama.cpp [ggml_llamacpp], with CUDA, and that llama.cpp reaches "roughly half the peak performance" of faster APIs such as NanoLLM.
- **Community guide [adsahu2026orinnanoguide].** This post (grey literature) estimates that VLMs up to about 4B fit on the Orin Nano 8 GB at Q4_K, using roughly 3–4 GB of RAM for 3–4B models.

**Academic studies.**

- Small-VLM quantization on Orin NX/AGX [shin2026smallvlmquant]: INT4 LLM quantization cuts VRAM sharply but **can slow decoding because of dequantization overhead**. SigLIP encoders also show unusually high INT8 latency on Jetson Ampere.
- LiteVLA-H [williams2026litevlah]: on an AGX Orin, latency is dominated by **multimodal prefill (about 94 %)**, not by decoding.
- LiteVLA-Edge [williams2026litevlaedge]: a 4-bit GGUF model in llama.cpp on Orin-class hardware takes 150.5 ms end to end (about 6.6 Hz).
- Energy studies:
  - [islam2025slmenergy] tested a Raspberry Pi 5, a Jetson Nano and a Jetson Orin Nano; the Orin Nano GPU was the most energy efficient.
  - [khatib2026edgecontinuum] found the AGX Orin uses less energy than a GPU server, but with higher latency.
  - [seymour2024llmsmall] characterizes Pythia models (70M–1.4B) on Orin.
- Systems work: [yang2026eaglevla] (asynchronous VLA on Orin with a llama.cpp-based engine) and [deng2026edgefm] (up to 1.49× faster than TRT Edge-LLM on Orin). [taherin2025vlascaling] finds that power-capped edge devices slow down non-linearly.

**What is missing.** No study measures **end-to-end grounding latency (image to box to 3-D goal) of Qwen3-VL-2B/4B/8B through Ollama/llama.cpp on an Orin Nano Super**, nor how that latency trades off against grounding accuracy. This is RQ2.

### C. Classic vs. open-vocabulary onboard detectors

- **Closed-vocabulary YOLO** is the standard onboard detector. On a UAV-oriented benchmark, YOLOv8n ran at 52 FPS on an Orin NX (65 FPS with INT8 PTQ), and the Raspberry Pi 5 did not meet real-time requirements [rey2025yolo].
- **Open-vocabulary detectors** are slower but still near real time.
  - YOLO-World: 35.4 AP on LVIS at 52 FPS on a V100 [cheng2024yoloworld].
  - Grounding DINO: 52.5 zero-shot AP on COCO [liu2023groundingdino]. It runs at 6.2 FPS on an Orin Nano 8 GB Super [maheshwari2025jetpacksuper].
  - On an AGX Orin, NanoOWL (OWL-ViT, TensorRT) takes 9.8 ms with the patch32 backbone, while YOLO-World-S/X takes 26/46 ms [park2025openvocabedge]. The same study finds NanoOWL suits noun phrases and YOLO-World handles more complex referring expressions [park2025openvocabedge, jetsonailab_nanoowl].
- **Generative VLMs** run at about 1 Hz or slower, but they accept free-form instructions and can reason about relations. SPF reports that letting the VLM output obstacle boxes itself was faster and more accurate than adding YOLOv8n (1.08 s vs. 1.73 s, 88.8 % vs. 72.2 %) [hu2025spf].
- Follow Anything [maalouf2023followanything] runs open-set detect-and-track on a laptop GPU at 6–20 FPS for an MAV.
- Dimmig et al. [dimmig2023smallform] show fully onboard detection plus VO for aerial pick-and-place, with 93 % grasp success over 70 trials.

**Gap.** Edge open-vocabulary benchmarks use AGX Orin and static images. Drone systems either use YOLO (closed set) or put the VLM offboard. This motivates RQ1's table, which contrasts the perception approaches.

### D. Grounding datasets and model reports

- **RefCOCO, RefCOCO+ and RefCOCOg.** RefCOCO and RefCOCO+ come from the ReferIt game protocol [kazemzadeh2014referitgame] applied to COCO [lin2014coco] and are commonly cited via [yu2016refcoco]. RefCOCOg comes from [mao2016refcocog].
- **gRefCOCO** [liu2023gres] adds **multi-target and no-target** expressions. This matters because the VLM must be able to say "not present" during search.
- **RefDrone** [sun2025refdrone] is built on VisDrone2019-DET [zhu2020visdrone]. It has 17.9k expressions, 8.5k images and on average 3.8 targets per expression, including 847 no-target expressions. In zero-shot tests, large multimodal models "performed poorly", with the best (CogVLM) reaching 15.38 instance F1. They struggled most with **small objects**, which the authors attribute to input-resolution limits.
- **Model reports:**
  - Qwen3-VL [bai2025qwen3vl] comes in dense 2B/4B/8B/32B and MoE versions with 256K context. The 4B model card [qwen2025qwen3vl4bcard] claims "stronger 2D grounding" and 3-D grounding for embodied AI.
  - Qwen2.5-VL [bai2025qwen25vl] localizes with boxes or points.
  - LFM2 / LFM2-VL [amini2025lfm2, liquidai2025lfm2vl] come in 450M and 1.6B sizes, use a SigLIP2 NaFlex encoder at native resolution up to 512×512 with tiling, and claim up to 2× faster GPU inference.
  - Gemma 3 [gemmateam2025gemma3] ranges from 1B to 27B and supports vision.

### E. Depth-based reactive obstacle avoidance

**Classical basis.**

- VFH [borenstein1991vfh] reduces range data to a polar histogram and steers toward the nearest low-density opening that matches the target direction. This is close to the thesis's go-around, which selects among turned candidate directions inside the camera's view.
- 3DVFH+ [vanneste2014vfh3d] extends this idea to 3-D using an octomap.
- Time-to-contact goes back to Lee's τ [lee1976tau].

**PX4-specific options.**

- PX4 **Collision Prevention** [px4_collisionprevention] limits velocity setpoints in **Position mode only**. It blocks motion in directions without sensor data and uses a `CP_DELAY` that combines sensor delay with tracking delay (typically 0.1–0.5 s).
- The ROS-based **PX4-Avoidance** (3DVFH+ local planner, officially supporting the RealSense D435) was **archived in August 2024** [px4_avoidance].
- There is therefore no maintained PX4 component for depth-camera avoidance in OFFBOARD mode. A companion-side guard, like the thesis's, is the practical choice.

**Recent methods.**

- NMPC with adaptive control barrier functions, choosing the depth points with minimum TTC from RGB-D data [saviolo2024reactive].
- DWA-3D on a LiDAR octomap, with about 40 ms per cycle [bes2024dwa3d].
- EGO-Planner [zhou2020egoplanner], used by VLN on the Fly with a depth-based occupancy map.

The D435i's ideal range is 0.3–3 m, with an 87°×58° depth FOV and a minimum depth of about 28 cm [intel_d435i]. This range bounds the reachable TTC horizon.

### F. Active perception and active object search

The classics define active perception as controlling the sensor to gain information [bajcsy1988active, aloimonos1988active]. They also show that active search can reduce the complexity of visual search [tsotsos1992relative].

Modern VLM-era work:

- **VLFM** [yokoyama2023vlfm] builds a language-grounded value map over depth-based frontiers and was demonstrated on a Spot robot.
- **AirHunt** [chen2026airhunt] ranks frontiers by semantic value and distance, and selects keyframes to avoid redundant VLM queries.
- **Spatial-semantic uncertainty** [srivastava2026spatialsemantic] keeps *separate* beliefs over where to look and over which candidate is the target. Planners based on expected information gain (EIG) reached confident decisions in 75–92.5 % of trials, against 20 % for random search, and weighting semantic uncertainty more heavily reduced the number of VLM queries.

FlySearch [pardyl2025flysearch] shows that VLMs alone do not explore systematically. This supports keeping the search policy outside the VLM, as the thesis does with its sweep or belief-based next-best-view.

### G. Indoor GPS-denied state estimation

PX4 fuses external vision through `ODOMETRY` / `VISION_POSITION_ESTIMATE` into EKF2 (`EKF2_EV_CTRL`, `EKF2_EV_DELAY`, `EKF2_HGT_REF`). The stream should run at 30–50 Hz, and the frame transform to PX4's conventions must be explicit [px4_extpos, px4_vio].

VIO options:

- **cuVSLAM** [korovko2025cuvslam] is CUDA-accelerated for Jetson and supports 1 to 32 cameras plus an IMU. It is packaged as Isaac ROS Visual SLAM [nvidia_isaacrosvslam], which requires at least 30 Hz with jitter within ±2 ms and has supported RGB-D mode since 2026-02-02.
- **OpenVINS** [geneva2020openvins] and **VINS-Mono/Fusion** [qin2018vinsmono] are open-source alternatives.
- [jeon2021vio] benchmarks VINS-Mono/Fusion, Kimera, ALVIO, Stereo-MSCKF, ORB-SLAM2 and ROVIO on Jetson TX2, Xavier NX and AGX Xavier for MAVs.

**Gap.** The closest VLM drone work uses OptiTrack [tayar2026vln]. Taking Flight uses a ZED Mini for VIO, but with offboard models [lim2025takingflight]. No source found runs **D435i-based VIO concurrently with a 4B VLM on an Orin Nano**, where GPU and memory contention is the obvious risk.

### H. Simulation, PX4 and MAVLink/MAVSDK

- PX4 [meier2015px4] with Gazebo (Harmonic is the default from Ubuntu 22.04) provides `gz_x500_depth`, an X500 with a front depth camera, in lockstep simulation [px4_gazebo].
- OFFBOARD mode needs a setpoint stream faster than 2 Hz before and during the mode, and a configured failsafe (`COM_OF_LOSS_T`, `COM_OBL_RC_ACT`) [px4_offboard].
- MAVSDK's Offboard plugin supports `VelocityBodyYawspeed` and `VelocityNedYaw`, resends at 20 Hz, and requires an initial setpoint before `start()` [mavsdk_offboard].
- MAVLink is surveyed in [koubaa2019mavlink], and the X500 V2 + Pixhawk 6C kit is documented in [px4_x500v2].

**Gap.** No opened source quantifies the sim-to-real gap for a VLM-in-the-loop PX4/Gazebo pipeline. Taking Flight lists missing motion blur and imaging artifacts as a sim-to-real limitation [lim2025takingflight].

### I. Works very close to this thesis (flagged)

See Table 3.

- **[tayar2026vln]** is the nearest neighbour: same flight controller (Pixhawk 6C), same camera (D435i), Jetson Orin (NX) and an onboard quantized small VLM with depth back-projection.
- **[zheng2026onfly]** is the nearest in VLM choice: Qwen3-VL-4B with a 2B ablation on a Jetson Orin NX.
- **[lim2025takingflight]** is the nearest in compute: Orin Nano with Ollama, but the models run offboard.

---

## 2. Table 1: Onboard perception approaches

| key | year | platform / computer | perception | onboard / offboard | latency reported | notes |
|---|---|---|---|---|---|---|
| tayar2026vln | 2026 | MIRA quadrotor, Pixhawk 6C, **Jetson Orin NX**, D435i | VLM (Qwen-3.5-2B INT4), 3×3 cell grounding + depth | Onboard | TTFT median 0.60 s; total median 0.79 s, p95 0.88 s (1,865 queries) | 13/15 goal reaches; OptiTrack odometry |
| zheng2026onfly | 2026 | Custom quad, PX4, **Orin NX 16 GB**, D435, Mid360 | VLM (Qwen3-VL-4B-AWQ, TensorRT), pixel goal + depth | Onboard | Decision 0.81 s; monitoring 1.15 s (with memory) | Sim SR 67.8 %; real flights qualitative only |
| liu2026openflyv2 | 2026 | Quadrotor, **Orin NX** | VLA (single model) | Onboard | 10 Hz, 12-waypoint chunks | 48 % real-world success, zero-shot sim-to-real |
| williams2026litevlah | 2026 | **AGX Orin** | VLA 256M, dual-rate | Onboard | 50.65 ms action (19.7 Hz); 150 ms caption | 2.2 GB, 22.1 W; prefill is about 94 % of latency |
| chen2025gradnav | 2025 | Drone (compute not named) | VLA with MoE action head | Onboard | Not reported | Real-world SR 67 % (trained tasks), 50 % (unseen) |
| lim2025takingflight | 2025 | Custom quad, Pixhawk 6C Mini, **Orin Nano** + ZED Mini | LLM (Gemma3 4B etc.) + VLM (Gemma3 12B etc.) via Ollama | **Offboard** (RTX 3080 Ti) | Not measured | Best mission SR 40 % (sim) |
| hu2025spf | 2025 | DJI Tello EDU | VLM (Gemini/GPT-4.1/…), 2-D waypoint | Offboard (API; not stated explicitly) | About 1–3 s per inference; 1.08 s with VLM obstacle boxes | 92.7 % real-world SR |
| zhang2025vlfly | 2025 | Tello EDU + RTX 4070 laptop | CLIP retriever + LLaMA encoder | Offboard | 7–10 Hz control (real) | 83 % (direct), 70 % (indirect) |
| chen2026airhunt | 2026 | Custom quad, NUC 11 Pro, Mid360, 3 OAK cameras | VLM (Qwen-VL-Max API) + Grounding DINO + SAM2 | VLM offboard (4G); planning onboard | VLM >2 s (up to about 10 s); planner <31 ms | Asynchronous VLM/planner |
| chen2023typefly | 2023 | Drone (not specified on abstract page) | LLM (MiniSpec plans) | Not stated on abstract | Up to 62 % less response time | Token-efficient plan language |
| maalouf2023followanything | 2023 | MAV + laptop GPU (6–8 GB) | Open-set foundation models (detect/segment/track) | Offboard laptop | 6–20 FPS | Text/image/click queries |
| dimmig2023smallform | 2023 | Small aerial research vehicle | Object detection + VO (detector not named on abstract page) | Onboard | Not on abstract page | 93 % grasp success (70 trials) |
| rey2025yolo | 2025 | Orin Nano, Orin NX, RPi 5 | YOLOv8n/s | Onboard (edge) vs. cloud | YOLOv8n 52 FPS (NX), 65 FPS INT8 | RPi 5 not real-time |
| park2025openvocabedge | 2025 | **AGX Orin** 64 GB (static images) | Open-vocab detector: NanoOWL, YOLO-World (+ SAM variants) | Onboard (edge) | NanoOWL 9.8 ms; YOLO-World-S 26 ms | Robot-perception study, not flown |
| maheshwari2025jetpacksuper | 2025 | **Orin Nano 8 GB Super** | Open-vocab detector (Grounding DINO), VLMs, ViTs | Onboard (edge) | Grounding DINO 6.2 FPS; Qwen2-VL-2B 4.4 tok/s | Vendor benchmark |
| nvidia2026trtedgellm | 2026 | **Orin Nano 8 GB**, Orin NX 16 GB, AGX Thor | VLM (Qwen3-VL 2B/4B/8B), LLMs | Onboard (edge) | Orin Nano Qwen3-VL-2B: 190.9 ms prefill, 36.9 tok/s | 4B/8B listed for NX only |
| *this thesis* | 2026 | X500 V2, Pixhawk 6C, **Orin Nano Super 8 GB**, D435i | VLM (Qwen3-VL-4B Q4, Ollama) + OpenCV depth guard | Onboard | (to be measured) | 2-of-3 consensus; sweep / active search |

---

## 3. Table 2: Edge computers ("why Jetson Orin Nano 8 GB")

| device | brand | AI perf. (INT8 TOPS) | GPU | memory | power | price (citable) | use in papers | source keys |
|---|---|---|---|---|---|---|---|---|
| **Jetson Orin Nano 8 GB Super** (dev kit) | NVIDIA | 67 sparse / 33 dense (Super); 40/20 original | Ampere, 1,024 CUDA cores, 32 tensor cores, 1,020 MHz; 17 TFLOPS FP16 | 8 GB 128-bit LPDDR5, 102 GB/s (68 GB/s original) | 7 / 15 / 25 W (blog lists 15 W, 25 W, MAXN SUPER) | US$249 dev kit | VLM (Qwen3-VL-2B via TRT Edge-LLM; Qwen2-VL-2B, PaliGemma2), LLM, Grounding DINO, YOLOv8; Taking Flight (offboard models); this thesis | maheshwari2025jetpacksuper, aufranc2024orinnanosuper, nvidia2026trtedgellm, rey2025yolo, lim2025takingflight, islam2025slmenergy |
| Jetson Orin Nano 4 GB (Super) | NVIDIA | 34 / 17 | Ampere, 8.5 TFLOPS FP16 | 4 GB, 51 GB/s | 7 W, 25 W, MAXN SUPER | n/a | Small LLMs (Gemma2-2B 18.6 tok/s), PaliGemma2 | maheshwari2025jetpacksuper |
| Jetson Orin NX 8 GB (Super) | NVIDIA | 117 / 58 (Super); 70 (original) | Ampere, 19 TFLOPS FP16; DLA 40/20 TOPS | 8 GB, 102 GB/s | 10–40 W, MAXN SUPER | US$399 module (launch) | YOLOv8n 52/65 FPS | maheshwari2025jetpacksuper, halfacree_agxorin, rey2025yolo |
| **Jetson Orin NX 16 GB** (Super) | NVIDIA | 157 / 78 (Super); 100 (original) | Ampere, 19 TFLOPS FP16; DLA 80/40 TOPS | 16 GB, 102 GB/s | 10–40 W, MAXN SUPER | US$599 module (launch) | VLM onboard drones: VLN on the Fly (Qwen-3.5-2B), OnFly (Qwen3-VL-4B), OpenFlyV2 (VLA); TRT Edge-LLM Qwen3-VL-2/4/8B | maheshwari2025jetpacksuper, halfacree_agxorin, tayar2026vln, zheng2026onfly, liu2026openflyv2, nvidia2026trtedgellm |
| Jetson AGX Orin 32/64 GB | NVIDIA | 275 sparse / 138 dense (64 GB); 200 (32 GB) | Ampere, 2,048 CUDA cores, 64 tensor cores, 2× NVDLA | 32 or 64 GB LPDDR5, 204.8 GB/s | 15–60 W | US$1,999 dev kit; modules US$899 / 1,599 | VLA (LiteVLA-H), open-vocabulary detectors (NanoOWL, YOLO-World), LLM energy studies | halfacree_agxorin, williams2026litevlah, park2025openvocabedge, khatib2026edgecontinuum |
| Raspberry Pi 5 | Raspberry Pi | (CPU only) | VideoCore VII 800 MHz | 2/4/8/16 GB LPDDR4X-4267 | 5 V / 5 A supply | US$50–120 | YOLOv8 (not real-time), SLM energy | raspberrypi2025pi5brief, rey2025yolo, islam2025slmenergy |
| RPi 5 + AI Kit (Hailo-8L) | Raspberry Pi / Hailo | 13 TOPS | — (NPU over PCIe 3.0 x1) | uses Pi RAM | n/a | US$70 (kit) | CNN vision only: detection, segmentation, pose | patuck2024aikit |
| RPi 5 + AI HAT+ 2 (Hailo-10H) | Raspberry Pi / Hailo | 40 TOPS (INT4) | — (NPU) | 8 GB on-board RAM | n/a | US$130 | LLMs at 1–1.5B (Qwen2.5-1.5B, Llama3.2-1B), unnamed VLM demo | patuck2026aihat2 |
| ModalAI VOXL 2 | ModalAI (Qualcomm QRB5165) | 15 TOPS NPU | Adreno 650 | 8 GB LPDDR5 | n/a | n/a | PX4-native companion (16 g); no VLM papers found | px4_voxl2 |
| Intel NUC 11 Pro (AirHunt) | Intel | n/a | n/a | 8 GB (as stated in paper) | n/a | n/a | Planning onboard; VLM via cloud | chen2026airhunt |
| Laptop / workstation GPUs | NVIDIA (RTX 4070, RTX 3080 Ti) | n/a | discrete GPU | 16 GB (3080 Ti, as stated) | n/a | n/a | Offboard VLM/LLM for drones | zhang2025vlfly, lim2025takingflight |

**Argument for the Orin Nano 8 GB Super, drawn from the table:**

1. It is the cheapest board in the set (US$249 dev kit) that runs a 2–4B VLM on a GPU with CUDA/TensorRT/llama.cpp support. The Raspberry Pi options are limited to CNNs (Hailo-8L) or about 1.5B LLMs (Hailo-10H) [patuck2024aikit, patuck2026aihat2].
2. Super mode raised memory bandwidth from 68 to 102 GB/s, the same as the Orin NX [maheshwari2025jetpacksuper]. Bandwidth largely governs decode speed [shin2026smallvlmquant, islam2025slmenergy].
3. Its 25 W budget is far below the AGX Orin's 60 W and fits the X500 payload envelope.
4. The trade-off is memory: 8 GB is shared between the CPU and GPU. NVIDIA's own Edge-LLM dashboard lists only Qwen3-VL-2B, not 4B, for the Orin Nano [nvidia2026trtedgellm], and community estimates put about 4B as the upper limit at Q4 [adsahu2026orinnanoguide]. The thesis's choice of a 4-bit 4B model therefore sits at the edge of what the hardware supports. That is a finding worth reporting, together with the latency difference between 2B and 4B (RQ2).
5. Every published onboard VLM drone stack uses the Orin NX 16 GB, so showing the stack on an Orin Nano is a distinct contribution.

---

## 4. Table 3: Closest works to this thesis

| key | what they built | hardware | sensors | localization | results | relevance / difference to this thesis |
|---|---|---|---|---|---|---|
| **tayar2026vln** (VLN on the Fly) | Modular onboard VLN: VLM picks 1 of 9 grid cells; median depth in the cell is back-projected to a 3-D goal; EGO-Planner B-spline; RAPTOR policy on the FC | MIRA micro-UAV, **Pixhawk 6C**, Jetson **Orin NX** | **D435i** | **OptiTrack** fused by EKF2 | 13/15 (87 %) open-volume; 3/6 cluttered collision-free; VLM 0.79 s median | Same FC and camera. They use a coarse 3×3 grid instead of a bounding box, 3 consecutive agreeing predictions (thesis: 2 of 3) and motion capture (thesis: VIO). They have no search when the target is out of view and no reactive go-around (they rely on EGO-Planner). Their VLM marked visible objects as "not visible" in 44–53 % of frames. |
| **zheng2026onfly** (OnFly) | Dual-agent (decision + monitoring) zero-shot AVLN; semantic-geometric verifier; bearing-aware gate; Fast-Planner receding horizon | Custom quad, PX4, **Orin NX 16 GB** | D435 (640×480), Mid360 LiDAR | LiDAR-based | Sim SR 67.8 % (vs. 26.4 % baseline); collision rate 2.7 %; decision 0.81 s on NX | **Same VLM (Qwen3-VL-4B)**; 2B ablation drops SR to 43.4 %, an argument for 4B. Pixel output instead of box. Needs LiDAR; real flights only qualitative. |
| **lim2025takingflight** | Open-source natural-language agent: PX4 + ROS 2 + Ollama LLM/VLM | Custom quad (0.56 m), **Pixhawk 6C Mini**, **Jetson Orin Nano** | ZED Mini | ZED VIO | Best mission SR 40 % (Gemma3 + Gemma3, sim); real flights with Gemma | Same Orin Nano + Ollama idea, but the models ran on an **offboard RTX 3080 Ti**. Latency was not measured. Only forward + yaw-rate commands. A direct contrast for RQ3 (onboard vs. offboard). |
| **zhang2025vlfly** (VLFly) | LLM instruction encoder + CLIP goal retriever + waypoint planner; velocity commands from monocular camera | Tello EDU + RTX 4070 laptop | Monocular RGB | None (no localization) | 83 % (direct) / 70 % real | Same "fly to the X" task; offboard; no depth safety. |
| **hu2025spf** (See, Point, Fly) | VLM outputs 2-D waypoint + discrete depth label → 3-D displacement; adaptive step size | DJI Tello EDU | RGB only | None | 92.7 % real; adaptive step halves completion time | Cloud VLMs at about 1–3 s per step. The adaptive step and VLM-produced obstacle boxes are reusable ideas. No metric depth. |
| **chen2026airhunt** (AirHunt) | Asynchronous VLM reasoning + semantic-geometric frontier planner for open-set object search | Custom quad, NUC 11 Pro, NxtPX4v2 | Mid360, 3 OAK cameras | LiDAR | Sim SR 73.1 %; 3 real outdoor searches | Search strategy and keyframe selection are directly relevant to the active-sensing search. VLM via cloud. |
| **liu2026openflyv2** | Single onboard VLA (perception + navigation) | Quadrotor, **Orin NX** | Camera (details on abstract page limited) | Not stated on abstract page | 48 % real, 10 Hz | End-to-end alternative; the thesis keeps grounding, safety and control separate and inspectable (like tayar2026vln). |
| **williams2026litevlah** (LiteVLA-H) | 256M VLA with dual-rate scheduler (action 19.7 Hz, semantics 6.7 Hz) | **AGX Orin** | Camera | Not stated | 81.3 % hardware obstacle course | Shows that prefill dominates latency; dual-rate scheduling idea. |

---

## 5. Ideas we could use (actionable)

**Goal extraction from box and depth**

1. **Use a robust statistic inside the box, not the box centre.** VLN on the Fly takes the *median valid depth* inside the selected cell and rejects the goal if no valid depth lies within range. A single-pixel depth was valid in only about 15 % of frames [tayar2026vln]. For the thesis: take the median, or a low percentile to handle occlusion by foreground objects, over a shrunken inner region of the box, and require a minimum fraction of valid pixels.
2. **Clamp the goal altitude** to a narrow band around the current altitude, and reject goals outside a **bounded operating volume** around the take-off point [tayar2026vln].
3. **Cap depth to the trusted sensor range.** OnFly caps at D_max = 7 m and skips refinement beyond it [zheng2026onfly]. The D435i's ideal range is 0.3–3 m [intel_d435i], so a cap of about 3–4 m with "approach then re-query" is justified.
4. **Use a semantic-geometric verifier.** Move the VLM point to a nearby pixel with similar features that also passes a depth-clearance check, with the obstacle mask dilated by 0.2 m [zheng2026onfly]. The same idea applies to box-centre refinement.

**Consensus and handling "not present"**

5. **Add a confidence threshold to the consensus.** VLN on the Fly requires 3 consecutive predictions of the same region, each above a confidence threshold, and keeps one active goal until completion. It suppresses nearby proposals and rapid updates [tayar2026vln]. The thesis's 2-of-3 vote can add the confidence threshold and the "single active goal" rule.
6. **Model false negatives explicitly in the search belief.** Small VLMs say "not visible" for present targets very often (44–53 % in [tayar2026vln]). The active-sensing belief update should use a detection likelihood well below 1 rather than treating "no" as certain.
7. **Evaluate no-target behaviour.** gRefCOCO and RefDrone contain no-target expressions [liu2023gres, sun2025refdrone]; report a no-target accuracy alongside the box IoU.
8. **Treat small and far targets as a known weakness.** Large multimodal models fail on small objects because of resolution limits [sun2025refdrone]. Remedies are to approach and re-query, or to crop or tile the image (LFM2-VL tiles at 512×512 [liquidai2025lfm2vl]).

**Latency handling**

9. **Decouple VLM and control.** OnFly runs a fast decision agent and an asynchronous monitor every 2 s [zheng2026onfly]. AirHunt runs the VLM and the planner asynchronously because a VLM query takes about two orders of magnitude longer than a planning cycle [chen2026airhunt]. LiteVLA-H uses dual-rate scheduling [williams2026litevlah]. For the thesis, the depth guard and velocity loop must never wait on the VLM.
10. **Shrink prefill by reducing image tokens.** Prefill is about 94 % of latency [williams2026litevlah]. TRT Edge-LLM caps visual tokens on Orin (`--max-image-tokens`) [nvidia2026trtedgellm]. Benchmark latency against input resolution on the Orin Nano.
11. **Do not assume INT4 is always faster.** Dequantization can slow decoding [shin2026smallvlmquant]. Compare Q4 against Q8 for the 2B model on the Orin Nano. The NVIDIA dashboards report prefill time and decode rate separately, so the thesis should do the same [nvidia2026trtedgellm].
12. **Query only on informative frames.** AirHunt and OnFly select keyframes by motion thresholds and feature novelty [chen2026airhunt, zheng2026onfly]. The search should query the VLM only after a significant yaw change.

**Search strategy**

13. **Keep the search policy outside the VLM.** VLMs explore poorly and small open models scored near zero in FlySearch, while a grid overlay on the image helped [pardyl2025flysearch].
14. **Separate "where to look" from "is this it".** Keep spatial and semantic beliefs separate and choose views by expected information gain. Weighting semantic uncertainty more heavily reduces VLM queries [srivastava2026spatialsemantic]. This maps directly to the thesis's belief over directions and next-best view.
15. **Score directions by value per cost.** Frontier-style scoring of semantic value against travel distance [chen2026airhunt, yokoyama2023vlfm] can rank yaw directions by expected gain per rotation time.

**Approach and safety layer**

16. **Turn before advancing when the target is off-centre.** OnFly's bearing-aware gate shrinks the forward range as d·exp(−½(θ/(σθ·θ_max))²) when the target is near the image edge [zheng2026onfly]. This is simple to add to the approach controller and also keeps the target inside the D435i's view.
17. **Plan for losing the target in the final metres.** In VLN on the Fly, failures happened when the target left the view near the goal [tayar2026vln]. Commit to the last confirmed 3-D goal once inside a radius, and stop at a stand-off distance above the D435i's minimum depth (about 0.28 m [intel_d435i]).
18. **Treat no-data as blocked and include delays in TTC.** Following PX4 Collision Prevention, directions without data are treated as blocked [px4_collisionprevention]. The braking distance should include sensor delay plus setpoint-tracking delay (`CP_DELAY`, typically 0.1–0.5 s) in the TTC threshold.
19. **Use VFH-style direction selection for the go-around.** Choose the opening closest to the goal direction from a polar histogram [borenstein1991vfh, vanneste2014vfh3d]. A short-term obstacle memory of recently blocked directions avoids oscillation. EGO-Planner [zhou2020egoplanner] is a heavier alternative used in [tayar2026vln].

**PX4 integration**

20. **Follow the OFFBOARD rules.** Stream setpoints before switching mode and keep the stream above 2 Hz (MAVSDK resends at 20 Hz). Configure `COM_OF_LOSS_T` and `COM_OBL_RC_ACT`, and check `is_active()`, because setpoint calls do not fail when a GCS takes over [px4_offboard, mavsdk_offboard].
21. **Set up the VIO pipeline carefully.** Stream `ODOMETRY` at 30–50 Hz and tune `EKF2_EV_DELAY` by minimizing innovations. Run the documented pre-flight axis checks [px4_extpos, px4_vio]. Isaac ROS cuVSLAM needs at least 30 Hz with jitter within ±2 ms and now supports RGB-D [nvidia_isaacrosvslam]. Measure GPU contention with the VLM on the 8 GB board.

**Practical pitfalls and deployment comparison**

22. **Check colour order and frame conventions.** Taking Flight reports RGB8 vs. BGR8 mismatches and NED/ENU transformation issues [lim2025takingflight]. Assert both in code.
23. **Use split computing as an RQ3 baseline.** For the onboard vs. ground-station comparison, the best placement depends on network conditions and image resolution [azimi2026cloudedge]. Adaptive split computing [bhattacharjya2025avery] and edge-vs-cloud streaming results (WebRTC; Qwen2-VL-2B under one second) [ahmad2026vlmedge] give the comparison points.

---

## 6. Could not verify or deliberately left out

- **NVIDIA product pages** (nvidia.com Jetson Orin and Orin Nano Super pages) were blocked during fetching. Orin specs therefore come from the NVIDIA Technical Blog [maheshwari2025jetpacksuper], CNX Software [aufranc2024orinnanosuper] and Hackster [halfacree_agxorin]. Cite the official NVIDIA datasheet in the final thesis if possible.
- **Intel RealSense D435i weight** was not listed on the opened product page.
- **Lee (1976)** page range is not verified, and the PubMed page was blocked by a captcha. The DOI, author and title are verified.
- **PX4 ICRA 2015 paper:** the middle author (D. Honegger) is not confirmed on an opened page.
- **Venues not confirmed:** CoRL for SPF, ECCV for [yu2016refcoco], ECCV 2024 for Grounding DINO and CVPR 2024 for YOLO-World. They are cited as arXiv.
- **No citable sources were found or opened for:** Gemma 4 (it appears only as a model name in the TRT Edge-LLM list), Qwen3.5 (used by [tayar2026vln], with no technical report opened) and an LFM2-VL paper beyond the LFM2 report and the blog.
- **Absolute latency figures** for [azimi2026cloudedge], [ahmad2026vlmedge], [chen2023typefly] and [islam2025slmenergy] are in the PDFs. Only the abstracts were read.
- **The Jetson AI Lab GenAI benchmark page** [jetsonailab_genaibench] gives a vLLM Llama-3.1-8B w4a16 result (44.2 tok/s, TTFT 32 ms) without naming the device, so it is not used in the tables.
- **Not found:** any peer-reviewed study of sim-to-real transfer for PX4/Gazebo VLM pipelines.
