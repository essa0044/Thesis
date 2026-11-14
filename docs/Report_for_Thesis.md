# Report for Thesis

Oct 29, 2026 · @essa ahmad

Running record of the NeuroDrone MSc work: every design decision, software version, dataset, test and result so far. New work gets added here; the thesis will be written from it. Last update 5 Nov 2026: sections 11–15 added (all 32 EvTTC recordings, SNN design and resolution sweep, newer VLMs and two-step strategies, ODA real flights, TartanAir); the old sections 11–14 are now 16–19. Section 10 was added on 31 Oct, section 9 on 30 Oct.

**New thesis direction (decided 5 Nov, evening).** The SNN is dropped from the thesis to keep the scope manageable. The thesis system is now: Qwen3-VL on a Jetson Orin Nano picks the goal from the RealSense D435i RGB image, a classical depth-based obstacle check on the same camera's depth image stops the drone, and PX4 on the Pixhawk 6C flies it (X500 V2). The SNN and event-camera work (sections 4, 7–12, 14, 15) stays as a personal side study in simulation only: is it really better than a classical detector? Simulation moves to PX4 SITL + Gazebo (`gz_x500_depth`) on the Ubuntu PC (section 15.2).

## 1. Thesis question and system

The thesis tests whether a small drone can run a vision-language model (VLM) onboard, with an event camera and a spiking neural network (SNN) as a fast safety reflex. Working title: *Onboard vision-language control for a small drone with a neuromorphic safety reflex: a feasibility study on edge hardware.* A multi-drone (swarm/MCP) variant was considered and parked as future work.

**Research questions**

1. RQ1: Can a small quantised VLM on a Jetson Orin-class board find a named target fast and accurately enough to steer by?
2. RQ2: How much mission time does VLM latency cost when reflexes handle obstacles, and does it ever cost safety?
3. RQ3: How early and how reliably does the SNN reflex detect approaching obstacles, and which ones does it miss?
4. RQ4: How much power does each part draw on the same board?

**Division of labour.** PX4 on the flight controller keeps the drone stable; the onboard computer only sends velocity setpoints over MAVLink in OFFBOARD mode, and the safety pilot can take OFFBOARD away at any time. The SNN looming reflex (event camera, \~100 Hz) and a depth-camera reflex stop or sidestep; the VLM picks the goal every 1–10 s. A slow or dead VLM only costs progress; a silent reflex makes the drone hold still, then land. This is the same split as the supervisor's reference work ([Paredes-Vallés et al.](https://arxiv.org/abs/2303.08778)), whose spiking network sends thrust and attitude commands to PX4; ours commands one level higher because the SNN is a reflex, not a full controller.

&#91;embedded content: planned onboard stack · 2 cameras, 3 modules, 1 coordinator, PX4\]

Only the SNN reflex and the VLM have been measured so far (on public data, sections 6 and 8); the depth reflex, coordinator and PX4 link exist in code and the synthetic simulator only.

**Why an SNN and not a classic event algorithm.** The looming detector computes the same thing either way: outward edge motion around the image centre. The thesis keeps it as an SNN because the detector is then event-driven end to end, its cost can be counted in spikes and synaptic operations (the numbers a neuromorphic chip such as SynSense Speck or BrainChip Akida bills for), and it could later move off the Jetson onto such a chip. The fair comparison is therefore the fully spiking detector against the earlier hybrid (non-spiking front end) on the same events; both are kept in the code (section 4).

**Planned hardware** (nothing bought yet; prices checked 12 Oct): NVIDIA Jetson Orin Nano Super (€559), iniVation DVXplorer Micro event camera 640×480 (€2,400), Intel RealSense D435i, Holybro X500 V2 kit with Pixhawk 6C, H-Flow optical-flow sensor. Bench phase €3,533, flight phase €1,573. Estimated all-up mass 1,681 g, thrust-to-weight 2.7, about 11 min of hover.

## 2. Software, hardware and versions

All benchmark numbers in sections 6, 8, 9 and 10 come from one workstation on 29–31 Oct 2026 (the weekend runs added the q8\_0 and bf16 tags of the three Qwen3-VL instruct models and qwen3-vl:30b-a3b-instruct and qwen3-vl:32b-instruct at q4\_K\_M, all from the Ollama library); the environment was captured from the machine itself (`results/environment.txt`).

| Component | Version / model | Notes |
| --- | --- | --- |
| Code | `neurodrone` 0.1.0 (Python package) | 128 automated tests pass on the workstation (147 s); 136 with the 30 Oct additions (run in the cloud workspace) |
| Workstation | AMD Ryzen Threadripper PRO 5955WX (16 cores), 128 GB RAM, NVIDIA RTX 6000 Ada (48 GB) | university PC; no admin rights, no WSL |
| OS / GPU driver | Windows 11 Education 10.0.26200; NVIDIA driver 546.12 | driver too old for Ollama ≥ 0.30 on CUDA; cannot be updated (no admin rights), so every VLM latency here is on Vulkan |
| Python | 3.14.6 (64-bit), venv | numpy 2.5.4, h5py 3.16.0, hdf5plugin 7.1.0, pyarrow 26.0.0, pandas 3.1.0, pillow 13.0.0, requests 2.34.2, huggingface\_hub 2.3.0, PyYAML 6.0.3, pytest 9.2.0 |
| VLM server | Ollama 0.40.13, Vulkan backend | `OLLAMA_VULKAN=1`, CUDA hidden, context 8192, 1 request at a time, all models kept loaded |
| Models | qwen3-vl:2b-instruct (afc393e3572c, 1.9 GB), qwen3-vl:4b-instruct (84f7ed8042fa, 3.3 GB), qwen3-vl:8b-instruct (70e58e37beb1, 6.2 GB), qwen2.5vl:3b (fb90415cde1e, 3.2 GB, q4\_K\_M), qwen2.5vl:3b-fp16 (0aeee3542efe, 7.5 GB) | IDs are Ollama digests. The plain `qwen3-vl:2b/4b/8b` tags are the *thinking* variants and were not used |
| Tried, not used | Ollama 0.24.0 (portable, CUDA) | runs Qwen2.5-VL but crashes on Qwen3-VL (`GGML_ASSERT` in llama-context.cpp) |

**Earlier machines.**

- 11 Oct: test suite (85 tests) on Python 3.13, NumPy 2.5, x86.
- 13 Oct: first real VLM test on the laptop: Intel i5-12500H, RTX 3050 Laptop (4 GB), Ollama 0.33.3, qwen2.5vl:3b (Q4\_K\_M).
- 16 Oct: synthetic-simulator comparison of the two detectors (section 4), on this cloud workspace's CPU.

**Scripts written for the benchmarks** (all in `neurodrone/scripts/`; the first three have tests on fake data in the real file formats):

- `get_datasets.py`: downloads every dataset, resumable, Drive files via Drive's direct-download URL.
- `bench_grounding.py`: VLM grounding benchmark (section 5).
- `eval_events.py`: `inspect`, `run`, `sweep` for EvTTC and MVSEC (section 7).
- `report_results.py`: writes `results/REPORT.md` from the raw result files.
- `wait_models.py`: waits until Ollama lists the needed models.
- `select_beta_cv.py`: leave-one-sequence-out choice of a detector's setting (section 9.1).
- `analyse_onsets.py`: labels every alarm onset by phase and by the ego car's speed and rotation (section 9.2).
- `neurodrone/ttc_classical.py`: the classical baseline detector (section 9.3); `day/jobs/extract_slices.py` cuts the training slices it was developed on.

**How the workstation was driven.** No shell was available on the PC from this session. A small job runner (`day/runner.ps1`, started by double-clicking `1_runner.bat`) runs every new `.cmd` file placed in `day/jobs` and logs it to `results/logs/jobs/`; all runs today went through it.

## 3. Datasets

Every result so far comes from five public datasets with their own labels; no self-made or self-labelled images are used, so every number can be re-run by anyone. All files were downloaded on 29 Oct 2026 from the authors' own hosting by `scripts/get_datasets.py`.

| Dataset | What it is | Source | Licence | Used | Scored samples |
| --- | --- | --- | --- | --- | --- |
| RefCOCO (UNC splits) | COCO photos + referring expressions, one box each | [lmms-lab-encoder/RefCOCO](https://huggingface.co/datasets/lmms-lab-encoder/RefCOCO) (lmms-eval copy), 1.4 GB | none stated for the annotations (the refer API code is Apache-2.0); COCO images, under Flickr's terms | testA, testB, val | 500 per split (+50 calibration), 250 in the size test |
| gRefCOCO | RefCOCO images + no-target and multi-target expressions (GRES, CVPR 2023) | [FudanCVL/gRefCOCO](https://huggingface.co/datasets/FudanCVL/gRefCOCO), 179 MB | none stated (checked 30 Oct); COCO images, under Flickr's terms | val | 252 multi-target + 248 no-target |
| RefDrone | VisDrone2019 drone photos with referring expressions; single, multi or no target | [sunzc-sunny/RefDrone](https://huggingface.co/datasets/sunzc-sunny/RefDrone), 887 MB | CC BY 4.0 | test (single- and no-target only) | 510 targets + 146 no-target; 84 % of targets cover < 1 % of the image |
| EvTTC | car-mounted Prophesee EVK4 event cameras (1280×720) approaching a dummy car or pedestrian; time-to-collision ground truth from lidar + GNSS/INS at 100 Hz | [nail-hnu.github.io/EvTTC](https://nail-hnu.github.io/EvTTC/) (Google Drive), about 65 GB used | BSD-3-Clause-Clear | 12 of 32 sequences (events, left camera); ground truth of all 32 | 2 training + 10 held-out approaches |
| MVSEC indoor\_flying | hexacopter flights in a room, DAVIS346 event camera (346×260), Vicon pose + depth ground truth | [daniilidis-group.github.io/mvsec](https://daniilidis-group.github.io/mvsec/) (Google Drive), 20 GB | CC BY-SA 4.0 | flights 1–4 (4.5 min in total) | false-alarm rate during real drone flight |

**EvTTC sequences used.** Training: CCRs-1-low-100%, CCRs-1-medium-100%. Held-out: CCRs-2-low-100%, CCRs-3-low-100%, CCRs-1-low-0% (near miss: the target is in the next lane), CCRs-side-low, CPNA-low (pedestrian), CCRs-1-low-50%, CCRs-2-medium-100%, CCRm-low-100%, CPLA-low (pedestrian), CCRs-side-medium. The ground truth shows that 100 %, 50 % and side sequences brake to about 3 m from the target (collision course), while 0 % sequences pass it 7 m away.

**Sampling.** Each VLM task draws its samples with a fixed seed (`seed 0`, keyed by task name); the exact sample list is saved next to the results (`samples__<task>__seed0.json`). The first 50 samples of each task are a calibration set that is never scored (section 5).

## 4. Looming detectors

Three detectors look for the same cue (the third, a classical baseline added on 30 Oct, is described at the end of this section), an obstacle on a collision course growing in view (edges moving outward on both sides of the image centre), and both are in the code: the fully spiking detector (`neurodrone/snn_spiking.py`, default, `snn.impl: spiking`) and the earlier hybrid (`neurodrone/snn.py`, `snn.impl: hybrid`), kept unchanged as the baseline. None of them needs training.

| Layer | Spiking detector (16 Oct) | Hybrid baseline |
| --- | --- | --- |
| 0 input | one spike per camera event (brighter / darker) | same events |
| 1 motion | coincidence neurons (time-difference encoders, Milde et al. 2018), 2 per pixel × polarity × motion baseline: a neighbour's spike on the image-centre side opens a 2–150 ms window, the pixel's own spike fires OUT; IN is the mirror. 77,856 neurons at 128×96, \~1.9 M at 640×480. Exact per event | timestamp comparisons in NumPy, packet by packet |
| 2 region | 9 leaky integrate-and-fire expansion neurons (left/centre/right thirds + left and right halves of 3 strips); OUT pushes up, IN pushes down, leak 50 ms; adaptive threshold theta0 + beta × recent region activity | Poisson z-score per sector, then leaky integrate-and-fire neurons, threshold theta (z units) |
| 3 alarm | 3 coincidence neurons, one per strip: left- and right-half expansion spikes within 20 ms | min(z\_left, z\_right) |
| readout | alarm = at least 2 alarm spikes within \~70 ms | same rule |

**Key settings** (`config/default.yaml`): spiking theta0 = 6, beta = 0.12, tau 50 ms, 1 ms clock, coincidence 20 ms, rate\_alarm 15; hybrid theta = 4. Both were tuned on the synthetic simulator with a 1.5–2× margin above the false-alarm cliff. An inhibitory "veto" layer was tried and removed (no effect in any test).

**Properties that matter for the thesis.**

- The spiking detector gives the same answer however the event stream is cut into packets (6.19 s first-alarm TTC on a 3 m wall under three cuttings); the hybrid gives 4.42–5.24 s on the same stream.
- Its cost can be counted for neuromorphic hardware: per second of synthetic approach flight, 9,816 events → 11,628 layer-1, 91 layer-2 and 36 layer-3 spikes, \~102,000 synaptic operations. Converted to an energy floor in section 9.6.
- On real event rates the adaptive term dominates the threshold: a layer-2 neuron fires when roughly (OUT − IN) / (OUT + IN) > beta, so beta is the one setting that matters and theta0 hardly does (section 8).

**Synthetic-simulator results (16 Oct, 128×96, 1.5 m/s, 5 noise seeds).** These check the logic; they do not predict real performance.

| Obstacle | Spiking first-alarm TTC (min / median / max) | Hybrid |
| --- | --- | --- |
| textured wall 3 m | 6.2 / 6.2 / 7.1 s | 4.4 / 4.6 / 5.5 s |
| textured box 1.5 m | 4.1 / 4.2 / 4.4 s | 3.3 / 3.5 / 3.5 s |
| blank wall 3 m | 7.3 / 7.3 / 7.3 s | 5.6 / 5.6 / 6.0 s |
| pole 0.4 m | 2.7 / 3.5 / 3.5 s, missed in 1 of 5 | 2.5 / 2.5 / 2.7 s |
| pole 0.15 m | missed | missed |

No false alarms for either detector in 7 scenes × 5 seeds (hover with 3° shake, forward flight with 5° shake, 10× sensor noise, 30°/s turns, an object passing to the side). In the closed loop both detectors end all 9 scenarios the same way; the spiking one stops further from its target (6.3 m vs 4.9 m). Weakness: attitude shake delays the spiking alarm more (2 m box: 8.7 m at 0°, 1.7 m at 3°; hybrid 6.1 m and 2.8 m); gyro derotation is the planned fix. CPU per 10 ms packet: spiking 0.47 ms vs hybrid 0.13 ms at 128×96, 3.6 ms vs 1.5 ms at 640×480. One coordinator setting changed with it: `arrive_w_frac` 0.20 → 0.15.

**Third detector, added 30 Oct: a classical baseline** (`neurodrone/ttc_classical.py`, `snn.impl: classical`). A standard non-spiking method from the event-vision literature that computes the same cue: normal optical flow from plane fits to the time surface (Benosman et al. 2014), then a least-squares fit of the expansion rate, i.e. the inverse time to contact (Clady et al. 2014); alarm when the estimated TTC stays below τ. It exists to answer whether the SNN does better than a normal algorithm on the same events. Its tests: it registers as a detector, handles an empty packet, alarms before contact on a synthetic approach, and stays silent under pure rotation. Design, development and results: sections 9.3–9.4.

## 5. VLM benchmark: method

The VLM is used only for visual grounding: given a target description it returns a box, and the coordinator turns the box into a bearing. The benchmark (`scripts/bench_grounding.py`) measures exactly that, through the same OpenAI-compatible `/v1/chat/completions` call and the same parser (`neurodrone/vlm.py`) the drone uses.

**Prompts.**

- *Drone prompt* (the one that flies). System: "You are the visual grounding module of a small drone. You reply with ONE JSON object and nothing else." User: the image, then "Mission target: … If the target is visible, give its bounding box in normalised image coordinates … as \[x1, y1, x2, y2\]. If it is not visible, say found=false. Reply with ONLY JSON, either {"found": true, "box": \[…\]} or {"found": false}."
- *Native prompt* (harness check, RefCOCO only): Qwen's own grounding format, `[{"bbox_2d": [x1, y1, x2, y2], "label": …}]`, "if it is not in the image, output \[\]".

**Request settings.** Temperature 0, at most 256 output tokens, one request at a time, 300 s timeout. Images are resized keeping the aspect ratio so the longest side is at most 1024 px (sides rounded to multiples of 28) and sent as JPEG, quality 90. Failed requests are recorded and retried on the next run; a model whose first 10 requests all fail is skipped.

**Coordinate calibration.** The models do not answer in the 0–1 coordinates the prompt asks for: Qwen3-VL answers in 0–1000, and Qwen2.5-VL in pixels of the image *as the server resized it* (Ollama scales every image to at least 1024 image tokens, so a 644×420 frame is answered in 1120×728 pixels). The first 50 samples of each task pick each model's convention by best mean IoU among five candidates (0–1000, pixels sent, 0–1, Qwen2.5 resized pixels, Qwen3 resized pixels); that choice is frozen and the 50 samples are never scored.

**Parser rules** (shared with the drone): the first balanced JSON object wins; for a repeated `"box"` key the first value is kept; a cut-off answer still yields `found` and its first box; a bare `[x1, y1, x2, y2]` is a box; `[]` means not found.

**Metrics.**

- Accuracy at IoU ≥ 0.5 on samples whose target is present (the standard RefCOCO metric); for gRefCOCO multi-target expressions, a box matching any one of the referred objects counts.
- Centre-in-box rate (enough to steer by), horizontal error of the box centre.
- "Not found" said on a present target (refusals).
- No-target correct: share of no-target samples answered not found.
- Accuracy by target size (< 1 %, 1–10 %, > 10 % of the image area).
- Valid-answer rate, latency median and 90th percentile, prompt tokens.
- 95 % confidence intervals: Wilson score interval.

**Runs.** A1–A4: drone prompt, 5 models × 5 tasks, 500 scored samples each (RefDrone 656). Harness check: native prompt, 3 models × 3 RefCOCO splits. A5 image size: longest side 1024 / 768 / 512 / 384 px, Qwen3-VL 2B and 4B, RefCOCO testA and RefDrone, 200 scored targets each, same samples at every size. Every raw answer is kept in JSONL, so a run can be re-scored without asking the model again; all summaries were re-scored with the final parser.

## 6. VLM results (29 Oct)

Qwen3-VL-4B-instruct is the best grounding model for the drone: 70 % correct on drone photos where every other model stays at or below 62 %, at 0.57 s per image on the workstation. All numbers: drone prompt, 500 scored samples per task (RefDrone 656), Ollama 0.40.13 on Vulkan.

&#91;embedded content: results/vlm/full/summary.json · 29 Oct 2026 · 5 models × 5 tasks\]

RefCOCO bars are the mean of testA, testB and val; per-split numbers and 95 % CIs are in `results/REPORT.md`.

**Behaviour per model (drone prompt).** Refusals = "not found" on a present target.

| Model | Refusals, RefCOCO | Refusals, RefDrone | No-target correct, gRefCOCO / RefDrone | RefDrone, targets < 1 % of image | Latency median (s) |
| --- | --- | --- | --- | --- | --- |
| qwen3-vl:2b-instruct | 8–9 % | 44 % | 82 % / 80 % | 36 % | 0.37 |
| qwen3-vl:4b-instruct | 1–2 % | 12 % | 69 % / 49 % | 68 % | 0.57 |
| qwen3-vl:8b-instruct | 3–4 % | 21 % | 82 % / 76 % | 58 % | 0.84 |
| qwen2.5vl:3b (q4\_K\_M) | 33–42 % | 52 % | 89 % / 81 % | 32 % | 0.49 |
| qwen2.5vl:3b-fp16 | 41–49 % | 61 % | 92 % / 90 % | 27 % | 0.47 |

**Harness check: drone prompt vs each model's own grounding prompt, RefCOCO testA / testB / val (acc %).**

| Model | Drone prompt | Native prompt | Refusals, native | Published |
| --- | --- | --- | --- | --- |
| qwen2.5vl:3b (q4\_K\_M) | 58.6 / 46.0 / 54.2 | 53.0 / 56.0 / 53.4 | 33–41 % | 91.7 / 84.0 / 89.1 ([Qwen2.5-VL report](https://arxiv.org/abs/2502.13923), Table 6) |
| qwen2.5vl:3b-fp16 | 52.4 / 42.4 / 49.0 | 63.2 / 60.8 / 60.6 | 27–32 % | same model |
| qwen3-vl:2b-instruct | 81.2 / 76.6 / 79.0 | 83.6 / 82.6 / 85.2 | 0 % | not checked |

**Image size (A5), acc % at longest side 1024 / 768 / 512 / 384 px, same samples.**

| Model | RefCOCO testA | RefDrone |
| --- | --- | --- |
| qwen3-vl:2b-instruct | 78.0 / 78.0 / 74.5 / 74.0 | 40.3 / 41.7 / 42.7 / 39.8 |
| qwen3-vl:4b-instruct | 86.5 / 86.5 / 87.0 / 86.5 | 65.4 / 65.9 / 64.0 / 57.8 |

**What the results say.**

- Qwen2.5-VL-3B's gap to its published scores is refusals, not wrong boxes: when it answers, 84–89 % of its boxes are correct (native prompt), close to the published 84–92 %. Unlike the published evaluation, both our prompts allow a "not found" answer, and in this Ollama build Qwen2.5-VL uses it on 27–49 % of present targets. Its high no-target scores are the same habit seen from the other side.
- Qwen3-VL refuses rarely and finds most targets. 8B is not better than 4B except on RefCOCO testB (82.4 vs 80.8 %), and is 1.5× slower.
- The drone prompt costs Qwen3-VL-2B 2–6 points against its native prompt on RefCOCO.
- The accuracy / hallucination trade-off is clearest on RefDrone: 4B finds 70 % of targets but answers "found" on half of the absent ones; 2B is cautious (80 % no-target correct) but misses most targets. The mission logic should confirm a detection over several frames before acting.
- Image size between 384 and 1024 px barely changes RefCOCO; on drone photos 4B loses 8 points at 384 px. Ollama upsamples small images to at least 1024 image tokens, so smaller images do not cut latency on this server.
- Latency is Vulkan on an RTX 6000 Ada; it ranks models against each other but says nothing about Jetson speed (experiment 5.1).

## 7. Event-camera benchmark: method

Both detectors run on identical 10 ms event packets from real recordings (`scripts/eval_events.py`) and are scored against each dataset's own ground truth.

**EvTTC handling.**

- Events from `prophesee/event_cam_left/{x, y, t, p}` (t in µs); left camera 1280×720, fx = 1263 px (about 54° wide by this calibration; the EvTTC paper quotes 41°, but the events follow the calibration).
- Coordinates are divided by 3 (to 426×240, about 8 px per degree), close to the DVXplorer Micro the drone would carry. Event counts are not reduced: the 3× binning keeps the HD sensor's event rate (1.6–9.5 M events/s), higher than a 640×480 sensor would give.
- Ground truth `gt_ttc.csv` (tab-separated, no header: index, time s, distance m, relative speed m/s, TTC s) is on the same clock as the events (offset 0, checked per sequence). It covers only the approach, from TTC 10–24 s down to about 1 s.

**MVSEC handling.** Events from `davis/left/events` (346×260, timestamps in s). TTC is computed from ground truth: median depth in the central 30 % of `depth_image_rect`, divided by the forward speed along the optical axis from `pose` (world velocity rotated into the camera frame). Alarms outside windows with TTC < 3 s count as false.

**Metrics (EvTTC).**

- First-alarm TTC: ground-truth TTC at the first alarm onset with TTC ≤ 6 s (higher = earlier warning).
- Early onsets: alarm onsets with TTC > 6 s or before the ground truth starts.
- Alarm share in the last 3 s before contact; CPU time per 10 ms packet (Python, workstation).
- Categories: collision course (100 %, 50 %, side), near miss (0 %), pedestrian dummy (CP…).

**Train / test split and selection rule.** Settings are tuned on two training sequences only (CCRs-1-low-100%, CCRs-1-medium-100%). The rule, fixed in code (`eval_events.choose()`) before any held-out sequence was run: keep settings with no early onset and an alarm on both training sequences; among them take the earliest worst-case first-alarm TTC; ties go to the less sensitive setting. Grid: spiking theta0 ∈ {6, 24, 96} × beta ∈ {0.12, 0.2, 0.3, 0.4, 0.5, 0.7}, hybrid theta ∈ {4, 6, 8, 12, 16}. A first, narrower grid (beta 0.08–0.2) found no qualifying setting and is kept in `results/events_sweep_narrow_grid`.

**Runs.** Default settings on all 12 EvTTC sequences + 4 MVSEC flights; the chosen settings on the same; the chosen settings with the image rows limited to 10–60 % (road removed, B5); and, as a sensitivity analysis rather than tuning, the spiking detector at beta 0.16, 0.2, 0.25, 0.3, 0.35 (theta0 96) on everything.

## 8. Event-camera results (29 Oct)

The spiking detector has a usable operating range on real event data (beta about 0.2: 8 of 9 car approaches alarmed about 5 s before contact, 1.6 false alarms per minute on drone flights), while the hybrid baseline never alarms in the last 6 s before contact at any threshold. The setting the fixed rule picked on two training sequences does not carry over.

**Tuning on the training sequences.** Only one spiking setting qualified: theta0 96, beta 0.4 (first alarms at TTC 5.9 s and 4.3 s, no early onset). beta ≤ 0.3 always gave early onsets; beta ≥ 0.5 never alarmed. theta0 mattered only at the margin: at beta 0.4, raising it from 24 to 96 removed the last early onset. No hybrid setting qualified (the rule fell back to theta 6): it alarmed only early.

**Every sequence, default and chosen settings.** First-alarm TTC in s / early onsets; "–" = no alarm at TTC ≤ 6 s.

| Sequence | Category | Split | Events (M) | Spiking beta 0.12 (default) | Spiking beta 0.4 (chosen) | Hybrid theta 4 (default) |
| --- | --- | --- | --- | --- | --- | --- |
| CCRs-1-low-100% | collision | train | 44 | 4.1 / 3 | 5.9 / 0 | – / 4 |
| CCRs-1-medium-100% | collision | train | 49 | 2.4 / 4 | 4.3 / 0 | – / 6 |
| CCRs-2-low-100% | collision | test | 62 | 3.1 / 0 | – / 0 | – / 4 |
| CCRs-3-low-100% | collision | test | 131 | 6.0 / 9 | – / 0 | – / 11 |
| CCRs-1-low-50% | collision | test | 46 | 5.1 / 7 | – / 14 | – / 10 |
| CCRs-2-medium-100% | collision | test | 96 | 1.9 / 1 | – / 0 | – / 5 |
| CCRm-low-100% | collision | test | 192 | 5.3 / 25 | – / 0 | – / 11 |
| CCRs-side-low | collision | test | 59 | 3.2 / 7 | – / 14 | – / 5 |
| CCRs-side-medium | collision | test | 54 | 2.2 / 10 | – / 0 | – / 5 |
| CCRs-1-low-0% | near miss | test | 42 | 5.2 / 5 | – / 6 | – / 5 |
| CPNA-low | pedestrian | test | 141 | 3.9 / 21 | – / 0 | – / 6 |
| CPLA-low | pedestrian | test | 144 | 3.9 / 30 | – / 0 | – / 14 |

**MVSEC drone flights, false alarm onsets per minute (flights 1 / 2 / 3 / 4).** Spiking beta 0.12: 20.5 / 9.2 / 17.2 / 9.1. Spiking beta 0.4: 0 on all four. Hybrid theta 4: 46.1 / 34.6 / 32.5 / 51.5; theta 6: 43.5 / 35.3 / 26.8 / 15.1.

**Trade-off over beta (sensitivity analysis on all sequences, not tuning).**

&#91;embedded content: results/events\_beta\_curve, events\_default, events\_tuned · 29 Oct 2026 · 12 EvTTC sequences, 4 MVSEC flights\]

The median first-alarm TTC stays between 4.5 and 5.9 s from beta 0.16 upward (3.2 s at 0.12). The near-miss sequence alarms at every beta. Choosing beta from these curves would be tuning on the test set, so it is picked by cross-validation instead (section 9.1).

**Further findings.**

- Road removed (rows 10–60 %, B5) with the chosen settings: the spiking detector is silent on every sequence; the hybrid alarms only on the two pedestrian sequences (TTC 6.0 s and 5.7 s), with 4–5 early onsets each.
- Early onsets at the default beta: 122 in total, 68 of them before the ground truth starts and 54 at TTC > 6 s; 43 more come after the ground truth ends and are not scored. Checked on 30 Oct: the early onsets happen only while the car is driving (section 9.2).
- The near miss alarms at every beta tested: the detector cannot tell a car in the next lane from one on its own path.
- CPU per 10 ms packet (Python, workstation): spiking 2–17 ms, hybrid 1–9 ms, growing with the event rate. At the highest rates (CCRm-low-100%, 9.5 M events/s) the spiking detector needs 16.5 ms per 10 ms packet, slower than real time in Python on this PC.

## 9. Second round of results (30 Oct)

Same workstation and software as section 2, same public recordings as section 7; no new data. These runs close the open checks listed on 29 Oct and add a classical baseline.

**Short answer.** On the same recordings and with the same selection rule, the spiking detector beats a standard classical event-camera method (normal optical flow + expansion fit) at every setting tried. At the setting each detector's own cross-validation picked, both catch 8 of 9 car approaches, but the spiking detector gives 1.6 false alarms per minute on the drone flights and the classical one 10.9. This is a result about **detecting** an approaching obstacle, not about low-level flight control: in this design PX4 does the low-level control and the detector only decides when to brake (section 1). Still open: at the cross-validated setting the spiking detector misses both pedestrians (the classical one catches them), the near miss still alarms, and most early alarms happen while the car is already driving towards the target.

### 9.1 Choosing beta by cross-validation

**Why.** On 29 Oct the rule picked beta on two training sequences and the choice (0.4) was silent on all held-out approaches (section 8). Picking beta from the trade-off curve over all sequences would be tuning on the test set.

**Method** (`scripts/select_beta_cv.py`). Leave one sequence out: each of the 12 EvTTC sequences is held out in turn, beta is chosen on the other 11, then scored on the held-out one. Candidates: beta 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4 (theta0 96), one run each over every sequence (`results/events_beta_curve*`). Rule, fixed in code before running: among the betas whose alarm reaches at least 85 % of the training collision approaches (an onset at TTC ≤ 6 s), take the one with the fewest early onsets on the training sequences; ties go to the larger (less sensitive) beta. MVSEC is never used for the choice; it is scored at the beta chosen most often.

**Result** (`results/events_cv/cv.json`). Beta 0.2 was chosen in 8 of 12 folds, 0.22 and 0.25 in 2 each, so the choice is stable.

| Held out | What it shows |
| --- | --- |
| 9 car approaches (collision course) | 8 alarmed; first alarm 1.8–5.9 s before contact, median 5.2 s. Missed: CCRs-3-low-100% (no alarm at all) |
| 2 pedestrian dummies | 0 alarmed |
| 1 near miss (car in the next lane) | alarmed, at TTC 2.3 s |
| early onsets, all 12 held-out sequences | 41 |
| MVSEC drone flights at beta 0.2 | 7 false alarm onsets in 4.48 min = 1.56 per minute |

This is the number to quote for the spiking detector: 8 of 9 car approaches caught about 5 s before contact with 1.6 false alarms per minute on drone flights, with beta picked without looking at the sequence being scored.

### 9.2 When the early alarms happen: the ego-motion check

**Method** (`scripts/analyse_onsets.py`). Every alarm onset in the saved traces is labelled by when it happens (before the ground truth starts, at TTC > 6 s, at TTC ≤ 6 s, after the ground truth ends) and by what the ego car is doing at that moment, from EvTTC's own data: speed from the GNSS/INS (`integratedNavigation/data/velocity`; still = below 0.5 m/s) and rotation from the lidar's IMU (`livox/imu/angular_vel`; turning = above 0.1 rad/s). The test is the onset **rate** per minute spent in each state outside the scored window, not the raw count, because the car drives most of the time anyway.

**Result** (`results/onsets/onsets.json`, spiking detector, all 12 sequences).

| beta | early onsets | while moving (1.31 min) | while still (0.97 min) | median ego speed at an early onset |
| --- | --- | --- | --- | --- |
| 0.12 | 74 | 72 (55 per min) | 2 (2.1 per min) | 6.3 m/s |
| 0.2 | 26 | 26 (20 per min) | 0 | 5.5 m/s |
| 0.4 | 34 | 34 (26 per min) | 0 | 5.6 m/s |

- **Early alarms happen while the car is driving, almost never while it stands still.** At beta 0.2, 24 of the 26 early onsets fall while the target is already being approached (ground truth running, TTC > 6 s), 2 before.
- **Not yet separated:** forward motion makes the whole static scene expand (background looming), but an onset at TTC 7–10 s can also be the real target seen early. Both fit these numbers. Separating them needs the image position of each alarm compared with the target, or the ego speed fed into the detector (a drone knows its own velocity from PX4).
- **Rotation could not be tested:** the car never turned faster than 0.1 rad/s outside the scored window in these recordings. Derotation for a shaking drone stays untested on real data.
- **The hybrid baseline does the opposite** (`results/onsets_hybrid_default`, theta 4): 86 early onsets, 58 of them while the car stands still (60 per min vs 21 per min while moving), and no onset at all inside the ground-truth window. Its z-score reacts to noise in a quiet scene, not to looming, which explains why it never alarmed near contact (section 8).

### 9.3 A classical baseline: optical flow + expansion fit

**Why.** The hybrid baseline is not a fair stand-in for "normal" event-camera vision (9.2). The question "is the SNN better than a normal algorithm?" needs a standard, published, non-spiking method that computes the same cue.

**Method** (`neurodrone/ttc_classical.py`, `snn.impl: classical`; same input, same output as the other two detectors):

1. **Normal optical flow per event** from a plane fitted to the surface of active events (Benosman et al. 2014): for each event, the latest timestamps of its 5×5 neighbourhood (same polarity, within 100 ms) are fitted with t = a·dx + b·dy + c; the flow is (a, b) / (a² + b²) px/s. Only the first event of each burst at a pixel counts (50 ms refractory), at least 4 neighbours per fit, speeds above 3,000 px/s dropped.
2. **Expansion rate** (Clady et al. 2014): for an approaching surface the flow is (p − c) / TTC around the focus of expansion c. The inverse TTC is fitted to all flow vectors of a 10 ms packet at once by least squares (at least 100 vectors, at most 20,000 events per packet).
3. **Alarm** when the smoothed TTC estimate (50 ms) is ≤ τ for 2 packets in a row. τ is its one setting, like beta for the spiking detector.

Same pixels as the spiking detector (rows, minimum radius, focus of expansion at the image centre). One run with saved traces; the alarm for each τ in 1, 2, 3, 4, 5, 6, 7, 8, 10, 12 s is re-derived from the stored estimates (tested to give exactly what a separate run gives).

**How it was built (methods log).** Version 1 used a 500 ms window. On real data its expansion came out negative during approaches: in textured scenes a neighbour the edge has not reached yet still holds the time of an earlier edge, and those stale timestamps flip the fitted direction. A robust refit (drop far-off neighbours) did not fix real data and broke the synthetic test. Variants were then compared only on slices of the two training sequences at TTC 8, 5 and 3 s (`results/slices`), which gave version 2 (100 ms window, no refit). Version 1's results are kept (`results/events_classical_v1`) and not used.

**Its TTC number.** Unlike the spiking detector it outputs a TTC estimate, but a rough one: median error per sequence 35–426 % of the true TTC (median over sequences 62 %), and on the busiest sequences it has an estimate for only 15–41 % of the approach. CPU per 10 ms packet (Python, workstation): median 4.9 ms on EvTTC, 0.9–1.4 ms on MVSEC; the spiking detector 4.5 ms and 0.6–0.7 ms.

### 9.4 Spiking vs classical: same recordings, same rule

Both detectors were run over all 12 EvTTC sequences and the 4 MVSEC flights at each of their settings (spiking: 10 beta values; classical: 10 τ values), and each had its setting picked by the same leave-one-sequence-out rule as in 9.1 (for the classical one, ties go to the lower τ).

&#91;embedded content: results/events\_beta\_curve\*, events\_classical, events\_cv, events\_cv\_classical · 30 Oct 2026 · 12 EvTTC sequences, 4 MVSEC flights\]

| Cross-validated result | Spiking (beta) | Classical (τ) |
| --- | --- | --- |
| Setting picked | beta 0.2 (8 of 12 folds) | τ 4 s (12 of 12 folds) |
| Car approaches caught (of 9) | 8; missed CCRs-3-low-100% | 8; missed CCRm-low-100% |
| First alarm before contact, median (range) | 5.2 s (1.8–5.9) | 4.5 s (1.5–6.0) |
| Pedestrian dummies caught (of 2) | 0 | 2 |
| Near miss alarmed | yes | yes |
| Early onsets, 12 held-out sequences | 41 | 38 |
| False alarms per minute, MVSEC drone flights | **1.6** | **10.9** |
| CPU per 10 ms packet, median EvTTC / MVSEC | 4.5 ms / 0.6–0.7 ms | 4.9 ms / 0.9–1.4 ms |

- **Cars:** for every classical setting there is a spiking setting that catches at least as many car approaches with fewer false alarms on the drone flights (chart). Example: 8 of 9 with no false alarm in 4.5 min of drone flight (beta 0.25) against 8 of 9 at 10.9 per minute (τ 4 s).
- **Pedestrians:** the classical detector at τ 4 s catches both; the spiking one only at beta 0.16 or lower. At beta 0.16 it catches all 9 cars and both pedestrians at 5.1 false alarms per minute, about half the classical rate at τ 4 s (which catches 8 cars). So the pedestrian miss at beta 0.2 comes from the setting, not from a limit of the spiking detector. The rule, fixed in advance, counts only car approaches; a rule that also required pedestrians would pick a lower beta.
- **Early onsets on EvTTC** are about the same (41 vs 38), and **neither** separates the near miss from a real collision.
- **Likely reason (not tested separately):** the flow fit is thrown off by textured scenes and fast motion (stale timestamps; on the busiest sequences valid estimates for only 15–41 % of the approach, 9.3), while the spiking detector only compares spike times of neighbouring pixels and its adaptive threshold scales with the scene's activity.
- **Limits of the comparison:** one classical method family (time-surface flow + global expansion fit), developed only on slices of the two training sequences; a better-engineered or learned method might do better. Both run in Python on a PC. The false-alarm rates rest on 4.5 min of indoor drone flight.

### 9.5 A lower event rate (closer to the drone's camera)

**Why.** EvTTC's HD cameras give 1.6–9.5 M events/s even after 3× binning (section 7); the small camera a drone would carry gives fewer.

**Method** (`eval_events.py --dedup-us`). Per binned pixel and polarity, only the first event in each 1 ms or 5 ms window is kept. Spiking detector at the cross-validated setting (theta0 96, beta 0.2), hybrid at theta 4; all 12 EvTTC sequences (`results/events_dedup_1000`, `events_dedup_5000`). This changes only the event rate, not the sensor's noise, lens or resolution.

| Events kept | Spiking: cars caught (of 9) | Spiking: pedestrians (of 2) | Spiking: early onsets | Hybrid: cars caught |
| --- | --- | --- | --- | --- |
| 100 % (as recorded) | 8 | 0 | 26 | 0 |
| 73 % (1 ms merge) | 8 | 0 | 23 | 0 |
| 43 % (5 ms merge) | 7 | 0 | 30 | 0 |

The spiking detector holds up with less than half the events; the hybrid catches nothing at any rate. The classical detector and the MVSEC flights were not run at the lower rates (to do if the comparison needs it).

### 9.6 Energy floor of the spiking detector

**Method** (`report_results.py`, energy table). The spiking detector counts its synaptic operations. These are multiplied by Loihi's minimum energy per synaptic spike operation, 23.6 pJ (Davies et al. 2018, IEEE Micro; a pre-silicon figure at 0.75 V). Beta 0.2, every sequence (`results/events_beta_curve2/b0.2`).

| Data | Events per s | Synaptic ops per s | Synaptic energy floor |
| --- | --- | --- | --- |
| MVSEC drone flights 1–4 | 0.2–0.4 M | 2.1–4.7 M | 50–110 µW |
| EvTTC car recordings (12) | 1.6–9.4 M | 19–118 M | 0.44–2.8 mW |

**What this is not.** It is a floor for the synapses only. Neuron updates are not counted: the same paper lists about 52 pJ per idle neuron per time step, so with the detector's 1 ms clock the 77,856 neurons of the 128×96 version alone would add about 4 mW if every neuron is updated every step. Spike routing, I/O and the camera are not counted either, and the detector has not been mapped onto a real chip (at 640×480 it needs about 1.9 M neurons, more than one Loihi chip's 131,072 neuron compartments). The figure shows that the synaptic work is small (microwatts on drone-like scenes); it is not a measured or predicted chip power. RQ4 needs the Jetson measurements.

### 9.7 VLM: second seed

**Method.** The same benchmark as section 5 (drone prompt, 1024 px, 500 scored samples per task, 50 calibration samples) with seed 1, which draws a different set of samples (`results/vlm/seed1`). Only the three Qwen3-VL instruct models were re-run; Qwen2.5-VL was not, since its weakness comes from refusals rather than from the sample draw. Same server (Ollama 0.40.13, Vulkan). About 1.5 h of GPU time, finished 12:31.

| Accuracy at IoU 0.5, % (95 % CI) | Seed | Qwen3-VL 2B | Qwen3-VL 4B | Qwen3-VL 8B |
| --- | --- | --- | --- | --- |
| RefCOCO, mean of testA / testB / val | 0 | 78.9 | 85.4 | 85.5 |
|  | 1 | 80.5 | 87.3 | 86.5 |
| gRefCOCO val, targets present | 0 | 83.7 (79–88) | 89.3 (85–93) | 84.5 (80–88) |
|  | 1 | 82.7 (78–87) | 85.9 (81–90) | 77.9 (72–83) |
| RefDrone test, drone photos | 0 | 41.0 (37–45) | 70.4 (66–74) | 62.2 (58–66) |
|  | 1 | 42.2 (38–47) | 68.0 (64–72) | 62.2 (58–66) |
| Median latency per query, s | 0 / 1 | 0.36–0.38 / 0.35–0.40 | 0.56–0.60 / 0.54–0.58 | 0.80–0.85 / 0.79–0.85 |

- **The ranking holds.** 4B is best on drone photos in both seeds (68–70 % vs 62 % for 8B and 41–42 % for 2B), and 4B and 8B tie on RefCOCO in both. The one change: 8B and 2B swap places on gRefCOCO, within their confidence intervals.
- **Run-to-run spread** is up to about 4 points on most tasks and 6.6 points at most (8B on gRefCOCO), consistent with the ±3–4 point CIs of 250–500 samples. Differences smaller than that between models should not be claimed.
- **Latency is reproducible** to within 0.03 s.
- **Conclusion for the thesis:** Qwen3-VL 4B instruct stays the model to fly (RQ1), now backed by two independent sample draws.

### 9.8 What is shown now, and what is not

**Shown, on public data, in Python on a PC:**

- The spiking looming detector works on real event-camera recordings: with beta picked by cross-validation it alarms on 8 of 9 car approaches about 5 s before contact, at 1.6 false alarms per minute on drone flights (9.1).
- It beats a standard classical event-camera method (flow + expansion fit) on the same data: at every classical setting a spiking setting catches at least as many car approaches with fewer false alarms (9.4). It also beats the old hybrid baseline, which reacts to noise, not looming (9.2).
- It keeps working with less than half the events (9.5), and its synaptic work is in the microwatt range on drone-like scenes (9.6, a floor, not a chip measurement).
- The VLM ranking is stable across two sample draws; Qwen3-VL 4B is the pick (9.7).

**Not shown yet:**

- Anything about flight: no closed-loop test in PX4 simulation or on hardware, so nothing yet on collisions avoided, mission time, or RQ2.
- Low-level control: by design PX4 does it; the comparison here is about detection only.
- Pedestrians at the cross-validated setting (0 of 2), near-miss rejection (it alarms), and whether the early alarms come from background expansion or from seeing the target early (9.2).
- Timing and power on the Jetson (RQ1 latency on the real board, RQ4).

## 10. Weekend runs (30–31 Oct)

Same workstation, software and rules as sections 2 and 9. Jobs 090 (GPU, Fri 15:18 – Sat 07:41), 094 (CPU, Sat 07:31 – 08:21) and 093 (downloads). The VLM part is complete. The detector part ran on **17 of the 32 EvTTC recordings**: Google Drive answered "quota exceeded" and let only 5 of the 20 missing recordings through (none after Fri 21:25). Job 093 keeps retrying until Sun \~11:20 and re-runs everything if more arrive; the numbers below would then be updated.

**Short answer.**

1. Qwen3-VL 4B at the default q4 quantisation stays the model to fly: 69 % on all 1,254 drone-photo queries, the same as its q8 and bf16 versions and as the 30B and 32B models, at 0.6 s per query here.
2. Ollama gives the model about the same number of image tokens (\~1,200) whatever image size is sent, so sending larger images does not help small drone targets; cropping the image would.
3. On 17 recordings (11 car approaches, 4 pedestrian dummies, 2 near misses) the spiking detector still beats the classical baseline at every setting, now counting pedestrians too.
4. The spiking detector's CPU cost follows the event rate, not the resolution; the classical one's grows with resolution.

### 10.1 VLM on the complete test sets

**Method.** `bench_grounding.py --full`: every RefDrone test query with one target or none (1,304), every gRefCOCO val ref (4,237), every RefCOCO testA and testB expression (5,657 and 5,095), minus 50 calibration samples each. Drone prompt, 1024 px, q4\_K\_M (`results/vlm/fullsets`).

| Accuracy at IoU 0.5, % | Qwen3-VL 2B | Qwen3-VL 4B | Qwen3-VL 8B |
| --- | --- | --- | --- |
| RefDrone test, 1,254 scored (95 % CI) | 43.9 (41–47) | **69.3 (67–72)** | 63.2 (60–66) |
| of which small targets, < 1 % of the image (921) | 38.5 | **65.9** | 59.4 |
| RefDrone, target absent: answered "not found" (152) | **79.6** | 48.7 | 75.7 |
| gRefCOCO val, 4,187 | 83.8 | **86.9** | 84.0 |
| RefCOCO testA, 5,607 | 83.5 | **90.6** | 89.5 |
| RefCOCO testB, 5,045 | 78.1 | **83.4** | 83.1 |
| Median latency per query | 0.38 s | 0.56–0.61 s | 0.83–0.84 s |

CIs on the gRefCOCO and RefCOCO rows are about ±1 point.

- **The sample estimates were right:** 4B on RefDrone 69.3 % on everything vs 70.4 % and 68.0 % on the two 500-sample draws (9.7).
- **4B's weak spot:** when the named object is not in the photo, it still draws a box in half the cases (48.7 % answered "not found"), against 76–80 % for 2B and 8B. On the drone, a detection must be confirmed over several frames before the drone steers to it (already planned for the parser, section 13).
- **Published check:** the Qwen3-VL technical report gives RefCOCO-avg (RefCOCO, RefCOCO+ and RefCOCOg, bf16, its own prompt) of 85.6 / 89.0 / 89.1 for 2B / 4B / 8B Instruct. Our RefCOCO testA/testB mean (drone prompt, q4, Ollama on Vulkan) is 80.8 / 87.0 / 86.3: same order, 2–5 points lower. Not the same benchmark, so a sanity check of the harness only.

### 10.2 Quantisation: q4 vs q8 vs bf16

**Method.** The q8\_0 and bf16 versions of the three models (`results/vlm/quant`): RefDrone in full, and RefCOCO testA / testB / gRefCOCO val on the seed-0 500-sample sets, i.e. exactly the samples the q4\_K\_M models answered in section 6. Every version ran fully on the GPU (checked in Ollama's `/api/ps`).

| Model | Download | RefDrone (1,254) | RefCOCO testA | RefCOCO testB | gRefCOCO val | Median latency |
| --- | --- | --- | --- | --- | --- | --- |
| 2B q4\_K\_M | 1.9 GB | 43.9 | 81.2 | 76.6 | 83.7 | 0.38 s |
| 2B q8\_0 | 2.7 GB | 50.9 | 83.8 | 78.6 | 83.7 | 0.40 s |
| 2B bf16 | 4.3 GB | 51.5 | 83.4 | 78.4 | 83.7 | 0.55 s |
| **4B q4\_K\_M** | **3.3 GB** | **69.3** | **88.8** | **80.8** | **89.3** | **0.57 s** |
| 4B q8\_0 | 5.1 GB | 69.1 | 89.2 | 81.2 | 87.3 | 0.60 s |
| 4B bf16 | 8.9 GB | 69.4 | 89.4 | 80.8 | 87.7 | 1.01 s |
| 8B q4\_K\_M | 6.2 GB | 63.2 | 87.6 | 82.4 | 84.5 | 0.84 s |
| 8B q8\_0 | 9.9 GB | 63.9 | 88.0 | 83.0 | 86.5 | 0.90 s |
| 8B bf16 | 18 GB | 63.9 | 88.0 | 82.8 | 86.1 | 1.65 s |

Accuracy at IoU 0.5 in %; RefDrone CIs about ±3 points, the 500-sample columns about ±3–4.

- **4B and 8B lose nothing measurable at q4** (within 1 point on drone photos, within 2 elsewhere), and q4 runs 1.7–2× faster than bf16.
- **2B loses 7 points on drone photos at q4** (43.9 % vs 50.9 / 51.5 %; CIs 41–47 vs 48–54), mostly by saying "not found" when the target is there (42 % of targets vs 34 %). If 2B is ever used, it should be q8\_0 (2.7 GB, same speed).
- **Choice for the Jetson:** 4B at q4\_K\_M (3.3 GB), with 2B at q8\_0 as the lighter fallback. Latencies here are the workstation on Vulkan; on the Jetson, with far less memory bandwidth, bf16 should fall further behind (expected, not measured).

### 10.3 Image size, and Ollama's image-token cap

**Method.** RefDrone in full at different longest image sides (`results/vlm/size_refdrone`; 1024 px from 10.1).

| Longest side sent | 512 | 768 | 1024 | 1280 | 1536 | 2048 |
| --- | --- | --- | --- | --- | --- | --- |
| 4B, all targets (%) | 65.2 | 67.7 | 69.3 | 69.4 | 69.3 | 69.7 |
| 4B, small targets < 1 % (%) | 61.0 | 64.0 | 65.9 | 66.2 | 66.3 | 66.8 |
| 2B, all targets (%) | – | – | 43.9 | – | 48.1 | 48.5 |
| Prompt tokens (median) | 1,180 | 1,180 | 1,221 | 1,223 | 1,204 | 1,204 |
| 4B latency (median) | 0.54 s | 0.54 s | 0.61 s | 0.58 s | 0.58 s | 0.58 s |

- **The model sees about the same number of pixels whatever is sent.** The prompt holds \~1,200 tokens at every size: Ollama resizes every image to roughly 1,000 image tokens, scaling small images up and large ones down. That is why latency hardly changes.
- Below \~1,000 px, detail is lost before Ollama scales the image up again (512 px: −4 points). Above \~1,280 px nothing changes for 4B. For 2B, 1,536 px is 4 points better than 1,024 px at the same token count; the likely reason is that a 1,024-px image is scaled up (blurred) while a 1,536-px one is scaled down (sharpened).
- **Consequence:** with Ollama, sending bigger images cannot give small drone targets more pixels. Cropping can: zooming the VLM onto the region where the target is expected, or splitting the frame into tiles. Not tested; a candidate for the Jetson work. Send at least \~1,280 px meanwhile.

### 10.4 Large reference models

**Why.** To see how much the Jetson-sized models give up. Neither model below fits an 8 GB Orin Nano; they are references only (`results/vlm/large`, same sets as 10.2).

| Model (q4\_K\_M) | Size | RefDrone (1,254) | small targets | target absent: "not found" | RefCOCO testA | RefCOCO testB | gRefCOCO val | Median latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL 4B (for comparison) | 3.3 GB | 69.3 | 65.9 | 48.7 | 88.8 | 80.8 | 89.3 | 0.57 s |
| Qwen3-VL 30B-A3B (mixture of experts, 3B active) | 19.6 GB | 69.4 | 65.8 | 66.4 | 90.0 | 86.4 | 89.3 | 0.83 s |
| Qwen3-VL 32B (dense) | 20.9 GB | 68.5 | 65.0 | 71.7 | 91.2 | 87.8 | 87.3 | 2.5 s |

- **On drone photos, 7–8× larger models are no better than 4B** (69.4 and 68.5 vs 69.3 %, also on small targets). The limit there is probably the image-token cap and the tiny targets (10.3), not model size.
- They are better at saying "not found" when the object is absent (66–72 % vs 49 %) and on RefCOCO testB (+6 to +7 points).
- 30B-A3B gave unreadable answers to 7 % of the gRefCOCO queries (valid rate 93 %); its gRefCOCO score counts those as wrong.

### 10.5 Detectors on 17 recordings

**Data.** The 12 recordings of section 9 plus the 5 that came through: CCRs-2-high-100%, CCRs-side-high (car approaches at high speed), CCRs-1-high-0% (near miss), CPLA-medium, CPNAO-high (pedestrian dummies). Together: 11 car approaches, 4 pedestrian dummies, 2 near misses, plus the 4 MVSEC flights (`results/events_avail`). Same detector settings as section 9; the spiking detector at 10 beta values, the classical one at 10 TTC thresholds, the hybrid at theta 4.

&#91;embedded content: results/events\_avail (spiking b\*, classical, MVSEC) · 31 Oct 2026 · 17 EvTTC recordings, 4 MVSEC flights\]

| Setting | Cars (of 11) | Pedestrians (of 4) | Near misses alarmed (of 2) | Early onsets | False alarms per min, MVSEC |
| --- | --- | --- | --- | --- | --- |
| Spiking, beta 0.16 | 11 | 4 | 2 | 50 | 5.1 |
| Spiking, beta 0.18 | 11 | 2 | 2 | 43 | 2.9 |
| Spiking, beta 0.2 | 9 | 1 | 2 | 36 | 1.6 |
| Spiking, beta 0.25 | 9 | 0 | 1 | 30 | 0 |
| Classical, TTC 3 s | 4 | 2 | 2 | 7 | 0.7 |
| Classical, TTC 4 s | 10 | 4 | 2 | 46 | 10.9 |
| Classical, TTC 5 s | 10 | 4 | 2 | 118 | 19.4 |
| Hybrid, theta 4 | 0 | 0 | 2 | 108 | 38.1 |

**Cross-validation, two rules.** Rule A is the rule of 9.1 (fixed 29 Oct): the hit rate counts car approaches only. Rule B (fixed 30 Oct, before these runs) counts car approaches and pedestrian dummies.

| Leave-one-recording-out result | Spiking, rule A | Spiking, rule B | Classical, rule A = B |
| --- | --- | --- | --- |
| Setting picked | beta 0.18 (15 of 17 folds; 0.25 in 2) | beta 0.18 (17 of 17) | TTC 4 s (17 of 17) |
| Cars caught (of 11) | 9; missed CCRs-3-low-100%, CCRs-side-high | 11 | 10; missed CCRm-low-100% |
| First alarm before contact, median (range) | 4.7 s (1.6–5.9) | 4.7 s (1.6–5.9) | 4.2 s (1.5–6.0) |
| Pedestrians caught (of 4) | 2; missed CPLA-low, CPNA-low | 2 | 4 |
| Near misses alarmed (of 2) | 2 | 2 | 2 |
| Early onsets, all held-out recordings | 43 | 43 | 46 |
| False alarms per min, MVSEC | **2.9** | **2.9** | **10.9** |

- **The comparison of 9.4 holds on more data.** For every classical setting there is a spiking setting that catches at least as many approaches with fewer false alarms (chart). At beta 0.16 the spiking detector catches all 11 cars and all 4 pedestrians at 5.1 false alarms per minute, against 10 cars, 4 pedestrians and 10.9 per minute for the classical detector at its chosen 4 s.
- **The cross-validated settings still trade off differently:** the spiking detector (beta 0.18) gives a quarter of the false alarms; the classical one catches all pedestrians.
- **Rule A's two missed cars are a rule effect:** in those two folds the held-out car was the one beta 0.25 misses, and 0.25 looked best on the rest. Rule B, which also needs pedestrians, kept beta 0.18 and caught all 11.
- **The choice of beta moved from 0.2 (12 recordings) to 0.18 (17 recordings).** The two new pedestrian recordings and the high-speed approaches favour the more sensitive setting.
- **Early alarms** (`onsets_spiking`): as in 9.2, they come only while the car moves (at beta 0.18: 42 while moving, 1 while standing still). The hybrid again fires mostly while the car stands still (76 of 108).
- **Still the same open problems:** the near misses alarm, and the early alarms are not yet separated into background expansion and a target seen early.

### 10.6 Lower event rate and finer resolution, both detectors

**Lower event rate** (`results/events_avail/dedup_*`; method as in 9.5, now with the classical detector too; 17 EvTTC recordings, settings fixed before the run: spiking beta 0.2, classical 4 s, hybrid theta 4).

| Events kept | Spiking: cars / pedestrians / early onsets | Classical: cars / pedestrians / early onsets | Hybrid: cars |
| --- | --- | --- | --- |
| 100 % | 9 / 1 / 36 | 10 / 4 / 46 | 0 |
| 70 % (1 ms merge) | 9 / 0 / 34 | 10 / 4 / 48 | 0 |
| 40 % (5 ms merge) | 7 / 0 / 30 | 10 / 4 / 49 | 0 |

The classical detector is unaffected by the lower rate; the spiking one loses 2 of 11 car approaches at 40 % of the events. A new weakness, not visible before the classical baseline existed. Beta was not re-chosen at the lower rates; a lower rate may simply need a lower beta (not tested).

**Finer resolution: 640×360** (2× instead of 3× binning, closer to the drone camera's 640×480; `results/events_avail/scale2`; both detectors at all their settings, then the same cross-validation, rule A). MVSEC is not affected (its camera is 346×260), so its false-alarm rates are those of the same settings in 10.5.

| Cross-validated, 17 recordings | Spiking at 426×240 | Spiking at 640×360 | Classical at 426×240 | Classical at 640×360 |
| --- | --- | --- | --- | --- |
| Setting picked | beta 0.18 | beta 0.2 (14 of 17 folds) | 4 s | 6 s (15 of 17 folds) |
| Cars (of 11) | 9 | 9 | 10 | 9 |
| Pedestrians (of 4) | 2 | **4** | 4 | 4 |
| Early onsets | 43 | **34** | 46 | 77 |
| False alarms per min, MVSEC | 2.9 | **1.6** | 10.9 | 23.6 |

At the finer resolution the spiking detector catches all pedestrians with fewer early alarms and fewer false alarms; the classical one needs a higher threshold and then alarms early much more often. Both are good news for the drone's 640×480 camera, with the caveat that the classical baseline was developed at 426×240 (its 5×5 neighbourhood was not re-tuned).

### 10.7 Clean CPU timing

**Method** (`results/events_avail/timing`). One process, one detector at a time, after the GPU queue had finished and with no other job running (only the download job's occasional web requests). Five EvTTC recordings from 2.4 to 9.4 M events/s, the four MVSEC flights, and three resolutions. Mean CPU time per 10 ms packet in Python on one core of the workstation; real time = 10 ms.

| Recording (events per s) | Spiking 426×240 | Spiking 640×360 | Spiking 1280×720 | Classical 426×240 | Classical 640×360 | Classical 1280×720 | Hybrid 426×240 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CCRs-side-low (2.4 M) | 3.4 | 3.8 | – | 3.3 | 4.8 | – | 1.5 |
| CCRs-1-low-100% (2.6 M) | 3.5 | 4.5 | 4.5 | 4.3 | 6.3 | 14.0 | 1.6 |
| CPNA-low (5.1 M) | 7.2 | 8.8 | – | 8.2 | 11.3 | – | 3.5 |
| CCRs-3-low-100% (7.2 M) | 11.1 | 14.2 | 14.1 | 10.0 | 13.7 | 20.1 | 5.4 |
| CCRm-low-100% (9.4 M) | 13.7 | 16.9 | – | 13.0 | 18.3 | – | 7.1 |
| MVSEC flights 1–4 (0.2–0.4 M, 346×260) | 0.55–0.69 |  |  | 0.90–1.40 |  |  | 0.18–0.28 |

- **At drone-like event rates (MVSEC) the spiking detector needs 0.6–0.7 ms per 10 ms packet**, about 6–7 % of one core in plain Python; the classical one 0.9–1.4 ms.
- **The spiking detector's cost follows the events, not the pixels:** from 640×360 to the full 1280×720 it stays the same (4.5 → 4.5 ms; 14.2 → 14.1 ms). The classical one grows 1.5–2.2× over the same step, probably because fewer events share a pixel at finer resolution, so more flow fits survive its per-pixel filter.
- At the car recordings' highest rates (above \~7 M events/s) both fall behind real time in Python on one core. 95th percentiles are up to 4–5× the mean (bursts of events).
- These are workstation numbers. The Jetson's ARM cores are slower; they still have to be measured there (section 13).

### 10.8 What changed in the conclusions

- **RQ1 (VLM).** Qwen3-VL 4B instruct at q4\_K\_M is confirmed on the complete test sets (69 % on drone photos, 0.6 s on this GPU). Quantisation costs it nothing; larger models do not help on drone photos; sending larger images does not help because Ollama caps the image tokens. The levers left for small targets are cropping or tiling. Its weak spot is answering with a box when the target is absent, so detections need confirmation over several frames.
- **RQ3 (detector).** With 17 recordings and pedestrians counted, the spiking detector still beats the classical baseline at every setting. Cross-validated, it gives 9–11 of 11 car approaches and 2 of 4 pedestrians at 2.9 false alarms per minute, against 10, 4 and 10.9 for the classical one. At 640×360 it catches all 4 pedestrians at 1.6 per minute. New weakness: at 40 % of the events it loses cars where the classical one does not.
- **Compute.** The spiking detector's cost scales with the event rate, not the resolution, which favours a higher-resolution camera. At drone-like rates it needs 6–7 % of one workstation core in Python.
- **Still open:** near misses alarm; the cause of the early alarms; 15 EvTTC recordings still blocked by Drive; everything is still in Python on a PC, nothing yet in the loop or on the Jetson.

## 11. All 32 EvTTC recordings (2–3 Nov)

On all 32 recordings the spiking detector catches more car approaches than the classical baseline (17/18 vs 14/18) with about half the early alarms (113 vs 213). Google Drive's download limit reset on Sat 31 Oct; the last 15 recordings arrived then (job 095), and the weekend pipeline re-ran everything on the full set (`results/events32`). These numbers supersede the 17-recording ones in 10.5–10.6.

**Method.** Same detectors and scoring as section 9. Spiking at 10 beta values (theta0 96), classical at 10 TTC thresholds, hybrid at theta 4. Leave-one-recording-out choice of the setting: rule A counts car approaches only (fixed 29 Oct), rule B counts cars and pedestrian dummies (fixed 30 Oct, before these runs).

| Detector, rule | Setting chosen | Cars caught | Pedestrians | Near misses alarmed | Early alarms | MVSEC false alarms / min |
| --- | --- | --- | --- | --- | --- | --- |
| Spiking, A = B | beta 0.16 (32/32 folds) | 17/18 | 8/9 | 5/5 | 113 | 5.1 |
| Classical, A | TTC 5 s (31 folds) | 14/18 | 9/9 | 5/5 | 213 | 19.4 |
| Classical, B |  | 14/18 | 9/9 | 4/5 | 170 |  |
| Spiking, 640×360 | beta 0.18 | 15/18 | 9/9 |  | 108 |  |
| Classical, 640×360 | TTC 12 s | 13/18 | 7/9 |  | 294 |  |

Median first alarm of the spiking detector: 4.1 s before contact. The 1 ms / 5 ms event-merge and the clean CPU timing runs of that weekend were not analysed further; the 5 ms merge was redone with beta chosen by CV (section 12).

## 12. SNN design and resolution sweep (4 Nov)

One pixel distance instead of two (4 neurons per pixel instead of 8) works as well as the full design and gave about 10× fewer false alarms on the MVSEC drone flights; below 426×240 most pedestrians are missed. Job 103, all 32 EvTTC recordings, `results/sweep`.

**Method.** Each variant: spiking detector at the same 10 beta values, then the leave-one-recording-out choice of beta (rule A shown; rule B gave the same choices except where noted). New settings in `eval_events.py`: an `sn_` prefix sets top-level fields for every detector (`sn_deltas_px`), and `sn_merge_polarity` feeds every event as OFF. Resolutions are EvTTC's 1280×720 binned by 8, 6, 4, 3, 2 and 1; theta0 either 96 (as at 426×240) or scaled with the pixel count.

| Variant | Neurons / pixel | Beta chosen | Cars | Pedestrians | Near misses | Early alarms | MVSEC false alarms / min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reference, 426×240 (section 11) | 8 | 0.16 | 17/18 | 8/9 | 5/5 | 113 | 5.1 |
| One distance, 3 px | 4 | 0.18 | 16/18 | 9/9 | 5/5 | 125 | 0.45 |
| One distance, 7 px | 4 | 0.18 | 17/18 | 5/9 | 4/5 | 77 | 0.45 |
| ON/OFF merged | 4 | 0.12 | 16/18 | 3/9 | 4/5 | 108 | 3.3 |
| Merged + one distance 3 px | 2 | 0.12 | 11/18 | 2/9 | 4/5 | 63 | 0.67 |
| 5 ms event merge | 8 | 0.12 | 15/18 | 8/9 | 5/5 | 113 |  |
| 160×90, theta0 96 / 14 | 8 | 0.12 | 18/18 | 3/9 | 4/5 | 110 / 179 |  |
| 213×120, theta0 96 / 24 | 8 | 0.14 / 0.12 | 17/18 / 15/18 | 3/9 / 6/9 | 4/5 / 5/5 | 105 / 247 |  |
| 320×180, theta0 96 / 54 | 8 | 0.16 | 17/18 / 18/18 | 3/9 | 4/5 / 5/5 | 75 / 92 |  |
| 640×360, theta0 216 | 8 | 0.20 | 16/18 | 7/9 | 4/5 | 64 |  |
| 1280×720, theta0 96 / 864 | 8 | 0.25 | 16/18 | 8/9 / 7/9 | 4/5 | 64 / 53 |  |

Classical baseline at the same resolutions (TTC threshold by CV): 160×90 caught 18/18, 9/9, 5/5 with 1002 early alarms; 213×120 17/18, 9/9, 5/5, 274; 320×180 15/18, 9/9, 5/5, 488; 1280×720 16/18, 7/9, 5/5, 72; 5 ms merge 14/18, 9/9, 5/5, 221.

- **One distance (3 px) is the candidate design**: half the layer-1 neurons, all pedestrians, false alarms on the drone flights 5.1 → 0.45 per min. Its beta (0.18) differs from the reference's (0.16), which also lowers MVSEC false alarms; to be confirmed at equal beta.
- **Pedestrians need resolution**: at 320×180 and below the SNN catches 3/9 pedestrian dummies whatever theta0. 426×240 is the lowest size that keeps them.
- **Above 426×240** nothing more is caught, but early alarms drop to 53–64.

## 13. VLM: newer models, two-step strategies, thinking (4–5 Nov)

No newer small model beats Qwen3-VL-4B-instruct overall; zoom-in adds 3 points on RefDrone at twice the latency, and thinking mode makes grounding worse and 20–40× slower. Jobs 102/104 (inside 105) and 116; Ollama 0.40.13 on Vulkan, native API (`/api/chat`, `"think": false`), q4\_K\_M.

**New models** (`results/vlm/new_models`; RefDrone test in full = 1,254 scored, the others the seed-0 500-sample sets; "not there" = correct "not found" on samples without a target).

| Model | RefDrone box (IoU ≥ 0.5) | RefDrone "not there" | RefCOCO testA | RefCOCO testB | gRefCOCO | gRefCOCO "not there" | Median latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B-instruct | 0.69 | 0.51 | 0.90 | 0.84 | 0.89 | 0.71 | 0.75 s |
| Qwen3.5-9B | 0.72 | 0.30 | 0.83 | 0.81 | 0.75 | 0.69 | 0.83 s |
| Qwen3.5-4B | 0.61 | 0.74 | 0.81 | 0.75 | 0.66 | 0.86 | 0.65 s |
| Qwen3.5-2B | 0.66 | 0.14 | 0.73 | 0.66 | 0.62 | 0.59 | 0.43 s |
| Qwen3.5-0.8B | 0.54 | 0.00 | 0.67 | 0.61 | 0.67 | 0.01 | 0.58 s |
| Gemma 4 12B | 0.20 | 0.36 | 0.50 | 0.62 | 0.46 | 0.67 | 0.85 s |
| Gemma 4 E4B | 0.18 | 0.41 | 0.53 | 0.49 | 0.66 | 0.73 | 0.52 s |
| Gemma 4 E2B | 0.14 | 0.84 | 0.38 | 0.43 | 0.51 | 0.82 | 0.32 s |

Qwen3.5-2B and 0.8B answer "found" to almost everything, which inflates their box scores; Qwen3.5-2B also gives `{"found": true}` without a box in 3–21 % of answers. Gemma 4 boxes are in \[y1, x1, y2, x2\] order (handled).

**Two-step strategies** (`scripts/bench_zoom.py`, `results/vlm/zoom`; RefDrone, 1,206 scored samples). Zoom = ask again on a crop around the first box; check = yes/no question on the box; tiles = ask four overlapping quarter tiles when the full image says "not found"; tiles + check = a tile's box must pass the yes/no check.

| Model, strategy | Box (IoU ≥ 0.5) | "Not there" | Median latency |
| --- | --- | --- | --- |
| Qwen3-VL-4B, one image | 0.69 | 0.51 | 0.75 s |
| Qwen3-VL-4B, zoom | 0.72 | 0.51 | 1.45 s |
| Qwen3-VL-4B, yes/no check | 0.35 | 0.91 | 1.25 s |
| Qwen3-VL-4B, tiles | 0.74 | 0.29 | 0.81 s |
| Qwen3-VL-4B, tiles + check | 0.70 | 0.50 | 0.81 s |
| Qwen3.5-4B, one image / zoom / tiles / tiles + check | 0.61 / 0.64 / 0.68 / 0.64 | 0.74 / 0.74 / 0.46 / 0.68 | 0.63 / 1.25 / 0.71 / 0.71 s |
| Qwen3-VL-2B, one image / zoom / tiles / tiles + check | 0.44 / 0.46 / 0.58 / 0.54 | 0.80 / 0.80 / 0.59 / 0.68 | 0.37 / 0.71 / 1.07 / 1.09 s |

On gRefCOCO and RefCOCO testA none of the strategies changes Qwen3-VL-4B by more than 1–2 points except the yes/no check, which throws away about half of the correct boxes everywhere. Qwen3-VL-8B strategies: running (job 116).

**Stricter prompt** ("only found=true if every detail matches", `results/vlm/strict`): Qwen3-VL-4B RefDrone 0.69 → 0.67 box, 0.51 → 0.59 "not there"; Qwen3.5-4B got worse on RefDrone (0.61 → 0.55 box, 0.74 → 0.42 "not there") but better boxes on gRefCOCO (0.66 → 0.80).

**Thinking mode on** (300 + 50 samples, `results/vlm/think`, up to 4,096 tokens): Qwen3.5-4B RefDrone 0.53 box, 0.21 "not there", 11 s per answer, a third of answers unreadable; Qwen3.5-9B 0.51 / 0.25 / 25 s. Not usable on a drone.

**Fixes made on the way**: the Ollama started at login ran on CUDA and crashed on this driver (every request failed; restarted on Vulkan, nothing lost); the parser now reads a list of boxes (`"box": [[…]]`), the `box_2d` key, `<think>` blocks, and 0–1 numbers from models that usually answer 0–1000; the follow-up model is picked by the mean of RefDrone box and "not there" (`pick_winner.py`), so a model that always says "found" cannot win.

### 13.1 VLM for the Jetson: 3×3 regions, image size, 4-bit, new small models (5–6 Nov)

Jobs 116, 125, 127 and 131 on the workstation GPU (Ollama 0.40.13, Vulkan). Samples as before (seed 0, 500 or 1,000 per task; seed 1 as a second draw).

**Qwen3-VL-8B (job 116).** RefDrone 0.628 on the full image, 0.712 with tiles; gRefCOCO 0.835 / 0.851; 0.8–0.96 s per answer. Not better than the 4B, and twice the size: the 4B stays.

**3×3 regions as in VLN on the Fly (job 125, new `bench_grid.py`).** The model names one of nine regions with a confidence, or says not visible. Compared with the region of the centre of the box the same model drew on the same samples:

| Model | RefDrone: named / from box | gRefCOCO: named / from box | Time, named / box (RefDrone) |
| --- | --- | --- | --- |
| Qwen3-VL-4B | 0.52 / 0.76 | 0.57 / 0.88 | 0.57 / 0.75 s |
| Qwen3-VL-2B | 0.42 / 0.49 | 0.53 / 0.84 | 0.38 / 0.37 s |
| Qwen3-VL-8B | 0.46 / 0.69 | 0.62 / 0.84 | 0.76 / 0.90 s |
| Qwen3.5-4B | 0.54 / 0.69 | 0.57 / 0.68 | 0.51 / 0.63 s |
| Gemma 4 E2B | 0.52 / – | 0.60 / – | 0.27 s / – |
| LFM3-VL 2B | 0.60 / – | 0.68 / – | 0.27 s / – |

Naming a region is clearly worse than drawing a box and taking its centre, and only 15–25 % faster for the 4B. The stated confidence is useless as a gate: 77–96 % of answers say ≥ 0.7, right or wrong. Not-visible answers are also worse (4B: 24 % correct on RefDrone's absent targets). Kept: boxes; the region or 3-D goal is computed from the box (median depth in the box, as the paper does per region); a goal is accepted only when several answers agree.

**Image size (Qwen3-VL-4B, jobs 125/127).** RefDrone 0.62 / 0.66 / 0.66 / 0.69 at 448 / 640 / 896 / 1,024 px (seed 0); seed 1: 0.672 at 896 px vs 0.674 at 1,024 px. gRefCOCO flat at 0.86–0.89. Drone views lose accuracy below \~900 px; 896 px is enough on the Jetson. The 2B stays at 0.42–0.47 on RefDrone at every size.

**4-bit vs 8-bit.** Qwen3-VL-4B: RefDrone 0.692 (q4) vs 0.694 (q8) on seed 0, 0.674 vs 0.694 on seed 1; gRefCOCO 0.889 vs 0.901 and 0.859 vs 0.847: no consistent difference, q4\_K\_M stays. The 2B loses more at 4-bit (0.42 vs 0.51 on RefDrone).

**New small models (jobs 124–127).**

- **LFM3-VL 2B** (Liquid AI, 1.1 GB, only a 2B exists): RefDrone 0.585 full image, **0.682 with tiles**, at 0.24–0.32 s; not-visible correct 0.85; gRefCOCO 0.67 / 0.75 with tiles; RefCOCO testA 0.85. On drone views it matches Qwen3-VL-4B at about half the time; on ground photos it is clearly weaker. Seed-1 check and RefCOCO testA/B strategies: job 131.
- **Bonsai-2 2B / 4B** (ternary weights): unusable for boxes (RefDrone ≤ 0.02 with every strategy, gRefCOCO ≤ 0.23).
- **Nemotron-3.5-Bolt E4B** (NVIDIA, 6.2 GB, y-x coordinates): RefDrone 0.07–0.19, gRefCOCO 0.51–0.53.

**Job 131 (6 Nov).** Qwen3-VL-4B at 896 px on RefDrone: full 0.683, zoom 0.706, **tiles 0.734** (0.62 s), tiles + confirm 0.696; the best drone-view result so far. RefCOCO testA / testB: 4B full 0.91 / 0.836, tiles 0.911 / 0.84; LFM3-VL 2B full 0.816 / 0.776, tiles 0.848 / 0.797. LFM3-VL 2B on seed 1: RefDrone 0.592 full / 0.689 tiles, gRefCOCO 0.702 / 0.766: seed 0 confirmed.

**Decision for the Jetson:** Qwen3-VL-4B instruct, q4\_K\_M, 896 px; LFM3-VL 2B with tiles as the fast alternative to measure on the Orin Nano itself.

## 14. ODA: real drone flights at poles (4–5 Nov)

On a drone flying forward none of the three detectors separates a pole ahead from the expansion of the whole scene: at an 87–89 % catch rate the alarm is on for 34–41 % of clear flight. The arena walls explain little; turning raises false alarms about 2.4×. Jobs 100–117, `scripts/oda.py`, `scripts/eval_oda.py`, `results/oda` … `oda4`.

**Data.** TU Delft ODA ([Dupeyroux et al. 2021](https://data.4tu.nl/)): 1,369 indoor flights at 1–2 poles, DVS 240×180 events, IMU 334–500 Hz, OptiTrack 80 Hz, read from the ROS bags (the CSV copies have one clock per sensor). One 98 GB zip from 4TU (md5 checked) holding a second zip; only the bags and the trial list are unpacked. 3 bags are unreadable: 1,329 flights scored, 83 of them at 1 lux.

**Labels (ours).** From OptiTrack and the pole positions: on course = the pole will pass within 0.5 m ahead; clear = pole behind, passing more than 1.0 m away, or slower than 0.3 m/s; in between not scored. TTC = distance / closing speed. From round 3, heading for the arena net/wall within 3 s is its own state (arena 8.9 × 8.8 m, estimated from where all flights went, +0.5 m).

**Scoring.** Per flight: a threat = at least 0.25 s on course; hit = alarm on course or within 0.3 s after; first-alarm TTC; false alarms = alarm onsets while clear, per clear minute; alarm-on share = share of clear time with the alarm on. Setting chosen by 5-fold CV over flights (fixed seed): catch rate ≥ 85 % on the training folds, then fewest false onsets.

**Data problems found and fixed** (each with a test):

- 71 flights (483–553) give positions in millimetres; rounds 1–2 scored them as all-clear flights. Converted from round 3.
- 7 flights start before the camera clock was synchronised (flight 9: 56 % of events on the sensor's own clock); events and IMU samples outside the OptiTrack time span are dropped.
- OptiTrack repeats the last pose while it has lost the drone; frozen samples are treated as dropouts.
- The raw gyro is mostly propeller vibration (std about 0.5 rad/s about z, while the 50 ms-smoothed rate stays mostly below 0.3 rad/s). The turning/straight split of rounds 1–3 used the raw gyro and is not valid; round 4 uses the smoothed rate (> 0.3 rad/s = turning).

| Round (job), held-out | Detector, setting | Flights with a threat | Caught | First alarm (median) | False alarms / min | Alarm on, clear flight |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (109), 24 SNN settings | Spiking, theta0 24, beta 0.08 | 976 | 70 % | 4.3 s | 41 |  |
|  | Classical, TTC 4 s | 976 | 88 % | 5.6 s | 52 |  |
|  | Hybrid, theta 2 | 976 | 84 % | 5.5 s | 59 |  |
| 2 (110), 60 SNN settings | Spiking, theta0 6, beta 0.04 | 976 | 88 % | 6.5 s | 50 | 40 % |
|  | Spiking one distance (2 px) | 976 | 88 % | 6.2 s | 49 | 40 % |
|  | Classical, TTC 4 s | 976 | 88 % | 5.6 s | 52 | 37 % |
|  | Hybrid, theta 1.5 | 976 | 88 % | 5.9 s | 59 | 38 % |
| 3 (114), walls + mm fix | Spiking, theta0 6, beta 0.04 | 1,035 | 89 % | 6.7 s | 52 | 41 % |
|  | Spiking one distance (2 px) | 1,035 | 87 % | 6.2 s | 50 | 37 % |
|  | Classical, TTC 4 s | 1,035 | 87 % | 5.6 s | 49 | 34 % |
|  | Hybrid, theta 1.5 | 1,035 | 86 % | 5.9 s | 61 | 35 % |
| 4 (117), + 500 ms alarm hold | Spiking, theta0 12, beta 0.04 | 1,035 | 87 % | 8.5 s | 23 | 57 % |
|  | Spiking one distance (4 px) | 1,035 | 89 % | 8.3 s | 23 | 59 % |
|  | Classical, TTC 8 s | 1,035 | 95 % | 8.9 s | 19 | 69 % |
|  | Hybrid, theta 2 | 1,035 | 88 % | 8.3 s | 24 | 56 % |

- **Walls**: heading for a net happens for 23.6 min in total against about 150 min of clear flight; the alarm is on 55–61 % of that time, but counting it as "not clear" changes false alarms per minute by less than 2 %.
- **Alarm-on budget** (round 2, best setting on all flights, so optimistic, more so for the SNN with 42 settings vs 10): with the alarm on at most 10 % / 20 % of clear time, the SNN catches 51 % / 74 % of pole threats, the classical method 27 % / 60 %. Counted as onsets the classical method looks better: the SNN alarm flickers.
- **Alarm hold** (round 4): CV picked the longest hold (500 ms) for every detector; false onsets halve, but the alarm is on longer. It moves the problem rather than solving it.
- **Turning** (round 4, smoothed rate): spiking 48 vs 20 false alarms/min turning vs straight, classical 38 vs 16. Ignoring alarms while turning (gate at 0.3 / 0.6 rad/s) was never chosen by CV: it also drops real alarms. Derotation is the next step.
- **Dim light** (1 lux, round 1): spiking 72 % → 30 % caught, classical 89 % → 78 %.
- **SNN thresholds** on this 240×180 sensor: theta0 6–24 instead of 96 on EvTTC.

## 15. TartanAir V2: simulated flights with true depth (4–5 Nov)

No usable result yet: the first run scored each 1–2 minute flight as one trial, so every detector "caught" it; the re-run with per-episode scoring (job 122) and a second environment (job 121) are running. Purpose: labels from true depth count every surface (walls included), and the exact poses allow derotation.

**Data** ([TartanAir V2](https://huggingface.co/datasets/theairlabcmu/tartanair2), CC BY 4.0 / BSD-3). AbandonedFactory, 9 easy trajectories, 14 min of flight, 7.4 billion simulated events (640×640, about 4–5 million per s), camera poses at 1 kHz, depth at 10 Hz (640×640 RGBA PNG = float32 metres), pinhole with 90° field of view. 43 GB downloaded; events per trajectory in one blosc-compressed HDF5 file (`events/{x,y,t,p}`, `ms_to_idx`). Speed median 1.7–2.2 m/s, up to 5.5 m/s. Loader `scripts/tartanair.py`, evaluation `scripts/eval_tartanair.py`.

**Labels (ours).** Every depth pixel is a 3-D point; with the camera velocity from the poses, a point is on course if it will pass within r\_hit of the camera ahead. A frame is a threat when at least 50 px are on course with closest approach under 2 s, clear when nothing on course under 4 s. With r\_hit = 0.5 m, 1,735 of 8,176 frames (21 %) were threats: the flights often brush past surfaces. Job 122 also scores r\_hit = 0.3 m.

**Runs.** Events binned to 320×320, repeated events per pixel within 1 ms merged (70–81 % of events kept). Speed: 20 s of flight takes 50–63 s for one SNN + the classical detector in one process. Job 120 (spiking 9 settings, classical, hybrid) scored per trajectory: discarded. Job 122: spiking 12 settings, one-distance 4, classical, hybrid, with the per-packet alarms saved (`results/tartanair2/*/traces`) so that labels and scoring can change without re-running; scored per threat episode (threat stretches, gaps < 0.5 s merged), with 0 / 250 / 500 ms alarm hold, CV over trajectories (3 folds). Next environment: OldTownSummer (30 GB downloading), then ConstructionSite (80 GB).

### 15.1 TartanAir: final results on three environments (5–6 Nov)

Jobs 122, 123, 126 and 128. 22 trajectories (AbandonedFactory 9, OldTownSummer 3, ConstructionSite 10), 262 threat episodes (on course = passing within 0.3 m). Scored per threat episode from saved traces; settings chosen by 3-fold cross-validation over trajectories (at least 85 % of episodes caught on the training folds, then the least alarm-on time while clear), reported on the held-out folds. Half resolution (320 × 320) unless noted. Cells: episodes caught / share of clear flight with the alarm on / false alarm onsets per minute.

| Detector | AbandonedFactory | OldTownSummer | ConstructionSite | All pooled |
| --- | --- | --- | --- | --- |
| Spiking (sensitive grid, job 126) | 88 % / 35 % / 38 | 100 % / 36 % / 98 | 92 % / 40 % / 94 | **95 % / 39 % / 86** |
| Classical (event flow + expansion) | 85 % / 48 % / 46 | 76 % / 55 % / 50 | 88 % / 51 % / 52 | **87 % / 34 % / 74** |
| Hybrid | 98 % / 27 % / 97 | 83 % / 39 % / 86 | 82 % / 33 % / 80 | **95 % / 37 % / 94** |
| Spiking, one distance (3 px) | – | 14 % / 0 % / 1 | 30 % / 4 % / 4 | 28 % / 5 % / 4 |

At a 70 % target the pooled operating points are: spiking 70 % / 24 % / 17, classical 71 % / 23 % / 57, hybrid 82 % / 15 % / 96. Median warning time (first alarm before contact) 0.5–0.8 s pooled; on ConstructionSite classical and hybrid warn only 0.2–0.3 s before. Turning raises false alarms for classical and hybrid (pooled hybrid 102 vs 69 per minute), barely for spiking.

**Reading.** With its sensitive settings on all three environments, the spiking detector reaches the same trade-off as the hybrid and close to the classical: about 90–95 % caught costs the alarm being on for a third or more of clear flight. The detectors lie on one curve; the limit is the input (forward flight makes the whole scene expand), as on ODA (87–89 % at 34–41 %). The first run of the new environments (job 123) used a grid that stopped at medium sensitivity and showed spiking at 71 %; job 126 corrected that.

**Full resolution (640 × 640, AbandonedFactory, job 128).** Classical 80 % / 13 % / 27 (half resolution: 85 % / 48 % / 46); hybrid 89 % / 15 % / 65 (98 % / 27 % / 97); spiking 89 % / 35 % / 35 (unchanged). Resolution helps the flow-based detectors a lot; one environment only so far.

**Depth-camera baseline (job 129, new `eval_tartanair.py depth`).** The classical check the thesis now uses: the closest obstacle in a corridor (radius 0.3 / 0.5 / 0.8 m) along the direction of flight, alarm when distance / speed is below τ (swept in the scoring). A RealSense-like version adds 0.3–10 m range, 5 % dropout, noise σ = 0.01 · Z² and 0.1 m/s velocity noise. First result, AbandonedFactory with the ideal sensor: 89 % of 73 episodes caught, no false alarms, median warning 0.76 s. This is an upper bound: the labels come from the same true depth. The final numbers for the realistic sensor and all three environments are in the depth baseline paragraph below.

**Depth baseline, final (jobs 129, 132; 6 Nov).** RealSense-like sensor alone, pooled over the three environments (218 threat episodes; depth frames are 10 Hz, so episodes are counted on a coarser grid than the event runs' 262): **87 % caught, alarm on 6 % of clear flight, 7 false onsets per minute**, median warning 0.9 s (41 % warned ≥ 1 s). At a 70 % target: 76 % caught, alarm on 0.3 %, 1.2 per minute. Per environment (85 % target): AbandonedFactory 88 % / 0 % / 1.9, OldTownSummer 86 % / 1 % / 3.0, ConstructionSite 80 % / 10 % / 14.9. The ideal sensor gives almost the same (pooled 86 % / 6 % / 4.0). Compared with the event detectors (34–39 % alarm-on and 74–94 per minute for 87–95 % caught), the depth check is far cleaner; part of that is because the labels come from the same true depth (an upper bound), but the gap is large.

**Ego-rotation compensation (jobs 130 and 134, side study).** Events re-projected with the exact camera orientation from the 1 kHz poses, as if the camera had not rotated since the start of each 100 ms window; translation untouched. Pooled over all three environments (22 held-out trajectories, 262 episodes, r 0.3). Each cell shows with / without derotation.

- Spiking, target 85 %: caught 93 / 95 %, alarm on in clear flight 27 / 39 %, false alarms 72 / 86 per min. Fewer false alarms and less alarm-on time, a few fewer catches.
- Classical, target 85 %: caught 89 / 87 %, alarm on 47 / 34 %, false alarms 34 / 74 per min. The clearest gain: about half the false alarms at the same catch rate, paid for with more alarm-on time.
- Hybrid, target 85 %: caught 92 / 95 %, alarm on 22 / 37 %, false alarms 114 / 94 per min. Worse.
- Classical, target 70 %: caught 73 / 71 %, alarm on 15 / 23 %, false alarms 60 / 57 per min. No gain.
- Hybrid, target 70 %: caught 71 / 82 %, alarm on 13 / 15 %, false alarms 15 / 96 per min. Far fewer false alarms, but about 11 points fewer catches.

For comparison, the depth baseline (section above) catches 87 %, is on 6 % of the time and gives 7 false alarms per min, so none of the event detectors, with or without derotation, beats it. Test run from job 133 (AbandonedFactory, spiking, derotated): 97 % caught, alarm on 30 %, 62 false alarms per min, against 88 / 35 / 38 without.

Caveat: the TartanAir poses are exact. On a real drone the rotation must come from the IMU or the estimator, so these numbers are an upper bound for what derotation can give on hardware.

**Code fixes in this round.** Trace files now carry the environment in their name (two environments both have P000, P001 …); pooled scoring keeps only settings run on every trajectory; the depth detector's tau sweep needs one frame (10 Hz), the event classical two packets in a row.

### 15.2 PX4 SITL + Gazebo: the simulation kit (6 Nov)

For the new thesis stack, tested here as far as possible without Gazebo (unit tests), to run on the Ubuntu PC:

- `sim/setup_sim_ubuntu.sh`: PX4 (newest stable tag) with its Ubuntu script (Gazebo Harmonic, no NuttX toolchain), a SITL build of `gz_x500_depth`, the Gazebo Python bindings, a venv with MAVSDK-Python (gRPC, < 4), QGroundControl dependencies and an Ollama check; each step prints OK / FAIL.
- `sim/worlds/nd_test.sdf`, `nd_obstacle.sdf`: PX4's default world plus a red box, blue barrel, green box and white box; the obstacle world adds a yellow pole between the start and the red box.
- `neurodrone/mission.py`: box + depth image → 3-D goal (median depth in the box, mapped between the RGB and depth cameras by ray angle) 1.5 m short of the target at the current altitude; a goal is accepted when 2 of the last 3 VLM answers agree within 1.5 m; turn to it, then fly at ≤ 1 m/s; the depth guard stops any motion with an obstacle in a 0.6 m corridor closer than 1.2 m or under 2 s to contact, and also when the depth image is missing or older than 0.5 s.
- `neurodrone/gz_camera.py` (RGB 1920 × 1080, hfov 1.204 rad; depth 640 × 480, hfov 1.274 rad, `/depth_camera`) and `scripts/sim_mission.py` (VLM thread, 20 Hz control loop over MAVSDK OFFBOARD velocity, JSONL log, `--no-fly` check mode, `--save-frames`).
- `VehicleState` now also carries local NED position, velocity, roll and pitch.

### 15.3 Closed-loop guard test in PX4 SIH (10 Nov, cloud Ubuntu 24.04)

No Linux PC with a user account was available for Gazebo, and WSL2 is blocked by group policy on the university Windows PCs. As a smaller check, the depth guard was flown against PX4's own built-in simulator (SIH, no Gazebo, no cameras, no VLM). It tests the PX4 flight stack, the MAVSDK OFFBOARD velocity setpoints and the guard function `neurodrone.mission.guard` together. It does not test the VLM, a real camera, Gazebo physics or the Jetson.

**Method.** PX4 v1.17.0 (commit d6f12ad1c4f70ad3230afd7d86e971421e02fef4) runs the SIH quadrotor. `scripts/sim_sih_guard.py` connects with MAVSDK-Python 3.17.4 (gRPC), takes off to 2 m, switches to OFFBOARD and flies north at 1 m/s with body-frame velocity commands at 20 Hz. SIH has no camera, so the depth image (160 × 120, hfov 1.274 rad, as the x500\_depth camera) is rendered from the drone's pose by ray casting against boxes and the ground, with the RealSense-like noise of the TartanAir depth baseline (σ = 0.01 · Z², 5 % dropout, 0.3–10 m). The guard settings are those of the mission script: corridor radius 0.6 m, stop distance 1.2 m, time to contact 2 s. A proportional controller holds the starting heading and altitude, because PX4 does not hold the heading in this mode (a first attempt without it let the heading swing to 14° and the drone drift 3.4 m sideways). Three cases, three seeds each; obstacles are placed relative to the start point.

| Case | Setup | Result (3 seeds) |
| --- | --- | --- |
| wall | wall across the path, near face 8 m ahead | stopped 3 / 3; guard first blocked at 2.2 m before the wall; closest approach 1.91 / 1.33 / 1.56 m; final gap 2.32 / 1.33 / 1.31 m |
| free | nothing in the way, goal 6 m ahead | reached 3 / 3, guard never blocked |
| beside | pillar 2.5 m to the side of the path | reached 3 / 3, guard never blocked (outside the corridor) |

All 9 runs met the criteria (wall: stopped and never closer than 0.3 m; free and beside: goal reached without a block). The guard fired at about 2.2 m because the time-to-contact rule (2 s at 1 m/s) triggers before the 1.2 m stop distance. The drone drifted up to 2.8 m sideways over a run, because only velocities are commanded and no position is held; the guard does not depend on this, but it is a property of the test, not of a real flight.

**Limits.** The simulator is simplified (no wind, perfect sensors apart from the depth noise I added), the world is three boxes, and the depth image is not a rendering, so lighting, reflective surfaces and the RGB-to-depth alignment of the real RealSense are not covered. The result shows that the control and guard logic works in closed loop; the real-camera behaviour has to come from the hardware tests.

**Reproduce.** On Ubuntu 24.04 with sudo and internet: `bash sim/run_sih.sh`. It installs the packages, fetches PX4 v1.17.0, builds it, creates a venv with `mavsdk<4`, starts PX4, runs the test and writes `results/sih/sih_guard_summary.json` (settings, versions, one row per run), the step logs `<case>_<seed>.jsonl` and the PX4 log. Seeds 0–2 and the world are fixed in the script. The run in this report was done on a cloud container with 2 cores and 7 GB RAM; PX4 was first set up by hand with the same steps (two clone problems had to be worked around, which the script now handles), and `run_sih.sh` itself was only run on that existing checkout, not from an empty directory. The NeuroDrone commit hash is not in the summary file (the cloud copy is not a git checkout): cite the commit of the repository that contains these files.

### 15.4 Full stack in PX4 + Gazebo on WSL2 (10 Nov, personal laptop)

**Setup.** Windows 11 laptop (RTX 3050 Laptop, 4 GB; 16 threads; WSL2 given 10 GB RAM), WSL2 Ubuntu 24.04 with mirrored networking so that Ubuntu reaches Ollama 0.40.13 on Windows at 127.0.0.1:11434. `sim/setup_sim_ubuntu.sh` installed PX4 v1.17.0, Gazebo Harmonic (gz-sim 8.15.0), the gz Python bindings and MAVSDK-Python (< 4); all 8 unit tests passed. Two fixes were needed: the PX4 build with 16 parallel jobs ran WSL out of memory (built with 4 jobs), and in PX4 v1.17 `DONT_RUN=1` no longer stops `make` from starting PX4 + Gazebo after the build (the script now skips the build when the binary exists). Manual `commander takeoff` needs `param set NAV_DLL_ACT 0` without a ground station.

**No-fly check** (`sim_mission.py --no-fly`, world nd\_test, "the red box"). Qwen3-VL-4B-Instruct found the red box in every answer; the box drawn on the saved frames sat on the red box. Depth distance 11.4 m, goal 10 m ahead (1.5 m stand-off), accepted after the second agreeing answer. VLM latency 29 s for the first call (model loading), then 5–7 s per answer on this GPU, which also renders Gazebo. The guard reported "no fresh depth" for a few seconds (depth image older than 0.5 s, GPU shared between Gazebo and the VLM) and blocked, as designed.

**First full flight** (same world and instruction). Takeoff, OFFBOARD at about 2.5 m, goal accepted after two answers (about 15 s after takeoff), then a straight flight of 10 m at 1 m/s, slowing near the goal, "arrived" about 1.5 m before the box, hover, landing. As the drone approached, the depth distance fell 11.4 → 9.8 → 7.3 → 5.0 → 3.9 m while the goal stayed within 1 m of the first estimate; at close range the VLM stopped finding the box (it fills or leaves the frame), and the consensus kept the accepted goal. The guard saw nothing in its corridor, which is correct: the boxes are lower than the flight altitude. MAVSDK printed "User callback queue slow" repeatedly: the telemetry callbacks fall behind while the Python process decodes 1920 × 1080 frames and encodes the VLM image; it did not affect this flight but should be fixed before timing measurements.

**Next.** The obstacle world (nd\_obstacle, yellow pole in the path) to test the guard in Gazebo; then a fixed test list (each target, with and without the pole) repeated several times, with the JSONL logs kept as results.

**Obstacle world, stop-only guard** (nd\_obstacle: yellow pole, radius 0.15 m, 6 m along the path, 0.2 m off the line to the red box; corridor radius 0.6 m). The guard blocked at 2.0 m (time-to-contact rule) and the drone stopped 1.3 m before the pole. After about 8 s the straight corridor cleared and the drone flew past the pole: the first VLM goal was 1.1 m to the side of the box and the drone had drifted about 0.2 m while hovering, so the pole left the 0.6 m corridor. Measured from the mission log (`mission_20261110_213607.jsonl`, 12 Nov; first estimated as 0.45–0.65 m), the pole's surface came within 0.64 m of the drone's centre; with propeller tips at about 0.42 m that is 22 cm of margin. A stop-only guard checks only the commanded direction, and a corridor measured from the centre leaves little room once the drone's size is counted.

**Change: wider corridor and a reactive go-around** (`mission.Avoider`). The corridor radius default is now 0.8 m (about 0.4 m propeller tip + 0.4 m margin). When the straight path is blocked, directions turned 10°, 20°, 30° to either side are tried (all inside the depth camera's view, heading kept towards the goal); the first whose corridor is free for 2.5 m is flown at 0.5 m/s, and kept for 2 s after the straight path is free again; if none is free the drone stops and hovers. No map: only what the camera sees in that frame. `--no-avoid` restores stop-only. Three unit tests added (go around the free side and commit; stop at a wall; stop on stale depth); 11 mission tests pass.

**Obstacle world with go-around.** Blocked at 2.1 m, went around to the right (10°, then 20°, then 10°, at 0.5 m/s) for about 7 s, back to the goal direction once the corridor was free, arrived about 1.5 m before the red box, hovered and landed. Measured from the mission log (`mission_20261110_220405.jsonl`), the drone passed the pole 1.01 m centre to centre: the pole's surface was 0.86 m from the drone's centre, 0.44 m beyond the propeller tips. One run each so far; the depth image in Gazebo has no noise, so the real RealSense will be worse.

### 15.5 Test series set-up and indoor positioning (12 Nov)

**Backup and working copy.** Everything thesis-related on the personal laptop is in the GitHub repository essa0044/Thesis (public): this report (`docs/`, Markdown and Word; from now on the Markdown file is the one that is edited), the laptop's files (`personal-pc/`, including the WSL folder with the four mission logs of 10 Nov) and the working code in `neurodrone/`. The lab workstation will be added under `workstation/` to compare versions.

**Fix: "User callback queue slow".** The MAVSDK telemetry callbacks fell behind because the Python process decoded and copied every 1920 × 1080 Gazebo RGB frame (about 6 MB, 30 times a second) inside the camera callback, and PX4 SITL streams some telemetry at 50 Hz or more. Now the camera callbacks only keep the newest message and decode it when it is used (the VLM needs an RGB frame every few seconds, the guard a depth frame every control step), the subscriptions are throttled to 5 Hz (RGB) and 20 Hz (depth) where the gz-transport Python bindings allow it, and the telemetry streams the mission uses are requested at 25 Hz (battery and in-air at 2 Hz). Not yet checked in WSL.

**Mission logs for analysis.** Each log now has a `start` record (instruction, model, world, all settings), an `offboard` record and an `end` record (outcome arrived / timeout / stopped / error, duration, final position, accepted goal). A run gives up and lands after `--max-time` (default 180 s). Ollama is asked to keep the model loaded for 30 min, so only the first run of a series pays the model loading time.

**Test series.** `sim/run_series.sh` flies "the red box", "the blue barrel" and "the green box" in two worlds, `nd_test` (no obstacles) and the new `nd_obstacles` (a yellow pole, radius 0.15 m, on the path to each target, 0.2 m off the straight line: red box pole at 6 m, barrel pole at 55 % of the way, green box pole at 55 %), 5 runs each = 30 runs, every run in a fresh PX4 + Gazebo, turning at 8°/s while no goal is accepted (the barrel and green box are near the edge of the camera's view at the start). `scripts/analyze_series.py` turns the logs into the results table: success (arrived, right target, final stand-off 0.5–2.5 m from the target's surface, no propeller contact with a pole), time to goal, stand-off, closest propeller clearance to a pole (propeller tip radius 0.42 m: X500 V2 arm 0.25 m + half a 13-inch propeller), time stopped and time going around, VLM calls, share found and latency (first call separately). Positions come from PX4's estimate in simulation; on the real drone they will come from the motion-capture system. 4 + 1 new tests (analysis on a synthetic log in the real world file; camera throttling and lazy decoding with stand-in gz modules); 16 tests pass.

**First series attempt (12 Nov, 1 run per case): VLM too slow, fixed.** All 6 runs timed out after 180 s without a single VLM answer: every call took longer than the client's 60 s limit. Ollama (started by Windows at login) had loaded Qwen3-VL-4B 54 % on the CPU and 46 % on the GPU (`ollama ps`, 4.0 GB at a 4096-token context): with Gazebo using the RTX 3050's 4 GB, most of the model did not fit. A timed-out call also kept Ollama busy, so the next call queued behind it. Changes: the series loads the model before any Gazebo starts (`scripts/vlm_warmup.py`: unload, load with the missions' settings, report the GPU share, time 3 calls on a test image, stop the series if the median is over 30 s); the context is fixed at 2048 tokens (less GPU memory; the same value in every call, or Ollama reloads the model); the client waits up to 240 s; the model is kept loaded for 60 min. On the Jetson the VLM will have the GPU to itself, so this is a property of the laptop set-up, not of the method.

**Second attempt (12 Nov, 1 run per case): search while turning, fixed.** With the VLM at about 5.5 s per call (`vlm_warmup.py`: model 50 % on the GPU), the runs again ended 0 of 6: the target was found once at the start, but the drone kept turning at its search rate during the next calls, so the object had left the picture before the next answer and two answers never agreed. The search is now turn-stop-look (`mission.Searcher`): turn to the next heading of a sweep around the start heading (0, +30, -30, +60, -60 … 180°), hold still until an answer arrives for a picture taken after the turn ended, stay while the answers find the object (the consensus then accepts the goal), and turn on when an answer finds nothing. The search turns at up to 20°/s. 2 tests added; 18 pass. This is also how the real drone has to search: with a VLM that answers every few seconds, it must hold still for each look.

**VLM stalls while Gazebo runs: graphics memory (14 Nov).** The series still got no VLM answer during flights, and the load order was not the cause: reloaded while Gazebo ran, the model still did not answer within 240 s, while without Gazebo it answered in 5.8 s. Task Manager during the test: RAM 87–94 % (WSL about 35–39 %, Ollama's llama-server about 22 %), CPU 52–76 %, and the RTX 3050's dedicated memory full (4.0 of 4.0 GB, with shared memory in use). Ollama reported the same 50 % GPU share whether or not Gazebo was running, so it did not see Gazebo's use of the GPU (the WSL rendering goes through Windows' GPU sharing); Gazebo's camera rendering and half of the model did not fit in 4 GB together, and the GPU work stalled. Test with Gazebo running (`vlm_warmup.py --num-gpu`): 0 layers on the GPU (CPU only): 57.8 s for the first call, then 6.3 s; 12 layers on the GPU (34 %): no answer within 240 s. So on this laptop the VLM runs on the CPU (`NUM_GPU=0`) while Gazebo uses the GPU, with the Gazebo window off (`HEADLESS=1`) and other programs closed. Thesis note: this is a limit of running simulator and VLM on one 4 GB laptop GPU, not of the method; on the Jetson the VLM has the GPU to itself, and its latency has to be measured there.

**Active sensing for the search (14 Nov).** The turn-stop-look sweep moves the camera in order to see (the "active" half of active perception, Bajcsy 1988; active vision, Aloimonos et al. 1988), but its order of views is fixed. A second search, `mission.ActiveSearcher` (`--search active`), lets the answers choose the next view. It keeps a belief over 36 directions (10° each, uniform at the start). An answer that finds nothing multiplies the directions inside the 69° view by 0.3 (the VLM sometimes misses, so they are not ruled out). An answer that finds the target puts 80 % of the belief on the box's direction and turns the drone to centre the box, so the confirming answer comes from a different picture: at temperature 0 two answers on the same picture are nearly copies and are weak evidence for the 2-of-3 consensus. When nothing is found, the next view is the heading whose view holds the most belief, minus 0.15 per 180° of turn; the view just seen is skipped. Research question for the series: sweep vs active search, compared on success, time to goal and number of VLM calls in the same worlds (`SEARCH=sweep` / `SEARCH=active bash sim/run_series.sh`; the table now has a search column). 2 tests added; 20 pass.

**Depth noise stays off in Gazebo, on purpose.** Gazebo's depth image is perfect (no noise, no missing pixels). It is left that way: the difference between the perfect simulated world and real flight with the RealSense is a research question of the thesis, answered by flying the same test list in simulation and in the lab.

**Indoor positioning: visual-inertial odometry from the RealSense (updated 14 Nov).** Velocity commands need a position estimate in PX4, and indoors there is no GPS; the IMUs alone drift within seconds. VLN on the Fly (Tayar et al., USP; the same Pixhawk 6C + RealSense D435i + Jetson combination) flew with an OptiTrack motion-capture system fused by PX4's EKF2, but the lab has no motion capture and nothing is to be bought. So the drone estimates its own position from its camera: visual-inertial odometry (VIO) on the Jetson from the D435i's stereo infrared images and its IMU (first choice NVIDIA Isaac ROS Visual SLAM / cuVSLAM, which supports the RealSense on Jetson; OpenVINS as an alternative), sent to PX4 as MAVLink odometry, with EKF2 using external vision for position, velocity and heading (`EKF2_EV_CTRL`, `EKF2_HGT_REF` = vision, GPS and compass off, `EKF2_EV_DELAY` tuned), as in PX4's visual-inertial-odometry guide. Consequences for the thesis: (1) the D435i's IR dot projector helps depth but hurts VIO tracking, so it is either alternated frame by frame or switched off at some cost to the guard's depth; (2) VIO and the VLM share the Jetson's GPU and 8 GB of memory, and VIO must never stall; (3) there is no external ground truth, so distances (stopping distance, clearance, stand-off) are measured with floor markings and a tape measure or video, and VIO drift with return-to-start tests; (4) VIO can lose track in plain or dim rooms, so the safety pilot and PX4 failsafes must be ready. If an optical-flow sensor or motion capture becomes available later, PX4 takes it through the same estimator interface.

### 15.6 Pilot series: sweep vs active search (14 Nov, laptop, VLM on the CPU)

**Set-up.** `HEADLESS=1 NUM_GPU=0 bash sim/run_all.sh` at commit 9e988ef: 3 targets × 2 worlds × 2 runs, first with the sweep, then with the active search (24 runs, 18:16–19:50), Gazebo without window, Qwen3-VL-4B-Instruct entirely on the CPU, 180 s limit. Logs, frames and PX4 output: `neurodrone/results/series/20261114_1816_sweep` and `20261114_1903_active`; tables re-made with the analysis of 14 Nov evening (simulator errors counted separately).

Sweep:

| world | target | search | runs | success | arrived | right target | time to goal (s, median) | stand-off (m, median) | min prop clearance (m) | going around (s, median) | VLM latency (s, median) | VLM calls (median) | sim errors (not counted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nd_obstacles | the blue barrel | sweep | 2 | 1/2 | 1/2 | 2/2 | 152.6 | 3.32 | 0.44 | 5.65 | 53.95 | 2.5 | 0 |
| nd_obstacles | the green box | sweep | 2 | 0/2 | 2/2 | 2/2 | 122.85 | 5.94 | 1.33 | 0.0 | 52.97 | 2.0 | 0 |
| nd_obstacles | the red box | sweep | 2 | 1/2 | 1/2 | 2/2 | 138.3 | 3.17 | 0.5 | 7.0 | 59.45 | 2.0 | 0 |
| nd_test | the blue barrel | sweep | 1 | 1/1 | 1/1 | 1/1 | 130.8 | 1.82 | – | 0.0 | 51.61 | 2 | 1 |
| nd_test | the green box | sweep | 2 | 2/2 | 2/2 | 2/2 | 146.65 | 1.63 | – | 0.0 | 54.99 | 2.0 | 0 |
| nd_test | the red box | sweep | 2 | 1/2 | 1/2 | 1/2 | 134.9 | 5.05 | – | 0.35 | 57.8 | 2.0 | 0 |

Active search:

| world | target | search | runs | success | arrived | right target | time to goal (s, median) | stand-off (m, median) | min prop clearance (m) | going around (s, median) | VLM latency (s, median) | VLM calls (median) | sim errors (not counted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nd_obstacles | the blue barrel | active | 1 | 1/1 | 1/1 | 1/1 | 133.0 | 1.94 | 0.44 | 6.5 | 51.0 | 2 | 1 |
| nd_obstacles | the green box | active | 2 | 0/2 | 1/2 | 2/2 | 123.0 | 5.5 | 0.64 | 3.35 | 52.2 | 2.5 | 0 |
| nd_obstacles | the red box | active | 2 | 2/2 | 2/2 | 2/2 | 143.65 | 1.93 | 0.53 | 7.0 | 54.38 | 2.0 | 0 |
| nd_test | the blue barrel | active | 2 | 2/2 | 2/2 | 2/2 | 135.2 | 1.87 | – | 0.0 | 54.37 | 2.0 | 0 |
| nd_test | the green box | active | 2 | 2/2 | 2/2 | 2/2 | 143.65 | 1.65 | – | 0.0 | 55.71 | 2.0 | 0 |
| nd_test | the red box | active | 2 | 2/2 | 2/2 | 2/2 | 137.2 | 1.86 | – | 0.0 | 54.6 | 2.0 | 0 |

**Overall:** sweep 6 of 11 runs successful, active search 9 of 11 (one simulator error each, not counted). Every run that succeeded needed 2 VLM answers (the consensus minimum).

**What the failures showed:**
1. **The VLM took 51–58 s per answer in flight (CPU only).** The check before each run said about 6 s because it sent the same test picture twice and Ollama reuses its work on a picture it has just processed; only the first call in the check was honest (57.8 s). With 2 answers needed, goals were accepted after 100–150 s; three runs ran out of the 180 s. Fixed: the check uses a new picture for every call; the limit is 360 s.
2. **Green box behind a pole: goal 6 m short (sweep 2 of 2, active 1 of 2).** At the start the green box is cut off at the right edge of the picture (box x 0.91–1.00) and the pole stands in front of that part of the image: the median depth in the box was the pole's (4.85 m), so the goal was 1.5 m before the pole and the drone "arrived" there. In one active run the drone first turned to centre the box; the depth was then 9.1 m, which is right (it then ran out of time at the pole). Fixed: no goal from a box that touches the left or right image edge (`mission.box_at_edge`); the sweep treats such a find as "not found" and moves to the next view, the active search turns to centre it. This is an argument for active sensing: centring the object gives a correct distance.
3. **The guard saw the floor.** In one empty-world run the drone flew at 1.2–1.4 m and pitched forward while accelerating; the guard's corridor tilted with the camera and the floor fell inside it ("obstacle 0.4 m ahead", 15 s blocked). Fixed: the corridor is now horizontal (roll and pitch removed with the drone's attitude) and goals are at the mission altitude, not the altitude at the moment of the picture. Test added.
4. **Simulator errors:** 2 runs never left the ground (PX4 takeoff timed out; arming denied). Not counted; the series now retries such a run once.

These are pilot results with known bugs; the series is to be repeated with the fixes (5 runs per case).

## 16. Problems met and how they were fixed

Every problem that changed a method or a number, newest first. These belong in the thesis's methods and limitations sections.

| Date | Problem | Fix / consequence |
| --- | --- | --- |
| 5 Nov | TartanAir first run scored each 1–2 min trajectory as one trial, so every detector "caught" it | re-run with per-packet alarm traces saved and scoring per threat episode (job 122); job 120 results discarded |
| 5 Nov | CPU idle 23:50–07:40: the first TartanAir run finished and the check-in that should have queued the next run did not happen | jobs 121/122 queued at 07:40; no results lost |
| 4 Nov | ODA raw gyro is mostly propeller vibration (std \~0.5 rad/s); the turning/straight split of ODA rounds 1–3 measured vibration, not turning | 50 ms-smoothed turn rate (> 0.3 rad/s = turning) from round 4; the earlier "turning is not the cause" withdrawn |
| 4 Nov | 71 ODA flights (483–553) give OptiTrack positions in millimetres | converted to metres (oda.fix\_mm); ODA rounds 1–2 had them as all-clear flights |
| 4 Nov | Some ODA flights start before the camera clock is synchronised (flight 9: 56 % of events at \~0 s) | events and IMU samples outside the OptiTrack time span dropped (7 flights affected) |
| 4 Nov | The 4TU ODA file is a zip inside a zip; inside it, dataset/1-50.zip is not a valid zip | only the outer archive's zip is unpacked; subset zips ignored |
| 4 Nov | Qwen3.5 / Gemma 4 mix 0–1 and 0–1000 box numbers, Qwen3.5 sends a list of boxes, Gemma uses y-x order | parser reads all three; stored answers re-scored before the follow-up model was picked |
| 4 Nov | The Ollama started by the app at login runs on CUDA, which crashes on this driver; every request of job 102 failed | Ollama restarted on Vulkan (job 105); the speed check now restarts it once if a model gives no answers |
| 4 Nov | Hugging Face slowed the TartanAir event download to \~3 MB/s (33 GB took \~3 h) | accepted; downloads resume after errors |
| 31 Oct | Sending larger images did not change the prompt length: Ollama resizes every image to about 1,000 image tokens | the image-size test reports the token count beside accuracy; cropping or tiling noted as the way to give small targets more pixels (10.3) |
| 30 Oct | Google Drive answered "quota exceeded" for the 20 EvTTC recordings not yet downloaded; only 5 came through by Sat morning | the download job was replaced by one that retries every 45 min for 44 h (job 093); the detector runs went ahead on the 17 recordings available (results/events\_avail), to be repeated if more arrive |
| 30 Oct | Risk that a model falls back to the CPU or runs very slowly on Vulkan and blocks the weekend queue | a 12-request speed check before each extra model (skip if the median is above 6–8 s) and a check that the model sits fully on the GPU; none had to be skipped |
| 30 Oct | The NVIDIA driver cannot be updated on the workstation (no admin rights) | Vulkan stays for every workstation VLM number; CUDA latencies will come from the Jetson |
| 30 Oct | Classical baseline v1 (500 ms window) estimated shrinking, not growing, during real approaches: neighbours not yet reached by an edge hold timestamps of earlier edges | 100 ms window, chosen on training-sequence slices only; v1 results kept apart (events\_classical\_v1) and not used |
| 30 Oct | A robust refit of the classical flow did not help on real data and broke the synthetic approach test | dropped (refit\_iters 0); the code path stays for the record |
| 30 Oct | Ego-motion check: the EvTTC car never turns faster than 0.1 rad/s outside the scored window | the rotation part cannot be tested on EvTTC; reported as untested (9.2) |
| 30 Oct | EvTTC field of view: the calibration gives about 54°, the paper quotes 41° | the calibration is used (the events follow it); section 7 text corrected (8 px per degree, not 7) |
| 30 Oct | RefCOCO and gRefCOCO state no licence for their annotations | used for research benchmarking only, with citation, nothing redistributed; noted in the dataset table |
| 29 Oct | Qwen2.5-VL answered `[]` (its "not found") and the parser counted it as an unreadable answer | `[]` now means not found; all saved answers re-scored, no new model calls |
| 29 Oct | Spiking settings chosen on two training sequences were silent on all 10 held-out approaches | reported as a failure to generalise; the beta trade-off over all sequences reported separately as a sensitivity analysis |
| 29 Oct | First tuning grid (beta 0.08–0.2) alarmed long before the danger; at real event rates beta dominates the threshold | wider grid (beta up to 0.7), same training sequences and rule; first grid kept |
| 29 Oct | Small Qwen3-VL models on drone photos listed dozens of boxes until the token limit (cut-off JSON) | parser keeps the first box and recovers cut-off answers (drone code too) |
| 29 Oct | gRefCOCO val has no single-target expressions (1,499 no-target, 2,736 multi-target) | multi-target expressions scored as correct if the box matches any referred object |
| 29 Oct | Qwen2.5-VL answers in pixels of the server's internally resized image (≥ 1024 image tokens) | two resize-aware coordinate modes, chosen per model on the calibration samples; the drone parser needs the same if Qwen2.5-VL flies |
| 29 Oct | Ollama 0.40 on CUDA crashed ("device kernel image is invalid"): its compressed kernels need driver 550+, the PC has 546.12 and no admin rights ([ollama#16449](https://github.com/ollama/ollama/issues/16449)) | Ollama 0.24 tried (crashes on Qwen3-VL); all runs on Ollama 0.40 with the Vulkan backend. Latencies are Vulkan numbers |
| 29 Oct | Plain `qwen3-vl` tags are the slow thinking variants | `-instruct` tags used |
| 29 Oct | Google Drive refused gdown 6.5.1 after a few files; two files hit Drive's per-file quota | direct-download URL with resume; quota files succeeded on retry |
| 29 Oct | The benchmark's resume logic kept failed requests as done (a smoke run had hit models not yet downloaded) | failed requests are retried; first smoke outputs moved aside |
| 29 Oct | No WSL and no admin rights on the workstation | PX4 SITL (experiment 5.5) not run |
| 16 Oct | Spiking detector alarmed earlier, causing one unnecessary sidestep before the target | coordinator `arrive_w_frac` 0.20 → 0.15 |
| 13 Oct | qwen2.5vl:3b on a 4 GB laptop GPU answered `@@@@…` from the 4th image on (model split 57 % CPU / 43 % GPU, 25–28 s per image) | qwen3-vl:2b, which fits the GPU, tested next: 32/32 correct, about 15 s per image |

## 17. What has happened so far, step by step

Three weeks in: the software prototype exists, both halves of the system have been measured on public data, and nothing has run on real hardware or in PX4 simulation yet.

1. **11 Oct – Taking stock.** Read the existing prototype (`neurodrone/`, about 3,700 lines of Python: SNN looming detector, coordinator state machine, VLM client, PX4 adapter, synthetic simulator) and re-ran its tests: 85/85 pass. Wrote the risks and open questions (everything synthetic, hardware unconfirmed, PX4 SITL needs Linux or WSL2). Agreed the thesis is the single-drone SNN + VLM system; the multi-drone MCP idea moved to future work. Created the thesis plan for the supervisor.
2. **12 Oct – Control split and hardware.** Fixed the division of labour (PX4 on a Pixhawk 6C does the low-level control; the Jetson sends velocity setpoints in OFFBOARD), as in the TU Delft paper. Priced the hardware in two phases (bench €3,533, flight €1,573) and built the weight and flight-time budget (1,681 g, about 11 min hover), plus the mass and inertia to put into PX4's simulated X500.
3. **13 Oct – First real VLM test (laptop).** qwen2.5vl:3b on a 4 GB RTX 3050 broke after 3 images (`@@@@…` output; model split across CPU and GPU, 25–28 s per image). The follow-up run: qwen3-vl:2b got 32/32 correct, centre within about 0.4 % of the image width, about 15 s per image; qwen2.5vl:3b also 32/32 but less precise (about 8 %) and slower.
4. **15 Oct – Workstation package.** Bundled the code and a test kit for the stronger lab PC.
5. **16 Oct – Why an SNN, and a fully spiking detector.** Discussed whether a classical event algorithm would do the same job (it would compute the same cue); decided to keep the SNN as the core and make it spiking end to end, keeping the old detector as the hybrid baseline. Implemented and tested the fully spiking detector (112 tests then) and compared both in the synthetic simulator (section 4). Wrote a short draft for the supervisor with the questions on scope, borrowing hardware, and the TU Delft paper.
6. **26 Oct – Workstation day plan.** Planned software-only tests for the lab PC on public datasets instead of self-made images, so the results are defensible: VLM grounding on RefCOCO / gRefCOCO / RefDrone, the detectors on EvTTC / MVSEC, PX4 SITL if possible.
7. **29 Oct – Workstation day.**
   1. Set up Python and a job runner on the PC (no admin rights, no WSL); 128 tests pass on Windows.
   2. Downloaded all datasets (about 88 GB), working around Google Drive refusals.
   3. Got the VLMs running despite the old GPU driver (Ollama on Vulkan).
   4. Smoke test, fixes (coordinate convention, gRefCOCO's make-up, parser), then the full VLM benchmark: 5 models, 5 tasks, harness check, image-size test (about 4 h of GPU time).
   5. Event detectors: tuning on 2 training sequences, runs on 12 EvTTC sequences and 4 MVSEC flights, the region-of-interest variant and the beta trade-off.
   6. Wrote `results/REPORT.md`, `results/NOTES_2026-10-29.md`, the first results in the thesis plan, and this report.
8. **30 Oct – Second workstation day (software only).**
   1. Closed the open checks from 29 Oct: beta by leave-one-sequence-out cross-validation, the ego-motion check of the early alarms, a lower event rate, an energy floor, the dataset licences.
   2. Second VLM seed for the three Qwen3-VL models (about 1.5 h of GPU time). The event jobs ran on the CPU beside it; never two GPU jobs at once.
   3. Built a classical baseline (optical flow + expansion fit). Version 1 failed on real data and was redesigned on training slices only; then it was compared with the spiking detector on the same recordings with the same rule.
   4. Confirmed the GPU driver cannot be updated. Added section 9 to this report.
9. **30–31 Oct – Weekend runs on the workstation (section 10).**
   1. GPU, Fri 15:18 – Sat 07:41: the three Qwen3-VL models on the complete test sets, their q8 and bf16 versions, image sizes from 512 to 2,048 px, and the 30B-A3B and 32B models (about 80,000 queries).
   2. Downloads: Google Drive let 5 of the 20 missing EvTTC recordings through; the job keeps retrying.
   3. CPU, Sat 07:31 – 08:21: all three detectors on the 17 recordings available, both cross-validation rules, a lower event rate and 640×360 for both detectors, then a clean CPU-timing run.
   4. New code, tested before the runs (140 tests): full-test-set mode for the VLM benchmark, all-recording and parallel modes for the detector runs, rule B for the cross-validation, a disk-space guard for downloads.

**5–6 Nov – Procurement, new direction, simulation kit.** Order list for the drone fixed (X500 V2 PX4 dev kit, Jetson Orin Nano Super, RealSense D435i, NVMe SSD, battery, charger, cables; optional list beside it). TartanAir finished on three environments (15.1); the VLM decisions for the Jetson made (13.1). The SNN left the thesis (top of this report); the depth-camera baseline, the full-resolution test and ego-rotation compensation run on the workstation; the PX4 + Gazebo kit is ready for the Ubuntu PC (15.2).

## 18. Next steps, all the way to the submitted thesis

The open checks from 29 Oct are closed (section 9), the long workstation runs are done (sections 10–14) and all 32 EvTTC recordings are in. Next: ego-motion (derotation, then background-relative looming) on TartanAir and ODA, then PX4 simulation, first in this cloud workspace. The GPU driver cannot be updated, so CUDA timings will come from the Jetson. No dates are set yet; milestones are to be agreed with the supervisor.

**Updated 6 Nov, after the change of direction** (this replaces the SNN-specific items below where they conflict):

1. Submit the order list; build and set up the drone when it arrives (JetPack on the NVMe from the Ubuntu PC).
2. Run the simulation kit on the Ubuntu PC: `--no-fly` check, then missions to each target in `nd_test`, then `nd_obstacle` (the guard must stop at the pole). Then: avoidance around a blocked corridor, and PX4 Collision Prevention as a comparison.
3. Thesis experiments in simulation: target reached, collisions and near misses, time to goal, VLM latency (Qwen3-VL-4B vs LFM3-VL 2B).
4. On the Jetson: real latencies at 896 px, q4; power; then outdoor flights with GPS, indoor with visual odometry (Isaac ROS VSLAM with the D435i).
5. Side study (simulation only): depth baseline vs event detectors on TartanAir (jobs 129/132), ego-rotation compensation (job 130), full resolution on the other two environments.

1) **Done on 30 Oct** (the open checks from 29 Oct, section 9).
   1. Beta by leave-one-sequence-out cross-validation (9.1), the ego-motion check of the early alarms (9.2), a lower event rate (9.5), an energy floor with its citation (9.6), the RefCOCO and gRefCOCO licences (section 3), a second VLM seed (9.7).
   2. A classical baseline and the like-for-like comparison with the spiking detector (9.3–9.4).
2) **Done on 30–31 Oct, the weekend runs** (section 10).
   1. VLM on the complete test sets, quantisation (q4 / q8 / bf16), image size and two large reference models (10.1–10.4).
   2. Detectors on 17 of 32 EvTTC recordings with cross-validation rules A and B, a lower event rate and 640×360 for both detectors, and a clean CPU-timing run (10.5–10.7).
3) **Done on 2–5 Nov** (sections 11–15).
   1. All 32 EvTTC recordings with cross-validation (11); SNN design and resolution sweep (12).
   2. Newer VLMs (Qwen3.5, Gemma 4), two-step strategies, stricter prompt, thinking mode (13).
   3. ODA real drone flights, four rounds incl. walls, alarm hold, turning (14); TartanAir set up (15).
4) **Detector work still open** (workstation, no new hardware).
   1. Derotation: rotate events back with the angular rate, first on TartanAir (exact poses), then ODA (smoothed gyro, axis mapping by contrast maximisation still to settle).
   2. Background-relative looming: alarm only when a region grows faster than the whole scene; the drone's speed and yaw rate as inhibitory inputs.
   3. TartanAir per-episode results (job 122) and the second and third environments (OldTownSummer, ConstructionSite).
   4. Confirm the one-distance design (3 px) at the same beta as the reference, on EvTTC, MVSEC and ODA.
   5. Alarm hold vs on-time: pick a hold that suits PX4 (no flicker) without keeping the alarm on most of the flight.
   6. Near miss: whether the left / centre / right neurons tell a car in the next lane from one on the path.
   7. Re-tune the classical detector's neighbourhood at 640×360 for a like-for-like comparison.
5) **Workstation limits.**
   1. The NVIDIA driver cannot be updated, so the workstation's VLM latencies stay on Vulkan.
   2. Gazebo still needs WSL2 or a Linux PC: ask IT once more, or the supervisor for a Linux machine.
6) **PX4 software-in-the-loop simulation.**
   1. First in this cloud workspace (Linux, no GPU): PX4 SITL with its built-in physics (SIH, no Gazebo, no camera) to test the MAVLink / OFFBOARD path and the reflex → brake → hold chain with synthetic events. Not tried yet.
   2. Then PX4 SITL + Gazebo with the X500 set to our mass (1.617 kg body) and scaled inertia, on WSL2 or a Linux PC; events from rendered video with v2e; the stack in the loop.
   3. Experiment 5.5: closed-loop ablation (SNN only, depth only, both, neither; textured, blank and thin obstacles): collisions, stops, mission time.
   4. Experiment 5.2: mission cost of VLM latency, using the measured latencies.
7) **Hardware decision with the supervisor.**
   1. Ask whether the lab can lend a Jetson Orin and an event camera (the camera is €2,400 of the €3,533 bench budget).
   2. Agree the scope (bench + simulation as the core, flight as a stretch goal) and the milestones.
8) **Jetson bench tests.**
   1. Experiment 5.1 on the Orin Nano: Qwen3-VL 4B instruct at q4\_K\_M (the choice from 10.2), 2B at q8\_0 as the fallback; image size (at least 1,280 px, 10.3) and cropping or tiling for small targets vs latency, accuracy and energy per query; these are also the CUDA latencies.
   2. Experiment 5.4: power, idle vs SNN vs VLM vs both.
   3. SNN and classical-detector CPU time on the Orin's ARM cores at the real camera resolution.
   4. Fix the drone's parser for whichever model flies (coordinate convention, multi-frame confirmation of detections, which 4B needs because it often draws a box when the target is absent, 10.1).
9) **Real event data from our own camera** (only if an event camera is available).
   1. Experiment 5.3: recordings with and without obstacles; false alarms, alarm distance, TTC; compare with EvTTC/MVSEC.
10) **Flight** *(stretch)*.
    1. Props-off bench run, then a tethered or netted test, then a supervised flight with a safety pilot.
11) **Writing the thesis from this report.**
    1. Chapter 1 Introduction and RQs ← section 1.
    2. Chapter 2 Background ← the related-work list in the thesis plan, plus the classical event-vision methods (Benosman 2014, Clady 2014).
    3. Chapter 3 System design ← sections 1 and 4.
    4. Chapter 4 Implementation ← sections 2, 4 and 11.
    5. Chapter 5 Experiments ← sections 3, 5 and 7 and the methods parts of 9 and 10 (+ Jetson and SITL work).
    6. Chapter 6 Results and discussion ← sections 6, 8, 9 and 10 (+ later results); limitations from 9.8, 10.8 and 11.
    7. Chapter 7 Conclusion and future work.
    8. Supervisor review, revision, submission.

## 19. Open questions and sources

**Open questions.**

- Deadline and milestones: not set yet.
- Can the lab lend a Jetson Orin and an event camera?
- Does the supervisor accept bench + simulation as the core, with real flight as a stretch goal?
- Can IT or the supervisor provide WSL2 or a Linux PC for Gazebo (the driver cannot be updated)?
- Early alarm onsets: they happen while the car drives (9.2, 10.5); still open whether they are background expansion or the target seen early.
- Which operating point does the thesis report for the spiking detector: rule A or B, 426×240 or 640×360 (10.5–10.6)? To fix before the remaining recordings arrive.
- The remaining 15 EvTTC recordings: blocked by Google Drive's download quota since 30 Oct.
- Licences (checked 30 Oct): RefDrone CC BY 4.0, EvTTC BSD-3-Clause-Clear, MVSEC CC BY-SA 4.0. RefCOCO and gRefCOCO state none for their annotations and use COCO images (COCO's terms: annotations CC BY 4.0, images under Flickr's terms). Used for research benchmarking with citation, nothing redistributed; to confirm with the supervisor that this is acceptable.

**Result files** (on the workstation, `C:\Users\esa0146\PycharmProjects\NeuroDrone-Thesis\results\`):

- `REPORT.md`: all tables with 95 % CIs, generated from the raw files by `report_results.py` (sections 6–9; the weekend runs are summarised in section 10 of this report).
- `NOTES_2026-10-29.md`: the methods notes of 29 Oct.
- `environment.txt`: versions.
- `vlm/{full,native,size,seed1}/`: raw answers, sample lists, summaries (29–30 Oct).
- `vlm/{fullsets,quant,size_refdrone,large}/`: the weekend VLM runs (10.1–10.4).
- `events_*`: per-sequence results and traces: `events_beta_curve*` (beta values), `events_cv` and `events_cv_classical` (cross-validation), `events_classical` (and `events_classical_v1`, not used), `events_dedup_1000` and `events_dedup_5000` (lower event rate).
- `events_avail/`: the weekend detector runs on 17 recordings (curves, cross-validation A and B, onsets, lower event rate, `scale2`, `timing`); `events32/` if the remaining recordings arrive.
- `onsets/` and `onsets_hybrid_default/`: the ego-motion check.
- `slices/`: the training-sequence slices the classical baseline was developed on.
- `logs/`: every job's log.

* `vlm/{grid,size,size_grid,quant,new,seed1,zoom,zoom_seed1,zoom_896}/`: 3×3 regions, image size, quantisation, new models, seed 1 and strategies (jobs 116, 125, 127, 131; section 13.1).
* `tartanair2/`: traces and scores; `score_{AF,OldTownSummer,ConstructionSite,ALL}_r{0.3,0.5}/`, `score_*2_*` (sensitive spiking grid, job 126), `score_AFfull_*` (full resolution, job 128), `depth/`, `depth_real/`, `score_real_*` (depth baseline, jobs 129/132), `derot/`, `score_derot_*` (job 130).
* `logs/jobs/`: the log of every job (100–132).

**Sources.**

- [Paredes-Vallés et al., fully neuromorphic vision and control for autonomous drone flight](https://arxiv.org/abs/2303.08778)
- [Qwen2.5-VL technical report](https://arxiv.org/abs/2502.13923) (published RefCOCO scores, Table 6)
- [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631) (RefCOCO-avg per model size, Tables 3–4)
- [RefCOCO, lmms-eval copy](https://huggingface.co/datasets/lmms-lab-encoder/RefCOCO); [refer, the RefCOCO API](https://github.com/lichengunc/refer) (code Apache-2.0)
- [gRefCOCO](https://huggingface.co/datasets/FudanCVL/gRefCOCO); [gRefCOCO repository](https://github.com/henghuiding/gRefCOCO)
- [COCO terms of use](https://cocodataset.org/#termsofuse)
- [RefDrone](https://huggingface.co/datasets/sunzc-sunny/RefDrone)
- [EvTTC](https://nail-hnu.github.io/EvTTC/)
- [MVSEC](https://daniilidis-group.github.io/mvsec/)
- [Benosman et al. 2014, Event-based visual flow, IEEE TNNLS 25(2)](https://doi.org/10.1109/TNNLS.2013.2273537) (time-surface plane-fit flow)
- [Clady et al. 2014, Asynchronous visual event-based time-to-contact, Front. Neurosci. 8:9](https://doi.org/10.3389/fnins.2014.00009) (TTC from event flow)
- [Davies et al. 2018, Loihi: a neuromorphic manycore processor with on-chip learning, IEEE Micro 38(1)](https://doi.org/10.1109/MM.2018.112130359) (Table 2: 23.6 pJ per synaptic spike op, 52 / 81 pJ per neuron update, 0.75 V, pre-silicon)
- [Ollama qwen3-vl tags](https://ollama.com/library/qwen3-vl/tags) (plain tags = thinking variants; q4\_K\_M, q8\_0 and bf16 sizes)
- [ollama/ollama#16449](https://github.com/ollama/ollama/issues/16449) (driver 550+ needed for CUDA kernels since Ollama 0.30)
