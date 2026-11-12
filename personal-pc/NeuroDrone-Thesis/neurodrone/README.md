# neurodrone — milestone 1

A drone stack with two nervous systems and a coordinator:

```
 event camera ──► spiking looming detector (SNN, ~100 Hz, always on) ─┐
                                                                      ├─► Coordinator ──► velocity setpoints ──► PX4 (does ALL stabilisation)
 depth camera ──► range reflex (fail-closed, independent backstop) ───┤        ▲
                                                                      │        │ "is that alarm my target or an obstacle?"
 RGB camera ────► VLM grounding (every 1–10 s, runs in its own server)┘────────┘
```

* **PX4 keeps the drone in the air.** Nothing here touches motors or attitude; we only stream
  body-frame velocity setpoints in OFFBOARD mode.
* **The SNN is a reflex**, not a pilot: it raises an alarm when something expands towards the drone.
* **The VLM is a grounding module**, not a pilot: it says *where the target is in the image* (box),
  from which the coordinator derives a bearing. When a reflex alarm fires, the drone brakes and the
  VLM looks again to decide *target (arrive) or obstacle (dodge)*. Reflex stops; semantics decide.
* **Never fly blind:** if the SNN (or the range sensor, once enabled) goes silent, the coordinator
  holds still, then lands. A dead or slow VLM only costs progress, never safety.

## Two looming detectors (`snn.impl` in the config)

* **`spiking` (default):** every stage is a spiking network: motion-detecting coincidence neurons,
  region expansion neurons with adaptive thresholds, and both-sides alarm neurons
  (`neurodrone/snn_spiking.py`).
* **`hybrid` (baseline):** the original detector (`neurodrone/snn.py`, unchanged): array maths plus
  6 spiking decision neurons.

Both take the same events and give the same `ThreatReport`. How the spiking network works, and how
the two compare (detection, false alarms, packet cutting, closed loop, CPU, spike counts): see
**SPIKING_DETECTOR.md**. Re-run the comparison with `scripts/compare_snn.py`.

## Status — what is verified and what is not

| Part | Status |
|---|---|
| Looming detectors (fully spiking = default, hybrid = baseline), coordinator, Brain, closed-loop sim | **112 automated tests pass** (`pytest`, ~2.5 min; checked 16 Oct 2026 on Python 3.13 + NumPy 2.5.4); earlier versions also passed on Python 3.12 + NumPy 2.4/2.5 and on Python 3.10 + NumPy 1.21.5/2.2.6 (x86-64 only; **not tried on ARM/Jetson**, Python 3.9 untested). Synthetic world only. |
| Pipeline (threads/asyncio) incl. failure injection (sensor unplugged, VLM garbage/hang, pilot not engaged) | tested with fakes |
| VLM client + parser | tested against a fake OpenAI-compatible server, and on 13 Oct 2026 with real Qwen3-VL 2B and Qwen2.5-VL 3B through Ollama on a laptop (RTX 3050 4 GB): 64/64 valid answers, 64/64 correct found/not-found. **Not yet run on a Jetson.** Set `vlm.coords: norm1000` for Qwen models in Ollama. |
| `Px4Vehicle` | API names verified against `mavsdk-grpc` 3.17.4. **Not run against PX4 SITL or hardware.** |
| `DavisLiveSource`, `OpenCVCamera`, `RealSenseClearance` | written from the documented APIs. **Untested (no hardware).** |
| Power sampler (`jtop`/`tegrastats` regexes) | patterns from memory. **Unverified on a real Jetson.** |
| SynSense **Speck** (on-chip SNN) | **not implemented** — see "Hardware notes". Both detectors run in NumPy on the Orin. |

## Quick start

```bash
pip install -e '.[dev]'                 # numpy, pyyaml, pillow, pytest
pytest                                  # everything above
python scripts/sim_closed_loop.py       # closed-loop scenarios, prints a table
python scripts/sweep_vlm_latency.py     # what does VLM latency cost? (sim)
python scripts/sweep_snn.py             # detection range vs false-alarm margin (sim)
python scripts/trace_scenario.py dodge_obstacle --goals   # WHY a scenario ended as it did (mode transitions)

# tune the SNN on YOUR recordings (the important one, see below):
python scripts/make_demo_events.py demo.npz && python scripts/replay_snn.py demo.npz --out trace.csv

# benchmark a VLM (serve it first: llama.cpp / Ollama / vLLM, anything OpenAI-compatible):
python scripts/bench_vlm.py --base-url http://127.0.0.1:8080/v1 --model qwen2.5-vl-3b \
    --images ./bench_images --instruction "the red door" -n 30 --power --out qwen3b.json

# bench the whole stack with a SIMULATED vehicle (props off, real sensors/VLM):
python scripts/run_px4.py --dry-run --instruction "the red door" --source davis --camera 0 --depth realsense

# PX4 SITL, then hardware (pip install -e '.[px4]'):
python scripts/run_px4.py --connect udpin://0.0.0.0:14540 --instruction "the red door" --depth realsense ...
python scripts/run_px4.py --connect serial:///dev/ttyTHS1:921600 ...   # Jetson UART -> Pixhawk TELEM2
```

`run_px4.py` defaults to **pilot engagement**: it streams zero setpoints and does nothing until *you*
switch the autopilot to OFFBOARD on the RC; switch back to take over. It **refuses to fly without the
range sensor** unless you pass `--allow-no-range` (reason below).

> **MAVSDK-Python v4 broke its API.** `pip install mavsdk` now gives a different, native binding.
> This project uses the gRPC wrapper (`pip install mavsdk-grpc`, import `mavsdk_grpc`; or `mavsdk<4`).

## Before any real flight

* SITL first, then props-off bench with `--dry-run`, then tethered/netted, open area, RC in hand with a
  second switch mapped to POSITION/HOLD as your instant escape. Local drone regulations and your lab's sign-off.
* Check your PX4 failsafe parameters (names vary by version; confirm in the PX4 docs): offboard-loss timeout
  and action (`COM_OF_LOSS_T`, `COM_OBL_*`), RC stick override (`COM_RC_OVERRIDE`), geofence (`GF_*`),
  battery (`BAT_*`), and on the Jetson link `MAV_1_CONFIG` / `MAV_1_MODE` / `SER_TEL2_BAUD`.
* Choose `coordinator.cruise_speed` from **measured** alarm range on your own recordings using
  `neurodrone/safety.py`. `brake_decel` and `reaction_s` are assumptions: measure them too.

## What the simulator showed (synthetic: logic checks, not performance predictions)

*These figures are for the **hybrid** detector (milestone 1). For the spiking detector and a side-by-side
comparison see SPIKING_DETECTOR.md.*

Detection range, 1.5 m/s, realistic packetisation, 8 seeds (min–median–max alarm distance):

| obstacle | alarm range | note |
|---|---|---|
| textured wall 3 m | 6.6 – 6.8 – 7.3 m | |
| textured box 2 m / 1.5 m | 5.9 – 6.1 – 6.6 m / 4.9 – 5.3 – 5.3 m | |
| blank (untextured) 3 m wall, dead ahead | 7.9 – 8.4 – 8.9 m | seen through its *edges* only |
| pole 0.4 m | 3.8 – 3.9 – 4.0 m | |
| **pole 0.15 m** | **missed** | thin things are invisible |

Stopping from 1.5 m/s with the assumed 2 m/s² and 0.4 s reaction takes ~1.2 m. Alarm time-to-contact
*falls* as speed rises (3 m wall: 6.3 s at 0.8 m/s → 2.7 s at 5 m/s), so do not extrapolate one number.

False alarms: none under attitude jitter up to 5° at 1–3 Hz (peaks ~57°/s), 30°/s yaw, or 3 Hz/pixel noise.
The false-alarm cliff is near θ≈1.5; the shipped θ=4 keeps a 2–2.7× margin (the rule used: most sensitive
setting that stays ≥2× above the cliff). **Re-derive this on real data**: `scripts/replay_snn.py`.

**What VLM latency costs** (oracle VLM, so timing only; `sweep_vlm_latency.py`):

| VLM latency | open approach: mission time / % time braking-or-waiting | with an obstacle |
|---|---|---|
| 0.5 s | 10.5 s / 14 % | 28.5 s |
| 2 s | 14.0 s / 34 % | 40 s |
| 4 s | 20.0 s / 54 % | – |
| 6–8 s | 32 s / 71 % (8 s) | 78 s (6 s) |

Every run arrived with no collision at every latency: latency costs *progress*, not *safety*.

SNN CPU cost (this sandbox's single x86 core, 346×260): 0.9 ms per 10 ms packet at 1 Mev/s, 2.4 ms at 3 Mev/s.
The spiking detector costs ~2.4–3.6× as much (3.6 ms per packet at 640×480 and ~1.7 Mev/s on a 2-core x86 sandbox).
The Orin's ARM cores will be slower (unmeasured): run `replay_snn.py` on the device.

## Known limitations (read these)

1. **Blank surfaces are invisible to the event-camera reflex.** A textureless wall taller than the field of
   view, approached off-centre, produced *no alarm* and the simulated drone flew into it
   (`blank_wall_snn_only`; asserted by a test so it can't silently change). Hence the range reflex.
2. **Thin obstacles** (poles, wires) are missed or detected late; the range sensor may miss wires too.
3. **Range sensor caveats:** depth cameras struggle with glass, black surfaces, sunlight and range beyond
   a few metres. `invalid_policy="blocked"` (default) fails closed: indoors safe, but it will not fly in open
   space with no returns; `"free"` accepts the risk outdoors.
4. **Reactive avoidance has no memory.** After a sidestep it can turn back toward a target hidden behind the
   obstacle and need an escalating second dodge; wide obstacles end in "stuck → hold". It needs a local map.
5. **Rotation corrupts the side (left/right) urgencies** (no gyro derotation); the alarm path is immune
   to rotation *false alarms* by construction, which is why the side-push is off by default. But rotation also
   *delays* real alarms, more so in the spiking detector: approaching a 2 m box with 3° of attitude shake it
   alarmed at 1.7 m (hybrid 2.8 m; both ~6–9 m without shake). Gyro derotation would fix both.
6. **"Arrived" is a heuristic:** alarm + VLM says the target is centred and ≥15 % of the image width (was 20 %,
   lowered on 16 Oct because the spiking detector alarms earlier). It ends ~5–8 m from the target (the alarm
   range); a proper standoff approach under the range sensor is next.
7. Small VLMs are poor at precise localisation; the prompt asks for a normalised box and the parser accepts
   0–1, pixel and 0–1000 coordinates, but nothing here measures *your* model's accuracy: use `bench_vlm.py` with labels.

## Things the tests caught in my own first design (so you can cite the reasoning)

* A signed sum over the centre sector reached 0.94 of the alarm threshold under mere attitude jitter with a
  textured object in view → replaced by mirrored left/right **coincidence neurons per radial band**
  (translation gives mirror pixels opposite sign; a static obstacle must straddle the direction of motion).
* Packet boundaries matter: aligned with the simulator's frames the 3 m wall alarmed 1.2 m earlier than for
  any other alignment. The simulator now cuts packets off-grid with jittered length (`PacketCutter`).
* Slowing down on SNN urgency starved its own alarm (looming flow ∝ speed): alarm at 1.3 m instead of 5 m. Removed.
* The advisory side-push drifted the drone off the target line (the target looks like an obstacle up close). Off.
* After a sidestep, pre-dodge VLM bearings point the wrong way → goals captured before a lateral move are invalidated.
* Alarm decisions used a ~3 s stale VLM box and the drone dodged its own target → brake, then adjudicate on a *fresh* image.
* The range-based speed cap sat exactly on the range-alarm boundary → 25 % buffer.

## Hardware notes

* **Event camera:** an iniVation DAVIS/DVXplorer works through `DavisLiveSource` (untested). For anything else,
  subclass `EventSource` and return `EventPacket`s (≤ 10 ms each).
* **SynSense Speck** runs its *own* on-chip convolutional SNN and hands you classification spikes, not raw events,
  so the NumPy detector here cannot run on it as-is: that needs a sinabs-trained network and training data, and I
  did not write the `samna` adapter without being able to check the API or know which kit you have.
  Whether the Speck *demo* kit exposes data to a host at all is something to confirm with SynSense.
* **VLM server:** any OpenAI-compatible endpoint (llama.cpp `llama-server --mmproj`, Ollama `/v1`, vLLM). On Jetson,
  CUDA-enabled builds usually come from `dusty-nv/jetson-containers`. Flags change between versions: check each project's docs.
* **Depth:** `RealSenseClearance` (pyrealsense2). Corridor/percentile logic is unit-tested; the camera I/O is not.

## Suggested thesis experiments (each is one script + a log)

1. `bench_vlm.py` grid: model × quantisation × image size → latency, tokens, parse rate, localisation error, J/query.
2. `sweep_vlm_latency.py` with *your measured* latencies → mission time vs VLM period (the table above, on your numbers).
3. `replay_snn.py` on real recordings: false alarms in obstacle-free flights, alarm TTC on known obstacles.
4. Power: SNN (CPU) vs VLM vs idle, same board (`--power`).
5. Ablation in SITL: SNN-only vs range-only vs both vs neither, against textured / blank / thin obstacles.

## Layout

```
neurodrone/  types config snn snn_spiking reflex coordinator safety vlm events depth camera vehicle runtime telemetry sim closedloop
scripts/     run_px4 bench_vlm replay_snn compare_snn sim_closed_loop trace_scenario sweep_vlm_latency sweep_snn make_demo_events
config/default.yaml   every tunable (a test keeps it in sync with the code)
tests/       112 tests: both detectors, coordinator, VLM, pipeline, scripts, simulator
```

Next steps I'd take: a Gazebo/PX4-SITL camera + event bridge (v2e can turn rendered video into events), a
standoff approach under the range sensor, a small obstacle memory for the dodge, gyro derotation, and the Speck path.
