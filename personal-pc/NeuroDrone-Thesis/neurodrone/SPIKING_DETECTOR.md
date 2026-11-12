# The fully spiking looming detector (16 Oct 2026)

**What changed:** the looming reflex is now a spiking network at every stage (`neurodrone/snn_spiking.py`),
and it is the default (`snn.impl: spiking`). The earlier detector (`neurodrone/snn.py`, now called the
*hybrid*) is unchanged and selectable with `snn.impl: hybrid`. It is the baseline for comparison.

**Why:** in the hybrid, only the last 6 neurons spiked. Motion detection, normalisation and the
"both sides" test were ordinary array maths. The new version does the same job with spiking
neurons throughout, using the same image geometry and the same motion-detector time window. It
needs no training.

All numbers below come from the **synthetic simulator** (128×96 event camera, ideal sensor model).
They compare the two designs and check the logic. **They do not predict performance on a real
sensor.** Both detectors must be re-tuned on real recordings (`scripts/replay_snn.py`).

## The network

| Layer | Neurons | What one neuron does | Replaces (in the hybrid) |
|---|---|---|---|
| 0 input | one spike per camera event | pixel got brighter / darker | – |
| 1 motion | **coincidence neurons**, 2 per pixel × polarity × motion baseline (OUT, IN). 77,856 at 128×96; ~1.9 million at 640×480 | A spike from the neighbour on the image-centre side *opens a window* (2–150 ms). A spike from its own pixel *fires it*, but only while the window is open. So OUT fires when an edge just moved outward; IN is the mirror image. These are "time-difference encoder" neurons, a known spiking motion detector (Milde et al. 2018). | timestamp comparisons in NumPy |
| 2 region | **9 expansion neurons** (leaky integrate-and-fire): left/centre/right thirds, plus the left and right half of 3 vertical strips | OUT spikes from its region push it up, IN spikes push it down, and it leaks back to 0 (50 ms). Its threshold rises when the region is busy (adaptive threshold), so texture or noise alone can't fire it. Only a clear outward majority does. | z-score normalisation |
| 3 alarm | **3 coincidence neurons**, one per strip | Fires when the strip's left-half and right-half expansion neurons spike within 20 ms of each other. An obstacle on a collision course expands on *both* sides; turning, shaking and sideways motion make one side contract, so that side stays silent. | `min(z_left, z_right)` |
| readout | spike-rate traces | alarm = at least 2 alarm spikes within ~70 ms (one alone never counts) | same |

**Tried and removed:** an inhibitory "veto" from contraction neurons in layer 3. Switching it off
changed nothing in any test scenario, because the both-sides coincidence already does its job. Every
remaining neuron type has a measurable effect.

**Simulation detail that matters:** the network is **independent of how the event stream is cut into
packets**. Layer 1 is evaluated exactly per event, so each event sees every earlier event, even in
the same packet. Layers 2–3 run on a global 1 ms clock. The hybrid only compares a packet with
*earlier* packets, so its result depends on where the packet boundaries fall (see below).

## Results: spiking vs hybrid, same event streams

**Detection.** Time-to-contact at the first alarm, flying at 1.5 m/s. Higher means earlier warning.
Min / median / max over 5 noise seeds.

| Obstacle | Spiking | Hybrid |
|---|---|---|
| textured wall 3 m | **6.2 / 6.2 / 7.1 s** | 4.4 / 4.6 / 5.5 s |
| textured box 1.5 m | **4.1 / 4.2 / 4.4 s** | 3.3 / 3.5 / 3.5 s |
| blank wall 3 m, dead ahead | **7.3 / 7.3 / 7.3 s** | 5.6 / 5.6 / 6.0 s |
| pole 0.4 m | 2.7 / 3.5 / 3.5 s, **missed in 1 of 5** | 2.5 / 2.5 / 2.7 s, never missed |
| pole 0.15 m | missed | missed |

**False alarms.** There were none for either detector, in any of the 7 scenes × 5 seeds: hovering
with 3° shake (object ahead or to the side), forward flight with 5° shake, with and without 10× sensor
noise, turning at 30°/s, 10× noise, and an object passing to the side. The spiking defaults keep a
1.5–2× margin above the settings where false alarms start (theta0 ≈ 3–4, beta ≈ 0.04–0.08); a test
enforces this.

**Packet cutting.** Same 3 m wall stream, cut three ways. Time-to-contact at the first alarm:

| | Default (jittered) | Aligned with sensor frames | Jittered, other seed |
|---|---|---|---|
| Spiking | 6.19 s | 6.19 s | 6.19 s |
| Hybrid | 4.42 s | 5.24 s | 5.24 s |

**Closed loop.** The whole system flying, with an oracle VLM. All 9 scenarios end the same way for
both detectors: 6 arrive, 2 hold or land as designed (no VLM; reflex failure), and both crash into the
blank wall without the range sensor (a known limitation, asserted by a test). The spiking detector alarms earlier, so the drone
stops further from its target: 6.3 m vs 4.9 m in `approach_target`.

**Weakness: shaking delays the alarm more.** Approaching a 2 m box, first alarm at:

| Attitude shake | 0° | 1.5° | 3° |
|---|---|---|---|
| Spiking | 8.7 m | 3.2 m | 1.7 m |
| Hybrid | 6.1 m | 3.8 m | 2.8 m |

Rotation makes one half of the image contract, which silences that half's expansion neuron, so the
both-sides coincidence fires late. A longer coincidence window (40 ms) didn't help. The proper fix
for both detectors is **gyro derotation**: subtracting the motion caused by rotation, measured by
the IMU. In the closed-loop `jitter_approach` scenario both still arrived without collision (minimum
clearance 1.1 m spiking vs 2.0 m hybrid).

**Compute (this computer, not the Jetson).**

| | Spiking | Hybrid |
|---|---|---|
| Per 10 ms packet, 128×96 sim scenes | 0.47 ms mean, 1.1 ms p95 | 0.13 ms mean, 0.35 ms p95 |
| Per 10 ms packet, 640×480, wall approach at ~1.7 M events/s | 3.6 ms | 1.5 ms |

So the spiking version uses about 2.4–3.6× the CPU. On the Orin's ARM cores it will be slower still,
which is **unmeasured**. If it's too slow at 640×480, halving the resolution is the usual remedy.

**Activity**, for energy estimates on neuromorphic hardware. Per second of approach flight (128×96):

| Events | Layer-1 spikes | Layer-2 spikes | Layer-3 spikes | Synaptic operations |
|---|---|---|---|---|
| 9,816 | 11,628 | 91 | 36 | ~102,000 |

Converting synaptic operations to energy needs a per-operation energy figure from a chip's
published specs. That's still to do, with a proper citation.

## One coordinator change

`coordinator.arrive_w_frac` went from 0.20 to 0.15. That's the minimum box size for an alarm to
count as "this is the target".

The reason: the spiking detector alarms earlier, when the target still looks smaller. At 0.20 it
made one unnecessary sidestep in front of its own target (`bearing_offset`). With the hybrid, all 9
closed-loop scenarios give identical results at 0.15 and 0.20.

## How to run

```bash
pytest                                              # 112 tests, ~2.5 min
python scripts/compare_snn.py --out results/snn_compare.json   # the tables above (1 seed), ~2 min
python scripts/sim_closed_loop.py --impl hybrid     # any scenario script, either detector
python scripts/sweep_snn.py --theta0 4 6 8 --beta 0.08 0.12    # threshold sweep (spiking)
python scripts/replay_snn.py rec.npz                # real recordings; detector from snn.impl in the config
```

## Still to do

1. **Re-tune on real event recordings.** Both detectors are tuned on synthetic events only.
2. **Measure CPU time on the Orin** at the real camera resolution.
3. **Gyro derotation** (fixes the delayed alarm under shaking, for both detectors).
4. **Energy estimate** from the synaptic-operation counts, with a cited per-operation figure.
