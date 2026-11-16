# Discussion {#sec:discussion}

## RQ1: How onboard perception works, and what it depends on

The perception module that came out of this work has a clear division of labour. The VLM answers one narrow question — where in this picture is the described object — and does so slowly and sometimes wrongly. Everything that has to be fast or reliable is done by classical code around it: the depth image turns a box into a place, a consensus of answers filters out invented objects, the search decides where to look, and a depth guard at 20 Hz keeps the drone away from obstacles whatever the VLM is doing. This split is what makes a model with seconds of delay usable on a moving vehicle.

The factors that decided whether the module worked were, in the order in which they appeared:

Table: Factors that determine whether onboard VLM perception works, with the evidence from this thesis. {#tbl:factors}

| Factor | What was found | Where |
|---|---|---|
| Model choice | 4B parameters at 4 bit were as accurate on drone views as 30B; the 2B lost a third of the targets | Chapter 4 |
| Answer format | boxes beat region names; a "not found" option changes some models' behaviour strongly; thinking modes are harmful | Chapter 4 |
| Hallucination | the best small model invents an absent object in about half of the cases; one answer must never move the drone | Chapters 3, 4 |
| Latency and motion | an answer is about a picture that is already old; the drone must hold still while looking, and answers to pictures taken while moving must be discarded | Chapter 5 |
| Shared compute | VLM and rendering (on the drone: VLM and VIO) competing for one GPU stalled the VLM completely | Chapter 5 |
| Sensor freshness | depth pictures can freeze or lag while still arriving; the guard needs the picture's age and a frozen-picture check, not just "the newest message" | Chapter 5 |
| Geometry of the box | depth inside a box can belong to an occluder; boxes at the image edge are unreliable | Chapter 5 |
| Memory | the 8 GB of the Jetson must hold the model, the odometry and the operating system together | Chapter 6 |

Several of these are not properties of the model at all but of the system around it, and most of them would not show up in a benchmark of the model alone. This is the main answer to RQ1: whether onboard perception works depends as much on timing, freshness and the handling of wrong answers as on the model's accuracy.

## RQ2: Which model

Qwen3-VL-4B-Instruct at 4 bit gave the best grounding on drone views of all models that fit an 8 GB computer (69 % at IoU 0.5 on RefDrone), with no measurable loss against 8- and 16-bit versions and no gain from models up to 32 B. Accuracy did not grow with size beyond 4 B on drone views, while latency grew roughly with the active parameter count (0.38 s for 2 B, 0.57 s for 4 B, 0.83 s for 8 B, 2.5 s for 32 B on the workstation). The likely reason is that small targets are limited by the number of image tokens the server gives each picture, not by the model's capacity; tiling the picture, which gives each target more tokens, was the only change that raised drone-view accuracy (to 73 %). The larger models' real advantage was in saying "not found" correctly, which the drone gets more cheaply from its consensus rule.

The answer comes with two caveats. First, latencies were measured on a desktop GPU; the Jetson measurement (stage 1) decides whether the 4B model is fast enough on board or whether the 2B-class LFM3-VL with tiles is the better trade. Second, RefDrone is the best available proxy, not the lab: its targets are cars and people seen from greater heights.

## RQ3: Deployment

The stack runs unchanged in simulation and on the drone because the interfaces were chosen to be standard: MAVLink and OFFBOARD velocity setpoints to PX4, an HTTP call to the VLM server, and camera images with time stamps. The VLM can therefore run on board (pipeline A) or on a ground station (pipeline B) without changing the mission code, while the guard always stays on board. Given Chapter 4's result that larger models do not draw better boxes on drone views, pipeline A loses little by running a small model, and gains independence from the network. The open risks of the deployment are measured in the staged plan: the Jetson's latency and memory with VLM and visual odometry together, the quality of the real depth on thin and dark obstacles, and the drift of the visual odometry, which replaces the motion capture used in comparable work.

## Sweep versus active search

The active search, which chooses the next view from the VLM's answers, needed fewer answers only when the fixed sweep's order was wrong for the target (the barrel on the left), and was slightly slower when the sweep's order happened to be right (the green box on the right), because it spends one extra turn centring a cut-off target. That extra turn is not wasted: a centred target gives a correct depth, which the edge rule otherwise has to reject. With five runs per case and a simple world the difference is small; the benefit should grow with more targets out of view and with a VLM whose answers are expensive, and it should not shrink on the Jetson, because the number of answers, not their duration, is what the active search saves.

## Limitations

- **No real flight yet.** All closed-loop results are from simulation with perfect depth and a VLM about ten times slower than expected on the Jetson. The deployment plan addresses this, but its results are not part of this draft.
- **Occlusion and memory.** The two design limits found in simulation — depth from an occluder inside the box, and a go-around without memory of the obstacle it just passed — caused all failures and close passes. Both have straightforward fixes (a depth-cluster check inside the box; keeping the side-step until the obstacle is behind the drone), which were not implemented so that the reported series tests one fixed design.
- **Benchmark scope.** One prompt per model; latency on a desktop GPU; RefDrone as the proxy for drone views.
- **Statistics.** Five runs per case can detect only large effects.

## Future work

The next steps follow from the limitations: the measurements on the Jetson and the lab flights of the staged plan; the occlusion check and the obstacle memory; a richer test world in which targets can be behind the drone; tiling the picture on the drone, which raised drone-view accuracy in the benchmark; and, if the Jetson allows it, two VLM calls per view (full picture and tiles) combined in the consensus.
