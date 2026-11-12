# NeuroDrone thesis: status, risks, open questions

11 Oct 2026. Written after reading HANDOFF.md and neurodrone/README.md.

## What I understood

- **Thesis:** can a drone run a vision-language model (VLM) onboard? A small VLM on a Jetson-class computer finds a named target every 1-10 s. An event camera + spiking neural network (SNN) reflex runs at ~100 Hz and stops the drone. A depth-camera reflex backs it up. PX4 does all the stabilisation.
- **Code:** `neurodrone/` (Python, ~3,700 lines): SNN looming detector, coordinator state machine, VLM client, PX4 adapter, synthetic simulator.
- **Checked today:** I re-ran the test suite here: 85/85 pass (Python 3.13, NumPy 2.5, x86).
- **Not tested at all:** PX4 SITL, any real sensor, any real VLM, the Jetson, the power logger. Every number so far comes from the synthetic simulator.

## Biggest risks

1. **Everything is synthetic.** Real event cameras add noise and light flicker. The SNN thresholds must be re-tuned on real recordings, and none exist yet.
2. **Hardware is unconfirmed.** Without a real event camera, the "neuromorphic" part becomes simulated events (v2e), which weakens the thesis claim. The full sensor set also needs a drone big enough to carry it.
3. **Integration is the time sink.** PX4 SITL + Gazebo + an event-camera bridge has never been run. Your PC runs Windows, so PX4 SITL needs WSL2 (Ubuntu) or a Linux machine.
4. **Small VLMs may be too slow or too imprecise on a Jetson.** This is cheap to find out early with `bench_vlm.py` (experiment 5.1 in the plan).
5. **Scope.** Real flight should stay a stretch goal; bench + simulation results are a complete bachelor thesis.
6. **Citations.** I checked the five arXiv links in the plan. Which "neuromorphic paper" your professor meant is still a guess (best guess: Paredes-Valles et al., TU Delft).

## What I need from you before the next step

1. Confirm the thesis is this SNN + VLM single-drone project. I put the MCP swarm idea from earlier in our chat under future work.
2. The hardware you actually have, or the lab provides:
   - Jetson model (Orin Nano / NX / AGX?)
   - event camera model, or none
   - depth camera model
   - drone frame, flight controller and PX4 version
   - your PC's GPU
3. Which neuromorphic paper your professor meant (title or PDF).
4. Your deadline, or how many weeks you have.
5. Your PC seems to have Ollama installed. Can I use it for a first VLM test?

## Next step (once I have 1 and 2)

Run experiment 5.1 (VLM benchmark) on whatever GPU you have, and set up PX4 SITL in parallel.

## Files in this folder

- `Thesis_Plan_ToDo.docx`: the to-do list for your professor (a Word copy of the shared doc).
- `Status_Risks_Questions.md`: this note.
