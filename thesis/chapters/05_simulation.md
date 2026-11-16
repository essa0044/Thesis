# Closed-Loop Simulation {#sec:sim}

Before any hardware was available, the complete stack was flown in PX4 software-in-the-loop simulation with Gazebo. The simulation answers two questions: does the chain from sentence to goal to safe flight work in closed loop, and does the active-sensing search need fewer VLM answers than a fixed sweep? On the way it also showed which practical factors break such a system — most of them timing problems — which feeds into RQ1.

## Set-up

**Software.** PX4 v1.17.0 [@meier2015px4] in software-in-the-loop mode with Gazebo Harmonic (gz-sim 8.15.0) [@gazebo], the stock X500 quadrotor with a depth camera (`gz_x500_depth`: an RGB camera of 1920 × 1080 pixels with a 69° horizontal field of view, and a depth camera of 640 × 480 pixels with a 73° field of view). The mission code of Chapter @sec:design connects through MAVSDK-Python [@mavsdk] and reads the camera images directly from Gazebo's transport layer. The VLM is Qwen3-VL-4B-Instruct (q4_K_M) in Ollama 0.40.16.

**Machine.** A Windows 11 laptop (Intel Core i5-12500H, NVIDIA RTX 3050 Laptop GPU with 4 GB) runs PX4, Gazebo and the mission code in WSL2 (Ubuntu 24.04) and Ollama on Windows. Gazebo renders its cameras on the NVIDIA GPU through WSL's Direct3D 12 route, without a window. The 4 GB GPU cannot hold Gazebo's rendering and the model at the same time, so the VLM runs entirely on the CPU and takes about **50 s per answer** — two orders of magnitude slower than on a desktop GPU and probably one order slower than on the Jetson. The simulation therefore tests the logic under an extreme VLM delay, not the timing of the real system.

**Worlds and targets.** Two worlds were built on PX4's default world (Figure @fig:worlds). Both contain four objects: a red box 12 m straight ahead of the start, a blue barrel at (8 m, 5 m to the left), a green box at (9 m, 6 m to the right) and a white box as a distractor. At the start, the red box is in the middle of the camera image, while the barrel and the green box are cut off at the left and right image edges. The world `nd_obstacles` adds one yellow pole (radius 0.15 m) on the way to each target, 0.2 m off the straight line, so that the drone must go around it.

![The two simulation worlds seen from above. The drone starts at the origin facing +x. Poles are only present in nd_obstacles.](figures/fig_worlds.png){#fig:worlds width=80%}

**Protocol.** Each target ("the red box", "the blue barrel", "the green box") was flown five times in each world with each search: 3 targets × 2 worlds × 5 runs × 2 searches = 60 runs. Every run starts a fresh PX4 and Gazebo. The drone takes off to 2 m, searches, flies to the accepted goal, hovers and lands, or gives up after 360 s. A run that fails because the simulator itself did not start (PX4 refuses to arm, take-off times out) is repeated once and not counted. Gazebo's depth image is perfect (no noise, no missing pixels); this was kept on purpose, so that the later comparison with the real RealSense isolates the sim-to-real difference.

**Success criteria.** A run succeeds when the drone (1) reports arrival, (2) at the right target, (3) with a final distance of 0.5–2.5 m between the drone's centre and the target's surface, and (4) without propeller contact with a pole. The propeller tips of the X500 V2 reach 0.42 m from its centre (arm 0.25 m plus half a 13-inch propeller); the closest *propeller clearance* is the smallest distance between a pole's surface and that circle. Positions are PX4's estimates, which in simulation are close to the true positions.

## Validating the simulation harness

The first series failed for reasons that had nothing to do with the method, and each failure revealed a factor that will matter on the real drone as well. They are listed here because they are the practical answer to the second part of RQ1: what makes onboard perception work or fail.

1. **The VLM did not answer at all** while Gazebo ran: model and renderer competed for the 4 GB GPU, and the GPU work stalled. Fix: the model on the CPU, Gazebo alone on the GPU. *Lesson:* VLM and other GPU users (on the Jetson: visual odometry) must be planned for together.
2. **The search never confirmed a target.** The drone kept turning while the VLM answered; by the time the answer came, the camera looked elsewhere, and two answers never agreed. Fix: turn, stop and look (Section @sec:search). *Lesson:* with a VLM that answers every few seconds, the drone must hold still for each look.
3. **Every view cost two answers,** because the answer already running when the drone stopped belonged to a picture taken while turning. Fix: the VLM worker is held while the drone turns and the first picture after a turn must be newer than the end of the turn.
4. **The goal was in front of a pole.** A target cut off at the image edge had the pole in front of the visible part; the median depth in the box was the pole's. Fix: no goal from a box touching the image edge (Section @sec:perception).
5. **The guard saw the floor** when the drone pitched forward. Fix: the guard's corridor is levelled with the drone's attitude.
6. **The guard was blind with fresh-looking images.** After a Windows update, Gazebo's depth pictures (a) froze while carrying new time stamps, and later (b) arrived 4–5 s late, because Gazebo rendered on the CPU and a 30 Hz full-HD RGB stream filled the message queue; the drone passed poles with almost no clearance. Fixes: Gazebo back on the GPU; the RGB camera rendered at 2 Hz (the VLM needs a picture every few seconds); and two new checks in the stack — the age of each picture in simulation time, and a watchdog that treats a depth picture as frozen if it does not change while the drone moves 0.5 m. *Lesson:* "the newest message" is not "a recent picture"; on the real drone a USB camera can freeze or lag in the same way, and the same checks apply with the RealSense's frame time stamps.
7. **The drone sank or rose during long searches,** because zero velocity left the altitude to PX4's estimate drift. Fix: the altitude is held actively in every phase.

After these fixes, the 60 runs below were flown in one session. Their health checks were clean: no frozen depth, 99 % of the depth images younger than 0.63 s, altitude while searching 1.4–2.2 m, no PX4 failsafe.

## Results

Table @tbl:simresults lists the results per case, and Figure @fig:simresults summarises them.

Table: Results of the final series (5 runs per row). Time to goal and stand-off are medians of the successful runs; propeller clearance is the minimum over all runs of the row. {#tbl:simresults}

| World | Target | Search | Success | VLM answers (median) | Time to goal (s) | Stand-off (m) | Min. propeller clearance (m) |
|---|---|---|---|---|---|---|---|
| empty | red box | sweep | 5/5 | 2 | 117 | 1.88 | – |
| empty | red box | active | 5/5 | 2 | 123 | 1.91 | – |
| empty | green box | sweep | 5/5 | 4 | 206 | 2.04 | – |
| empty | green box | active | 5/5 | 4 | 220 | 2.05 | – |
| empty | blue barrel | sweep | 5/5 | 5 | 263 | 1.89 | – |
| empty | blue barrel | active | 5/5 | 4 | 218 | 1.91 | – |
| poles | red box | sweep | 5/5 | 2 | 113 | 1.92 | 0.23 |
| poles | red box | active | 4/5 | 2 | 113 | 1.96 | 0.48 |
| poles | green box | sweep | 4/5 | 4 | 219 | 2.04 | 0.04 |
| poles | green box | active | 5/5 | 4 | 223 | 2.05 | 0.59 |
| poles | blue barrel | sweep | 4/5 | 6 | 241 | 1.98 | 0.40 |
| poles | blue barrel | active | 5/5 | 3 | 160 | 2.00 | 0.55 |

![Simulation results: (a) successful runs per target and search, (b) VLM answers needed per successful run.](figures/fig_sim_results.png){#fig:simresults width=100%}

**Overall.** The stack reached the named target in **57 of 60 runs, with no collision**: 28 of 30 with the sweep and 29 of 30 with the active search (Fisher's exact test p = 1.0, no difference in success at this sample size). The median stand-off was 2.0 m from the target's surface (1.5 m aim from the drone's centre plus braking). In the worlds with poles, the drone went around every pole it met except one, spending a median of 3–4 s in the go-around.

**Search effort.** The two searches differ only where the target starts (Table @tbl:effort). For the red box, in view at the start, both needed the minimum of two answers. For the green box, cut off at the right edge, both needed four; the sweep's first turn (+30°, to the right) happens to be the right one, and the active search was about 10 s slower because it first turns to centre the cut-off box (Mann-Whitney p = 0.03). For the barrel, cut off at the left edge, the sweep first looks to the right — the wrong side — and needed 4.7 answers on average, the active search 3.7 (p = 0.04), 46 s less.

Table: Search effort in the successful runs (means). {#tbl:effort}

| Target | At the start | VLM answers, sweep / active | Time to goal (s), sweep / active |
|---|---|---|---|
| Red box | centre of the image | 2.0 / 2.0 | 115 / 115 |
| Green box | cut off at the right edge | 4.0 / 4.0 | 213 / 223 |
| Blue barrel | cut off at the left edge | 4.7 / 3.7 | 248 / 202 |

With 50 s per answer, one answer is worth 50 s here; on the Jetson, with an expected few seconds per answer, the time saved will be small, but the number of answers — and with it the number of chances for a wrong one — stays lower. A world in which targets can be behind or far to the side of the drone would show the difference more clearly; it was not run.

## Failure analysis

The three failures and the two close passes all come from two limits of the design (Figure @fig:failures):

- **Occlusion inside the box (2 runs: sweep, green box; active, red box).** The pole stood in front of the target inside the VLM's box, the median depth in the box was the pole's, and the drone "arrived" 5.8 m and 7.3 m short of the target. The edge rule only covers boxes at the image edge. Possible fixes: take the farther of two depth clusters in the box, or check that the depth agrees with the size of the box.
- **No free way around (1 run: sweep, barrel).** At the pole none of the directions within ±30° was free for 2.5 m, and the drone stayed stopped until the time limit. Stopping is the safe outcome, but the mission fails.
- **Close passes after the go-around (sweep, green box: 0.04 m propeller clearance; sweep, red box: 0.23 m).** After the side-step, the pole left the camera's view; the drone turned back towards the goal and cut the corner beside the pole, which it no longer saw. Over all obstacle runs, the median propeller clearance was 0.53 m (sweep) and 0.59 m (active). The difference between the searches here is chance: it depends on the path after the go-around, not on the search. A short memory of the obstacle (keep the side-step until the obstacle is behind the drone according to its own position estimate) would remove it.

![Two failure modes seen from above: (a) the pole inside the VLM's box gives a goal in front of the pole; (b) after the side-step the pole is outside the camera's view and the drone cuts the corner.](figures/fig_failures.png){#fig:failures width=90%}

## Threats to validity

- **Perfect depth.** Gazebo's depth image has no noise, no holes and no reflections. The real RealSense will be worse, especially on thin poles, dark or shiny surfaces and at short range; this is measured in stages 3 and 8 of the deployment plan.
- **Camera geometry.** The simulated depth camera sees 73° horizontally; the real D435i sees about 87° and works best between about 0.3 and 3 m [@intel_d435i]. The guard's corridors and the go-around angles were chosen to stay inside the narrower simulated view.
- **VLM latency.** About 50 s per answer is unrealistically slow. It makes searches long and favours the search that needs fewer answers; it does not affect the guard, which runs at 20 Hz independently of the VLM.
- **Simple scenes.** Four objects with distinct colours on a flat plane, one pole per path. The VLM had an easy task; real lab images with clutter and similar objects will be harder (Chapter 4 suggests about 70 % found per answer on drone images).
- **Sample size.** Five runs per case can show large differences only; the success rates of the two searches cannot be told apart, and the effect on VLM answers is significant only for the barrel.
- **Simulator artefacts.** Several harness problems were found and fixed during the work; earlier series affected by them were discarded and are not reported. The final series ran without any of them.
