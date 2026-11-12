# Simulation kit: VLM goal + depth guard in PX4 SITL / Gazebo

What runs: the Gazebo X500 with a depth camera (`gz_x500_depth`, the PX4 model closest to your drone) ->
Qwen3-VL picks the target from the RGB image -> the depth image gives its distance -> a goal 1.5 m short of it ->
the drone turns to it and flies; a classical depth guard stops it if anything is in its path
(closer than 1.2 m or under 2 s to contact). No SNN.

## Setup (Ubuntu 22.04 / 24.04)
    bash sim/setup_sim_ubuntu.sh            # or OLLAMA_URL=http://<workstation-ip>:11434 bash sim/setup_sim_ubuntu.sh

## Run
    # terminal 1
    cd ~/PX4-Autopilot && PX4_GZ_WORLD=nd_test make px4_sitl gz_x500_depth
    # terminal 2 - first only look (no flying): checks cameras, VLM and guard
    ~/nd-venv/bin/python scripts/sim_mission.py --no-fly --instruction "the red box" --ollama http://<ip>:11434 --save-frames frames
    # then fly
    ~/nd-venv/bin/python scripts/sim_mission.py --instruction "the red box" --ollama http://<ip>:11434 --save-frames frames

Targets in the worlds: "the red box" (12 m ahead), "the blue barrel" (front left), "the green box" (front right),
"the white box" (far left). `nd_obstacle` adds a yellow pole between the start and the red box.

Every run writes `mission_<time>.jsonl` (VLM answers, goals, guard distances, commands). Ctrl+C lands.
Useful options: --vmax (m/s), --alt, --vlm-period, --model lfm3-vl:2b, --search-yaw 15 (turn while nothing found).
