#!/usr/bin/env bash
# Simulated test series: every target x every world x RUNS repetitions, each run in a fresh PX4 + Gazebo.
#
#   cd ~/Thesis/neurodrone && bash sim/run_series.sh                 # 3 targets x 2 worlds x 5 runs = 30 runs
#   RUNS=1 bash sim/run_series.sh                                    # quick check: 6 runs
#   HEADLESS=1 bash sim/run_series.sh                                # no Gazebo window (less GPU load)
#   TARGETS="the red box" WORLDS=nd_obstacles RUNS=3 bash sim/run_series.sh
#
# Needs: sim/setup_sim_ubuntu.sh done once (PX4 built, ~/nd-venv), Ollama reachable at OLLAMA (default
# 127.0.0.1:11434) with MODEL pulled. About 2-4 min per run. Writes OUT/<world>__<target>__<run>.jsonl, the PX4 output
# of every run, the VLM frames, and at the end OUT/summary.{csv,json} + the table (scripts/analyze_series.py).
# Stop with Ctrl+C: the current run is cleaned up and the table is made from the runs so far.
set -u
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
VENV="${VENV:-$HOME/nd-venv}"
OLLAMA="${OLLAMA:-http://127.0.0.1:11434}"
MODEL="${MODEL:-qwen3-vl:4b-instruct}"
RUNS="${RUNS:-5}"
WORLDS="${WORLDS:-nd_test nd_obstacles}"
TARGETS="${TARGETS:-the red box;the blue barrel;the green box}"
OUT="${OUT:-$KIT/results/series/$(date +%Y%m%d_%H%M)}"
HEADLESS="${HEADLESS:-0}"
MAX_TIME="${MAX_TIME:-180}"
SEARCH_YAW="${SEARCH_YAW:-8}"
NUM_CTX="${NUM_CTX:-2048}"      # same in the warm-up and every call, or Ollama reloads the model
VLM_MAX_MEDIAN="${VLM_MAX_MEDIAN:-30}"  # stop before the series if a VLM call takes longer than this (s)   # deg/s turn while no goal is accepted (the barrel and green box are near the view's edge)

mkdir -p "$OUT/px4" "$OUT/frames"
cp "$KIT"/sim/worlds/nd_*.sdf "$PX4_DIR/Tools/simulation/gz/worlds/"
curl -s --max-time 5 "$OLLAMA/api/version" >/dev/null || { echo "no Ollama at $OLLAMA"; exit 1; }
{
  echo "date: $(date -Is)"; echo "neurodrone commit: $(git -C "$KIT" rev-parse HEAD 2>/dev/null)"
  echo "px4: $(git -C "$PX4_DIR" describe --tags --always)"; echo "gazebo: $(gz sim --versions 2>/dev/null | tail -1)"
  echo "ollama: $(curl -s "$OLLAMA/api/version")"; echo "model: $MODEL"
  echo "runs: $RUNS  worlds: $WORLDS  targets: $TARGETS  max_time: $MAX_TIME  search_yaw: $SEARCH_YAW  headless: $HEADLESS"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
} > "$OUT/run_info.txt"
cat "$OUT/run_info.txt"

cleanup() {
  pkill -f "sleep infinity" 2>/dev/null; pkill -f "px4_sitl_default/bin/px4" 2>/dev/null
  pkill -f "gz sim" 2>/dev/null; pkill -f "sim_mission.py" 2>/dev/null; sleep 3
  pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f "px4_sitl_default/bin/px4" 2>/dev/null; true
}
summary() {
  "$VENV/bin/python" "$KIT/scripts/analyze_series.py" "$OUT"/*.jsonl --worlds "$KIT/sim/worlds" --out "$OUT/summary" \
    | tee "$OUT/summary.md"
}
trap 'echo; echo "interrupted"; cleanup; summary; exit 130' INT TERM

# Load the VLM before any Gazebo runs, so Ollama can put as much of it on the GPU as possible, and time it.
cleanup
echo; echo "=== VLM warm-up (no Gazebo running)"
( cd "$KIT" && "$VENV/bin/python" scripts/vlm_warmup.py --ollama "$OLLAMA" --model "$MODEL" --num-ctx "$NUM_CTX" \
    --max-median "$VLM_MAX_MEDIAN" ) | tee -a "$OUT/run_info.txt"
if [ "${PIPESTATUS[0]}" != 0 ]; then
  echo "The VLM is too slow for the series (see above). Close other programs that use the GPU (games, browsers with"
  echo "video, a running Gazebo) and run again, or allow more: VLM_MAX_MEDIAN=60 bash sim/run_series.sh"
  exit 1
fi

IFS=';' read -r -a TLIST <<< "$TARGETS"
n=0; total=$(( RUNS * $(wc -w <<< "$WORLDS") * ${#TLIST[@]} ))
for i in $(seq 1 "$RUNS"); do
  for w in $WORLDS; do
    for t in "${TLIST[@]}"; do
      n=$((n + 1)); slug=$(echo "$t" | sed 's/^the //; s/ /_/g'); name="${w}__${slug}__${i}"
      echo; echo "=== run $n/$total: $name   $(date +%H:%M:%S)"
      cleanup
      ( cd "$PX4_DIR" && sleep infinity | env PX4_GZ_WORLD="$w" HEADLESS="$([ "$HEADLESS" = 1 ] && echo 1)" \
          make px4_sitl gz_x500_depth ) > "$OUT/px4/$name.log" 2>&1 &
      ok=0
      for _ in $(seq 1 120); do
        grep -q "Ready for takeoff" "$OUT/px4/$name.log" 2>/dev/null && { ok=1; break; }; sleep 1
      done
      if [ $ok = 0 ]; then echo "PX4 not ready after 120 s - skipped (see $OUT/px4/$name.log)"; continue; fi
      sleep 5  # let the estimator settle
      ( cd "$KIT" && timeout $(( MAX_TIME + 120 )) "$VENV/bin/python" scripts/sim_mission.py --instruction "$t" \
          --ollama "$OLLAMA" --model "$MODEL" --max-time "$MAX_TIME" --search-yaw "$SEARCH_YAW" --num-ctx "$NUM_CTX" \
          --log "$OUT/$name.jsonl" --save-frames "$OUT/frames/$name" ) 2>&1 \
        | grep -v -E "FutureWarning|mavsdk-grpc|To carry on|To freeze|To migrate|_bootstrap" | tail -4
      sleep 8  # landing
    done
  done
done
cleanup
echo; echo "=== all runs done: $OUT"
summary
