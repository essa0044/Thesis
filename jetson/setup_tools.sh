#!/usr/bin/env bash
# Day one on the Jetson: the Python tools of stages 1-5 (neurodrone/), Ollama with the thesis model, benchmark data.
#
#   bash jetson/setup_tools.sh --check        # report only
#   bash jetson/setup_tools.sh                # venv + Python packages + unit tests
#   bash jetson/setup_tools.sh --ollama --model --data    # also install Ollama, pull the model, download the data
#
# Safe to run again; never deletes anything. Run as the normal user from the repository (git clone of essa0044/Thesis).
set -uo pipefail
CHECK=0; DO_OLLAMA=0; DO_MODEL=0; DO_DATA=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;; --ollama) DO_OLLAMA=1 ;; --model) DO_MODEL=1 ;; --data) DO_DATA=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option $arg"; exit 2 ;;
  esac
done
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ND="$REPO/neurodrone"
VENV="${ND_VENV:-$HOME/nd_venv}"
DATA="${ND_DATA:-$HOME/nd_data}"
MODEL="${ND_MODEL:-qwen3-vl:4b-instruct}"
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
miss() { printf '  \033[33mTODO\033[0m  %s\n' "$*"; }
run()  { if [[ $CHECK == 1 ]]; then miss "would run: $*"; else echo "  + $*"; "$@"; fi; }

echo "== Board"
[[ -f /etc/nv_tegra_release ]] && ok "$(head -1 /etc/nv_tegra_release)" || miss "not a Jetson?"
command -v nvpmodel >/dev/null && ok "power mode: $(nvpmodel -q 2>/dev/null | head -1)  (stage 1 wants MAXN_SUPER: jetson/setup_vio.sh sets it)"
free -m | awk '/Mem:/ {print "  memory: " $2 " MB total, " $7 " MB available"}'

echo "== Python environment $VENV (with the system packages: JetPack's OpenCV, a source-built pyrealsense2)"
if [[ -x "$VENV/bin/python" ]]; then ok "venv exists"; else run python3 -m venv --system-site-packages "$VENV"; fi
PY="$VENV/bin/python"
if [[ $CHECK == 0 ]]; then
  run "$PY" -m pip install -q --upgrade pip
  run "$PY" -m pip install -q numpy pillow huggingface_hub "mavsdk<4" pytest
fi
if [[ -x "$PY" ]] && "$PY" -c "import pyrealsense2" 2>/dev/null; then ok "pyrealsense2 importable"
elif [[ $CHECK == 0 ]] && "$PY" -m pip install -q pyrealsense2 2>/dev/null && "$PY" -c "import pyrealsense2" 2>/dev/null; then ok "pyrealsense2 from pip"
else miss "pyrealsense2 missing (stages 3-4): build librealsense with Python bindings, e.g.
          git clone https://github.com/IntelRealSense/librealsense && cd librealsense && mkdir build && cd build &&
          cmake .. -DBUILD_PYTHON_BINDINGS=true -DFORCE_RSUSB_BACKEND=true -DPYTHON_EXECUTABLE=$PY -DCMAKE_BUILD_TYPE=Release &&
          make -j4 && sudo make install   (then: export PYTHONPATH=\$PYTHONPATH:/usr/local/lib/python3/dist-packages ... see README)"
fi
if [[ -x "$PY" ]] && "$PY" -c "import cv2" 2>/dev/null; then ok "OpenCV $("$PY" -c 'import cv2; print(cv2.__version__)')"
else miss "OpenCV not importable (optional; Pillow is used instead): sudo apt install python3-opencv"; fi
command -v tegrastats >/dev/null && ok "tegrastats" || miss "tegrastats not found (stage 1 memory/temperature log)"

echo "== Ollama"
if command -v ollama >/dev/null; then ok "ollama $(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
elif [[ $DO_OLLAMA == 1 ]]; then
  if [[ $CHECK == 1 ]]; then miss "would install Ollama (curl -fsSL https://ollama.com/install.sh | sh)"
  else echo "  + Ollama's install script (detects JetPack)"; curl -fsSL https://ollama.com/install.sh | sh; fi
else miss "Ollama missing: bash jetson/setup_tools.sh --ollama"; fi
if command -v ollama >/dev/null && ollama list 2>/dev/null | grep -q "^$MODEL"; then ok "model $MODEL ($(ollama list | awk -v m="$MODEL" '$1==m {print $2}'))"
elif [[ $DO_MODEL == 1 ]]; then run ollama pull "$MODEL"
else miss "model $MODEL not pulled: bash jetson/setup_tools.sh --model  (workstation digest 84f7ed8042fa)"; fi

echo "== Benchmark data $DATA (stage 1 accuracy, ~2.2 GB)"
if [[ -f "$DATA/refdrone/refdrone_test_base64.tsv" && -f "$DATA/grefcoco/instances.json" ]]; then ok "RefDrone and gRefCOCO present"
elif [[ $DO_DATA == 1 ]]; then run "$PY" "$ND/scripts/get_vlm_data.py" --root "$DATA"
else miss "data missing: bash jetson/setup_tools.sh --data"; fi

echo "== Unit tests (fakes for camera, Ollama, PX4)"
if [[ $CHECK == 1 ]]; then miss "would run the tests"
else (cd "$ND" && "$PY" -m pytest -q tests 2>&1 | tail -3); fi

cat <<EOT

Stage 1:  $PY $ND/scripts/jetson_vlm_bench.py all --data $DATA --out ~/nd_results/stage1
          (about 4-5 h; nothing else running; read ~/nd_results/stage1/report.md)
Stage 3:  $PY $ND/scripts/realsense_bench.py depth --emitter both --out ~/nd_results/stage3/depth   (see README)
Stage 4:  $PY $ND/scripts/record_replay.py record --out ~/nd_rec/run01
EOT
