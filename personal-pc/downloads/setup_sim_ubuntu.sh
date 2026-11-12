#!/usr/bin/env bash
# Set up PX4 SITL + Gazebo Harmonic + the VLM mission script on Ubuntu 22.04 / 24.04.
#
#   bash sim/setup_sim_ubuntu.sh                 # everything
#   OLLAMA_URL=http://192.168.1.20:11434 bash sim/setup_sim_ubuntu.sh   # check a remote Ollama (Windows workstation)
#   PX4_DIR=~/PX4-Autopilot PX4_REF=v1.16.0 bash sim/setup_sim_ubuntu.sh
#
# Each step prints OK / FAIL; it is safe to run again (steps already done are skipped).
# Takes ~1-2 h the first time (PX4 clone + toolchain + first build). Needs sudo, internet, ~40 GB free.
set -u
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # the folder with neurodrone/, scripts/, sim/
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
PX4_REF="${PX4_REF:-}"                                       # empty: newest stable release tag
VENV="${VENV:-$HOME/nd-venv}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
MODEL="${MODEL:-qwen3-vl:4b-instruct}"

ok()   { echo -e "\e[32mOK\e[0m    $*"; }
fail() { echo -e "\e[31mFAIL\e[0m  $*"; }
step() { echo; echo "=== $*"; }

step "1. System check"
. /etc/os-release
echo "Ubuntu $VERSION_ID, $(nproc) cores, $(free -g | awk '/Mem:/{print $2}') GB RAM, $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' ') free in $HOME"
case "$VERSION_ID" in 22.04|24.04) ok "Ubuntu $VERSION_ID is supported by PX4 + Gazebo Harmonic";;
  *) fail "Ubuntu $VERSION_ID: PX4's Gazebo setup supports 22.04 / 24.04 only"; exit 1;; esac
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no NVIDIA GPU / driver found (Gazebo still works, slower)"
sudo -v || { fail "need sudo"; exit 1; }
sudo apt-get update -qq && sudo apt-get install -y -qq git curl python3-venv python3-pip >/dev/null && ok "base packages"

step "2. PX4-Autopilot source"
if [ ! -d "$PX4_DIR/.git" ]; then
  if [ -z "$PX4_REF" ]; then
    PX4_REF=$(git ls-remote --tags --refs https://github.com/PX4/PX4-Autopilot.git 'v*' | awk -F/ '{print $3}' \
              | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1)
  fi
  echo "cloning PX4 $PX4_REF (with submodules, a few GB)"
  git clone --recursive --branch "$PX4_REF" https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR" && ok "PX4 cloned" || { fail "clone"; exit 1; }
else
  ok "PX4 already at $PX4_DIR ($(git -C "$PX4_DIR" describe --tags --always))"
fi

step "3. PX4 toolchain + Gazebo (PX4's own script; simulation only, no NuttX/ARM toolchain)"
if command -v gz >/dev/null; then ok "Gazebo already installed: $(gz sim --versions 2>/dev/null | tail -1)"
else
  bash "$PX4_DIR/Tools/setup/ubuntu.sh" --no-nuttx && ok "PX4 setup script" || { fail "PX4 setup script"; exit 1; }
  echo "NOTE: log out and back in (or reboot) once after this step, then run this script again."
fi

step "4. Test worlds (red box, blue barrel, green box, white box; nd_obstacle adds a yellow pole in the way)"
WD="$PX4_DIR/Tools/simulation/gz/worlds"
cp "$KIT"/sim/worlds/nd_*.sdf "$WD"/ && ok "copied to $WD" || fail "copy worlds"

step "5. Build PX4 SITL for gz_x500_depth (first build ~10-20 min)"
# Newer PX4 releases start PX4 + Gazebo after the build even with DONT_RUN=1, so build with ninja directly once
# CMake has configured the tree, and use 4 jobs (16 parallel jobs ran WSL out of memory).
if [ -x "$PX4_DIR/build/px4_sitl_default/bin/px4" ]; then ok "PX4 SITL already built"
else
  ( cd "$PX4_DIR" && timeout 300 env DONT_RUN=1 HEADLESS=1 make px4_sitl j=4 </dev/null ) >/tmp/px4_build.log 2>&1
  ninja -C "$PX4_DIR/build/px4_sitl_default" -j4 >>/tmp/px4_build.log 2>&1
  pkill -f "px4_sitl_default/bin/px4" 2>/dev/null; pkill -f "gz sim" 2>/dev/null
  [ -x "$PX4_DIR/build/px4_sitl_default/bin/px4" ] && ok "PX4 SITL built" \
    || { fail "build - see /tmp/px4_build.log (tail below)"; tail -20 /tmp/px4_build.log; }
fi

step "6. Gazebo Python bindings (to read the camera images)"
sudo apt-get install -y -qq python3-gz-transport13 python3-gz-msgs10 >/dev/null 2>&1
python3 -c "import gz.transport13, gz.msgs10" 2>/dev/null && ok "gz.transport13 / gz.msgs10" || fail "gz python bindings (sudo apt install python3-gz-transport13 python3-gz-msgs10)"

step "7. Python venv for the mission script ($VENV, sees the system gz bindings)"
[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -q --upgrade pip >/dev/null
"$VENV/bin/pip" install -q "mavsdk<4" numpy pillow pytest >/dev/null && ok "mavsdk (gRPC), numpy, pillow" || fail "pip install"
"$VENV/bin/python" -c "import mavsdk, gz.transport13, numpy, PIL" && ok "imports" || fail "imports in the venv"
( cd "$KIT" && "$VENV/bin/python" -m pytest -q -p no:cacheprovider tests/test_mission.py tests/test_sim_mission.py 2>&1 | tail -1 )

step "8. QGroundControl (ground station)"
if ls "$HOME"/QGroundControl*.AppImage >/dev/null 2>&1; then ok "found $(ls "$HOME"/QGroundControl*.AppImage | head -1)"
else
  sudo usermod -aG dialout "$USER"; sudo apt-get remove -y -qq modemmanager >/dev/null 2>&1
  sudo apt-get install -y -qq gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl libfuse2 libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor-dev >/dev/null 2>&1
  echo "download the Linux AppImage from https://qgroundcontrol.com (Download -> Ubuntu) into $HOME, then: chmod +x ~/QGroundControl*.AppImage"
fi

step "9. VLM server (Ollama) at $OLLAMA_URL"
if curl -s --max-time 5 "$OLLAMA_URL/api/version" | grep -q version; then
  ok "Ollama answers: $(curl -s "$OLLAMA_URL/api/version")"
  curl -s "$OLLAMA_URL/api/tags" | grep -q "\"$MODEL\"" && ok "model $MODEL present" || fail "model $MODEL missing there: ollama pull $MODEL"
else
  fail "no Ollama at $OLLAMA_URL"
  echo "  - on the Windows workstation: set OLLAMA_HOST=0.0.0.0:11434, restart Ollama, allow TCP 11434 in the firewall,"
  echo "    then OLLAMA_URL=http://<workstation-ip>:11434 bash $0"
  echo "  - or here (needs a GPU for useful speed): curl -fsSL https://ollama.com/install.sh | sh && ollama pull $MODEL"
fi

step "Done. To fly:"
cat <<EOF
  terminal 1:  cd $PX4_DIR && PX4_GZ_WORLD=nd_test make px4_sitl gz_x500_depth
  terminal 2:  cd $KIT && $VENV/bin/python scripts/sim_mission.py --no-fly --instruction "the red box" --ollama $OLLAMA_URL --save-frames frames
               (cameras + VLM + guard only; check frames/ shows the right box)
  then:        $VENV/bin/python scripts/sim_mission.py --instruction "the red box" --ollama $OLLAMA_URL --save-frames frames
  obstacle:    PX4_GZ_WORLD=nd_obstacle ...  (the guard should stop the drone at the yellow pole)
EOF
