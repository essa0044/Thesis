#!/usr/bin/env bash
# Reproducible closed-loop guard test: PX4 (built-in SIH simulator, no Gazebo) + MAVSDK + synthetic depth.
#
#   bash sim/run_sih.sh                  # sets up (once), starts PX4, runs scripts/sim_sih_guard.py, stops PX4
#   PX4_REF=v1.17.0 OUT=results/sih bash sim/run_sih.sh
#
# Tested on Ubuntu 24.04 (x86_64), 2 cores / 7 GB RAM. Needs sudo for apt, internet for GitHub and PyPI.
# Everything is pinned: PX4 tag (PX4_REF), mavsdk<4, seeds 0..2 and the world are set in scripts/sim_sih_guard.py.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_REF="${PX4_REF:-v1.17.0}"
PX4_DIR="${PX4_DIR:-$HOME/px4-sih}"
VENV="${VENV:-$HOME/nd-venv}"
OUT="${OUT:-$KIT/results/sih}"
BUILD="$PX4_DIR/build/px4_sitl_default"

echo "== 1. packages"
sudo apt-get update -qq
sudo apt-get install -y -qq git cmake ninja-build g++ ccache libeigen3-dev libxml2-utils \
  python3-venv python3-pip python3-empy python3-jinja2 python3-yaml python3-numpy python3-serial \
  python3-toml python3-jsonschema python3-kconfiglib python3-packaging >/dev/null

echo "== 2. PX4 $PX4_REF (shallow clone; only the submodules a SITL build needs)"
if [ ! -d "$PX4_DIR/.git" ]; then
  git clone --depth 1 --branch "$PX4_REF" https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR"
fi
cd "$PX4_DIR"
# 'git clone --recursive' fails on an unrelated board submodule (gitlab) and leaves the others empty, so fetch each
# submodule at its pinned commit, skipping the NuttX / board / simulator ones that a SITL build does not use.
git submodule init >/dev/null 2>&1 || true
git submodule status | awk '{print $2}' | grep -vE 'nuttx|NuttX|boards/|Tools/simulation|vertiq|cyphal|test/|fuzz' | while read -r p; do
  sha=$(git ls-tree HEAD "$p" | awk '{print $3}')
  [ -n "$(ls -A "$p" 2>/dev/null | grep -v '^\.git$')" ] && continue
  git submodule update --init --depth 1 -- "$p" >/dev/null 2>&1 || true
  [ -n "$(ls -A "$p" 2>/dev/null | grep -v '^\.git$')" ] || ( cd "$p" && git fetch -q --depth 1 origin "$sha" && git checkout -q -f "$sha" )
done
( cd src/modules/mavlink/mavlink && git submodule update --init --depth 1 >/dev/null 2>&1 )   # pymavlink (mavgen)
echo "PX4 $(git describe --tags --always) commit $(git rev-parse HEAD)"

echo "== 3. build (first time ~10-15 min); DONT_RUN so make does not start PX4 itself"
[ -x "$BUILD/bin/px4" ] || ( DONT_RUN=1 make px4_sitl_default >/tmp/px4_build.log 2>&1 </dev/null || { tail -20 /tmp/px4_build.log; exit 1; } )

echo "== 4. python venv with MAVSDK (gRPC, < 4)"
[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -q "mavsdk<4" numpy pillow pytest

echo "== 5. start PX4 SIH (quadrotor, no GUI) and run the test"
mkdir -p "$BUILD/rootfs" "$OUT"
cd "$BUILD/rootfs"
PX4_SYS_AUTOSTART=10040 PX4_SIM_MODEL=sihsim_quadx ../bin/px4 -d -s ../etc/init.d-posix/rcS -t ../../../test_data ../etc \
  >"$OUT/px4_stdout.log" 2>&1 </dev/null &
PX4_PID=$!
trap 'kill $PX4_PID 2>/dev/null || true' EXIT
until grep -q "Ready for takeoff" "$OUT/px4_stdout.log" 2>/dev/null; do sleep 1; done
cd "$KIT"
PX4_DIR="$PX4_DIR" "$VENV/bin/python" scripts/sim_sih_guard.py --out "$OUT" "$@" | tee "$OUT/run.log"
echo "== done: $OUT/sih_guard_summary.json"
