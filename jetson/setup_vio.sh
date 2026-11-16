#!/usr/bin/env bash
# Stage 6: visual-inertial odometry (VIO) on the Jetson Orin Nano for PX4 - the set-up of Andrew Bernas,
# "GPS-Denied Drone with NVIDIA Jetson Orin Nano" (Hackster.io, June 2025; github.com/bandofpv/VSLAM-UAV), scripted.
# Isaac ROS Visual SLAM (release 3.2, ROS 2 Humble, Docker) on the D435i's infrared images + IMU; the pose goes through
# MAVROS (second container) over a USB-to-serial adapter into the Pixhawk's TELEM2 at 921600 baud.
#
#   bash jetson/setup_vio.sh --check            # only report what is done and what is missing (changes nothing)
#   bash jetson/setup_vio.sh                    # do the host set-up (asks for sudo), clone, configure
#   bash jetson/setup_vio.sh --build-mavros     # also build the MAVROS image (about an hour on the Orin Nano)
#
# Safe to run again: every step checks first. It never flashes firmware, never touches PX4 (that is
# neurodrone/scripts/px4_params.py) and never deletes anything. Run it on the Jetson as the normal user (not root).
# What it cannot do for you is listed at the end (camera firmware, IMU calibration, the launch itself).
set -uo pipefail

CHECK=0; BUILD_MAVROS=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --build-mavros) BUILD_MAVROS=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option $arg"; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_ROS_WS="${ISAAC_ROS_WS:-$HOME/workspaces/isaac_ros-dev}"
ISAAC_COMMON_BRANCH="release-3.2"
VSLAM_UAV_URL="https://github.com/bandofpv/VSLAM-UAV.git"
VSLAM_UAV_COMMIT="037198f0b10a85a1295da8e6cc7100e4fa121dec"   # the version read for the thesis (15 Jul 2025)
WANT_JETPACK="6.2"; WANT_RS_FW="5.13.0.50"
FAIL=0; TODO=()

ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
miss() { printf '  \033[33mTODO\033[0m  %s\n' "$*"; TODO+=("$*"); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=1; }
run()  { if [[ $CHECK == 1 ]]; then miss "would run: $*"; else echo "  + $*"; "$@" || bad "command failed: $*"; fi; }

if [[ $(id -u) -eq 0 ]]; then echo "run as the normal user, not root (sudo is asked where needed)"; exit 2; fi

echo "== 1. Board"
if [[ -f /etc/nv_tegra_release ]]; then ok "$(head -1 /etc/nv_tegra_release)"; else bad "not a Jetson (/etc/nv_tegra_release missing)"; fi
JP=$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null || true)
if [[ "$JP" == ${WANT_JETPACK}* ]]; then ok "JetPack $JP"; else miss "JetPack $WANT_JETPACK expected (found '${JP:-none}'): Isaac ROS 3.2 needs JetPack 6.2"; fi
ROOTDEV=$(findmnt -n -o SOURCE / 2>/dev/null || true)
if [[ "$ROOTDEV" == *nvme* ]]; then ok "system on the NVMe SSD ($ROOTDEV): Docker images are on the SSD too"
else miss "system not on the NVMe SSD ($ROOTDEV): put Docker's data-root on the SSD (Bernas: docker 'data-root' in /etc/docker/daemon.json), the images need ~30 GB"; fi

echo "== 2. Power mode MAXN SUPER and clocks"
MAXN_ID=$(grep -oP 'POWER_MODEL ID=\K[0-9]+(?= NAME=MAXN_SUPER)' /etc/nvpmodel.conf 2>/dev/null | head -1)
CUR_MODE=$(nvpmodel -q 2>/dev/null | head -1 || true)
if [[ "$CUR_MODE" == *MAXN_SUPER* ]]; then ok "power mode: $CUR_MODE"
elif [[ -n "$MAXN_ID" ]]; then run sudo nvpmodel -m "$MAXN_ID"; miss "reboot if nvpmodel asked for it, then run this script again"
else miss "MAXN_SUPER not found in /etc/nvpmodel.conf (JetPack 6.2 + Super firmware needed); current: ${CUR_MODE:-?}"; fi
if command -v jetson_clocks >/dev/null; then run sudo jetson_clocks; else miss "jetson_clocks not found"; fi

echo "== 3. Docker with the NVIDIA runtime"
if command -v docker >/dev/null; then ok "docker $(docker --version | cut -d' ' -f3 | tr -d ,)"; else bad "docker missing (comes with JetPack: sudo apt install nvidia-container docker.io)"; fi
if id -nG "$USER" | grep -qw docker; then ok "$USER is in the docker group"; else run sudo usermod -aG docker "$USER"; miss "log out and in again (docker group)"; fi
if command -v nvidia-ctk >/dev/null; then ok "nvidia-container-toolkit present"; else bad "nvidia-ctk missing: sudo apt install nvidia-container-toolkit"; fi
if [[ -f /etc/cdi/nvidia.yaml ]]; then ok "GPU CDI spec /etc/cdi/nvidia.yaml"
else run sudo nvidia-ctk cdi generate --mode=csv --output=/etc/cdi/nvidia.yaml; fi
if grep -q '"default-runtime"[[:space:]]*:[[:space:]]*"nvidia"' /etc/docker/daemon.json 2>/dev/null; then ok "docker default runtime = nvidia"
else
  if [[ $CHECK == 1 ]]; then miss "would set \"default-runtime\": \"nvidia\" in /etc/docker/daemon.json (backup kept)"
  else
    echo "  + set default-runtime nvidia in /etc/docker/daemon.json (backup: daemon.json.bak_nd)"
    sudo cp -n /etc/docker/daemon.json /etc/docker/daemon.json.bak_nd 2>/dev/null || true
    sudo python3 - <<'PY' || bad "could not edit /etc/docker/daemon.json"
import json, os
p = "/etc/docker/daemon.json"
d = json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else {}
d.setdefault("runtimes", {}).setdefault("nvidia", {"path": "nvidia-container-runtime", "runtimeArgs": []})
d["default-runtime"] = "nvidia"
json.dump(d, open(p, "w"), indent=4)
PY
    run sudo systemctl restart docker
  fi
fi

echo "== 4. Tools"
for p in git git-lfs jq curl; do
  if command -v "$p" >/dev/null; then ok "$p"; else run sudo apt-get install -y "$p"; fi
done
[[ $CHECK == 1 ]] || git lfs install --skip-repo >/dev/null 2>&1 || true

echo "== 5. RealSense udev rules (host)"
if ls /etc/udev/rules.d/*realsense* >/dev/null 2>&1; then ok "udev rules: $(ls /etc/udev/rules.d/*realsense* | head -1)"
else
  if [[ $CHECK == 1 ]]; then miss "would install librealsense's 99-realsense-libusb.rules"
  else
    TMP=$(mktemp -d)
    git clone -q --depth 1 https://github.com/IntelRealSense/librealsense.git "$TMP/librealsense" && \
      sudo cp "$TMP/librealsense/config/99-realsense-libusb.rules" /etc/udev/rules.d/ && \
      sudo udevadm control --reload-rules && sudo udevadm trigger && ok "udev rules installed" || bad "udev rules"
  fi
fi

echo "== 6. Isaac ROS workspace $ISAAC_ROS_WS"
if [[ $CHECK == 1 && ! -d "$ISAAC_ROS_WS" ]]; then miss "would create $ISAAC_ROS_WS"; else mkdir -p "$ISAAC_ROS_WS/src"; fi
if grep -q "export ISAAC_ROS_WS=" ~/.bashrc 2>/dev/null; then ok "ISAAC_ROS_WS in ~/.bashrc"
elif [[ $CHECK == 1 ]]; then miss "would add 'export ISAAC_ROS_WS=$ISAAC_ROS_WS' to ~/.bashrc"
else echo "export ISAAC_ROS_WS=$ISAAC_ROS_WS" >> ~/.bashrc; ok "ISAAC_ROS_WS added to ~/.bashrc"; fi
if [[ -d "$ISAAC_ROS_WS/src/isaac_ros_common/.git" ]]; then
  ok "isaac_ros_common ($(git -C "$ISAAC_ROS_WS/src/isaac_ros_common" rev-parse --abbrev-ref HEAD))"
else run git clone -b "$ISAAC_COMMON_BRANCH" https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git "$ISAAC_ROS_WS/src/isaac_ros_common"; fi
CFG="CONFIG_IMAGE_KEY=ros2_humble.realsense"
if grep -qx "$CFG" ~/.isaac_ros_common-config 2>/dev/null; then ok "~/.isaac_ros_common-config: $CFG"
elif [[ $CHECK == 1 ]]; then miss "would write $CFG to ~/.isaac_ros_common-config"
else echo "$CFG" > ~/.isaac_ros_common-config; ok "wrote ~/.isaac_ros_common-config ($CFG)"; fi
if [[ -d "$ISAAC_ROS_WS/VSLAM-UAV/.git" ]]; then ok "VSLAM-UAV at $(git -C "$ISAAC_ROS_WS/VSLAM-UAV" rev-parse --short HEAD)"
else
  run git clone "$VSLAM_UAV_URL" "$ISAAC_ROS_WS/VSLAM-UAV"
  [[ $CHECK == 1 ]] || git -C "$ISAAC_ROS_WS/VSLAM-UAV" checkout -q "$VSLAM_UAV_COMMIT" || bad "checkout $VSLAM_UAV_COMMIT"
fi
if [[ -d "$ISAAC_ROS_WS/VSLAM-UAV/vslam" ]]; then
  if [[ $CHECK == 1 ]]; then miss "would copy jetson/vio/nd_vslam_realsense.py into VSLAM-UAV/vslam/"
  else cp "$HERE/vio/nd_vslam_realsense.py" "$ISAAC_ROS_WS/VSLAM-UAV/vslam/" && ok "nd_vslam_realsense.py (colour + depth for the mission as well) copied"; fi
fi

echo "== 7. MAVROS container (Bernas's mavrospy image)"
if docker image inspect mavrospy-vslam:latest >/dev/null 2>&1; then ok "image mavrospy-vslam:latest"
elif [[ $BUILD_MAVROS == 1 ]]; then
  (cd "$ISAAC_ROS_WS/VSLAM-UAV/docker/mavrospy" && run docker build --network=host -t mavrospy-vslam:latest -f Dockerfile .)
else miss "build the MAVROS image: bash jetson/setup_vio.sh --build-mavros  (about an hour)"; fi

echo "== 8. Things to check by hand"
if command -v rs-fw-update >/dev/null; then
  FW=$(rs-fw-update -l 2>/dev/null | grep -oP 'firmware version: \K[0-9.]+' | head -1)
  if [[ "$FW" == "$WANT_RS_FW" ]]; then ok "D435i firmware $FW"; else miss "D435i firmware ${FW:-unknown}: Isaac ROS 3.2 / Bernas use $WANT_RS_FW (rs-fw-update -f <Signed_Image_UVC_5_13_0_50.bin>)"; fi
else miss "D435i firmware must be $WANT_RS_FW: check with rs-fw-update -l (librealsense tools)"; fi
if ls /dev/ttyUSB* >/dev/null 2>&1; then ok "USB-serial adapter: $(ls /dev/ttyUSB* | tr '\n' ' ')"; else miss "no /dev/ttyUSB*: plug the USB-to-serial adapter (TELEM2 of the Pixhawk)"; fi

echo
if [[ $FAIL == 1 ]]; then echo "Some steps FAILED (see above)."; fi
if [[ ${#TODO[@]} -gt 0 ]]; then echo "Still to do:"; for t in "${TODO[@]}"; do echo "  - $t"; done; fi
cat <<EOF

Next (jetson/README.md, stage 6):
  1. PX4:   python neurodrone/scripts/px4_params.py show --profile vio, then apply --yes, then reboot the Pixhawk
  2. VIO:   cd \$ISAAC_ROS_WS/src/isaac_ros_common && ./scripts/run_dev.sh -d \$ISAAC_ROS_WS
            (in the container)  cd /workspaces/isaac_ros-dev/VSLAM-UAV/vslam && ./vslam_launch.sh
  3. MAVROS: bash jetson/vio/run_mavros_vio.sh      (MAVROS + pose relay to /mavros/vision_pose/pose_cov ONLY;
            Bernas's own mavrospy.launch.py would also start an OFFBOARD flight pattern - do not use it here)
  4. Check: python neurodrone/scripts/vio_check.py drift --laps 3   (propellers off)
EOF
exit $FAIL
