#!/usr/bin/env bash
# Start MAVROS + the VIO pose relay (no flight-pattern node) in Bernas's mavrospy-vslam image.
#   bash jetson/vio/run_mavros_vio.sh [/dev/ttyUSB0]
# The image is built by: bash jetson/setup_vio.sh --build-mavros
set -euo pipefail
DEV="${1:-/dev/ttyUSB0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -e "$DEV" ]] || { echo "$DEV not found: plug the USB-to-serial adapter (Pixhawk TELEM2)"; exit 1; }
docker image inspect mavrospy-vslam:latest >/dev/null 2>&1 || { echo "image mavrospy-vslam:latest missing: bash jetson/setup_vio.sh --build-mavros"; exit 1; }
exec docker run -it --rm --name nd-mavros-vio --device="$DEV" --net=host --ipc=host -v "$HERE":/nd:ro \
  mavrospy-vslam:latest -c "source /opt/ros/humble/setup.bash && source /home/jetson/ros2_ws/install/setup.bash && \
  export ROS_DOMAIN_ID=1 && ros2 launch /nd/nd_mavros_vio.launch.py fcu_url:=$DEV:921600"
