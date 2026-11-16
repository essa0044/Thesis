"""MAVROS + the VIO pose relay ONLY - Bernas's vslam/mavrospy.launch.py without its OFFBOARD pattern node.

Bernas's launch file also starts `mavrospy` (square / circle / figure-8 flights in OFFBOARD). For stage 6 we want the
VIO pose in PX4 and nothing that commands the drone, so this file starts:
  - MAVROS (px4.launch) on the USB-to-serial adapter wired to the Pixhawk's TELEM2 (921600 baud)
  - topic_tools relay: /visual_slam/tracking/vo_pose_covariance -> /mavros/vision_pose/pose_cov
Run inside the mavrospy-vslam container with ROS_DOMAIN_ID=1 (jetson/vio/run_mavros_vio.sh does that).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fcu_url = DeclareLaunchArgument("fcu_url", default_value="/dev/ttyUSB0:921600",
                                    description="serial device of the USB-to-serial adapter (Pixhawk TELEM2)")
    px4_launch = os.path.expanduser("~/ros2_ws/install/mavros/share/mavros/launch/px4.launch")
    mavros = IncludeLaunchDescription(AnyLaunchDescriptionSource(px4_launch),
                                      launch_arguments={"fcu_url": LaunchConfiguration("fcu_url")}.items())
    relay = Node(package="topic_tools", executable="relay", name="relay_pose", output="screen",
                 arguments=["/visual_slam/tracking/vo_pose_covariance", "/mavros/vision_pose/pose_cov"])
    return LaunchDescription([fcu_url, mavros, relay])
