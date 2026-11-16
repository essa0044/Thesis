# Based on isaac_ros_vslam_realsense.py from github.com/bandofpv/VSLAM-UAV (itself NVIDIA's example,
# Apache-2.0, Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES).
#
# Changed for the thesis (stage 6b / 7, NOT YET TESTED ON THE DRONE): the same camera must also give the mission its
# colour picture (VLM) and depth (guard), because only one process can own the D435i. Colour and depth are therefore
# switched on here, and the mission reads them from this node's ROS 2 topics instead of opening the camera itself:
#     /camera/color/image_raw (+ camera_info)       colour, 1280x720 at 15 fps
#     /camera/depth/image_rect_raw (+ camera_info)  depth, same profile as the infrared images (shared stereo module)
# The IR dot projector stays OFF as in Bernas's set-up (it disturbs the visual odometry); stage 3 measures how much
# depth the guard loses without it. To check on day one: VSLAM still at >= 30 Hz (`ros2 topic hz
# /visual_slam/tracking/vo_pose_covariance`), CPU load, USB bandwidth. If not, go back to Bernas's file (infrared and
# IMU only) and lower the colour rate.
#
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    realsense_camera_node = Node(
        name='camera',
        namespace='camera',
        package='realsense2_camera',
        executable='realsense2_camera_node',
        parameters=[{
            'enable_infra1': True,
            'enable_infra2': True,
            'enable_color': True,                        # thesis: VLM picture
            'rgb_camera.color_profile': '1280x720x15',   # thesis
            'enable_depth': True,                        # thesis: guard and 3-D goal
            'align_depth.enable': False,                 # aligning at 90 Hz costs too much CPU; the mission maps by ray
            'depth_module.emitter_enabled': 0,
            'depth_module.profile': '640x360x90',
            'enable_gyro': True,
            'enable_accel': True,
            'gyro_fps': 200,
            'accel_fps': 200,
            'unite_imu_method': 2
        }],
    )

    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'enable_image_denoising': False,
            'rectified_images': True,
            'enable_imu_fusion': True,
            # IMU noise from Bernas's file; replace with your own Allan-variance values (README, stage 6)
            'gyro_noise_density': 0.000244,
            'gyro_random_walk': 0.000019393,
            'accel_noise_density': 0.001862,
            'accel_random_walk': 0.003,
            'calibration_frequency': 200.0,
            'image_jitter_threshold_ms': 22.00,
            'base_frame': 'camera_link',
            'imu_frame': 'camera_gyro_optical_frame',
            'enable_slam_visualization': False,          # thesis: no RViz on the drone
            'enable_landmarks_view': False,
            'enable_observations_view': False,
            'camera_optical_frames': [
                'camera_infra1_optical_frame',
                'camera_infra2_optical_frame',
            ],
        }],
        remappings=[
            ('visual_slam/image_0', 'camera/infra1/image_rect_raw'),
            ('visual_slam/camera_info_0', 'camera/infra1/camera_info'),
            ('visual_slam/image_1', 'camera/infra2/image_rect_raw'),
            ('visual_slam/camera_info_1', 'camera/infra2/camera_info'),
            ('visual_slam/imu', 'camera/imu'),
        ],
    )

    visual_slam_launch_container = ComposableNodeContainer(
        name='visual_slam_launch_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[visual_slam_node],
        output='screen',
    )

    return launch.LaunchDescription([visual_slam_launch_container, realsense_camera_node])
