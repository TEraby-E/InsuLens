"""Run the defect detector on real images, a directory, or a video."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    perception_share = get_package_share_directory("insulens_perception")
    detector_config = os.path.join(perception_share, "config", "detector.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "source",
                default_value="/root/insulens/datasets/cplid_yolo/images/val",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value=(
                    "/root/insulens/models/insulator_defect_yolov10s.pt"
                ),
            ),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument("visualize", default_value="true"),
            DeclareLaunchArgument("monitor", default_value="false"),
            Node(
                package="insulens_perception",
                executable="image_source",
                name="real_image_source",
                output="screen",
                parameters=[{"source": LaunchConfiguration("source")}],
            ),
            Node(
                package="insulens_perception",
                executable="detector",
                name="yolov10_insulator_detector",
                output="screen",
                parameters=[
                    detector_config,
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "device": LaunchConfiguration("device"),
                        "image_topic": "/insulens/real_camera/image_raw",
                    },
                ],
            ),
            Node(
                package="rqt_image_view",
                executable="rqt_image_view",
                arguments=["/insulens/detection_image"],
                condition=IfCondition(LaunchConfiguration("visualize")),
                output="screen",
            ),
            Node(
                package="insulens_perception",
                executable="inspection_monitor",
                name="inspection_monitor",
                condition=IfCondition(LaunchConfiguration("monitor")),
                output="screen",
            ),
        ]
    )
