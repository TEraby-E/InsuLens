"""Launch only the YOLOv10 detector for an existing image topic."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("insulens_perception")
    config = os.path.join(package_share, "config", "detector.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_path",
                default_value="auto",
            ),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument(
                "image_topic", default_value="/insulens/camera/image_raw"
            ),
            Node(
                package="insulens_perception",
                executable="detector",
                name="yolov10_insulator_detector",
                output="screen",
                parameters=[
                    config,
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "device": LaunchConfiguration("device"),
                        "image_topic": LaunchConfiguration("image_topic"),
                    },
                ],
            ),
        ]
    )
