"""Launch the complete Gazebo + patrol + YOLOv10 inspection system."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    gazebo_share = get_package_share_directory("gazebo_ros")
    simulation_share = get_package_share_directory("doggo_gazebo")
    perception_share = get_package_share_directory("doggo_perception")
    world = os.path.join(simulation_share, "worlds", "inspection.world")
    models = os.path.join(simulation_share, "models")
    detector_config = os.path.join(perception_share, "config", "detector.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_path",
                default_value="/root/doggo/models/insulator_yolov10s.pt",
            ),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("visualize", default_value="false"),
            SetEnvironmentVariable(
                "GAZEBO_MODEL_PATH",
                models
                + os.pathsep
                + os.environ.get("GAZEBO_MODEL_PATH", ""),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(gazebo_share, "launch", "gazebo.launch.py")
                ),
                launch_arguments={
                    "world": world,
                    "gui": LaunchConfiguration("gui"),
                }.items(),
            ),
            Node(
                package="doggo_gazebo",
                executable="patrol",
                name="inspection_patrol",
                output="screen",
            ),
            Node(
                package="doggo_perception",
                executable="detector",
                name="yolov10_insulator_detector",
                output="screen",
                parameters=[
                    detector_config,
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "device": LaunchConfiguration("device"),
                    },
                ],
            ),
            Node(
                package="rqt_image_view",
                executable="rqt_image_view",
                arguments=["/doggo/detection_image"],
                condition=IfCondition(LaunchConfiguration("visualize")),
                output="screen",
            ),
        ]
    )
