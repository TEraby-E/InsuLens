"""Start the Gazebo data-generation world and automatic label writer."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.actions import SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    gazebo_share = get_package_share_directory("gazebo_ros")
    doggo_share = get_package_share_directory("doggo_gazebo")
    world = os.path.join(doggo_share, "worlds", "insulator_dataset.world")
    model_path = os.path.join(doggo_share, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    generator = Node(
        package="doggo_gazebo",
        executable="generate_dataset",
        name="insulator_dataset_generator",
        output="screen",
        parameters=[
            {
                "output_dir": LaunchConfiguration("output_dir"),
                "num_samples": LaunchConfiguration("num_samples"),
                "seed": LaunchConfiguration("seed"),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "output_dir", default_value="/root/doggo/datasets/insulator_sim"
            ),
            DeclareLaunchArgument("num_samples", default_value="1200"),
            DeclareLaunchArgument("seed", default_value="42"),
            DeclareLaunchArgument("gui", default_value="true"),
            SetEnvironmentVariable(
                "GAZEBO_MODEL_PATH",
                model_path + os.pathsep + existing_model_path,
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(gazebo_share, "launch", "gazebo.launch.py")
                ),
                launch_arguments={
                    "world": world,
                    "gui": LaunchConfiguration("gui"),
                    "verbose": "false",
                }.items(),
            ),
            generator,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=generator,
                    on_exit=[Shutdown(reason="Dataset generation completed")],
                )
            ),
        ]
    )
