"""Serve the static rosbridge dashboard from the installed package share path."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    dashboard_dir = os.path.join(
        get_package_share_directory("insulens_perception"), "web", "dashboard"
    )
    port = LaunchConfiguration("port")
    return LaunchDescription(
        [
            DeclareLaunchArgument("port", default_value="8080"),
            ExecuteProcess(
                cmd=["python3", "-m", "http.server", port, "--directory", dashboard_dir],
                output="screen",
            ),
        ]
    )
