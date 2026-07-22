#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE_COUNT="${1:-1200}"
source /opt/ros/humble/setup.bash
source "${PROJECT_DIR}/.venv/bin/activate"
source "${PROJECT_DIR}/install/setup.bash"

if command -v xvfb-run >/dev/null 2>&1 && [[ -z "${DISPLAY:-}" ]]; then
  exec xvfb-run -a ros2 launch doggo_gazebo generate_dataset.launch.py \
    gui:=false num_samples:="${SAMPLE_COUNT}"
fi
exec ros2 launch doggo_gazebo generate_dataset.launch.py \
  gui:=true num_samples:="${SAMPLE_COUNT}"
