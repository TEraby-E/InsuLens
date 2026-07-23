#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"

# A symlink-install workspace is not relocatable. Remove only dangling generated
# links so colcon can recreate them when the project directory has moved.
if [[ -d "${PROJECT_DIR}/install" ]]; then
  find "${PROJECT_DIR}/install" -xtype l -delete
fi

colcon build --symlink-install
