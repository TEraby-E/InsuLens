#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}"
colcon build --symlink-install
