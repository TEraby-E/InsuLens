#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${1:-60}"
source /opt/ros/humble/setup.bash
source "${PROJECT_DIR}/.venv/bin/activate"
source "${PROJECT_DIR}/install/setup.bash"

exec ros2 run insulens_perception train_yolov10 \
  --data "${PROJECT_DIR}/datasets/insulator_sim/data.yaml" \
  --model yolov10s.yaml \
  --epochs "${EPOCHS}" \
  --device 0 \
  --project "${PROJECT_DIR}/runs"
