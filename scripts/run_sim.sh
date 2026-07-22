#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-${PROJECT_DIR}/models/insulator_yolov10s.pt}"
source /opt/ros/humble/setup.bash
source "${PROJECT_DIR}/.venv/bin/activate"
source "${PROJECT_DIR}/install/setup.bash"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found: ${MODEL_PATH}" >&2
  echo "Generate data and run ./scripts/train.sh first." >&2
  exit 1
fi

if command -v xvfb-run >/dev/null 2>&1 && [[ -z "${DISPLAY:-}" ]]; then
  exec xvfb-run -a ros2 launch doggo_bringup inspection_sim.launch.py \
    model_path:="${MODEL_PATH}" device:=cuda:0 gui:=false visualize:=false
fi

exec ros2 launch doggo_bringup inspection_sim.launch.py \
  model_path:="${MODEL_PATH}" device:=cuda:0 gui:=true visualize:=true
