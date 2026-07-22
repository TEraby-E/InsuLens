#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${1:-50}"
source /opt/ros/humble/setup.bash
source "${PROJECT_DIR}/.venv/bin/activate"
source "${PROJECT_DIR}/install/setup.bash"

if [[ ! -f "${PROJECT_DIR}/datasets/cplid_yolo/data.yaml" ]]; then
  echo "Prepared CPLID dataset not found. Run ./scripts/download_cplid.sh." >&2
  exit 1
fi

exec ros2 run doggo_perception train_yolov10 \
  --data "${PROJECT_DIR}/datasets/cplid_yolo/data.yaml" \
  --model "${PROJECT_DIR}/models/insulator_yolov10s.pt" \
  --epochs "${EPOCHS}" \
  --imgsz 768 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --project "${PROJECT_DIR}/runs" \
  --name insulator_defect_yolov10s \
  --export "${PROJECT_DIR}/models/insulator_defect_yolov10s.pt"

