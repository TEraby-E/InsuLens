#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble was not found at /opt/ros/humble." >&2
  exit 1
fi

if [[ -f "${VENV_DIR}/bin/activate" ]] && \
  ! grep -Fq "VIRTUAL_ENV=${VENV_DIR}" "${VENV_DIR}/bin/activate"; then
  echo "Virtual environment was created at a different project path; recreating it."
  rm -rf -- "${VENV_DIR}"
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  if [[ -d "${VENV_DIR}" ]]; then
    rm -rf -- "${VENV_DIR}"
  fi
  /usr/bin/python3.10 -m venv --system-site-packages "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install colcon-common-extensions

if ! python -c 'import torch' >/dev/null 2>&1; then
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo "Environment ready. Build with: ${PROJECT_DIR}/scripts/build.sh"
