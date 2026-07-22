#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_DIR}/datasets/raw"
ARCHIVE="${RAW_DIR}/CPLID.zip"
SOURCE_URL="https://github.com/InsulatorData/InsulatorDataSet/archive/refs/heads/master.zip"
DOWNLOAD_URL="${CPLID_DOWNLOAD_URL:-https://ghfast.top/${SOURCE_URL}}"

mkdir -p "${RAW_DIR}/cplid"
if [[ ! -f "${ARCHIVE}" ]]; then
  curl -L --retry 3 -C - "${DOWNLOAD_URL}" -o "${ARCHIVE}"
fi
unzip -q -o "${ARCHIVE}" -d "${RAW_DIR}/cplid"

python "${PROJECT_DIR}/scripts/prepare_cplid.py" \
  --source "${RAW_DIR}/cplid/InsulatorDataSet-master" \
  --output "${PROJECT_DIR}/datasets/cplid_yolo"

