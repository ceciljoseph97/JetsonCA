#!/usr/bin/env bash
# One-camera checks on Jetson (no radar).
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer conda libstdc++ (newer CXXABI) over system — pip OpenCV often needs this.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

python3 check_camera.py \
  --frames 30 \
  --infer \
  --infer-runs 10 \
  --save-preview artifacts/camera_preview.jpg \
  --out artifacts/camera_check_jetson.json
