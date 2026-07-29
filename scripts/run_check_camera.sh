#!/usr/bin/env bash
# One-camera checks on Jetson (no radar).
set -euo pipefail
cd "$(dirname "$0")/.."

# Help Miniforge find system libstdc++ if pip OpenCV needs CXXABI_1.3.15
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

python3 check_camera.py \
  --frames 30 \
  --infer \
  --infer-runs 10 \
  --save-preview artifacts/camera_preview.jpg \
  --out artifacts/camera_check_jetson.json
