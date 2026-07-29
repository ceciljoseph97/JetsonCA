#!/usr/bin/env bash
# Live USB-camera KPI bench on Jetson (no radar SDK required).
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
python3 benchmark.py \
  --live \
  --mode camera_only \
  --device cuda \
  --camera-device "${CAMERA_DEVICE:-0}" \
  --warmup 5 \
  --runs 30 \
  --out artifacts/benchmark_live_camera_only_kpi_jetson.json
