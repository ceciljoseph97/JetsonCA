#!/usr/bin/env bash
# Live multimodal inference with 1 camera, radar unavailable.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

python3 live_camera_only.py \
  --device cuda \
  --camera-device "${CAMERA_DEVICE:-0}" \
  --window 30 \
  --warmup 5 \
  --runs 30 \
  --duration-s "${DURATION_S:-20}" \
  --out artifacts/live_camera_only_jetson.json \
  --jsonl-out recordings/live_camera_only.jsonl
