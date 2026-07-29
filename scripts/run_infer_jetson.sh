#!/usr/bin/env bash
# Live cam1+radar2 headless inference on Jetson.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 infer_headless.py \
  --device cuda \
  --camera-device 0 \
  --window 30 \
  --print-every 1.0 \
  --jsonl-out recordings/live_preds.jsonl
