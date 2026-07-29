#!/usr/bin/env bash
# Run AI-DISCO KPI bench on Jetson (model-forward, synthetic tensors).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 benchmark.py \
  --device cuda \
  --mode both \
  --n-cameras 1 \
  --n-radars 2 \
  --warmup 5 \
  --runs 30 \
  --out artifacts/benchmark_cam1_radar2_kpi_jetson.json
