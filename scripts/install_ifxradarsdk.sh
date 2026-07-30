#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

WHEEL_DIR="third_party/ifxradarsdk"
SDK_DIR="third_party/radar_sdk"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

python3 -m pip install -U pip setuptools wheel build cmake ninja

shopt -s nullglob
WHEELS=("$WHEEL_DIR"/*.whl)
shopt -u nullglob

if [ "${#WHEELS[@]}" -gt 0 ]; then
  python3 -m pip install "${WHEELS[0]}"
else
  if [ ! -d "$SDK_DIR" ]; then
    echo "No ifxradarsdk wheel found in $WHEEL_DIR and no bundled radar_sdk in $SDK_DIR" >&2
    exit 2
  fi

  BUILD_DIR="$SDK_DIR/build"
  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"
  (
    cd "$BUILD_DIR"
    cmake -G Ninja ..
    cmake --build . --target wheel-ifxradarsdk
  )

  mapfile -t BUILT_WHEELS < <(python3 - <<'PY'
from pathlib import Path
for p in Path("third_party/radar_sdk/build").rglob("ifxradarsdk-*.whl"):
    print(p)
PY
)
  if [ "${#BUILT_WHEELS[@]}" -eq 0 ]; then
    echo "Built radar_sdk but could not find ifxradarsdk wheel" >&2
    exit 3
  fi
  python3 -m pip install "${BUILT_WHEELS[0]}"
fi

python3 -c "from ifxradarsdk.fmcw import DeviceFmcw; print(DeviceFmcw.get_list())"
