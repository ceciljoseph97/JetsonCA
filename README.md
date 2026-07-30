# JetsonCA

Jetson Nano–oriented slice of `Crossattention` for **cam1 + radar2** edge inference and KPI benchmarking.

Training / Tk GUI / large datasets stay in `../Crossattention`. This package is what you sync onto the Nano.

## Layout

| Path | Role |
|---|---|
| `model.py` | Multimodal cross-attention net |
| `benchmark.py` | AI-DISCO KPI bench (synthetic or `--live`) |
| `infer_headless.py` | Live inference, no GUI |
| `export_onnx.py` | ONNX export → TensorRT |
| `jetson_env.py` | Tegra detect + thread/memory tweaks |
| `realtime_multimodal.py` | Camera stream + checkpoint load |
| `radar_*.py` / `range_*.py` | BGT radar path (needs Infineon SDK) |
| `artifacts/best_multimodal_crossattention.pt` | Checkpoint |

**Not included:** `train.py`, `gui_app.py`, `data/`, evaluation notebooks.

## Setup on Jetson

```bash
# 1) JetPack CUDA torch (use NVIDIA wheel matching your JetPack — do NOT pip install torch from PyPI)
# Example placeholder — pick the correct wheel for JP 4.6 / 5.x:
# pip3 install torch torchvision --index-url https://...

cd ~/JetsonCA   # or wherever you cloned/copied
python3 -m pip install -r requirements-jetson.txt

# 2) Infineon radar SDK Python wrapper (required for --live with BGT radar)
# Uses a prebuilt wheel from third_party/ifxradarsdk/ if present, otherwise
# builds one from the bundled third_party/radar_sdk source tree.
chmod +x scripts/install_ifxradarsdk.sh
./scripts/install_ifxradarsdk.sh

# verify radar SDK import + discovery
python3 -c "from ifxradarsdk.fmcw import DeviceFmcw; print(DeviceFmcw.get_list())"

# optional: direct device open smoke test
python3 -c "from ifxradarsdk.fmcw import DeviceFmcw; d=DeviceFmcw(); print(d.get_sensor_information()); d.close()"
```

## Sync from Windows laptop

```powershell
# from Windows (OpenSSH / scp)
scp -r "D:\...\4_code\JetsonCA" user@<nano-ip>:~/

# or git
# on Nano: git clone <JetsonCA remote>
```

Then Remote-SSH into Nano and open `~/JetsonCA`.

## Benchmark (KPI gate)

```bash
chmod +x scripts/*.sh
./scripts/run_benchmark_jetson.sh

# or
python3 benchmark.py --device cuda --mode both --n-cameras 1 --n-radars 2 \
  --out artifacts/benchmark_cam1_radar2_kpi_jetson.json
```

Live sensors:

```bash
python3 benchmark.py --live --device cuda --n-cameras 1 --n-radars 2 \
  --out artifacts/benchmark_cam1_radar2_kpi_jetson_live.json
```

## Headless inference

```bash
./scripts/run_infer_jetson.sh
# Ctrl+C to stop
```

## TensorRT path (optional)

```bash
python3 export_onnx.py --checkpoint artifacts/best_multimodal_crossattention.pt
# on Nano with TensorRT:
# trtexec --onnx=artifacts/multimodal_crossattention.onnx \
#   --saveEngine=artifacts/model_fp16.engine --fp16
```

## One-camera checks (today — no radar)

Plug USB camera, then:

```bash
cd ~/AIDisco/JetsonCA
git pull
chmod +x scripts/run_check_camera.sh
./scripts/run_check_camera.sh
```

Or step-by-step:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
python check_camera.py --frames 30 --infer --out artifacts/camera_check_jetson.json
```

Writes:
- `artifacts/camera_check_jetson.json` — probe + capture + optional `camera_only` latency
- `artifacts/camera_preview.jpg` — last frame

Also useful (synthetic, no USB needed):

```bash
python benchmark.py --device cuda --mode camera_only \
  --out artifacts/benchmark_camera_only_jetson.json
```

### Live camera-only KPI bench (today — no radar)

Full AI-DISCO KPI report on USB frames (`radar_present=False`):

```bash
chmod +x scripts/run_benchmark_live_camera_only.sh
./scripts/run_benchmark_live_camera_only.sh

# or:
python benchmark.py --live --mode camera_only --device cuda --camera-device 0 \
  --out artifacts/benchmark_live_camera_only_kpi_jetson.json
```

### Live camera-only labels / jsonl (today)

```bash
cd ~/AIDisco/JetsonCA
git pull
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# timed latency + ~20s live prints / jsonl
chmod +x scripts/run_live_camera_only.sh
./scripts/run_live_camera_only.sh

# or:
python live_camera_only.py --device cuda --camera-device 0 --runs 30 --duration-s 20 \
  --out artifacts/live_camera_only_jetson.json

# continuous until Ctrl+C
python infer_headless.py --no-radar --device cuda --camera-device 0
```

How live modes differ:

| Command | Camera | Radar | Model flags |
|---|---|---|---|
| `benchmark.py` (no `--live`) | synthetic | synthetic | as `--mode` |
| `benchmark.py --live --mode camera_only` | USB live | none | `radar_present=False` + full KPI JSON |
| `live_camera_only.py` / `--no-radar` | USB live | none | labels/jsonl (+ light latency) |
| `benchmark.py --live` (default both) | USB live | needs BGT | both present |
| tomorrow `--live` + radars | USB | 1–2 BGT | full MM |

**Tomorrow:** radar SDK + dual BGT, then full live cam1+radar2.

## OpenCV note (Miniforge)

Synthetic `benchmark.py` no longer needs OpenCV. If `import cv2` fails with `CXXABI_1.3.15`,
the **system** libstdc++ is too old for the pip wheel — use conda’s newer one:

```bash
conda install -y -c conda-forge "libstdcxx-ng>=13"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
python -c "import cv2; print(cv2.__version__)"

# still broken?
pip uninstall -y opencv-python opencv-python-headless
conda install -y -c conda-forge opencv
```

## AI-DISCO KPI targets (ours)

- latency mean < 100 ms, p95 < 150 ms, ≥ 8 FPS  
- weights < 32 MB, params < 5 M, buffer < 64 MB  
- < 15 GFLOPs/inf, < 8000 MMACs/inf  
- temporal < 50 MMACs, cross-attn < 100 MMACs  

Host (RTX) already clears these; **re-run on Orin** for the latency/FPS platform gate.

## Relation to Crossattention

| Crossattention | JetsonCA |
|---|---|
| Develop / train / GUI | Deploy / bench / live |
| Full repo + data | Slim edge package |
| Windows + desktop GPU | Jetson Nano class |

Keep model code in sync manually (or cherry-pick) when `model.py` / checkpoint format changes.
