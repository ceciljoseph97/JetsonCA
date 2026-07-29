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

# optional Infineon radar SDK (same as host Exploration install)
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

## OpenCV note (Miniforge)

Synthetic `benchmark.py` no longer needs OpenCV. If `import cv2` fails with `CXXABI_1.3.15`, fix later for live camera:

```bash
conda install -y -c conda-forge "libstdcxx-ng>=13"
# or
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```


- latency mean < 100 ms, p95 < 150 ms, ≥ 8 FPS  
- weights < 32 MB, params < 5 M, buffer < 64 MB  
- < 15 GFLOPs/inf, < 8000 MMACs/inf  
- temporal < 50 MMACs, cross-attn < 100 MMACs  

Host (RTX) already clears these; **re-run on Nano** for the latency/FPS platform gate.

## Relation to Crossattention

| Crossattention | JetsonCA |
|---|---|
| Develop / train / GUI | Deploy / bench / live |
| Full repo + data | Slim edge package |
| Windows + desktop GPU | Jetson Nano class |

Keep model code in sync manually (or cherry-pick) when `model.py` / checkpoint format changes.
