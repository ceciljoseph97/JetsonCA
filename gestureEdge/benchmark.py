"""gestureEdge benchmark: latency / params / MACs (radar-only dual).

  python gestureEdge/benchmark.py --checkpoint artifacts/gesture_edge_soli/best_gesture_edge.pt
  python gestureEdge/benchmark.py --device cuda --runs 50 --mode both
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from gestureEdge.ckpt import load_ckpt
from gestureEdge.preprocess import SOLI_LABELS

MODES = ("both", "radar1_only", "radar2_only")


def _sync(device: torch.device):
  if device.type == "cuda" and torch.cuda.is_available():
    torch.cuda.synchronize(device)


def _reset_peak(device: torch.device):
  if device.type == "cuda" and torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(int(device.index or 0))


def _cuda_mem(device: torch.device) -> dict[str, float] | None:
  if device.type != "cuda" or not torch.cuda.is_available():
    return None
  idx = int(device.index or 0)
  return {
    "allocated_mb": float(torch.cuda.memory_allocated(idx)) / (1024.0 ** 2),
    "reserved_mb": float(torch.cuda.memory_reserved(idx)) / (1024.0 ** 2),
    "peak_allocated_mb": float(torch.cuda.max_memory_allocated(idx)) / (1024.0 ** 2),
    "peak_reserved_mb": float(torch.cuda.max_memory_reserved(idx)) / (1024.0 ** 2),
  }


def count_parameters(model: nn.Module) -> dict[str, int]:
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return {"total": int(total), "trainable": int(trainable)}


def weight_memory_mb(model: nn.Module) -> dict[str, float]:
  n = sum(p.numel() * p.element_size() for p in model.parameters())
  n += sum(b.numel() * b.element_size() for b in model.buffers())
  return {"bytes": float(n), "kb": n / 1024.0, "mb": n / (1024.0 ** 2)}


def estimate_macs(model: nn.Module, r1: torch.Tensor, r2: torch.Tensor) -> dict[str, float]:
  """Best-effort MACs via thop; falls back to 0."""
  try:
    from thop import profile as thop_profile

    macs, _ = thop_profile(model, inputs=(r1, r2), verbose=False)
    macs = float(macs)
    return {
      "total_macs": macs,
      "total_mops": macs / 1e6,
      "total_flops": macs * 2.0,
      "total_gflops": (macs * 2.0) / 1e9,
      "method": "thop",
    }
  except Exception as exc:
    return {
      "total_macs": 0.0,
      "total_mops": 0.0,
      "total_flops": 0.0,
      "total_gflops": 0.0,
      "method": "unavailable",
      "error": str(exc),
    }


def measure_latency(
  model: nn.Module,
  r1: torch.Tensor,
  r2: torch.Tensor,
  p1: torch.Tensor,
  p2: torch.Tensor,
  *,
  warmup: int,
  runs: int,
  device: torch.device,
) -> dict[str, Any]:
  model.eval()
  _reset_peak(device)
  with torch.no_grad():
    for _ in range(warmup):
      model(r1, r2, radar1_present=p1, radar2_present=p2)
    _sync(device)
    times: list[float] = []
    for _ in range(runs):
      _sync(device)
      t0 = time.perf_counter()
      model(r1, r2, radar1_present=p1, radar2_present=p2)
      _sync(device)
      times.append((time.perf_counter() - t0) * 1000.0)
  arr = np.asarray(times, dtype=np.float64)
  out: dict[str, Any] = {
    "runs": float(runs),
    "warmup": float(warmup),
    "latency_ms_mean": float(arr.mean()),
    "latency_ms_std": float(arr.std()),
    "latency_ms_min": float(arr.min()),
    "latency_ms_max": float(arr.max()),
    "latency_ms_p50": float(np.percentile(arr, 50)),
    "latency_ms_p95": float(np.percentile(arr, 95)),
    "latency_ms_p99": float(np.percentile(arr, 99)),
    "throughput_fps": float(1000.0 / max(arr.mean(), 1e-9)),
  }
  mem = _cuda_mem(device)
  if mem is not None:
    out["cuda_memory"] = mem
  return out


def _presence(mode: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
  if mode == "radar1_only":
    return torch.tensor([True], device=device), torch.tensor([False], device=device)
  if mode == "radar2_only":
    return torch.tensor([False], device=device), torch.tensor([True], device=device)
  return torch.tensor([True], device=device), torch.tensor([True], device=device)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
  device = torch.device(args.device)
  ckpt = Path(args.checkpoint)
  model, labels, cfg = load_ckpt(ckpt, str(device))
  window = int(getattr(args, "window", None) or cfg.get("window", 40) or 40)
  channels = int(cfg.get("in_channels", 3))
  r1 = torch.randn(1, window, channels, 32, 32, device=device)
  r2 = torch.randn(1, window, channels, 32, 32, device=device)

  params = count_parameters(model)
  weights = weight_memory_mb(model)
  do_mem = bool(getattr(args, "profile_memory", True))
  do_cmp = bool(getattr(args, "profile_compute", True))

  modes = list(MODES) if getattr(args, "all_modes", False) else [str(args.mode)]
  if modes[0] not in MODES:
    modes = ["both"]

  profiles = []
  compute_ref = None
  for mode in modes:
    p1, p2 = _presence(mode, device)
    lat = (
      measure_latency(
        model, r1, r2, p1, p2, warmup=int(args.warmup), runs=int(args.runs), device=device
      )
      if do_cmp
      else {}
    )
    compute = estimate_macs(model, r1, r2) if do_cmp and compute_ref is None else (compute_ref or {})
    if compute_ref is None and do_cmp:
      compute_ref = compute
    mean_s = float(lat.get("latency_ms_mean", 0.0) or 0.0) / 1000.0
    macs = float(compute.get("total_macs", 0.0) or 0.0)
    flops = float(compute.get("total_flops", 0.0) or 0.0)
    perf = {
      "achieved_mops_per_s": (macs / 1e6) / max(mean_s, 1e-9) if mean_s else 0.0,
      "achieved_gflops_per_s": (flops / 1e9) / max(mean_s, 1e-9) if mean_s else 0.0,
    }
    profiles.append(
      {
        "mode": mode,
        "input": {
          "window": window,
          "shape": [1, window, channels, 32, 32],
          "radar1_present": bool(p1.item()),
          "radar2_present": bool(p2.item()),
        },
        "latency": lat,
        "compute": compute,
        "performance": perf,
        "cuda_memory": lat.get("cuda_memory") if do_mem else None,
        "labels": labels,
      }
    )

  report: dict[str, Any] = {
    "model": "gestureEdge",
    "arch": "cnn_lstm_cross_dual",
    "checkpoint": str(ckpt.resolve()),
    "device": str(device),
    "labels": labels or list(SOLI_LABELS),
    "config": cfg,
    "benchmark_source": "synthetic",
    "parameters": params,
    "parameters_millions": round(params["total"] / 1e6, 4),
    "weight_memory": weights if do_mem else None,
    "buffer_memory_estimate": {
      "mb": (2 * window * channels * 32 * 32 * 4) / (1024.0 ** 2),
      "note": "2 radars × window × float32 RD frames",
    }
    if do_mem
    else None,
    "profiling": {
      "memory": do_mem,
      "resource": bool(getattr(args, "profile_resource", False)),
      "compute": do_cmp,
    },
    "platform": {
      "hostname": socket.gethostname(),
      "os": f"{platform.system()} {platform.release()}",
      "python": platform.python_version(),
      "torch": torch.__version__,
      "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" and torch.cuda.is_available() else None,
    },
    "deployment": {"profile": "gesture_edge_dual_radar", "n_radars": 2, "n_cameras": 0, "n_audio": 0},
    "profiles": profiles,
  }

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2), encoding="utf-8")
  report["wrote"] = str(out.resolve())
  return report


def format_profile_report(report: dict[str, Any]) -> str:
  flags = report.get("profiling") or {}
  plat = report.get("platform") or {}
  lines = [
    f"model       gestureEdge ({report.get('config', {}).get('arch') or 'cnn_lstm_cross_dual'})",
    f"checkpoint  {report.get('checkpoint')}",
    f"device      {report.get('device')}  gpu={plat.get('gpu_name') or 'cpu'}",
    f"platform    {plat.get('hostname')}  {plat.get('os')}",
    f"params      {report['parameters']['total']:,}  ({report.get('parameters_millions')} M)",
    f"labels      {', '.join(report.get('labels') or [])}",
  ]
  if flags.get("memory", True):
    wm = report.get("weight_memory") or {}
    bm = report.get("buffer_memory_estimate") or {}
    lines.append("- memory -")
    lines.append(f"  weights    {wm.get('mb', 0):.3f} MB")
    lines.append(f"  input buf  {bm.get('mb', 0):.3f} MB")
    for profile in report.get("profiles") or []:
      cuda = profile.get("cuda_memory") or (profile.get("latency") or {}).get("cuda_memory")
      if cuda:
        lines.append(
          f"  cuda peak [{profile['mode']}]  "
          f"alloc={cuda.get('peak_allocated_mb', 0):.1f} MB"
        )
  if flags.get("compute", True):
    lines.append("- compute -")
    for profile in report.get("profiles") or []:
      c = profile.get("compute") or {}
      lat = profile.get("latency") or {}
      perf = profile.get("performance") or {}
      lines.append(f"  [{profile.get('mode')}]")
      lines.append(
        f"    latency  mean={lat.get('latency_ms_mean', 0):.2f} ms  "
        f"p50={lat.get('latency_ms_p50', 0):.2f}  "
        f"p95={lat.get('latency_ms_p95', 0):.2f}  "
        f"fps={lat.get('throughput_fps', 0):.2f}"
      )
      if c.get("total_macs"):
        lines.append(
          f"    {c.get('total_gflops', 0):.4f} GFLOPs/inf  "
          f"{c.get('total_mops', 0):.2f} MMACs/inf  "
          f"achieved={perf.get('achieved_gflops_per_s', 0):.2f} GFLOP/s  "
          f"({c.get('method')})"
        )
      elif c.get("method") == "unavailable":
        lines.append(f"    FLOPs    unavailable ({c.get('error', 'no thop')})")
  if report.get("wrote"):
    lines.append(f"\nwrote {report['wrote']}")
  return "\n".join(lines)


def parse_args():
  p = argparse.ArgumentParser(description="gestureEdge latency / params / MACs")
  p.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("artifacts/gesture_edge_soli/best_gesture_edge.pt"),
  )
  p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--window", type=int, default=None)
  p.add_argument("--warmup", type=int, default=10)
  p.add_argument("--runs", type=int, default=30)
  p.add_argument("--mode", choices=MODES, default="both")
  p.add_argument("--all-modes", action="store_true")
  p.add_argument("--out", type=Path, default=Path("artifacts/benchmark_gesture_edge.json"))
  p.add_argument("--profile-memory", action="store_true", default=True)
  p.add_argument("--no-profile-memory", action="store_false", dest="profile_memory")
  p.add_argument("--profile-compute", action="store_true", default=True)
  p.add_argument("--no-profile-compute", action="store_false", dest="profile_compute")
  p.add_argument("--profile-resource", action="store_true", default=False)
  return p.parse_args()


def main():
  args = parse_args()
  report = run_benchmark(args)
  print(format_profile_report(report))


if __name__ == "__main__":
  main()
