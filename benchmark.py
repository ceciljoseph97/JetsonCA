#!/usr/bin/env python3
"""System-level benchmark: latency, params, FLOPs/MOPS + AI-DISCO KPI report.

Target deployment profile: 1 camera + 2 radars (dual-radar early-fused), Jetson Nano class.

Common commands:
  python benchmark.py
  python benchmark.py --all-modes
  python benchmark.py --device cpu
  python benchmark.py --device cuda --runs 100 --warmup 20
  python benchmark.py --mode radar_only
  python benchmark.py --out artifacts/benchmark_cam1_radar2_kpi_RTX_5090.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from jetson_env import apply_jetson_runtime_tweaks, default_device, is_jetson
from radar_utils import DualRadarSession, fuse_radar_streams_for_model
from realtime_multimodal import CameraStream, load_checkpoint, preprocess_camera_frame

# AI-DISCO KPIs for cam1+radar2 multimodal activity recognition on Jetson Nano–class edge.
# Tuned to OUR stack (windowed cross-attention), not external partner gesture budgets.
AIDISCO_KPI_TARGETS: dict[str, Any] = {
  "source": "AI-DISCO — cam1+radar2 multimodal activity recognition (Jetson Nano class)",
  "deployment_profile": "cam1_radar2",
  "target_platform": "jetson_nano_4gb",
  # Realtime forward (model only; sensor I/O separate)
  "latency_ms_mean_max": 100.0,  # >=10 FPS usable
  "latency_ms_p95_max": 150.0,
  "throughput_fps_min": 8.0,
  # Footprint on 4 GB shared RAM
  "weight_memory_mb_max": 32.0,
  "params_max": 5_000_000,
  "buffer_memory_mb_max": 64.0,
  # Compute envelope — keep under ~Nano FP32 comfort zone for windowed CNN+attn
  "gflops_per_inference_max": 15.0,
  "mmacs_per_inference_max": 8_000.0,
  # Architecture health: camera CNN may dominate, but temporal path must stay light
  "temporal_mmacs_max": 50.0,
  "cross_attn_mmacs_max": 100.0,
}

CLI_EXAMPLES = """Examples:
  python benchmark.py
      Run the default benchmark on the default checkpoint and write JSON to artifacts/.

  python benchmark.py --live
      Run a live benchmark with connected camera/radar devices.

  python benchmark.py --all-modes
      Benchmark both, radar_only, and camera_only in one run.

  python benchmark.py --device cpu
      Force CPU benchmarking.

  python benchmark.py --device cuda --runs 100 --warmup 20
      More stable GPU timing with longer warmup and more measured runs.

  python benchmark.py --mode radar_only
      Measure only the radar_present=true / camera_present=false path.

  python benchmark.py --out artifacts/benchmark_cam1_radar2_kpi_RTX_5090.json
      Save results to a machine-specific JSON filename.
"""

KEY_MEANINGS: dict[str, str] = {
  "parameters_millions": "Total parameter count in millions.",
  "weight_memory": "Parameter + registered buffer memory footprint of the model in RAM/VRAM.",
  "buffer_memory_estimate": "Approximate input-window buffer footprint for deployment, not peak activations.",
  "operation_specs": "Per-block theoretical operation breakdown for the primary mode.",
  "profiles": "Per-mode benchmark results; usually contains both, radar_only, and/or camera_only.",
  "total_macs": "Total multiply-accumulate operations for one forward pass.",
  "total_mops": "Total MACs divided by 1e6. Despite the name, this is a count per inference, not per second.",
  "total_flops": "Total floating-point operations for one forward pass, using FLOPs = 2 * MACs.",
  "total_gflops": "Total FLOPs divided by 1e9 for one forward pass.",
  "latency_ms_mean": "Average forward-pass latency in milliseconds over measured runs.",
  "latency_ms_p50": "Median forward-pass latency in milliseconds.",
  "latency_ms_p95": "95th percentile forward-pass latency in milliseconds.",
  "throughput_fps": "Approximate inferences per second, computed as 1000 / mean latency_ms.",
  "achieved_mops_per_s": "Theoretical MAC count divided by measured mean latency; effective achieved throughput in MOPS/s.",
  "achieved_gflops_per_s": "Theoretical FLOP count divided by measured mean latency; effective achieved throughput in GFLOPs/s.",
  "conv_share": "Fraction of total theoretical compute spent in convolution layers.",
  "conv_mmacs": "Millions of MACs spent in convolution layers.",
  "temporal_transformer_mmacs": "Millions of MACs spent in temporal self-attention + feed-forward blocks.",
  "cross_attention_mmacs": "Millions of MACs spent in radar-camera cross-attention blocks.",
  "mmacs_per_inference": "Total millions of MACs for one forward pass.",
  "gflops_per_inference": "Total billions of FLOPs for one forward pass.",
  "kpi.pass": "Boolean pass/fail results against the AI-DISCO deployment thresholds.",
  "kpi.pass_count": "Number of KPI checks passed.",
}


def count_parameters(model: nn.Module) -> dict[str, int]:
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return {"total": int(total), "trainable": int(trainable)}


def weight_memory_bytes(model: nn.Module) -> dict[str, float]:
  total_bytes = 0
  for p in model.parameters():
    total_bytes += p.numel() * p.element_size()
  for b in model.buffers():
    total_bytes += b.numel() * b.element_size()
  return {
    "bytes": float(total_bytes),
    "kb": float(total_bytes) / 1024.0,
    "mb": float(total_bytes) / (1024.0 ** 2),
  }


def estimate_activation_buffer_mb(
  *,
  batch: int,
  window: int,
  image_size: int,
  radar_h: int = 32,
  radar_w: int = 32,
  dtype_bytes: int = 4,
  n_cameras: int = 1,
  n_radars: int = 2,
) -> dict[str, float]:
  """Rough live-buffer estimate for cam1 + radar2 streaming window."""
  cam_elems = batch * window * n_cameras * 3 * image_size * image_size
  radar_elems = batch * window * n_radars * 3 * radar_h * radar_w
  # fused radar is one tensor after early fuse
  fused_radar_elems = batch * window * 3 * radar_h * radar_w
  total_elems = cam_elems + radar_elems + fused_radar_elems
  bytes_ = total_elems * dtype_bytes
  return {
    "bytes": float(bytes_),
    "kb": float(bytes_) / 1024.0,
    "mb": float(bytes_) / (1024.0 ** 2),
    "n_cameras": float(n_cameras),
    "n_radars": float(n_radars),
    "note": "input window buffers only (not full activation peak)",
  }


def collect_platform_info(device: torch.device) -> dict[str, Any]:
  info: dict[str, Any] = {
    "hostname": platform.node(),
    "os": platform.platform(),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "device": str(device),
    "cuda_available": bool(torch.cuda.is_available()),
    "machine": platform.machine(),
    "processor": platform.processor(),
  }
  if device.type == "cuda" and torch.cuda.is_available():
    idx = device.index or 0
    props = torch.cuda.get_device_properties(idx)
    info["gpu_name"] = props.name
    info["gpu_total_memory_mb"] = round(props.total_memory / (1024 ** 2), 1)
    info["jetson_like"] = "tegra" in props.name.lower() or "orin" in props.name.lower() or "nano" in props.name.lower()
  else:
    info["gpu_name"] = None
    info["jetson_like"] = False
  return info


def build_kpi_report(
  *,
  profile: dict[str, Any],
  params: dict[str, int],
  weight_mem: dict[str, float],
  buffer_mem: dict[str, float],
  targets: dict[str, Any] | None = None,
) -> dict[str, Any]:
  t = dict(targets or AIDISCO_KPI_TARGETS)
  compute = profile["compute"]
  latency = profile["latency"]
  perf = profile["performance"]
  specs = profile.get("compute", {}).get("operation_specs", {}) or {}

  mmacs = compute["total_mops"]
  mflops = compute["total_gflops"] * 1000.0
  gflops = compute["total_gflops"]
  mean_s = latency["latency_ms_mean"] / 1000.0
  mmacs_per_s_achieved = mmacs / max(mean_s, 1e-12)
  mflops_per_s_achieved = mflops / max(mean_s, 1e-12)
  gflops_per_s_achieved = gflops / max(mean_s, 1e-12)

  conv = specs.get("conv", {})
  temporal = specs.get("temporal_transformer", {})
  cross = specs.get("cross_attention", {})

  measured = {
    "deployment_profile": t["deployment_profile"],
    "latency_ms_mean": latency["latency_ms_mean"],
    "latency_ms_p50": latency["latency_ms_p50"],
    "latency_ms_p95": latency["latency_ms_p95"],
    "throughput_fps": latency["throughput_fps"],
    "mmacs_per_inference": mmacs,
    "mflops_per_inference": mflops,
    "gflops_per_inference": gflops,
    "mmacs_per_s_achieved": mmacs_per_s_achieved,
    "mflops_per_s_achieved": mflops_per_s_achieved,
    "gflops_per_s_achieved": gflops_per_s_achieved,
    "achieved_mops_per_s": perf["achieved_mops_per_s"],
    "achieved_gflops_per_s": perf["achieved_gflops_per_s"],
    "parameters": params["total"],
    "weight_memory_kb": weight_mem["kb"],
    "weight_memory_mb": weight_mem["mb"],
    "buffer_memory_mb": buffer_mem["mb"],
    "conv_mmacs": float(conv.get("mmacs", 0.0)),
    "conv_share": float(conv.get("share_of_total", 0.0)),
    "temporal_transformer_mmacs": float(temporal.get("mmacs", 0.0)),
    "temporal_transformer_share": float(temporal.get("share_of_total", 0.0)),
    "cross_attention_mmacs": float(cross.get("mmacs", 0.0)),
    "cross_attention_share": float(cross.get("share_of_total", 0.0)),
  }

  checks = {
    "latency_mean_lt_100ms": measured["latency_ms_mean"] < float(t["latency_ms_mean_max"]),
    "latency_p95_lt_150ms": measured["latency_ms_p95"] < float(t["latency_ms_p95_max"]),
    "throughput_ge_8fps": measured["throughput_fps"] >= float(t["throughput_fps_min"]),
    "weight_memory_lt_32mb": measured["weight_memory_mb"] < float(t["weight_memory_mb_max"]),
    "params_lt_5m": measured["parameters"] < int(t["params_max"]),
    "buffer_memory_lt_64mb": measured["buffer_memory_mb"] < float(t["buffer_memory_mb_max"]),
    "gflops_per_inf_lt_15": measured["gflops_per_inference"] < float(t["gflops_per_inference_max"]),
    "mmacs_per_inf_lt_8000": measured["mmacs_per_inference"] < float(t["mmacs_per_inference_max"]),
    "temporal_mmacs_lt_50": measured["temporal_transformer_mmacs"] < float(t["temporal_mmacs_max"]),
    "cross_attn_mmacs_lt_100": measured["cross_attention_mmacs"] < float(t["cross_attn_mmacs_max"]),
  }

  return {
    "targets": t,
    "measured": measured,
    "pass": checks,
    "pass_count": int(sum(1 for v in checks.values() if v)),
    "check_count": int(len(checks)),
    "interpretation": {
      "latency": "Model-forward latency on the bench device (sensor I/O not included).",
      "throughput_fps": "1000 / mean latency; usable realtime floor is 8 FPS.",
      "gflops_per_inference": "Theoretical cost per window; Nano class comfort < 15 GFLOPs.",
      "conv_share": "Frame CNN share — expected dominant for 112² × window.",
      "temporal_transformer_mmacs": "Per-modality TemporalEncoder (self-attn + FFN).",
      "cross_attention_mmacs": "Radar↔camera CrossAttentionBlock stack.",
      "jetson_note": (
        "Passing host CUDA latency does not guarantee Nano latency; "
        "re-run with --device cuda on the Jetson for platform gate."
      ),
    },
  }


def _conv2d_macs(module: nn.Conv2d, input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> int:
  # input: (N, C_in, H, W), output: (N, C_out, H_out, W_out)
  batch = int(output_shape[0])
  cout, hout, wout = int(output_shape[1]), int(output_shape[2]), int(output_shape[3])
  cin = int(module.in_channels)
  kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
  groups = int(module.groups)
  # MACs per output element = (C_in / groups) * Kh * Kw
  macs_per_out = (cin // groups) * kh * kw
  return batch * cout * hout * wout * macs_per_out


def _linear_macs(module: nn.Linear, input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> int:
  # last dim is features; all leading dims are batch-like
  out_elems = int(np.prod(output_shape[:-1])) if len(output_shape) > 1 else 1
  return out_elems * int(module.in_features) * int(module.out_features)


def _mha_macs(module: nn.MultiheadAttention, query_shape: tuple[int, ...]) -> int:
  # Approximate: QKV proj + attn matmul + out proj for batch_first (B, T, E)
  if len(query_shape) != 3:
    return 0
  batch, seq, embed = (int(x) for x in query_shape)
  # 3 * B * T * E * E  (Q,K,V) + B * H * T * T * (E/H) roughly as B*T*T*E + B*T*E*E out
  qkv = 3 * batch * seq * embed * embed
  attn = batch * seq * seq * embed
  out = batch * seq * embed * embed
  return qkv + attn + out


def _module_name(root: nn.Module, target: nn.Module) -> str:
  for name, module in root.named_modules():
    if module is target:
      return name or target.__class__.__name__
  return target.__class__.__name__


def _classify_block(name: str) -> str:
  n = name or ""
  if n.startswith("radar_encoder"):
    return "conv_radar"
  if n.startswith("camera_encoder"):
    return "conv_camera"
  if n.startswith("radar_temporal"):
    return "temporal_radar"
  if n.startswith("camera_temporal"):
    return "temporal_camera"
  if n.startswith("layers"):
    return "cross_attention"
  if n.startswith(("shared_proj", "activity_classifier", "classifier", "human_classifier")):
    return "head"
  return "other"


def _is_under_mha(root: nn.Module, module: nn.Module) -> bool:
  """True if module is nested inside a MultiheadAttention (avoid double-count)."""
  for parent in root.modules():
    if isinstance(parent, nn.MultiheadAttention) and parent is not module:
      for child in parent.modules():
        if child is module:
          return True
  return False


def _agg_op_bundle(records: list[dict[str, Any]]) -> dict[str, Any]:
  macs = int(sum(int(r["macs"]) for r in records))
  by_op: dict[str, int] = defaultdict(int)
  for r in records:
    by_op[str(r["op"])] += int(r["macs"])
  return {
    "macs": macs,
    "flops": macs * 2,
    "mops": macs / 1e6,
    "mmacs": macs / 1e6,
    "mflops": (macs * 2) / 1e6,
    "gflops": (macs * 2) / 1e9,
    "ops": len(records),
    "macs_by_op": {k: int(v) for k, v in by_op.items()},
    "mops_by_op": {k: float(v) / 1e6 for k, v in by_op.items()},
    "layers": records,
  }


def count_parameters_by_block(model: nn.Module) -> dict[str, dict[str, float]]:
  blocks: dict[str, int] = defaultdict(int)
  for name, param in model.named_parameters():
    block = _classify_block(name)
    blocks[block] += int(param.numel())
  out: dict[str, dict[str, float]] = {}
  for block, n in blocks.items():
    out[block] = {
      "parameters": float(n),
      "weight_kb": float(n * 4) / 1024.0,
      "weight_mb": float(n * 4) / (1024.0 ** 2),
    }
  return out


def build_operation_specs(records: list[dict[str, Any]], *, total_macs: int) -> dict[str, Any]:
  by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for rec in records:
    by_block[str(rec.get("block", "other"))].append(rec)

  conv_records = by_block.get("conv_radar", []) + by_block.get("conv_camera", [])
  temporal_records = by_block.get("temporal_radar", []) + by_block.get("temporal_camera", [])
  cross_records = by_block.get("cross_attention", [])
  head_records = by_block.get("head", [])

  def _share(macs: int) -> float:
    return float(macs) / float(total_macs) if total_macs > 0 else 0.0

  conv = _agg_op_bundle(conv_records)
  temporal = _agg_op_bundle(temporal_records)
  cross = _agg_op_bundle(cross_records)
  head = _agg_op_bundle(head_records)

  conv["share_of_total"] = _share(int(conv["macs"]))
  temporal["share_of_total"] = _share(int(temporal["macs"]))
  cross["share_of_total"] = _share(int(cross["macs"]))
  head["share_of_total"] = _share(int(head["macs"]))

  return {
    "conv": {
      **conv,
      "description": "Frame CNN encoders (radar_encoder + camera_encoder Conv2d)",
      "by_stream": {
        "radar": _agg_op_bundle(by_block.get("conv_radar", [])),
        "camera": _agg_op_bundle(by_block.get("conv_camera", [])),
      },
    },
    "temporal_transformer": {
      **temporal,
      "description": (
        "Per-modality TemporalEncoder (TransformerEncoderLayer: self-attn MHA + FFN linears)"
      ),
      "by_stream": {
        "radar": _agg_op_bundle(by_block.get("temporal_radar", [])),
        "camera": _agg_op_bundle(by_block.get("temporal_camera", [])),
      },
      "attn_macs": int(sum(int(r["macs"]) for r in temporal_records if r["op"] == "multihead_attention")),
      "ffn_macs": int(sum(int(r["macs"]) for r in temporal_records if r["op"] == "linear")),
    },
    "cross_attention": {
      **cross,
      "description": "CrossAttentionBlock stack (radar↔camera MHA + FFN)",
    },
    "head": {
      **head,
      "description": "shared_proj + activity/human classifiers",
    },
    "by_block": {k: _agg_op_bundle(v) for k, v in by_block.items()},
  }


class FlopCounter:
  """Hook-based MAC counter. FLOPs ≈ 2 * MACs for mul-add pairs."""

  def __init__(self, model: nn.Module):
    self.model = model
    self.records: list[dict[str, Any]] = []
    self._handles: list[Any] = []

  def __enter__(self) -> FlopCounter:
    self.records.clear()

    def conv_hook(module: nn.Conv2d, inputs, output):
      x = inputs[0]
      name = _module_name(self.model, module)
      macs = _conv2d_macs(module, tuple(x.shape), tuple(output.shape))
      self.records.append(
        {
          "op": "conv2d",
          "block": _classify_block(name),
          "name": name,
          "in_shape": list(x.shape),
          "out_shape": list(output.shape),
          "kernel": list(module.kernel_size) if isinstance(module.kernel_size, tuple) else [module.kernel_size] * 2,
          "macs": int(macs),
          "flops": int(macs * 2),
        }
      )

    def linear_hook(module: nn.Linear, inputs, output):
      if _is_under_mha(self.model, module):
        return  # counted inside multihead_attention formula
      x = inputs[0]
      name = _module_name(self.model, module)
      macs = _linear_macs(module, tuple(x.shape), tuple(output.shape))
      self.records.append(
        {
          "op": "linear",
          "block": _classify_block(name),
          "name": name,
          "in_shape": list(x.shape),
          "out_shape": list(output.shape),
          "macs": int(macs),
          "flops": int(macs * 2),
        }
      )

    def mha_hook(module: nn.MultiheadAttention, inputs, output):
      q = inputs[0]
      name = _module_name(self.model, module)
      macs = _mha_macs(module, tuple(q.shape))
      self.records.append(
        {
          "op": "multihead_attention",
          "block": _classify_block(name),
          "name": name,
          "in_shape": list(q.shape),
          "macs": int(macs),
          "flops": int(macs * 2),
        }
      )

    for module in self.model.modules():
      if isinstance(module, nn.Conv2d):
        self._handles.append(module.register_forward_hook(conv_hook))
      elif isinstance(module, nn.Linear):
        self._handles.append(module.register_forward_hook(linear_hook))
      elif isinstance(module, nn.MultiheadAttention):
        self._handles.append(module.register_forward_hook(mha_hook))
    return self

  def __exit__(self, exc_type, exc, tb):
    for h in self._handles:
      h.remove()
    self._handles.clear()

  def summary(self) -> dict[str, Any]:
    by_op: dict[str, int] = defaultdict(int)
    conv_layers: list[dict[str, Any]] = []
    total_macs = 0
    for rec in self.records:
      by_op[rec["op"]] += rec["macs"]
      total_macs += rec["macs"]
      if rec["op"] == "conv2d":
        conv_layers.append(rec)

    specs = build_operation_specs(self.records, total_macs=total_macs)
    return {
      "total_macs": int(total_macs),
      "total_flops": int(total_macs * 2),
      "total_mops": float(total_macs) / 1e6,
      "total_gflops": float(total_macs * 2) / 1e9,
      "macs_by_op": {k: int(v) for k, v in by_op.items()},
      "flops_by_op": {k: int(v * 2) for k, v in by_op.items()},
      "mops_by_op": {k: float(v) / 1e6 for k, v in by_op.items()},
      "conv2d_layers": conv_layers,
      "conv2d_macs": int(by_op.get("conv2d", 0)),
      "conv2d_mops": float(by_op.get("conv2d", 0)) / 1e6,
      "conv2d_flops": int(by_op.get("conv2d", 0) * 2),
      "operation_specs": specs,
    }


def _sync(device: torch.device):
  if device.type == "cuda":
    torch.cuda.synchronize(device)

def measure_latency(
  model: nn.Module,
  radar: torch.Tensor,
  camera: torch.Tensor,
  radar_present: torch.Tensor,
  camera_present: torch.Tensor,
  *,
  warmup: int,
  runs: int,
  device: torch.device,
) -> dict[str, float]:
  model.eval()
  with torch.no_grad():
    for _ in range(warmup):
      model(radar, camera, radar_present=radar_present, camera_present=camera_present)
    _sync(device)

    times_ms: list[float] = []
    for _ in range(runs):
      _sync(device)
      t0 = time.perf_counter()
      model(radar, camera, radar_present=radar_present, camera_present=camera_present)
      _sync(device)
      times_ms.append((time.perf_counter() - t0) * 1000.0)

  arr = np.asarray(times_ms, dtype=np.float64)
  return {
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


def _summarize_timing_series(values_ms: list[float]) -> dict[str, float]:
  arr = np.asarray(values_ms, dtype=np.float64)
  return {
    "count": float(arr.size),
    "mean_ms": float(arr.mean()),
    "std_ms": float(arr.std()),
    "min_ms": float(arr.min()),
    "max_ms": float(arr.max()),
    "p50_ms": float(np.percentile(arr, 50)),
    "p95_ms": float(np.percentile(arr, 95)),
    "p99_ms": float(np.percentile(arr, 99)),
  }


def make_inputs(
  *,
  batch: int,
  window: int,
  image_size: int,
  device: torch.device,
  dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
  radar = torch.randn(batch, window, 3, 32, 32, device=device, dtype=dtype)
  camera = torch.randn(batch, window, 3, image_size, image_size, device=device, dtype=dtype)
  return radar, camera


def profile_mode(
  model: nn.Module,
  *,
  mode: str,
  batch: int,
  window: int,
  image_size: int,
  device: torch.device,
  warmup: int,
  runs: int,
) -> dict[str, Any]:
  radar, camera = make_inputs(batch=batch, window=window, image_size=image_size, device=device)
  radar_on = mode in ("both", "radar_only")
  camera_on = mode in ("both", "camera_only")
  radar_present = torch.full((batch,), radar_on, dtype=torch.bool, device=device)
  camera_present = torch.full((batch,), camera_on, dtype=torch.bool, device=device)

  with FlopCounter(model) as counter:
    with torch.no_grad():
      model(radar, camera, radar_present=radar_present, camera_present=camera_present)
  compute = counter.summary()

  latency = measure_latency(
    model,
    radar,
    camera,
    radar_present,
    camera_present,
    warmup=warmup,
    runs=runs,
    device=device,
  )

  mean_s = latency["latency_ms_mean"] / 1000.0
  achieved_mops = compute["total_mops"] / max(mean_s, 1e-12)
  achieved_gflops = compute["total_gflops"] / max(mean_s, 1e-12)
  conv_mops_s = compute["conv2d_mops"] / max(mean_s, 1e-12)
  specs = compute.get("operation_specs", {})
  temporal_mops = float(specs.get("temporal_transformer", {}).get("mops", 0.0))
  temporal_mops_s = temporal_mops / max(mean_s, 1e-12)

  return {
    "mode": mode,
    "input": {
      "batch": batch,
      "window": window,
      "radar_shape": list(radar.shape),
      "camera_shape": list(camera.shape),
      "radar_present": radar_on,
      "camera_present": camera_on,
    },
    "compute": compute,
    "latency": latency,
    "performance": {
      "achieved_mops_per_s": float(achieved_mops),
      "achieved_gflops_per_s": float(achieved_gflops),
      "conv2d_mops_per_s": float(conv_mops_s),
      "temporal_transformer_mops_per_s": float(temporal_mops_s),
      "note": "achieved_* = theoretical ops / measured mean latency (single forward)",
    },
  }


def profile_live_mode(
  model: nn.Module,
  *,
  batch: int,
  window: int,
  image_size: int,
  device: torch.device,
  warmup: int,
  runs: int,
  num_rx: int,
  radar_profile: str,
  frame_rate_hz: float,
  camera_device: int,
  camera_width: int,
  camera_height: int,
  camera_fps: float,
  radar1_uuid: str | None,
  radar2_uuid: str | None,
  mirror_radar2: bool,
  min_range_m: float,
  max_range_m: float | None,
) -> dict[str, Any]:
  if batch != 1:
    raise ValueError("live mode currently supports --batch-size 1 only")

  radar_buffer: deque[torch.Tensor] = deque(maxlen=window)
  camera_buffer: deque[torch.Tensor] = deque(maxlen=window)
  radar_present = torch.ones((1,), dtype=torch.bool, device=device)
  camera_present = torch.ones((1,), dtype=torch.bool, device=device)

  model.eval()
  camera_stream = CameraStream(camera_device, camera_width, camera_height, camera_fps).start()
  latest_meta: dict[str, Any] = {}
  stage_times: dict[str, list[float]] = defaultdict(list)
  compute: dict[str, Any] | None = None

  try:
    with DualRadarSession(
      num_rx=num_rx,
      profile=radar_profile,
      frame_rate_hz=frame_rate_hz,
      radar1_uuid=radar1_uuid,
      radar2_uuid=radar2_uuid,
      mirror_radar2=mirror_radar2,
      min_range_m=min_range_m,
      max_range_m=max_range_m,
    ) as radar_session:
      if not radar_session.slots[0].available and not radar_session.slots[1].available:
        raise RuntimeError("No live radar device available for --live benchmark")

      warmup_remaining = int(warmup)
      measured = 0

      while measured < runs:
        t_loop0 = time.perf_counter()

        t0 = time.perf_counter()
        radar1, radar2 = radar_session.read_tensors()
        stage_times["radar_capture_ms"].append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        fused_radar, fuse_meta = fuse_radar_streams_for_model(
          radar1,
          radar2,
          mode="mean",
          mirror_radar2=mirror_radar2,
        )
        stage_times["radar_fuse_ms"].append((time.perf_counter() - t0) * 1000.0)
        latest_meta = dict(fuse_meta)
        latest_meta["status_text"] = radar_session.status_text

        if fused_radar is None:
          continue

        t0 = time.perf_counter()
        camera_rgb = camera_stream.get_latest()
        if camera_rgb is None:
          continue
        camera_tensor = preprocess_camera_frame(camera_rgb, image_size).cpu()
        stage_times["camera_fetch_preprocess_ms"].append((time.perf_counter() - t0) * 1000.0)

        radar_buffer.append(fused_radar.cpu())
        camera_buffer.append(camera_tensor)
        if len(radar_buffer) < window or len(camera_buffer) < window:
          continue

        radar_tensor = torch.stack(list(radar_buffer), dim=0).unsqueeze(0).to(device)
        camera_tensor_batched = torch.stack(list(camera_buffer), dim=0).unsqueeze(0).to(device)

        if compute is None:
          with FlopCounter(model) as counter:
            with torch.no_grad():
              model(
                radar_tensor,
                camera_tensor_batched,
                radar_present=radar_present,
                camera_present=camera_present,
              )
          compute = counter.summary()

        if warmup_remaining > 0:
          with torch.no_grad():
            model(
              radar_tensor,
              camera_tensor_batched,
              radar_present=radar_present,
              camera_present=camera_present,
            )
          _sync(device)
          warmup_remaining -= 1
          continue

        t0 = time.perf_counter()
        with torch.no_grad():
          model(
            radar_tensor,
            camera_tensor_batched,
            radar_present=radar_present,
            camera_present=camera_present,
          )
        _sync(device)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        stage_times["inference_ms"].append(inference_ms)
        stage_times["total_loop_ms"].append((time.perf_counter() - t_loop0) * 1000.0)
        measured += 1
  finally:
    camera_stream.stop()

  if compute is None:
    raise RuntimeError("Live benchmark did not gather enough frames to build a full input window")

  latency = {
    "runs": float(runs),
    "warmup": float(warmup),
    "latency_ms_mean": _summarize_timing_series(stage_times["inference_ms"])["mean_ms"],
    "latency_ms_std": _summarize_timing_series(stage_times["inference_ms"])["std_ms"],
    "latency_ms_min": _summarize_timing_series(stage_times["inference_ms"])["min_ms"],
    "latency_ms_max": _summarize_timing_series(stage_times["inference_ms"])["max_ms"],
    "latency_ms_p50": _summarize_timing_series(stage_times["inference_ms"])["p50_ms"],
    "latency_ms_p95": _summarize_timing_series(stage_times["inference_ms"])["p95_ms"],
    "latency_ms_p99": _summarize_timing_series(stage_times["inference_ms"])["p99_ms"],
    "throughput_fps": float(1000.0 / max(_summarize_timing_series(stage_times["inference_ms"])["mean_ms"], 1e-9)),
  }
  mean_s = latency["latency_ms_mean"] / 1000.0
  specs = compute.get("operation_specs", {})
  temporal_mops = float(specs.get("temporal_transformer", {}).get("mops", 0.0))

  return {
    "mode": "live_both",
    "input": {
      "batch": batch,
      "window": window,
      "radar_shape": [1, window, 3, 32, 32],
      "camera_shape": [1, window, 3, image_size, image_size],
      "radar_present": True,
      "camera_present": True,
    },
    "live_capture": {
      "camera_device": camera_device,
      "camera_width": camera_width,
      "camera_height": camera_height,
      "camera_fps": camera_fps,
      "num_rx": num_rx,
      "radar_profile": radar_profile,
      "frame_rate_hz": frame_rate_hz,
      "radar1_uuid": radar1_uuid,
      "radar2_uuid": radar2_uuid,
      "mirror_radar2": mirror_radar2,
      "fusion_meta_last": latest_meta,
      "timing": {
        key: _summarize_timing_series(values) for key, values in stage_times.items() if values
      },
      "notes": [
        "total_loop_ms includes live sensor polling, radar fusion, camera fetch/preprocess, and model inference.",
        "latency.* remains model-forward latency only so old KPI thresholds stay comparable.",
      ],
    },
    "compute": compute,
    "latency": latency,
    "performance": {
      "achieved_mops_per_s": float(compute["total_mops"] / max(mean_s, 1e-12)),
      "achieved_gflops_per_s": float(compute["total_gflops"] / max(mean_s, 1e-12)),
      "conv2d_mops_per_s": float(compute["conv2d_mops"] / max(mean_s, 1e-12)),
      "temporal_transformer_mops_per_s": float(temporal_mops / max(mean_s, 1e-12)),
      "note": "achieved_* = theoretical ops / measured mean inference latency on live data windows",
    },
  }


def run_benchmark(args) -> dict[str, Any]:
  device = torch.device(args.device)
  model, labels, config = load_checkpoint(args.checkpoint, str(device))
  model.eval()

  params = count_parameters(model)
  params_by_block = count_parameters_by_block(model)
  weight_mem = weight_memory_bytes(model)
  buffer_mem = estimate_activation_buffer_mb(
    batch=args.batch_size,
    window=args.window,
    image_size=int(config["image_size"]),
    n_cameras=args.n_cameras,
    n_radars=args.n_radars,
  )
  platform_info = collect_platform_info(device)

  if args.live:
    profiles = [
      profile_live_mode(
        model,
        batch=args.batch_size,
        window=args.window,
        image_size=int(config["image_size"]),
        device=device,
        warmup=args.warmup,
        runs=args.runs,
        num_rx=args.num_rx,
        radar_profile=args.radar_profile,
        frame_rate_hz=args.frame_rate,
        camera_device=args.camera_device,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        radar1_uuid=args.radar1_uuid,
        radar2_uuid=args.radar2_uuid,
        mirror_radar2=not args.no_mirror_radar2,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
      )
    ]
  else:
    modes = ["both", "radar_only", "camera_only"] if args.all_modes else [args.mode]
    profiles = [
      profile_mode(
        model,
        mode=mode,
        batch=args.batch_size,
        window=args.window,
        image_size=int(config["image_size"]),
        device=device,
        warmup=args.warmup,
        runs=args.runs,
      )
      for mode in modes
    ]

  # Primary KPI profile: multimodal both (cam1 + dual-radar fused as radar stream)
  primary = next((p for p in profiles if p["mode"] in ("both", "live_both")), profiles[0])
  kpi = build_kpi_report(
    profile=primary,
    params=params,
    weight_mem=weight_mem,
    buffer_mem=buffer_mem,
  )
  primary_specs = primary.get("compute", {}).get("operation_specs", {})

  report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "benchmark_source": "live" if args.live else "synthetic",
    "checkpoint": str(args.checkpoint),
    "device": str(device),
    "platform": platform_info,
    "deployment": {
      "profile": "cam1_radar2",
      "n_cameras": int(args.n_cameras),
      "n_radars": int(args.n_radars),
      "note": "Model sees 1 camera stream + 1 fused dual-radar stream (early fuse).",
    },
    "labels": labels,
    "config": config,
    "parameters": params,
    "parameters_millions": round(params["total"] / 1e6, 4),
    "parameters_by_block": params_by_block,
    "weight_memory": weight_mem,
    "buffer_memory_estimate": buffer_mem,
    "key_meanings": KEY_MEANINGS,
    "operation_specs": primary_specs,
    "kpi": kpi,
    "notes": [
      "FLOPs = 2 * MACs (mul-add counted as 2).",
      "MOPS / MMACs = MACs / 1e6.",
      "MFLOPs = FLOPs / 1e6.",
      "achieved_mops_per_s = theoretical MACs / measured mean latency.",
      "AI-DISCO KPI targets for Jetson Nano–class cam1+radar2 deployment.",
      "operation_specs.conv = radar+camera ConvFrameEncoder Conv2d cost.",
      "operation_specs.temporal_transformer = radar/camera TemporalEncoder "
      "(TransformerEncoderLayer self-attn + FFN).",
      "Linear layers nested under MultiheadAttention are not double-counted.",
      "Current forward always runs radar+camera encoders; modality flags zero tokens after encode "
      "(so radar_only/camera_only theoretical MACs match both until early-skip is added).",
    ],
    "profiles": profiles,
  }

  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

  return report


def _print_summary(report: dict[str, Any]):
  print(f"checkpoint: {report['checkpoint']}")
  print(f"device:     {report['device']}")
  plat = report.get("platform", {})
  print(
    f"platform:   {plat.get('hostname')}  os={plat.get('os')}  "
    f"python={plat.get('python')}  torch={plat.get('torch')}"
  )
  if plat.get("gpu_name"):
    print(f"gpu:        {plat['gpu_name']}  ({plat.get('gpu_total_memory_mb')} MB)")
  dep = report.get("deployment", {})
  print(f"deploy:     {dep.get('profile')}  cams={dep.get('n_cameras')}  radars={dep.get('n_radars')}")
  print(f"params:     {report['parameters']['total']:,} ({report['parameters_millions']} M)")
  wm = report.get("weight_memory", {})
  bm = report.get("buffer_memory_estimate", {})
  print(f"weights:    {wm.get('kb', 0):.1f} kB  ({wm.get('mb', 0):.3f} MB)")
  print(f"buffer~:    {bm.get('mb', 0):.3f} MB  (input window estimate)")

  kpi = report.get("kpi")
  if kpi:
    m = kpi["measured"]
    t = kpi["targets"]
    p = kpi["pass"]
    print("\n=== AI-DISCO KPI (cam1+radar2 -> Jetson Nano class) ===")
    print(
      f"  latency mean: {m['latency_ms_mean']:.2f} ms  "
      f"(< {t['latency_ms_mean_max']})  "
      f"{'PASS' if p['latency_mean_lt_100ms'] else 'FAIL'}"
    )
    print(
      f"  latency p95:  {m['latency_ms_p95']:.2f} ms  "
      f"(< {t['latency_ms_p95_max']})  "
      f"{'PASS' if p['latency_p95_lt_150ms'] else 'FAIL'}"
    )
    print(
      f"  throughput:   {m['throughput_fps']:.2f} FPS  "
      f"(>= {t['throughput_fps_min']})  "
      f"{'PASS' if p['throughput_ge_8fps'] else 'FAIL'}"
    )
    print(
      f"  GFLOPs/inf:   {m['gflops_per_inference']:.3f}  "
      f"(< {t['gflops_per_inference_max']})  "
      f"{'PASS' if p['gflops_per_inf_lt_15'] else 'FAIL'}"
    )
    print(
      f"  MMACs/inf:    {m['mmacs_per_inference']:.1f}  "
      f"(< {t['mmacs_per_inference_max']})  "
      f"{'PASS' if p['mmacs_per_inf_lt_8000'] else 'FAIL'}"
    )
    print(
      f"  weights:      {m['weight_memory_mb']:.3f} MB / {int(m['parameters']):,} params  "
      f"(< {t['weight_memory_mb_max']} MB, < {int(t['params_max']):,})  "
      f"{'PASS' if p['weight_memory_lt_32mb'] and p['params_lt_5m'] else 'FAIL'}"
    )
    print(
      f"  buffer~:      {m['buffer_memory_mb']:.3f} MB  "
      f"(< {t['buffer_memory_mb_max']})  "
      f"{'PASS' if p['buffer_memory_lt_64mb'] else 'FAIL'}"
    )
    print(
      f"  temporal:     {m['temporal_transformer_mmacs']:.2f} MMACs  "
      f"(< {t['temporal_mmacs_max']})  "
      f"{'PASS' if p['temporal_mmacs_lt_50'] else 'FAIL'}"
    )
    print(
      f"  cross-attn:   {m['cross_attention_mmacs']:.2f} MMACs  "
      f"(< {t['cross_attn_mmacs_max']})  "
      f"{'PASS' if p['cross_attn_mmacs_lt_100'] else 'FAIL'}"
    )
    print(
      f"  conv share:   {100.0 * m['conv_share']:.1f}%  "
      f"({m['conv_mmacs']:.1f} MMACs)  [info]"
    )
    print(f"  score:        {kpi['pass_count']}/{kpi['check_count']} checks passed")
    print(f"  note:         {kpi['interpretation']['jetson_note']}")
  params_by_block = report.get("parameters_by_block") or {}
  if params_by_block:
    print("\n=== Parameters by block ===")
    for block in (
      "conv_radar",
      "conv_camera",
      "temporal_radar",
      "temporal_camera",
      "cross_attention",
      "head",
      "other",
    ):
      if block not in params_by_block:
        continue
      info = params_by_block[block]
      print(
        f"  {block:18s}  params={int(info['parameters']):>9,}  "
        f"weights={info['weight_kb']:.1f} kB"
      )

  specs = report.get("operation_specs") or {}
  if specs:
    print("\n=== Operation specs (mode=both) ===")
    for key, title in (
      ("conv", "CONV (frame CNN)"),
      ("temporal_transformer", "TEMPORAL TRANSFORMER"),
      ("cross_attention", "CROSS-ATTENTION"),
      ("head", "HEAD"),
    ):
      block = specs.get(key)
      if not block:
        continue
      print(
        f"  [{title}]  MMACs={block['mmacs']:.3f}  MFLOPs={block['mflops']:.3f}  "
        f"share={100.0 * block.get('share_of_total', 0.0):.2f}%  ops={block['ops']}"
      )
      print(f"    macs_by_op: {block.get('macs_by_op', {})}")
      if key == "conv":
        for stream, sub in (block.get("by_stream") or {}).items():
          print(
            f"    {stream}: MMACs={sub['mmacs']:.3f}  MFLOPs={sub['mflops']:.3f}  "
            f"layers={sub['ops']}"
          )
        for layer in block.get("layers") or []:
          print(
            f"      {layer['name']}: out={layer.get('out_shape')}  "
            f"k={layer.get('kernel')}  MOPS={layer['macs']/1e6:.3f}"
          )
      if key == "temporal_transformer":
        print(
          f"    attn_MMACs={block.get('attn_macs', 0)/1e6:.3f}  "
          f"ffn_MMACs={block.get('ffn_macs', 0)/1e6:.3f}"
        )
        for stream, sub in (block.get("by_stream") or {}).items():
          print(
            f"    {stream}: MMACs={sub['mmacs']:.3f}  MFLOPs={sub['mflops']:.3f}  "
            f"ops={sub['ops']}  by_op={sub.get('macs_by_op', {})}"
          )
        for layer in block.get("layers") or []:
          print(
            f"      {layer['name']}: op={layer['op']}  in={layer.get('in_shape')}  "
            f"MOPS={layer['macs']/1e6:.3f}"
          )

  for profile in report["profiles"]:
    c = profile["compute"]
    lat = profile["latency"]
    perf = profile["performance"]
    print(f"\n=== mode={profile['mode']} ===")
    print(
      f"  MACs={c['total_macs']:,}  MOPS={c['total_mops']:.3f}  "
      f"FLOPs={c['total_flops']:,}  GFLOPs={c['total_gflops']:.4f}"
    )
    print(
      f"  conv2d: MACs={c['conv2d_macs']:,}  MOPS={c['conv2d_mops']:.3f}  "
      f"FLOPs={c['conv2d_flops']:,}"
    )
    specs_m = c.get("operation_specs") or {}
    if specs_m:
      conv_b = specs_m.get("conv", {})
      temp_b = specs_m.get("temporal_transformer", {})
      print(
        f"  conv_block: MMACs={conv_b.get('mmacs', 0):.3f}  "
        f"share={100.0 * conv_b.get('share_of_total', 0.0):.2f}%"
      )
      print(
        f"  temporal_transformer: MMACs={temp_b.get('mmacs', 0):.3f}  "
        f"attn={temp_b.get('attn_macs', 0)/1e6:.3f}  "
        f"ffn={temp_b.get('ffn_macs', 0)/1e6:.3f}  "
        f"share={100.0 * temp_b.get('share_of_total', 0.0):.2f}%"
      )
    print(f"  macs_by_op: {c['macs_by_op']}")
    print(
      f"  latency_ms: mean={lat['latency_ms_mean']:.3f}  "
      f"p50={lat['latency_ms_p50']:.3f}  p95={lat['latency_ms_p95']:.3f}  "
      f"fps={lat['throughput_fps']:.2f}"
    )
    print(
      f"  achieved: {perf['achieved_mops_per_s']:.1f} MOPS/s  "
      f"{perf['achieved_gflops_per_s']:.3f} GFLOP/s  "
      f"conv={perf['conv2d_mops_per_s']:.1f} MOPS/s  "
      f"temporal={perf.get('temporal_transformer_mops_per_s', 0.0):.1f} MOPS/s"
    )


def parse_args():
  parser = argparse.ArgumentParser(
    description="Benchmark multimodal model latency + FLOPs/MOPS + AI-DISCO KPIs (JetsonCA)",
    epilog=CLI_EXAMPLES,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_multimodal_crossattention.pt"))
  parser.add_argument("--device", type=str, default=default_device())
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--window", type=int, default=30, help="Temporal window length (frames)")
  parser.add_argument("--warmup", type=int, default=5 if is_jetson() else 10)
  parser.add_argument("--runs", type=int, default=30 if is_jetson() else 50)
  parser.add_argument("--mode", choices=("both", "radar_only", "camera_only"), default="both")
  parser.add_argument("--all-modes", action="store_true", help="Benchmark both + radar_only + camera_only")
  parser.add_argument("--live", action="store_true", help="Use actual connected camera/radar devices instead of synthetic tensors")
  parser.add_argument("--n-cameras", type=int, default=1, help="Deployment camera count (KPI buffer est.)")
  parser.add_argument("--n-radars", type=int, default=2, help="Deployment radar count (KPI buffer est.)")
  parser.add_argument("--camera-device", type=int, default=0, help="Live mode: camera index")
  parser.add_argument("--camera-width", type=int, default=640, help="Live mode: requested camera width")
  parser.add_argument("--camera-height", type=int, default=480, help="Live mode: requested camera height")
  parser.add_argument("--camera-fps", type=float, default=15.0, help="Live mode: requested camera FPS")
  parser.add_argument("--num-rx", type=int, default=3, help="Live mode: radar receiver channels")
  parser.add_argument("--radar-profile", type=str, default="safe", help="Live mode: radar configuration profile")
  parser.add_argument("--frame-rate", type=float, default=5.0, help="Live mode: radar frame rate")
  parser.add_argument("--radar1-uuid", type=str, default=None, help="Live mode: optional primary radar UUID")
  parser.add_argument("--radar2-uuid", type=str, default=None, help="Live mode: optional secondary radar UUID or __none__")
  parser.add_argument("--no-mirror-radar2", action="store_true", help="Live mode: do not mirror radar1 when radar2 is missing")
  parser.add_argument("--min-range-m", type=float, default=0.0, help="Live mode: minimum radar range gate in meters")
  parser.add_argument("--max-range-m", type=float, default=None, help="Live mode: maximum radar range gate in meters")
  default_out = (
    Path("artifacts/benchmark_cam1_radar2_kpi_jetson.json")
    if is_jetson()
    else Path("artifacts/benchmark_cam1_radar2_kpi.json")
  )
  parser.add_argument("--out", type=Path, default=default_out)
  return parser.parse_args()


def main():
  args = parse_args()
  tweaks = apply_jetson_runtime_tweaks()
  print(f"runtime: {tweaks}")
  report = run_benchmark(args)
  _print_summary(report)
  if args.out is not None:
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
  main()
