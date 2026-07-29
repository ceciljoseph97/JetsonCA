"""Jetson Nano / Tegra runtime helpers."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def is_jetson() -> bool:
  try:
    return Path("/etc/nv_tegra_release").exists()
  except OSError:
    return False


def default_device() -> str:
  try:
    import torch

    if torch.cuda.is_available():
      return "cuda"
  except Exception:
    pass
  return "cpu"


def apply_jetson_runtime_tweaks(*, threads: int = 2) -> dict[str, object]:
  """Reduce host-thread / allocator pressure on Nano-class boards."""
  info: dict[str, object] = {
    "is_jetson": is_jetson(),
    "machine": platform.machine(),
    "threads": threads,
  }
  os.environ.setdefault("OMP_NUM_THREADS", str(threads))
  os.environ.setdefault("MKL_NUM_THREADS", str(threads))
  os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
  try:
    import torch

    torch.set_num_threads(threads)
    if torch.cuda.is_available():
      # Prefer expandable segments when available; ignore on older JetPack torch.
      try:
        torch.cuda.empty_cache()
      except Exception:
        pass
      info["cuda"] = True
      info["device_name"] = torch.cuda.get_device_name(0)
    else:
      info["cuda"] = False
  except Exception as exc:
    info["torch_error"] = str(exc)
  return info
