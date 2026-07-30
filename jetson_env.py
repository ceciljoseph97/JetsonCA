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


def ensure_conda_lib_path(*, reexec: bool = False) -> str | None:
  """Prefer conda libstdc++ over the older system one (fixes cv2 CXXABI errors).

  Pip opencv on Jetson often needs CXXABI_1.3.15+, while
  /lib/aarch64-linux-gnu/libstdc++.so.6 from JetPack is older.

  Changing LD_LIBRARY_PATH via os.environ is ignored by the already-running
  dynamic linker — pass reexec=True (gui entrypoints) to restart the process.
  """
  import sys

  prefix = os.environ.get("CONDA_PREFIX")
  if not prefix:
    return None
  lib = str(Path(prefix) / "lib")
  if not Path(lib).is_dir():
    return None
  current = os.environ.get("LD_LIBRARY_PATH", "")
  parts = [p for p in current.split(":") if p]
  if lib in parts and parts[0] == lib:
    return lib
  # Put conda lib first.
  parts = [p for p in parts if p != lib]
  os.environ["LD_LIBRARY_PATH"] = lib + ((":" + ":".join(parts)) if parts else "")
  if reexec and os.environ.get("_JETSONCA_LIB_REEXEC") != "1":
    os.environ["_JETSONCA_LIB_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
  return lib


def apply_jetson_runtime_tweaks(*, threads: int = 2) -> dict[str, object]:
  """Reduce host-thread / allocator pressure on Nano-class boards."""
  conda_lib = ensure_conda_lib_path(reexec=False)
  info: dict[str, object] = {
    "is_jetson": is_jetson(),
    "machine": platform.machine(),
    "threads": threads,
    "conda_lib": conda_lib,
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
