"""Range gating and target-distance estimation for FMCW range-doppler maps."""

from __future__ import annotations

import numpy as np

from radar_bgt import PROFILES

DEFAULT_PROFILE = "safe"


def profile_metrics(profile: str = DEFAULT_PROFILE) -> dict[str, float]:
  params = PROFILES.get(profile, PROFILES[DEFAULT_PROFILE])
  return {
    "range_resolution_m": float(params["range_resolution_m"]),
    "max_range_m": float(params["max_range_m"]),
    "profile": profile,
  }


def metrics_from_meta(meta: dict | None) -> dict[str, float]:
  if not meta:
    return profile_metrics(DEFAULT_PROFILE)
  profile = str(meta.get("radar_profile", DEFAULT_PROFILE))
  metrics = profile_metrics(profile)
  if "range_resolution_m" in meta:
    metrics["range_resolution_m"] = float(meta["range_resolution_m"])
  if "max_range_m" in meta:
    metrics["max_range_m"] = float(meta["max_range_m"])
  metrics["profile"] = profile
  return metrics


def _range_mask(num_bins: int, range_resolution_m: float, min_range_m: float, max_range_m: float) -> np.ndarray:
  range_axis_m = np.arange(num_bins, dtype=np.float32) * float(range_resolution_m)
  return (range_axis_m >= float(min_range_m)) & (range_axis_m <= float(max_range_m))


def apply_range_gate_raw(
  radar_maps,
  *,
  range_resolution_m: float,
  min_range_m: float = 0.0,
  max_range_m: float,
) -> np.ndarray:
  """Gate raw per-antenna maps shaped (num_ant, range_bins, doppler_bins)."""
  maps = np.asarray(radar_maps)
  if np.iscomplexobj(maps):
    maps = np.abs(maps)
  maps = maps.astype(np.float32, copy=False)
  if maps.ndim != 3:
    raise ValueError(f"Expected raw radar maps (ant, range, doppler), got {maps.shape}")
  mask = _range_mask(maps.shape[1], range_resolution_m, min_range_m, max_range_m)
  gated = maps.copy()
  gated[:, ~mask, :] = 0.0
  return gated


def apply_range_gate_tensor(
  radar_tensor: np.ndarray,
  *,
  max_range_m: float,
  profile_max_range_m: float,
  min_range_m: float = 0.0,
  range_axis: int = 1,
) -> np.ndarray:
  """
  Gate saved/processed tensors shaped (..., H, W).
  Uses a linear range-axis approximation after resize to 32x32.
  """
  tensor = np.asarray(radar_tensor, dtype=np.float32)
  if tensor.ndim < 2:
    return tensor

  if range_axis < 0:
    range_axis = tensor.ndim + range_axis
  num_bins = tensor.shape[range_axis]
  profile_max_range_m = max(float(profile_max_range_m), 1e-6)

  min_bin = int(np.floor((float(min_range_m) / profile_max_range_m) * num_bins))
  max_bin = int(np.ceil((float(max_range_m) / profile_max_range_m) * num_bins))
  min_bin = int(np.clip(min_bin, 0, num_bins))
  max_bin = int(np.clip(max_bin, min_bin + 1, num_bins))

  gated = tensor.copy()
  sl = [slice(None)] * gated.ndim
  sl[range_axis] = slice(0, min_bin)
  gated[tuple(sl)] = 0.0
  sl[range_axis] = slice(max_bin, num_bins)
  gated[tuple(sl)] = 0.0
  return gated


def estimate_peak_range_m(
  radar_tensor: np.ndarray,
  *,
  profile_max_range_m: float,
  range_axis: int = 1,
) -> float:
  """Estimate dominant target range from a processed radar tensor."""
  tensor = np.asarray(radar_tensor, dtype=np.float32)
  if tensor.ndim == 3:
    energy = np.max(tensor, axis=0)
  elif tensor.ndim == 2:
    energy = tensor
  else:
    return 0.0

  if range_axis < 0:
    range_axis = energy.ndim + range_axis
  range_profile = energy.max(axis=-1) if energy.ndim == 2 else energy
  peak_bin = int(np.argmax(range_profile))
  num_bins = range_profile.shape[0]
  return float(peak_bin / max(num_bins - 1, 1) * profile_max_range_m)


def in_recognition_range(
  target_range_m: float,
  *,
  min_range_m: float,
  max_range_m: float,
) -> bool:
  return float(min_range_m) <= float(target_range_m) <= float(max_range_m)
