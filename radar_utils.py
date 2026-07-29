"""Shared radar display, dual-device helpers, label hierarchy, and temporal diagnostics."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from label_hierarchy import format_hierarchy, hierarchy_dict, label_hierarchy
from preprocessing import do_inference_processing
from radar_bgt import configure_bgt_device
from range_doppler import DopplerAlgo
from range_gating import profile_metrics

try:
  from ifxradarsdk.common.exceptions import ErrorFrameAcquisitionFailed
  from ifxradarsdk.fmcw import DeviceFmcw
except ImportError:  # pragma: no cover - allows import without hardware SDK
  ErrorFrameAcquisitionFailed = Exception  # type: ignore[misc, assignment]
  DeviceFmcw = None  # type: ignore[misc, assignment]

FUSE_MODES = ("max", "mean", "sum")
RX_MODES = ("fuse", "rx0", "rx1", "rx2")
CROSS_SENSOR_MODES = ("side_by_side", "max", "radar1", "radar2")

def list_radar_uuids() -> list[str]:
  if DeviceFmcw is None:
    return []
  try:
    return list(DeviceFmcw.get_list())
  except Exception:
    return []


def fuse_dual_radar_tensors(
  radar1: np.ndarray | torch.Tensor,
  radar2: np.ndarray | torch.Tensor | None,
  mode: str = "mean",
) -> np.ndarray | torch.Tensor:
  """Fuse two radar clip/frame tensors for model input. Keeps shape (..., 3, H, W)."""
  if radar2 is None:
    return radar1
  if isinstance(radar1, torch.Tensor):
    if mode == "max":
      return torch.maximum(radar1, radar2)
    return 0.5 * (radar1 + radar2)
  radar1_np = np.asarray(radar1, dtype=np.float32)
  radar2_np = np.asarray(radar2, dtype=np.float32)
  if mode == "max":
    return np.maximum(radar1_np, radar2_np)
  return 0.5 * (radar1_np + radar2_np)


def fuse_radar_streams_for_model(
  radar1: torch.Tensor | None,
  radar2: torch.Tensor | None,
  *,
  mode: str = "mean",
  mirror_radar2: bool = False,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
  """
  Build model input from 0–2 live radar streams. Never silently treats loss as OK:
  metadata reports which sensors contributed.
  """
  meta: dict[str, Any] = {
    "radar1_live": radar1 is not None,
    "radar2_live": radar2 is not None,
    "radar2_mirrored": False,
    "fusion": "none",
    "degraded": False,
  }
  r1, r2 = radar1, radar2
  if r2 is None and r1 is not None and mirror_radar2:
    r2 = r1.clone() if isinstance(r1, torch.Tensor) else torch.from_numpy(np.asarray(r1).copy())
    meta["radar2_mirrored"] = True
    meta["radar2_live"] = True

  if r1 is not None and r2 is not None and not meta["radar2_mirrored"]:
    meta["fusion"] = f"dual_{mode}"
    return fuse_dual_radar_tensors(r1, r2, mode=mode), meta
  if r1 is not None and r2 is not None and meta["radar2_mirrored"]:
    meta["fusion"] = f"dual_{mode}_mirrored"
    return fuse_dual_radar_tensors(r1, r2, mode=mode), meta
  if r1 is not None:
    meta["fusion"] = "radar1_only"
    meta["degraded"] = True
    return r1, meta
  if r2 is not None:
    meta["fusion"] = "radar2_only"
    meta["degraded"] = True
    return r2, meta
  return None, meta


def format_sensor_reliability(
  *,
  radar1_live: bool,
  radar2_live: bool,
  radar2_mirrored: bool,
  radar_hw1: bool,
  radar_hw2: bool,
  fusion: str,
  camera_live: bool,
  camera2_live: bool,
  camera_inference: int,
  camera_monitor: int | None,
  camera2_in_model: bool = False,
  dual_camera_fusion: str = "none",
) -> str:
  def _radar_slot(live: bool, hw: bool, mirrored: bool, name: str) -> str:
    if mirrored:
      return f"{name}=mirrored"
    if live and hw:
      return f"{name}=OK"
    if hw and not live:
      return f"{name}=DROPPED"
    return f"{name}=missing"

  parts = [
    _radar_slot(radar1_live, radar_hw1, False, "R1"),
    _radar_slot(radar2_live, radar_hw2, radar2_mirrored, "R2"),
    f"fuse={fusion}",
    f"cam[{camera_inference}]={'OK' if camera_live else 'no frame'}→model",
  ]
  if camera_monitor is not None and camera_monitor >= 0:
    tag = "→model" if camera2_in_model else "view-only"
    parts.append(f"cam2[{camera_monitor}]={'OK' if camera2_live else 'no frame'}({tag})")
  if dual_camera_fusion not in ("none", "", "cam1_only"):
    parts.append(f"cam_fuse={dual_camera_fusion}")
  elif dual_camera_fusion == "cam1_only":
    parts.append("cam_fuse=cam1")
  degraded = (not radar1_live or not radar2_live) and fusion not in ("none",)
  if degraded and not radar2_mirrored:
    parts.append("DEGRADED")
  return " | ".join(parts)
def fuse_channels(radar_tensor: np.ndarray, fuse_mode: str = "max", rx_mode: str = "fuse") -> np.ndarray:
  tensor = np.asarray(radar_tensor, dtype=np.float32)
  if rx_mode.startswith("rx"):
    channel = int(rx_mode[2:])
    channel = int(np.clip(channel, 0, tensor.shape[0] - 1))
    return tensor[channel]

  if fuse_mode == "mean":
    return tensor.mean(axis=0)
  if fuse_mode == "sum":
    return tensor.sum(axis=0)
  return tensor.max(axis=0)


def fused_to_rgb(fused_map: np.ndarray, *, log_scale: bool = True) -> np.ndarray:
  """Viridis colormap with robust contrast so single peaks don't crush the map."""
  channel = np.asarray(fused_map, dtype=np.float32)
  if log_scale:
    channel = np.log1p(np.maximum(channel, 0.0))

  finite = channel[np.isfinite(channel)]
  if finite.size == 0:
    return np.zeros((*channel.shape, 3), dtype=np.uint8)

  lo = float(np.percentile(finite, 5.0))
  hi = float(np.percentile(finite, 99.5))
  if hi <= lo + 1e-6:
    lo = float(finite.min())
    hi = max(float(finite.max()), lo + 1e-6)

  channel = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
  channel = np.uint8(channel * 255.0)
  colored = cv2.applyColorMap(channel, cv2.COLORMAP_VIRIDIS)
  return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def render_radar_panel(
  radar_tensor: np.ndarray,
  fuse_mode: str = "max",
  rx_mode: str = "fuse",
  overlay_text: str | None = None,
  sensor_title: str | None = None,
) -> np.ndarray:
  fused = fuse_channels(radar_tensor, fuse_mode=fuse_mode, rx_mode=rx_mode)
  # Drop fully-empty range bins so the gated ROI fills the panel instead of
  # a thin energy strip in a sea of zeros.
  active = np.where(fused.max(axis=1) > 1e-6)[0]
  if active.size >= 2:
    fused = fused[int(active[0]) : int(active[-1]) + 1]
  rgb = fused_to_rgb(fused)
  # Text overlays are drawn later at display resolution in the GUI.
  _ = overlay_text
  _ = sensor_title
  return rgb


def combine_sensor_panels(
  radar1_rgb: np.ndarray,
  radar2_rgb: np.ndarray,
  cross_sensor_mode: str = "side_by_side",
) -> np.ndarray:
  if cross_sensor_mode == "radar1":
    return radar1_rgb.copy()
  if cross_sensor_mode == "radar2":
    return radar2_rgb.copy()

  h = max(radar1_rgb.shape[0], radar2_rgb.shape[0])
  w = max(radar1_rgb.shape[1], radar2_rgb.shape[1])

  def _resize(img: np.ndarray) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
      return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

  left = _resize(radar1_rgb)
  right = _resize(radar2_rgb)
  if cross_sensor_mode == "max":
    return np.maximum(left, right)
  return np.concatenate([left, right], axis=1)


def process_raw_frame(
  raw: np.ndarray,
  algo: DopplerAlgo,
  num_rx: int,
  *,
  radar_profile: str = "safe",
  min_range_m: float = 0.0,
  max_range_m: float | None = None,
) -> torch.Tensor:
  """
  Match training: normalize+resize the full RD map first, then apply the same
  approximate 32x32 range gate used by the dataset. Raw-gating before min-max
  (previous path) changed intensity stats vs training and hurt live accuracy.
  """
  from range_gating import apply_range_gate_tensor

  antenna_maps = [algo.compute_doppler_map(raw[i, :, :], i) for i in range(num_rx)]
  metrics = profile_metrics(radar_profile)
  tensor = do_inference_processing(antenna_maps).squeeze(0).cpu()

  if max_range_m is None:
    return tensor

  gated = apply_range_gate_tensor(
    tensor.numpy(),
    max_range_m=float(max_range_m),
    profile_max_range_m=metrics["max_range_m"],
    min_range_m=float(min_range_m),
  )
  return torch.from_numpy(np.asarray(gated, dtype=np.float32))


@dataclass
class RadarDeviceSlot:
  uuid: str | None
  device: Any = None
  algo: DopplerAlgo | None = None
  available: bool = False
  label: str = "radar"


class DualRadarSession:
  """Open up to two BGT60 devices; optional mirror radar1 into radar2 when only one is present."""

  def __init__(
    self,
    num_rx: int,
    profile: str,
    frame_rate_hz: float,
    radar1_uuid: str | None = None,
    radar2_uuid: str | None = None,
    mirror_radar2: bool = True,
    min_range_m: float = 0.0,
    max_range_m: float | None = None,
  ):
    self.num_rx = num_rx
    self.profile = profile
    self.frame_rate_hz = frame_rate_hz
    self.mirror_radar2 = mirror_radar2
    self.min_range_m = min_range_m
    self.metrics = profile_metrics(profile)
    self.max_range_m = float(max_range_m if max_range_m is not None else self.metrics["max_range_m"])
    self.uuids = list_radar_uuids()

    primary_uuid = radar1_uuid
    if primary_uuid is None and self.uuids:
      primary_uuid = self.uuids[0]

    secondary_uuid = radar2_uuid
    if radar2_uuid == "__none__":
      secondary_uuid = None
    elif secondary_uuid is None and len(self.uuids) > 1:
      secondary_uuid = self.uuids[1]

    self.slots: list[RadarDeviceSlot] = [
      RadarDeviceSlot(uuid=primary_uuid, label="radar1"),
      RadarDeviceSlot(uuid=secondary_uuid, label="radar2"),
    ]
    self._devices: list[Any] = []
    self._miss_streak: list[int] = [0, 0]

  def __enter__(self) -> DualRadarSession:
    if DeviceFmcw is None:
      return self

    open_plan: list[tuple[RadarDeviceSlot, str | None]] = [
      (self.slots[0], self.slots[0].uuid),
      (self.slots[1], self.slots[1].uuid),
    ]
    if self.slots[0].uuid is None:
      open_plan[0] = (self.slots[0], "__default__")

    for slot, uuid in open_plan:
      if uuid is None:
        continue
      try:
        device = DeviceFmcw() if uuid == "__default__" else DeviceFmcw(uuid=uuid)
        cfg = configure_bgt_device(
          device,
          self.num_rx,
          profile=self.profile,
          frame_rate_hz=self.frame_rate_hz,
        )
        slot.device = device
        slot.algo = DopplerAlgo(cfg, self.num_rx)
        slot.available = True
        self._devices.append(device)
      except Exception:
        slot.available = False

    if not self.slots[0].available and self.slots[1].available:
      self.slots[0], self.slots[1] = self.slots[1], self.slots[0]
      self.slots[0].label = "radar1"
      self.slots[1].label = "radar2"

    return self

  def __exit__(self, exc_type, exc, tb):
    for device in self._devices:
      try:
        device.close()
      except Exception:
        pass
    self._devices.clear()

  @property
  def status_text(self) -> str:
    r1 = "ok" if self.slots[0].available else "missing"
    if self.slots[1].available:
      r2 = "ok"
    elif self.mirror_radar2 and self.slots[0].available:
      r2 = "mirror"
    else:
      r2 = "missing"
    live1 = "live" if self._miss_streak[0] == 0 and self.slots[0].available else "stale"
    live2 = (
      "live"
      if self._miss_streak[1] == 0 and self.slots[1].available
      else ("n/a" if not self.slots[1].available else "stale")
    )
    return f"radar1={r1}/{live1} radar2={r2}/{live2}"

  def set_mirror_radar2(self, enabled: bool):
    self.mirror_radar2 = bool(enabled)

  def set_range_limits(self, min_range_m: float, max_range_m: float):
    self.min_range_m = float(min_range_m)
    self.max_range_m = float(max_range_m)

  def read_tensors(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    tensors: list[torch.Tensor | None] = [None, None]
    for idx, slot in enumerate(self.slots):
      if not slot.available or slot.device is None or slot.algo is None:
        self._miss_streak[idx] += 1
        continue
      try:
        raw = slot.device.get_next_frame()[0]
        tensors[idx] = process_raw_frame(
          raw,
          slot.algo,
          self.num_rx,
          radar_profile=self.profile,
          min_range_m=self.min_range_m,
          max_range_m=self.max_range_m,
        )
        self._miss_streak[idx] = 0
      except ErrorFrameAcquisitionFailed:
        self._miss_streak[idx] += 1
        continue

    radar1 = tensors[0]
    radar2 = tensors[1]
    return radar1, radar2


@dataclass
class TemporalFrameRecord:
  frame_index: int
  rgb: np.ndarray
  prediction: str = ""
  confidence: float = 0.0
  hierarchy_text: str = ""


class TemporalDiagnosticLogger:
  def __init__(self, output_path: Path | None = None):
    self.output_path = output_path
    self.events: list[dict[str, Any]] = []
    self._pending: dict[str, Any] | None = None

  def on_prediction(
    self,
    frame_index: int,
    prediction: str,
    confidence: float,
    expected_label: str,
    t_minus_1: TemporalFrameRecord | None,
    t_frame: TemporalFrameRecord,
  ):
    if self._pending is not None:
      self._finalize_pending(t_frame)

    self._pending = {
      "timestamp_s": time.time(),
      "frame_index": frame_index,
      "prediction": prediction,
      "confidence": float(confidence),
      "hierarchy": hierarchy_dict(prediction, confidence),
      "expected_label": expected_label or "",
      "temporal": {
        "t_minus_1": self._frame_payload(t_minus_1),
        "t": self._frame_payload(t_frame),
        "t_plus_1": None,
      },
    }

  def on_frame(self, frame_record: TemporalFrameRecord):
    if self._pending is not None and self._pending["temporal"]["t_plus_1"] is None:
      if frame_record.frame_index > int(self._pending["frame_index"]):
        self._pending["temporal"]["t_plus_1"] = self._frame_payload(frame_record)
        self._finalize_pending(None)

  def flush(self):
    if self._pending is not None:
      self._finalize_pending(None)

  def _frame_payload(self, record: TemporalFrameRecord | None) -> dict[str, Any] | None:
    if record is None:
      return None
    return {
      "frame_index": record.frame_index,
      "prediction": record.prediction,
      "confidence": float(record.confidence),
      "hierarchy_text": record.hierarchy_text,
    }

  def _finalize_pending(self, _next_frame: TemporalFrameRecord | None):
    if self._pending is None:
      return
    self.events.append(self._pending)
    self._pending = None
    if self.output_path is not None:
      self.output_path.parent.mkdir(parents=True, exist_ok=True)
      self.output_path.write_text(json.dumps(self.events, indent=2), encoding="utf-8")


class TemporalStripBuffer:
  def __init__(self, maxlen: int = 5):
    self.frames: deque[TemporalFrameRecord] = deque(maxlen=maxlen)

  def append(self, record: TemporalFrameRecord):
    self.frames.append(record)

  def triple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    items = list(self.frames)
    while len(items) < 3:
      items.insert(0, TemporalFrameRecord(frame_index=-1, rgb=blank))

    # Newest frame is "t". With maxlen>3 we can show farther history so motion is visible.
    t0 = items[-1]
    t_m1 = items[-2] if len(items) >= 2 else TemporalFrameRecord(frame_index=-1, rgb=blank)
    t_m2 = items[-3] if len(items) >= 3 else TemporalFrameRecord(frame_index=-1, rgb=blank)
    return (
      t_m2.rgb,
      t_m1.rgb,
      t0.rgb,
      f"t-2 (#{t_m2.frame_index})",
      f"t-1 (#{t_m1.frame_index})",
      f"t (#{t0.frame_index})",
    )
