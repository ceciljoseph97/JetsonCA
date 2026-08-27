#!/usr/bin/env python3
"""Detector + sensor preprocessing in front of the multimodal classifier.

Radar path (existing DSP kept as-is, detector sits on RD):
  raw ADC → MTI + range–Doppler FFT → |RD| → per-ch norm → 32×32
  → range gate → CA-CFAR peak list → gate / peak mask

Camera path:
  RGB stream → (optional) MOG2 / frame-diff BG → motion mask → bbox ROI
  → align / resize / normalize (unchanged model input prep)

This module does **not** retrain the CNN. It provides:
  - detections for gating ("run classifier only if target present")
  - overlays for poster / GUI
  - optional camera crop to motion ROI (inference-only; train mismatch if used blindly)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from radar_utils import fuse_channels
from range_gating import apply_range_gate_tensor, estimate_peak_range_m


@dataclass
class RadarDetection:
  range_bin: int
  doppler_bin: int
  range_m: float
  snr_db: float
  power: float


@dataclass
class CameraDetection:
  bbox_xyxy: tuple[int, int, int, int]  # x0,y0,x1,y1 in pixel coords of input frame
  area_px: int
  motion_score: float
  mask: np.ndarray | None = None


@dataclass
class DetectionResult:
  radar_hits: list[RadarDetection] = field(default_factory=list)
  camera: CameraDetection | None = None
  radar_triggered: bool = False
  camera_triggered: bool = False
  gate_open: bool = False
  rd_cfar_mask: np.ndarray | None = None  # (H, W) bool
  camera_fg_mask: np.ndarray | None = None
  meta: dict[str, Any] = field(default_factory=dict)

  @property
  def best_radar(self) -> RadarDetection | None:
    if not self.radar_hits:
      return None
    return max(self.radar_hits, key=lambda h: h.snr_db)


def preprocess_radar_tensor(
  radar: np.ndarray,
  *,
  min_range_m: float = 0.0,
  max_range_m: float | None = None,
  profile_max_range_m: float = 5.0,
) -> np.ndarray:
  """Apply the same range gate used by dataset/live paths on a stored RD tensor."""
  radar = np.asarray(radar, dtype=np.float32)
  if max_range_m is None:
    return radar
  return apply_range_gate_tensor(
    radar,
    max_range_m=float(max_range_m),
    profile_max_range_m=float(profile_max_range_m),
    min_range_m=float(min_range_m),
  )


def rd_power_map(radar_chw: np.ndarray, *, fuse_mode: str = "max") -> np.ndarray:
  """(C,H,W) or (H,W) → non-negative power map (H,W)."""
  x = np.asarray(radar_chw, dtype=np.float32)
  if x.ndim == 3:
    x = fuse_channels(x, fuse_mode=fuse_mode, rx_mode="fuse")
  x = np.maximum(x, 0.0)
  return x


def ca_cfar_2d(
  power: np.ndarray,
  *,
  guard: int = 1,
  train: int = 3,
  pfa: float = 1e-3,
  min_snr_db: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
  """Cell-averaging CFAR on a 2D RD power map (vectorized)."""
  power = np.asarray(power, dtype=np.float32)
  if power.ndim != 2:
    raise ValueError(f"CFAR expects (H,W), got {power.shape}")

  outer = train + guard
  k_outer = 2 * outer + 1
  k_guard = 2 * guard + 1
  n_train = max(k_outer * k_outer - k_guard * k_guard, 1)
  alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)

  # Box-filter sums via integral / uniform filter
  sum_outer = cv2.blur(power, (k_outer, k_outer), borderType=cv2.BORDER_REFLECT) * (k_outer * k_outer)
  sum_guard = cv2.blur(power, (k_guard, k_guard), borderType=cv2.BORDER_REFLECT) * (k_guard * k_guard)
  noise = np.maximum((sum_outer - sum_guard) / float(n_train), 1e-12)
  snr_db = 10.0 * np.log10(np.maximum(power, 1e-12) / noise)
  mask = (power > (alpha * noise)) & (snr_db >= min_snr_db)
  # invalidate border where window incomplete
  mask[:outer, :] = False
  mask[-outer:, :] = False
  mask[:, :outer] = False
  mask[:, -outer:] = False
  return mask.astype(bool), snr_db.astype(np.float32)


def extract_radar_detections(
  power: np.ndarray,
  mask: np.ndarray,
  snr_db: np.ndarray,
  *,
  profile_max_range_m: float,
  max_hits: int = 5,
) -> list[RadarDetection]:
  """Non-max suppression style: take local peaks inside CFAR mask."""
  h, w = power.shape
  hits: list[RadarDetection] = []
  ys, xs = np.where(mask)
  order = np.argsort(snr_db[ys, xs])[::-1]
  taken = np.zeros_like(mask, dtype=bool)

  for idx in order:
    y = int(ys[idx])
    x = int(xs[idx])
    if taken[y, x]:
      continue
    # suppress neighborhood
    y0, y1 = max(0, y - 1), min(h, y + 2)
    x0, x1 = max(0, x - 1), min(w, x + 2)
    if power[y, x] < power[y0:y1, x0:x1].max() - 1e-8:
      continue
    taken[y0:y1, x0:x1] = True
    range_m = float(y / max(h - 1, 1) * profile_max_range_m)
    hits.append(
      RadarDetection(
        range_bin=y,
        doppler_bin=x,
        range_m=range_m,
        snr_db=float(snr_db[y, x]),
        power=float(power[y, x]),
      )
    )
    if len(hits) >= max_hits:
      break
  return hits


def camera_motion_mask(
  frames_rgb: np.ndarray,
  *,
  method: str = "mog2",
  diff_thresh: int = 25,
  min_area: int = 400,
) -> tuple[np.ndarray, CameraDetection | None]:
  """Build FG mask from a short RGB clip (T,H,W,3) uint8 and a bbox if large enough."""
  frames = np.asarray(frames_rgb)
  if frames.ndim != 4 or frames.shape[0] < 2:
    raise ValueError("camera_motion_mask expects (T,H,W,3) with T>=2")

  h, w = frames.shape[1], frames.shape[2]
  if method == "diff":
    # accumulate abs-diff vs median background
    bg = np.median(frames.astype(np.float32), axis=0)
    acc = np.zeros((h, w), dtype=np.float32)
    for fr in frames:
      d = cv2.absdiff(fr.astype(np.float32), bg)
      gray = cv2.cvtColor(np.clip(d, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
      acc = np.maximum(acc, gray.astype(np.float32))
    _, mask = cv2.threshold(acc.astype(np.uint8), diff_thresh, 255, cv2.THRESH_BINARY)
  else:
    sub = cv2.createBackgroundSubtractorMOG2(history=min(60, int(frames.shape[0]) * 2), varThreshold=16, detectShadows=False)
    mask = np.zeros((h, w), dtype=np.uint8)
    for fr in frames:
      bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
      fg = sub.apply(bgr)
      mask = np.maximum(mask, fg)
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)

  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  best = None
  best_area = 0
  for cnt in contours:
    area = int(cv2.contourArea(cnt))
    if area < min_area or area <= best_area:
      continue
    x, y, bw, bh = cv2.boundingRect(cnt)
    best_area = area
    best = CameraDetection(
      bbox_xyxy=(int(x), int(y), int(x + bw), int(y + bh)),
      area_px=area,
      motion_score=float(area) / float(h * w),
      mask=mask,
    )

  return mask, best


def crop_camera_to_bbox(
  frame_rgb: np.ndarray,
  bbox_xyxy: tuple[int, int, int, int],
  *,
  pad: float = 0.15,
) -> np.ndarray:
  h, w = frame_rgb.shape[:2]
  x0, y0, x1, y1 = bbox_xyxy
  bw, bh = x1 - x0, y1 - y0
  x0 = int(max(0, x0 - pad * bw))
  y0 = int(max(0, y0 - pad * bh))
  x1 = int(min(w, x1 + pad * bw))
  y1 = int(min(h, y1 + pad * bh))
  crop = frame_rgb[y0:y1, x0:x1]
  if crop.size == 0:
    return frame_rgb
  return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)


@dataclass
class DetectorConfig:
  # radar CFAR
  cfar_guard: int = 1
  cfar_train: int = 3
  cfar_pfa: float = 1e-3
  min_snr_db: float = 6.0
  max_radar_hits: int = 5
  require_radar: bool = True
  # camera
  camera_method: str = "mog2"  # mog2 | diff
  min_motion_area: int = 400
  require_camera: bool = False
  # range gate (preprocessing)
  min_range_m: float = 0.0
  max_range_m: float | None = None
  profile_max_range_m: float = 5.0


class MultimodalDetector:
  """Preprocess sensors + CFAR/camera FG detector used as a gate before classification."""

  def __init__(self, config: DetectorConfig | None = None):
    self.cfg = config or DetectorConfig()

  def run(
    self,
    radar_seq: np.ndarray,
    camera_seq: np.ndarray,
    *,
    frame_index: int | None = None,
  ) -> DetectionResult:
    """
    Parameters
    ----------
    radar_seq : (T,C,H,W) gated or ungated RD clip
    camera_seq : (T,H,W,3) uint8 RGB aligned to radar timeline
    """
    cfg = self.cfg
    radar_seq = preprocess_radar_tensor(
      radar_seq,
      min_range_m=cfg.min_range_m,
      max_range_m=cfg.max_range_m,
      profile_max_range_m=cfg.profile_max_range_m,
    )
    t = int(radar_seq.shape[0] // 2 if frame_index is None else frame_index)
    t = int(np.clip(t, 0, radar_seq.shape[0] - 1))
    frame_rd = radar_seq[t]
    power = rd_power_map(frame_rd)
    mask, snr = ca_cfar_2d(
      power,
      guard=cfg.cfar_guard,
      train=cfg.cfar_train,
      pfa=cfg.cfar_pfa,
      min_snr_db=cfg.min_snr_db,
    )
    hits = extract_radar_detections(
      power,
      mask,
      snr,
      profile_max_range_m=cfg.profile_max_range_m,
      max_hits=cfg.max_radar_hits,
    )
    # also keep classic peak for meta
    peak_m = estimate_peak_range_m(frame_rd, profile_max_range_m=cfg.profile_max_range_m)

    cam_mask, cam_det = camera_motion_mask(
      camera_seq,
      method=cfg.camera_method,
      min_area=cfg.min_motion_area,
    )

    radar_ok = len(hits) > 0
    camera_ok = cam_det is not None
    if cfg.require_radar and cfg.require_camera:
      gate = radar_ok and camera_ok
    elif cfg.require_radar:
      gate = radar_ok
    elif cfg.require_camera:
      gate = camera_ok
    else:
      gate = radar_ok or camera_ok

    return DetectionResult(
      radar_hits=hits,
      camera=cam_det,
      radar_triggered=radar_ok,
      camera_triggered=camera_ok,
      gate_open=gate,
      rd_cfar_mask=mask,
      camera_fg_mask=cam_mask,
      meta={
        "frame_index": t,
        "peak_range_m": peak_m,
        "n_cfar_cells": int(mask.sum()),
        "n_radar_hits": len(hits),
      },
    )


def _draw_text_badge(
  frame_rgb: np.ndarray,
  text: str,
  *,
  org: tuple[int, int],
  fg: tuple[int, int, int],
  bg: tuple[int, int, int] = (12, 12, 12),
  font_scale: float = 0.55,
  thickness: int = 1,
  pad: int = 6,
  alpha: float = 0.72,
  anchor: str = "tl",
) -> None:
  """Opaque-ish label badge so status text stays readable over heatmaps."""
  font = cv2.FONT_HERSHEY_SIMPLEX
  (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
  x, y = org
  box_w = tw + 2 * pad
  box_h = th + baseline + 2 * pad
  if anchor == "tr":
    x0 = max(0, x - box_w)
    y0 = max(0, y)
  elif anchor == "bl":
    x0 = max(0, x)
    y0 = max(0, y - box_h)
  elif anchor == "br":
    x0 = max(0, x - box_w)
    y0 = max(0, y - box_h)
  else:
    x0 = max(0, x)
    y0 = max(0, y)
  x1 = min(frame_rgb.shape[1], x0 + box_w)
  y1 = min(frame_rgb.shape[0], y0 + box_h)
  if x1 <= x0 or y1 <= y0:
    return
  overlay = frame_rgb.copy()
  cv2.rectangle(overlay, (x0, y0), (x1, y1), bg, -1)
  cv2.addWeighted(overlay, alpha, frame_rgb, 1.0 - alpha, 0, dst=frame_rgb)
  cv2.rectangle(frame_rgb, (x0, y0), (x1, y1), fg, 1)
  text_x = x0 + pad
  text_y = y0 + pad + th
  cv2.putText(frame_rgb, text, (text_x, text_y), font, font_scale, fg, thickness, cv2.LINE_AA)


def draw_gate_badge(frame_rgb: np.ndarray, *, gate_open: bool, detect_prob: float | None = None) -> np.ndarray:
  """Stamp learned/classical GATE status on camera frame (top-right)."""
  out = np.asarray(frame_rgb)
  status = "GATE OPEN" if gate_open else "GATE CLOSED"
  if detect_prob is not None:
    status = f"{status}  det={detect_prob:.2f}"
  color = (60, 220, 100) if gate_open else (255, 90, 90)
  _draw_text_badge(
    out,
    status,
    org=(out.shape[1] - 8, 8),
    fg=color,
    bg=(10, 10, 10),
    font_scale=0.55,
    thickness=1,
    pad=7,
    alpha=0.78,
    anchor="tr",
  )
  return out


def overlay_camera_detection(frame_rgb: np.ndarray, det: DetectionResult, *, alpha: float = 0.35) -> np.ndarray:
  out = frame_rgb.copy()
  if det.camera_fg_mask is not None:
    heat = cv2.applyColorMap(det.camera_fg_mask, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(heat, alpha, out, 1.0 - alpha, 0)
  if det.camera is not None:
    x0, y0, x1, y1 = det.camera.bbox_xyxy
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 64, 64), 2)
    _draw_text_badge(
      out,
      f"cam ROI area={det.camera.area_px}",
      org=(x0, max(0, y0 - 28)),
      fg=(255, 120, 120),
      font_scale=0.45,
      thickness=1,
      pad=4,
      alpha=0.65,
      anchor="tl",
    )
  draw_gate_badge(out, gate_open=bool(det.gate_open))
  return out


def dilate_mask(mask: np.ndarray, k: int = 1) -> np.ndarray:
  if k <= 0:
    return mask.astype(bool)
  kernel = np.ones((2 * k + 1, 2 * k + 1), dtype=np.uint8)
  return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def apply_cfar_mask_to_radar_seq(
  radar_seq: np.ndarray,
  *,
  profile_max_range_m: float,
  guard: int = 1,
  train: int = 3,
  pfa: float = 1e-3,
  min_snr_db: float = 6.0,
  dilate: int = 1,
  keep_ratio_if_empty: float = 0.35,
) -> np.ndarray:
  """Soft-hard CFAR ROI mask on (T,C,H,W) RD clip for train/infer consistency.

  Cells outside dilated CFAR support are attenuated (not hard-zeroed if no hits,
  to avoid empty tensors on idle clips).
  """
  radar = np.asarray(radar_seq, dtype=np.float32).copy()
  if radar.ndim != 4:
    raise ValueError(f"expected (T,C,H,W), got {radar.shape}")

  out = np.zeros_like(radar)
  for t in range(radar.shape[0]):
    power = rd_power_map(radar[t])
    mask, _snr = ca_cfar_2d(
      power,
      guard=guard,
      train=train,
      pfa=pfa,
      min_snr_db=min_snr_db,
    )
    mask = dilate_mask(mask, dilate)
    if not mask.any():
      # keep a weak full map so background / idle still trains
      gain = np.full(power.shape, keep_ratio_if_empty, dtype=np.float32)
    else:
      gain = np.where(mask, 1.0, 0.05).astype(np.float32)
    out[t] = radar[t] * gain[None, :, :]
  return out


def apply_camera_roi_crop_seq(
  camera_seq: np.ndarray,
  *,
  method: str = "mog2",
  min_area: int = 400,
  pad: float = 0.15,
) -> np.ndarray:
  """Crop RGB clip (T,H,W,3) to motion ROI when available; else identity."""
  frames = np.asarray(camera_seq)
  if frames.ndim != 4 or frames.shape[0] < 2:
    return frames
  _mask, det = camera_motion_mask(frames, method=method, min_area=min_area)
  if det is None:
    return frames
  return np.stack([crop_camera_to_bbox(fr, det.bbox_xyxy, pad=pad) for fr in frames], axis=0)


def format_rd_detection_label(det: DetectionResult) -> str | None:
  """Short CFAR summary for UI labels (not drawn onto the RD map)."""
  best = det.best_radar
  if best is None:
    n = len(det.radar_hits)
    return f"CFAR×{n}" if n else None
  return f"CFAR {best.range_m:.2f}m"


def overlay_rd_detection(rd_rgb: np.ndarray, det: DetectionResult) -> np.ndarray:
  """Keep RD map clean — CFAR is reported via format_rd_detection_label()."""
  _ = det
  return np.asarray(rd_rgb)
