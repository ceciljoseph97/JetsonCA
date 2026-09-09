"""Checkpoint load + camera preprocess without OpenCV (Jetson synthetic bench safe)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from label_hierarchy import is_background_label
from model import MultiModalCrossAttentionNet


WALKING_BG_AUDIO_V1 = Path("artifacts/walking_bg_audio_v1/best_multimodal_crossattention.pt")
WALK_WAVE_SNAP_V1 = Path("artifacts/walk_wave_snap_v1/best_multimodal_crossattention.pt")


def default_checkpoint() -> Path:
  """Prefer latest walk/wave/snap audio ckpt; fall back to walking / Crossattention siblings."""
  here = Path(__file__).resolve().parent
  ca = here.parent / "Crossattention"
  candidates = [
    here / WALK_WAVE_SNAP_V1,
    ca / WALK_WAVE_SNAP_V1,
    Path.cwd() / WALK_WAVE_SNAP_V1,
    here / WALKING_BG_AUDIO_V1,
    ca / "artifacts" / "walking_bg_audio_v1" / "best_multimodal_crossattention.pt",
    Path.cwd() / WALKING_BG_AUDIO_V1,
    here / "artifacts" / "best_multimodal_crossattention_detect.pt",
    here / "artifacts" / "best_multimodal_crossattention.pt",
  ]
  for path in candidates:
    if path.exists():
      return path
  return here / WALK_WAVE_SNAP_V1


def resolve_window_len(config: dict, cli_window: int | None = 30) -> int:
  """Prefer checkpoint seq_len from Crossattention train; else CLI/default."""
  cfg = int(config.get("seq_len") or 0) or 0
  if cfg > 0:
    return cfg
  return int(cli_window or 30) or 30


def audio_off_kwargs(model, device) -> dict:
  """Mask audio as absent when the ckpt has an audio branch but no mic is wired."""
  if not getattr(model, "enable_audio", False):
    return {}
  return {"audio_present": torch.zeros(1, dtype=torch.bool, device=device)}


def _cfg_int(config: dict, key: str, default: int) -> int:
  value = config.get(key, default)
  return default if value is None else int(value)


def load_checkpoint(path: Path | str, device: str):
  try:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
  except TypeError:
    checkpoint = torch.load(path, map_location=device)
  config = dict(checkpoint["config"])
  labels = list(checkpoint.get("activity_labels", checkpoint["labels"]))
  all_labels = list(checkpoint.get("all_labels") or labels)
  if not any(is_background_label(x) for x in all_labels):
    all_labels = ["background", *labels]
  config["all_labels"] = all_labels

  model = MultiModalCrossAttentionNet(
    num_classes=len(labels),
    num_activity_classes=_cfg_int(config, "num_activity_classes", len(labels)),
    num_coarse_classes=_cfg_int(config, "num_coarse_classes", len(labels)),
    num_subaction_classes=_cfg_int(config, "num_subaction_classes", len(labels)),
    model_dim=int(config["model_dim"]),
    num_heads=int(config["num_heads"]),
    num_layers=int(config["num_layers"]),
    dropout=float(config["dropout"]),
    modality_dropout=0.0,
    temporal_mode=str(config.get("temporal_mode", "none")),
    enable_human_head=bool(config.get("enable_human_head", False)),
    enable_detect_head=bool(config.get("enable_detect_head", False)),
    enable_reliability_gates=bool(config.get("enable_reliability_gates", False)),
    enable_audio=bool(config.get("enable_audio", False)),
    audio_n_mels=_cfg_int(config, "audio_n_mels", 32),
    audio_mel_width=_cfg_int(config, "audio_mel_width", 32),
  ).to(device)

  state = dict(checkpoint["model_state"])
  remapped = {}
  for key, value in state.items():
    if key.startswith("classifier."):
      remapped[key.replace("classifier.", "activity_classifier.", 1)] = value
    else:
      remapped[key] = value

  missing, _unexpected = model.load_state_dict(remapped, strict=False)
  # Legacy single-radar ckpts: clone radar1 → radar2 so dual-encoder path is sane.
  if any(k.startswith("radar2_encoder.") for k in missing):
    model.radar2_encoder.load_state_dict(model.radar_encoder.state_dict())
  if any(k.startswith("radar2_temporal.") for k in missing):
    model.radar2_temporal.load_state_dict(model.radar_temporal.state_dict())

  # Only fuse hierarchy when trained coarse/sub heads exist in the checkpoint.
  config["use_hierarchical_fusion"] = any(k.startswith("coarse_classifier.") for k in remapped)
  config["has_detect_head"] = any(k.startswith("detect_classifier.") for k in remapped) or bool(
    config.get("enable_detect_head", False)
  )

  model.eval()
  return model, labels, config


def preprocess_camera_frame(frame_rgb: np.ndarray, image_size: int) -> torch.Tensor:
  """RGB HxWxC uint8/float -> normalized CHW tensor. Uses torch resize (no cv2)."""
  arr = np.asarray(frame_rgb)
  if arr.dtype != np.float32:
    arr = arr.astype(np.float32)
  if arr.max() > 1.5:
    arr = arr / 255.0
  tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
  tensor = F.interpolate(tensor, size=(image_size, image_size), mode="area")
  tensor = tensor.squeeze(0)
  return (tensor - 0.5) / 0.5
