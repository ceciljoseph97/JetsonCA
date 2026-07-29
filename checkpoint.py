"""Checkpoint load + camera preprocess without OpenCV (Jetson synthetic bench safe)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import MultiModalCrossAttentionNet


def load_checkpoint(path: Path | str, device: str):
  try:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
  except TypeError:
    checkpoint = torch.load(path, map_location=device)
  config = checkpoint["config"]
  labels = checkpoint.get("activity_labels", checkpoint["labels"])
  model = MultiModalCrossAttentionNet(
    num_classes=len(labels),
    num_activity_classes=int(config.get("num_activity_classes", len(labels))),
    model_dim=int(config["model_dim"]),
    num_heads=int(config["num_heads"]),
    num_layers=int(config["num_layers"]),
    dropout=float(config["dropout"]),
    modality_dropout=0.0,
    temporal_mode=str(config.get("temporal_mode", "none")),
    enable_human_head=bool(config.get("enable_human_head", False)),
  ).to(device)

  state = dict(checkpoint["model_state"])
  remapped = {}
  for key, value in state.items():
    if key.startswith("classifier."):
      remapped[key.replace("classifier.", "activity_classifier.", 1)] = value
    else:
      remapped[key] = value
  model.load_state_dict(remapped, strict=False)
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
