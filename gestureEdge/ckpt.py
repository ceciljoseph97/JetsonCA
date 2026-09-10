"""Checkpoint load for gestureEdge."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import GestureEdgeNet
from .preprocess import normalize_label_list


def load_ckpt(path: Path, device: str):
  try:
    ckpt = torch.load(path, map_location=device, weights_only=False)
  except TypeError:
    ckpt = torch.load(path, map_location=device)
  labels = normalize_label_list(ckpt.get("labels"))
  cfg = dict(ckpt.get("config") or {})
  model = GestureEdgeNet(
    num_classes=len(labels),
    in_channels=int(cfg.get("in_channels", 3)),
    cnn_features=int(cfg.get("cnn_features", 256)),
    lstm_hidden=int(cfg.get("lstm_hidden", 256)),
    modality_dropout=0.0,
  ).to(device)
  model.load_state_dict(ckpt["model_state"], strict=True)
  model.eval()
  return model, labels, cfg
