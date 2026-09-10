"""Upstream CNN+LSTM from 4uf04eG/FMCW-gesture-recognition.

Architecture is copied so the pickled FinetuneGestureNet checkpoint
(`trained_model_finetune_7cl_25ep_custom_split.pt`) can unpickle. Do not
“improve” conv padding / LSTM layout — weights are a full object pickle.

Live wrapper matches GestureEdgeNet: radar1/radar2 (B,T,C,H,W) → logits.
https://github.com/4uf04eG/FMCW-gesture-recognition
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# sklearn LabelEncoder.classes_ order from their encoder_7 comment.
FMCW_7_LABELS: tuple[str, ...] = (
  "finger_circle",
  "finger_rub",
  "no-action",
  "palm_hold",
  "pull",
  "push",
  "swipe",
)

UPSTREAM_WINDOW = 12  # Debouncer.memory_length in realtime_recognition.py


class FeatureExtractor(torch.nn.Module):
  def __init__(self, in_features, out_features) -> None:
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_features, 32, 3)
    self.norm1 = torch.nn.BatchNorm2d(32)
    self.pool1 = torch.nn.MaxPool2d(2)

    self.conv2 = torch.nn.Conv2d(32, 64, 3)
    self.norm2 = torch.nn.BatchNorm2d(64)
    self.pool2 = torch.nn.MaxPool2d(2)

    self.conv3 = torch.nn.Conv2d(64, 128, 3)
    self.norm3 = torch.nn.BatchNorm2d(128)
    self.pool3 = torch.nn.MaxPool2d(2)

    self.conv4 = torch.nn.Conv2d(128, 256, 3)
    self.norm4 = torch.nn.BatchNorm2d(256)
    self.pool4 = torch.nn.MaxPool2d(2)

    self.relu = torch.nn.ReLU()
    self.flatten = torch.nn.Flatten()
    self.fc = torch.nn.Linear(512, out_features)
    self.dropout = torch.nn.Dropout(0.5)

  def forward(self, x):
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.relu(x)
    x = self.pool1(x)

    x = self.conv2(x)
    x = self.norm2(x)
    x = self.relu(x)
    x = self.pool2(x)

    x = self.conv3(x)
    x = self.norm3(x)
    x = self.relu(x)
    x = self.pool3(x)

    x = self.flatten(x)
    x = self.fc(x)
    x = self.dropout(x)
    return x


class GestureNet(torch.nn.Module):
  def __init__(
    self,
    num_input_channels=4,
    num_cnn_features=256,
    num_rnn_hidden_size=256,
    num_classes=7,
  ) -> None:
    super().__init__()
    self.num_rnn_hidden_size = num_rnn_hidden_size
    self.frame_model = FeatureExtractor(num_input_channels, num_cnn_features)
    self.temporal_model = torch.nn.LSTM(input_size=num_cnn_features, hidden_size=num_rnn_hidden_size)
    self.fc1 = torch.nn.Linear(num_rnn_hidden_size, num_rnn_hidden_size // 2)
    self.relu = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(num_rnn_hidden_size // 2, num_classes)

  def forward(self, x):
    hidden = None
    for frame in x:
      features = self.frame_model(frame)
      features = torch.unsqueeze(features, 0)
      out, hidden = self.temporal_model(features, hidden)
    out = self.fc1(out)
    out = self.relu(out)
    out = self.fc2(out)
    return out


class FinetuneGestureNet(GestureNet):
  def __init__(self, num_classes=7, weights_path=None):
    super().__init__(num_input_channels=3, num_classes=12)
    self.fc2 = torch.nn.Sequential(
      torch.nn.Linear(self.num_rnn_hidden_size // 2, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 64),
      torch.nn.ReLU(),
      torch.nn.Linear(64, num_classes),
    )
    if weights_path:
      loaded = torch.load(weights_path, map_location="cpu", weights_only=False)
      src = loaded.state_dict() if isinstance(loaded, torch.nn.Module) else loaded
      self.load_state_dict(src, strict=False)


class ActionDebouncer:
  """Working debounce (upstream Debouncer compares a list to an int and never fires)."""

  def __init__(
    self,
    detect_threshold: float = 0.6,
    noise_threshold: float = 0.3,
    min_num_detections: int = 3,
  ):
    self.detect_threshold = float(detect_threshold)
    self.noise_threshold = float(noise_threshold)
    self.min_num_detections = int(min_num_detections)
    self.memory: list[int] = []

  def update(self, probs: np.ndarray) -> Optional[int]:
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    hits = np.where(p > self.detect_threshold)[0]
    if hits.size != 1:
      self.memory.clear()
      return None
    if int(np.sum(p <= self.noise_threshold)) < (p.size - 1):
      self.memory.clear()
      return None
    action = int(hits[0])
    self.memory.append(action)
    if len(self.memory) > self.min_num_detections:
      self.memory = self.memory[-self.min_num_detections :]
    if len(self.memory) >= self.min_num_detections and all(x == action for x in self.memory):
      return action
    return None


def infer_num_classes(net: nn.Module) -> int:
  fc2 = getattr(net, "fc2", None)
  if isinstance(fc2, nn.Sequential):
    for m in reversed(list(fc2.modules())):
      if isinstance(m, nn.Linear):
        return int(m.out_features)
  if isinstance(fc2, nn.Linear):
    return int(fc2.out_features)
  return 7


class UpstreamFmcwAdapter(nn.Module):
  """Single-radar GitHub net behind the dual-radar GestureEdgeNet call signature."""

  def __init__(self, net: nn.Module):
    super().__init__()
    self.net = net
    self.num_classes = infer_num_classes(net)

  def forward(
    self,
    radar1: torch.Tensor,
    radar2: torch.Tensor | None = None,
    *,
    radar1_present: torch.Tensor | None = None,
    radar2_present: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    if radar2 is None:
      radar2 = radar1
    b = int(radar1.shape[0])
    device = radar1.device
    if radar1_present is None:
      radar1_present = torch.ones(b, dtype=torch.bool, device=device)
    if radar2_present is None:
      radar2_present = torch.ones(b, dtype=torch.bool, device=device)

    w1 = radar1_present.to(dtype=radar1.dtype)[:, None, None, None, None]
    w2 = radar2_present.to(dtype=radar2.dtype)[:, None, None, None, None]
    fused = (radar1 * w1 + radar2 * w2) / (w1 + w2).clamp_min(1e-6)
    # GitHub GestureNet: x is (T, B, C, H, W); `for frame in x`.
    x = fused.permute(1, 0, 2, 3, 4).contiguous()
    raw = self.net(x)
    if raw.dim() == 3:
      logits = raw.reshape(raw.shape[0], raw.shape[1], -1)[-1]
    else:
      logits = raw.reshape(b, -1)
    logits = logits.reshape(b, -1)
    denom = (radar1_present.float() + radar2_present.float()).clamp_min(1e-6)
    reliance = torch.stack([radar1_present.float() / denom, radar2_present.float() / denom], dim=-1)
    return {
      "logits": logits,
      "reliance": reliance,
      "radar1_present": radar1_present,
      "radar2_present": radar2_present,
    }
