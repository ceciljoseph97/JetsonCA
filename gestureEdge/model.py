"""CNN + LSTM (+ simple dual-radar cross-attn), styled after 4uf04eG/FMCW-gesture-recognition."""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameCNN(nn.Module):
  """Spatial CNN on one RD frame (C,H,W) → feature vector."""

  def __init__(self, in_channels: int = 3, out_features: int = 256):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(in_channels, 32, 3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, 3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      nn.Conv2d(64, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      nn.AdaptiveAvgPool2d(1),
    )
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(128, out_features),
      nn.ReLU(inplace=True),
      nn.Dropout(0.5),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(self.net(x))


class GestureEdgeNet(nn.Module):
  """
  Shared CNN → optional simple cross-attn between radar1/radar2 frame feats
  → LSTM over time → FC classes.

  Inputs: radar1/radar2 (B, T, C, H, W), present masks (B,).
  """

  def __init__(
    self,
    num_classes: int = 11,
    *,
    in_channels: int = 3,
    cnn_features: int = 256,
    lstm_hidden: int = 256,
    modality_dropout: float = 0.0,
  ):
    super().__init__()
    self.num_classes = int(num_classes)
    self.cnn_features = int(cnn_features)
    self.lstm_hidden = int(lstm_hidden)
    self.modality_dropout = float(modality_dropout)

    self.cnn = FrameCNN(in_channels, cnn_features)
    self.cross = nn.MultiheadAttention(cnn_features, num_heads=4, batch_first=True)
    self.cross_norm = nn.LayerNorm(cnn_features)
    self.lstm = nn.LSTM(input_size=cnn_features, hidden_size=lstm_hidden, batch_first=True)
    self.head = nn.Sequential(
      nn.Linear(lstm_hidden, lstm_hidden // 2),
      nn.ReLU(inplace=True),
      nn.Dropout(0.3),
      nn.Linear(lstm_hidden // 2, self.num_classes),
    )

  def _drop(self, p1: torch.Tensor, p2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (not self.training) or self.modality_dropout <= 0:
      return p1, p2
    b = p1.shape[0]
    device = p1.device
    drop = torch.rand(b, device=device) < self.modality_dropout
    drop1 = drop & (torch.rand(b, device=device) < 0.5)
    drop2 = drop & ~drop1
    k1, k2 = p1 & ~drop1, p2 & ~drop2
    none = ~(k1 | k2)
    k1 = torch.where(none, p1, k1)
    k2 = torch.where(none & ~k1, p2, k2)
    return k1, k2

  def encode_frames(self, radar: torch.Tensor) -> torch.Tensor:
    """(B,T,C,H,W) → (B,T,F)."""
    b, t, c, h, w = radar.shape
    flat = radar.reshape(b * t, c, h, w)
    feat = self.cnn(flat).reshape(b, t, -1)
    return feat

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
    b, t = radar1.shape[:2]
    device = radar1.device
    if radar1_present is None:
      radar1_present = torch.ones(b, dtype=torch.bool, device=device)
    if radar2_present is None:
      radar2_present = torch.ones(b, dtype=torch.bool, device=device)
    radar1_present, radar2_present = self._drop(radar1_present, radar2_present)

    f1 = self.encode_frames(radar1)
    f2 = self.encode_frames(radar2)

    # Simple dual fuse per timestep: cross-attn then presence-weighted blend.
    # Treat each batch item's time axis as sequence length for MHA over the *two* streams:
    # stack as (B*T, 2, F)
    stacked = torch.stack([f1, f2], dim=2).reshape(b * t, 2, self.cnn_features)
    attn_out, _ = self.cross(stacked, stacked, stacked, need_weights=False)
    attn_out = self.cross_norm(stacked + attn_out)
    e1 = attn_out[:, 0].reshape(b, t, -1)
    e2 = attn_out[:, 1].reshape(b, t, -1)
    w1 = radar1_present.to(e1.dtype)[:, None, None]
    w2 = radar2_present.to(e2.dtype)[:, None, None]
    fused = (e1 * w1 + e2 * w2) / (w1 + w2).clamp_min(1e-6)

    lstm_out, _ = self.lstm(fused)
    logits = self.head(lstm_out[:, -1])
    reliance = torch.stack(
      [
        radar1_present.float() / (radar1_present.float() + radar2_present.float()).clamp_min(1e-6),
        radar2_present.float() / (radar1_present.float() + radar2_present.float()).clamp_min(1e-6),
      ],
      dim=-1,
    )
    return {
      "logits": logits,
      "reliance": reliance,
      "radar1_present": radar1_present,
      "radar2_present": radar2_present,
    }
