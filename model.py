from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFrameEncoder(nn.Module):
  def __init__(self, in_channels: int, hidden_dim: int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(hidden_dim),
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d(1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.net(x)
    return x.flatten(1)


class FeedForwardBlock(nn.Module):
  def __init__(self, dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
    super().__init__()
    self.net = nn.Sequential(
      nn.LayerNorm(dim),
      nn.Linear(dim, dim * mlp_ratio),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(dim * mlp_ratio, dim),
      nn.Dropout(dropout),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.net(x)


class TemporalEncoder(nn.Module):
  def __init__(self, dim: int, mode: str = "transformer", dropout: float = 0.1, num_heads: int = 4):
    super().__init__()
    self.mode = mode
    if mode == "none":
      self.net = nn.Identity()
    elif mode == "bilstm":
      hidden = max(dim // 2, 1)
      self.net = nn.LSTM(
        input_size=dim,
        hidden_size=hidden,
        num_layers=1,
        batch_first=True,
        bidirectional=True,
      )
      self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Dropout(dropout))
    elif mode == "transformer":
      layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=num_heads,
        dim_feedforward=dim * 4,
        dropout=dropout,
        batch_first=True,
        activation="gelu",
        norm_first=True,
      )
      self.net = nn.TransformerEncoder(layer, num_layers=1)
    else:
      raise ValueError(f"Unsupported temporal mode: {mode}")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.mode == "none":
      return x
    if self.mode == "bilstm":
      out, _ = self.net(x)
      return x + self.proj(out)
    return x + self.net(x)


class CrossAttentionBlock(nn.Module):
  def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
    super().__init__()
    self.radar_to_camera = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
    self.camera_to_radar = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
    self.radar_norm = nn.LayerNorm(dim)
    self.camera_norm = nn.LayerNorm(dim)
    self.radar_ffn = FeedForwardBlock(dim, dropout=dropout)
    self.camera_ffn = FeedForwardBlock(dim, dropout=dropout)

  def forward(
    self,
    radar_tokens: torch.Tensor,
    camera_tokens: torch.Tensor,
    radar_present: torch.Tensor,
    camera_present: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    radar_context, _ = self.radar_to_camera(radar_tokens, camera_tokens, camera_tokens, need_weights=False)
    camera_context, _ = self.camera_to_radar(camera_tokens, radar_tokens, radar_tokens, need_weights=False)

    radar_context = radar_context * camera_present[:, None, None].to(radar_context.dtype)
    camera_context = camera_context * radar_present[:, None, None].to(camera_context.dtype)

    radar_tokens = self.radar_norm(radar_tokens + radar_context)
    camera_tokens = self.camera_norm(camera_tokens + camera_context)

    radar_tokens = self.radar_ffn(radar_tokens)
    camera_tokens = self.camera_ffn(camera_tokens)
    return radar_tokens, camera_tokens


class MultiModalCrossAttentionNet(nn.Module):
  def __init__(
    self,
    num_classes: int,
    radar_channels: int = 3,
    camera_channels: int = 3,
    model_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    max_len: int = 64,
    modality_dropout: float = 0.3,
    temporal_mode: str = "transformer",
    num_activity_classes: int | None = None,
    enable_human_head: bool = True,
  ):
    super().__init__()
    self.modality_dropout = modality_dropout
    self.enable_human_head = enable_human_head
    self.num_activity_classes = num_activity_classes or num_classes

    self.radar_encoder = ConvFrameEncoder(radar_channels, model_dim)
    self.camera_encoder = ConvFrameEncoder(camera_channels, model_dim)
    self.radar_temporal = TemporalEncoder(model_dim, mode=temporal_mode, dropout=dropout, num_heads=num_heads)
    self.camera_temporal = TemporalEncoder(model_dim, mode=temporal_mode, dropout=dropout, num_heads=num_heads)

    self.radar_pos = nn.Parameter(torch.randn(1, max_len, model_dim) * 0.02)
    self.camera_pos = nn.Parameter(torch.randn(1, max_len, model_dim) * 0.02)
    self.radar_modality = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
    self.camera_modality = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

    self.layers = nn.ModuleList(
      [CrossAttentionBlock(model_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
    )

    self.shared_proj = nn.Sequential(
      nn.LayerNorm(model_dim),
      nn.Linear(model_dim, model_dim),
      nn.GELU(),
      nn.Dropout(dropout),
    )
    self.activity_classifier = nn.Sequential(
      nn.LayerNorm(model_dim),
      nn.Linear(model_dim, model_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(model_dim, self.num_activity_classes),
    )
    self.classifier = self.activity_classifier
    self.human_classifier = None
    if self.enable_human_head:
      self.human_classifier = nn.Sequential(
        nn.LayerNorm(model_dim),
        nn.Linear(model_dim, model_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(model_dim, 2),
      )

  def _encode_sequence(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch, time_steps = x.shape[:2]
    x = x.reshape(batch * time_steps, *x.shape[2:])
    x = encoder(x)
    return x.reshape(batch, time_steps, -1)

  def _apply_modality_dropout(
    self,
    radar_present: torch.Tensor,
    camera_present: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if (not self.training) or self.modality_dropout <= 0:
      return radar_present, camera_present

    keep_radar = radar_present.clone()
    keep_camera = camera_present.clone()
    random_values = torch.rand(radar_present.shape[0], 2, device=radar_present.device)

    drop_radar = (random_values[:, 0] < self.modality_dropout) & keep_radar
    drop_camera = (random_values[:, 1] < self.modality_dropout) & keep_camera

    both_dropped = drop_radar & drop_camera
    if both_dropped.any():
      drop_radar = drop_radar & ~both_dropped

    keep_radar = keep_radar & ~drop_radar
    keep_camera = keep_camera & ~drop_camera
    return keep_radar, keep_camera

  def forward(
    self,
    radar: torch.Tensor,
    camera: torch.Tensor,
    radar_present: torch.Tensor | None = None,
    camera_present: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    device = radar.device
    batch_size, radar_len = radar.shape[:2]
    camera_len = camera.shape[1]

    if radar_present is None:
      radar_present = torch.ones(batch_size, dtype=torch.bool, device=device)
    if camera_present is None:
      camera_present = torch.ones(batch_size, dtype=torch.bool, device=device)

    radar_present, camera_present = self._apply_modality_dropout(radar_present, camera_present)

    radar_tokens = self._encode_sequence(self.radar_encoder, radar)
    camera_tokens = self._encode_sequence(self.camera_encoder, camera)

    radar_tokens = radar_tokens + self.radar_pos[:, :radar_len] + self.radar_modality
    camera_tokens = camera_tokens + self.camera_pos[:, :camera_len] + self.camera_modality

    radar_tokens = radar_tokens * radar_present[:, None, None].to(radar_tokens.dtype)
    camera_tokens = camera_tokens * camera_present[:, None, None].to(camera_tokens.dtype)
    radar_tokens = self.radar_temporal(radar_tokens)
    camera_tokens = self.camera_temporal(camera_tokens)
    radar_tokens = radar_tokens * radar_present[:, None, None].to(radar_tokens.dtype)
    camera_tokens = camera_tokens * camera_present[:, None, None].to(camera_tokens.dtype)

    for layer in self.layers:
      radar_tokens, camera_tokens = layer(radar_tokens, camera_tokens, radar_present, camera_present)
      radar_tokens = radar_tokens * radar_present[:, None, None].to(radar_tokens.dtype)
      camera_tokens = camera_tokens * camera_present[:, None, None].to(camera_tokens.dtype)

    radar_shared = self.shared_proj(radar_tokens.mean(dim=1))
    camera_shared = self.shared_proj(camera_tokens.mean(dim=1))

    present_count = radar_present.to(torch.float32) + camera_present.to(torch.float32)
    present_count = present_count.clamp_min(1.0).unsqueeze(1)
    fused = (
      radar_shared * radar_present[:, None].to(radar_shared.dtype)
      + camera_shared * camera_present[:, None].to(camera_shared.dtype)
    ) / present_count

    logits = self.activity_classifier(fused)
    human_logits = None
    if self.human_classifier is not None:
      human_logits = self.human_classifier(fused)

    align_mask = radar_present & camera_present
    if align_mask.any():
      alignment_loss = 1.0 - F.cosine_similarity(
        radar_shared[align_mask],
        camera_shared[align_mask],
        dim=-1,
      ).mean()
    else:
      alignment_loss = logits.new_tensor(0.0)

    return {
      "logits": logits,
      "activity_logits": logits,
      "human_logits": human_logits,
      "fused": fused,
      "radar_shared": radar_shared,
      "camera_shared": camera_shared,
      "alignment_loss": alignment_loss,
      "radar_present": radar_present,
      "camera_present": camera_present,
    }
