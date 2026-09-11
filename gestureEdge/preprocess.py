"""Soli DSP + live RD preprocess (32×32 per frame, 3 RX channels)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# DeepSoli file IDs 0..10 → names (GitHub table / arxiv:2602.04436 Fig.5).
# ID 11 = Background (skipped in train by default).
SOLI_ID_TO_NAME: dict[int, str] = {
  0: "Pinch Index",
  1: "Palm Tilt",
  2: "Finger Slide",
  3: "Pinch Pinky",
  4: "Slow Swipe",
  5: "Fast Swipe",
  6: "Push",
  7: "Pull",
  8: "Finger Rub",
  9: "Circle",
  10: "Palm Hold",
  11: "Background",
}
SOLI_LABELS = tuple(SOLI_ID_TO_NAME[i] for i in range(11))
# Drive / 3-class subset (Soli file IDs).
DRIVE_SOLI_IDS: tuple[int, ...] = (6, 7, 10)  # Push, Pull, Palm Hold
DRIVE_LABELS: tuple[str, ...] = tuple(SOLI_ID_TO_NAME[i] for i in DRIVE_SOLI_IDS)
DRIVE_ID_TO_CLASS: dict[int, int] = {gid: i for i, gid in enumerate(DRIVE_SOLI_IDS)}
DRIVE_SLUGS: tuple[str, ...] = tuple(n.replace(" ", "_") for n in DRIVE_LABELS)
# BGT finetune / Collect: same 3 + Palm Tilt as class 3 (keeps 3-class head rows).
BGT_SOLI_IDS: tuple[int, ...] = (6, 7, 10, 1)
BGT_LABELS: tuple[str, ...] = tuple(SOLI_ID_TO_NAME[i] for i in BGT_SOLI_IDS)
BGT_SLUGS: tuple[str, ...] = tuple(n.replace(" ", "_") for n in BGT_LABELS)
FRAME_HW = (32, 32)


def soli_name(gesture_id: int) -> str:
  return SOLI_ID_TO_NAME.get(int(gesture_id), str(gesture_id))


def normalize_label_list(labels: list | tuple | None) -> list[str]:
  """Map ckpt labels (numeric strings or names) → canonical gesture names."""
  if not labels:
    return list(SOLI_LABELS)
  out: list[str] = []
  for lab in labels:
    s = str(lab)
    if s.isdigit():
      out.append(soli_name(int(s)))
    else:
      out.append(s)
  return out


def normalize_channels(chw: np.ndarray) -> np.ndarray:
  """Per-channel min-max to [0,1] like FMCW-gesture-recognition do_preprocessing."""
  x = np.asarray(chw, dtype=np.float32)
  out = np.empty_like(x)
  for i in range(x.shape[0]):
    lo = float(x[i].min())
    hi = float(x[i].max())
    out[i] = (x[i] - lo) / (hi - lo + 1e-6)
  return out


def soli_frame_to_chw(ch0: np.ndarray, ch1: np.ndarray, ch2: np.ndarray) -> np.ndarray:
  """One Soli frame (1024,)×3 → (3,32,32)."""
  planes = []
  for ch in (ch0, ch1, ch2):
    v = np.asarray(ch, dtype=np.float32).reshape(-1)
    if v.size < FRAME_HW[0] * FRAME_HW[1]:
      v = np.pad(v, (0, FRAME_HW[0] * FRAME_HW[1] - v.size))
    planes.append(v[: FRAME_HW[0] * FRAME_HW[1]].reshape(FRAME_HW))
  return normalize_channels(np.stack(planes, axis=0))


def soli_clip_to_tensor(ch0: np.ndarray, ch1: np.ndarray, ch2: np.ndarray) -> torch.Tensor:
  """(T,1024)×3 → (T,3,32,32) float32 tensor."""
  t = int(ch0.shape[0])
  frames = [soli_frame_to_chw(ch0[i], ch1[i], ch2[i]) for i in range(t)]
  return torch.from_numpy(np.stack(frames, axis=0))


def window_clip(
  clip: torch.Tensor,
  *,
  window: int = 40,
  train: bool = False,
  rng: np.random.Generator | None = None,
) -> torch.Tensor:
  """(T,C,H,W) → (window,C,H,W)."""
  t = int(clip.shape[0])
  win = max(1, int(window))
  if t >= win:
    if train:
      rng = rng or np.random.default_rng()
      start = int(rng.integers(0, t - win + 1))
    else:
      start = max(0, (t - win) // 2)
    return clip[start : start + win]
  pad = clip[:1].expand(win - t, *clip.shape[1:])
  return torch.cat([pad, clip], dim=0)


def live_rd_to_frame(frame: torch.Tensor | np.ndarray) -> torch.Tensor:
  """Live BGT RD (3,H,W) → (3,32,32) min-max per channel."""
  if isinstance(frame, torch.Tensor):
    x = frame.detach().float().cpu().numpy()
  else:
    x = np.asarray(frame, dtype=np.float32)
  if x.ndim == 4:
    x = x[-1]
  if x.shape[0] != 3:
    # pad / trim channels
    c = np.zeros((3, x.shape[-2], x.shape[-1]), dtype=np.float32)
    n = min(3, x.shape[0])
    c[:n] = x[:n]
    x = c
  x = normalize_channels(x)
  t = torch.from_numpy(x).unsqueeze(0)
  t = F.interpolate(t, size=FRAME_HW, mode="bilinear", align_corners=False).squeeze(0)
  return t
