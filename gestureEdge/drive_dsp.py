"""Doppler-centroid drive features (shared by Drive DSP and Infer calibrator)."""

from __future__ import annotations

import numpy as np

from range_gating import _as_rd_hw

_PUSH_FRAME = -0.62
_PULL_FRAME = -0.32
_HIST_N = 8
_PUSH_VOTES = 3
_PULL_VOTES = 3
_HOLD_VOTES = 6
_FLICK_LATCH = 5
_RANGE_STD_HONK = 1.35
_IMBALANCE_HONK = 0.32
_HONK_MID = 4


def energy_ok(rd) -> bool:
  hw = _as_rd_hw(rd)
  if hw.size < 8:
    return False
  tmp = np.asarray(hw, dtype=np.float32).copy()
  tmp[:2, :] = 0.0
  return float(tmp.max()) >= 0.22


def hand_present(rd) -> bool:
  hw = _as_rd_hw(rd)
  if hw.size < 8:
    return False
  tmp = np.asarray(hw, dtype=np.float32).copy()
  tmp[:2, :] = 0.0
  peak = float(tmp.max())
  if peak < 0.35:
    return False
  tot = float(tmp.sum()) + 1e-6
  r, d = np.unravel_index(int(np.argmax(tmp)), tmp.shape)
  r0, r1 = max(0, r - 2), min(tmp.shape[0], r + 3)
  d0, d1 = max(0, d - 2), min(tmp.shape[1], d + 3)
  frac = float(tmp[r0:r1, d0:d1].sum()) / tot
  return frac >= 0.10 and r >= 3


def range_peak(rd) -> float:
  """Range-bin of the energy peak (H=range). Tilt walks this; Hold stays put."""
  hw = _as_rd_hw(rd)
  if hw.size < 8:
    return 0.0
  tmp = np.asarray(hw, dtype=np.float32).copy()
  if tmp.shape[0] >= 2:
    tmp[:2, :] = 0.0
  e = tmp.sum(axis=1)
  return float(np.argmax(e))


def rd_energy(rd) -> float:
  hw = _as_rd_hw(rd)
  if hw.size < 8:
    return 0.0
  tmp = np.asarray(hw, dtype=np.float32).copy()
  if tmp.shape[0] >= 2:
    tmp[:2, :] = 0.0
  return float(tmp.sum())


def dual_imbalance(rd1, rd2) -> float:
  e1 = rd_energy(rd1)
  e2 = rd_energy(rd2)
  return abs(e1 - e2) / (e1 + e2 + 1e-6)


def honk_cue(range_hist, *, n_mid: int, present: bool, imbalance: float = 0.0) -> bool:
  """Palm tilt: mid-band, hand present, range wander and/or R1–R2 imbalance."""
  rs = np.asarray(range_hist, dtype=np.float32)
  if (not present) or n_mid < _HONK_MID or rs.size < 6:
    return False
  rstd = float(rs.std())
  if rstd >= _RANGE_STD_HONK:
    return True
  return imbalance >= _IMBALANCE_HONK and rstd >= 0.8


def doppler_centroid(tchw) -> float:
  x = np.asarray(tchw, dtype=np.float32)
  if hasattr(tchw, "detach"):
    x = tchw.detach().cpu().numpy().astype(np.float32)
  if x.ndim == 4:
    hw = x.mean(axis=(0, 1))
  elif x.ndim == 3:
    hw = x.max(axis=0)
  else:
    hw = x
  hw = np.asarray(hw, dtype=np.float32).copy()
  if hw.shape[0] >= 2:
    hw[:2, :] = 0.0
  energy = hw.sum(axis=0)
  mid = hw.shape[1] // 2
  bins = np.arange(hw.shape[1], dtype=np.float32) - float(mid)
  return float((energy * bins).sum() / (float(energy.sum()) + 1e-6))


def _as_tchw(frames) -> np.ndarray:
  if hasattr(frames, "detach"):
    x = frames.detach().cpu().numpy().astype(np.float32)
  else:
    x = np.asarray(frames, dtype=np.float32)
  if x.ndim == 3:
    x = x[None, ...]
  return x


def frame_centroids(tchw) -> np.ndarray:
  x = _as_tchw(tchw)
  return np.asarray([doppler_centroid(x[i]) for i in range(len(x))], dtype=np.float32)


def vote_vector(tchw, *, hist_n: int = _HIST_N) -> np.ndarray:
  """[c_mean, n_push, n_pull, n_mid, present, energy] on last hist_n frames."""
  x = _as_tchw(tchw)
  cs = frame_centroids(x[-hist_n:])
  n_push = float((cs <= _PUSH_FRAME).sum())
  n_pull = float((cs >= _PULL_FRAME).sum())
  n_mid = float(((cs > _PUSH_FRAME) & (cs < _PULL_FRAME)).sum())
  last = x[-1]
  return np.asarray(
    [float(cs.mean()), n_push, n_pull, n_mid, float(hand_present(last)), float(energy_ok(last))],
    dtype=np.float32,
  )
