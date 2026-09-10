"""DeepSoli dsp/*.h5 loader from SoliData.zip (or extracted folder)."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import SOLI_LABELS, soli_clip_to_tensor, window_clip


@dataclass(frozen=True)
class SoliRef:
  key: str
  gesture: int
  session: str
  rep: str


def _is_zip(path: Path) -> bool:
  return path.is_file() and path.suffix.lower() == ".zip"


def index_soli(data_path: Path, *, max_label: int = 10) -> list[SoliRef]:
  data_path = Path(data_path).resolve()
  refs: list[SoliRef] = []
  if _is_zip(data_path):
    with zipfile.ZipFile(data_path) as zf:
      names = sorted(n for n in zf.namelist() if n.endswith(".h5"))
  else:
    root = data_path / "dsp" if (data_path / "dsp").is_dir() else data_path
    names = [str(p.relative_to(data_path)).replace("\\", "/") for p in sorted(root.rglob("*.h5"))]
    if not names and data_path.is_dir():
      names = [str(p) for p in sorted(Path(data_path).rglob("*.h5"))]

  for name in names:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 3:
      continue
    try:
      gesture = int(parts[0])
    except ValueError:
      continue
    if gesture < 0 or gesture > max_label:
      continue
    refs.append(SoliRef(key=name, gesture=gesture, session=parts[1], rep=parts[2]))
  if not refs:
    raise FileNotFoundError(f"No Soli .h5 under {data_path}")
  return refs


def split_by_session(refs: list[SoliRef], *, val_ratio: float = 0.2, seed: int = 0):
  sessions = sorted({r.session for r in refs})
  rng = np.random.default_rng(seed)
  rng.shuffle(sessions)
  n_val = max(1, int(round(len(sessions) * val_ratio))) if len(sessions) > 1 else 0
  val_s = set(sessions[:n_val]) if n_val else set()
  train = [r for r in refs if r.session not in val_s]
  val = [r for r in refs if r.session in val_s] if val_s else train[-max(1, len(train) // 10) :]
  return train, val


class SoliGestureDataset(Dataset):
  def __init__(
    self,
    data_path: Path,
    refs: list[SoliRef],
    *,
    window: int = 40,
    train: bool = False,
    seed: int = 0,
  ):
    self.data_path = Path(data_path)
    self.refs = list(refs)
    self.window = int(window)
    self.train = bool(train)
    self._rng = np.random.default_rng(seed)
    self._zip = zipfile.ZipFile(self.data_path) if _is_zip(self.data_path) else None

  def __getstate__(self):
    s = self.__dict__.copy()
    s["_zip"] = None
    return s

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._zip = None

  def __len__(self) -> int:
    return len(self.refs)

  def _open_h5(self, key: str):
    if self._zip is None and _is_zip(self.data_path):
      self._zip = zipfile.ZipFile(self.data_path)
    if self._zip is not None:
      return h5py.File(io.BytesIO(self._zip.read(key)), "r")
    path = Path(key)
    if not path.is_file():
      path = self.data_path / key
    return h5py.File(path, "r")

  def __getitem__(self, index: int) -> dict:
    ref = self.refs[index]
    with self._open_h5(ref.key) as h:
      ch0 = np.asarray(h["ch0"], dtype=np.float32)
      ch1 = np.asarray(h["ch1"], dtype=np.float32)
      ch2 = np.asarray(h["ch2"], dtype=np.float32)
    clip = soli_clip_to_tensor(ch0, ch1, ch2)
    radar1 = window_clip(clip, window=self.window, train=self.train, rng=self._rng)
    # Dual-radar training: mirror (Soli is single sensor).
    if self.train and float(self._rng.random()) < 0.5:
      radar2 = radar1 + 0.01 * torch.randn_like(radar1)
    else:
      radar2 = radar1
    return {
      "radar1": radar1,
      "radar2": radar2,
      "radar1_present": True,
      "radar2_present": True,
      "label": int(ref.gesture),
    }

  def close(self):
    if self._zip is not None:
      self._zip.close()
      self._zip = None
