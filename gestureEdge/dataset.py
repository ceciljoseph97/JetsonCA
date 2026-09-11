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

from .preprocess import BGT_LABELS, BGT_SLUGS, SOLI_LABELS, soli_clip_to_tensor, window_clip


@dataclass(frozen=True)
class SoliRef:
  key: str
  gesture: int
  session: str
  rep: str


def _is_zip(path: Path) -> bool:
  return path.is_file() and path.suffix.lower() == ".zip"


def index_soli(
  data_path: Path,
  *,
  max_label: int = 10,
  allowed: set[int] | None = None,
) -> list[SoliRef]:
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
    if allowed is not None and gesture not in allowed:
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
    label_map: dict[int, int] | None = None,
  ):
    self.data_path = Path(data_path)
    self.refs = list(refs)
    self.window = int(window)
    self.train = bool(train)
    self.label_map = dict(label_map) if label_map else None
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
    gid = int(ref.gesture)
    label = int(self.label_map[gid]) if self.label_map is not None else gid
    return {
      "radar1": radar1,
      "radar2": radar2,
      "radar1_present": True,
      "radar2_present": True,
      "label": label,
    }

  def close(self):
    if self._zip is not None:
      self._zip.close()
      self._zip = None


def _slug_to_label(slug: str) -> int | None:
  key = slug.replace(" ", "_").lower()
  for i, name in enumerate(BGT_SLUGS):
    if name.lower() == key or BGT_LABELS[i].lower() == slug.lower():
      return i
  return None


def index_bgt_drive(root: Path) -> list[tuple[Path, int]]:
  root = Path(root)
  if not root.is_dir():
    raise FileNotFoundError(f"No BGT drive clips under {root}")
  out: list[tuple[Path, int]] = []
  for path in sorted(root.rglob("*.npz")):
    lab = _slug_to_label(path.parent.name)
    if lab is None:
      continue
    out.append((path, lab))
  if not out:
    raise FileNotFoundError(f"No Push/Pull/Palm_Hold/Palm_Tilt .npz under {root}")
  return out


def split_bgt_files(
  files: list[tuple[Path, int]],
  *,
  val_ratio: float = 0.2,
  seed: int = 0,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
  by: dict[int, list[tuple[Path, int]]] = {}
  for item in files:
    by.setdefault(item[1], []).append(item)
  rng = np.random.default_rng(seed)
  train: list[tuple[Path, int]] = []
  val: list[tuple[Path, int]] = []
  for lab in sorted(by):
    items = list(by[lab])
    rng.shuffle(items)
    n = len(items)
    if n <= 1:
      train.extend(items)
      continue
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1)
    val.extend(items[:n_val])
    train.extend(items[n_val:])
  if not train:
    raise RuntimeError("no BGT train clips")
  if not val:
    val = list(train)
  return train, val


class BgtDriveDataset(Dataset):
  """Live BGT clips saved as npz (T,3,32,32) per radar."""

  def __init__(
    self,
    files: list[tuple[Path, int]],
    *,
    window: int = 40,
    train: bool = False,
    seed: int = 0,
  ):
    self.files = list(files)
    self.window = int(window)
    self.train = bool(train)
    self._rng = np.random.default_rng(seed)

  def __len__(self) -> int:
    return len(self.files)

  def __getitem__(self, index: int) -> dict:
    path, label = self.files[index]
    z = np.load(path)
    r1 = torch.from_numpy(np.asarray(z["radar1"], dtype=np.float32))
    if "radar2" in z.files:
      r2 = torch.from_numpy(np.asarray(z["radar2"], dtype=np.float32))
    else:
      r2 = r1
    if r1.ndim == 3:
      r1 = r1.unsqueeze(0)
    if r2.ndim == 3:
      r2 = r2.unsqueeze(0)
    r1 = window_clip(r1, window=self.window, train=self.train, rng=self._rng)
    r2 = window_clip(r2, window=self.window, train=self.train, rng=self._rng)
    if self.train and float(self._rng.random()) < 0.3:
      r1 = (r1 + 0.02 * torch.randn_like(r1)).clamp(0, 1)
      r2 = (r2 + 0.02 * torch.randn_like(r2)).clamp(0, 1)
    return {
      "radar1": r1,
      "radar2": r2,
      "radar1_present": True,
      "radar2_present": True,
      "label": int(label),
    }

  def close(self):
    return
