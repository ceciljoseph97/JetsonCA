"""Save / count live BGT Push-Pull-Hold + Palm Tilt clips.

  python gestureEdge/collect_drive.py --out artifacts/gesture_edge_bgt --reps 15
  python gestureEdge/collect_drive.py --class "Palm Tilt" --reps 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from gestureEdge.preprocess import BGT_LABELS, DRIVE_LABELS, SOLI_ID_TO_NAME, live_rd_to_frame


COLLECT_LABELS: tuple[str, ...] = tuple(BGT_LABELS)


def slug_for(label: str) -> str:
  return str(label).replace(" ", "_")


def class_dir(root: Path, label: str) -> Path:
  d = Path(root) / slug_for(label)
  d.mkdir(parents=True, exist_ok=True)
  return d


def _label_id(label: str) -> int:
  if label in DRIVE_LABELS:
    return list(DRIVE_LABELS).index(label)
  for gid, name in SOLI_ID_TO_NAME.items():
    if name.lower() == label.lower() or name.replace(" ", "_").lower() == slug_for(label).lower():
      return int(gid)
  raise ValueError(f"Unknown collect label {label}")


def count_clips(root: Path) -> dict[str, int]:
  root = Path(root)
  out = {}
  for name in COLLECT_LABELS:
    d = root / slug_for(name)
    out[name] = len(list(d.glob("*.npz"))) if d.is_dir() else 0
  return out


def next_idx(folder: Path) -> int:
  existing = sorted(folder.glob("*.npz"))
  if not existing:
    return 0
  nums = []
  for p in existing:
    try:
      nums.append(int(p.stem))
    except ValueError:
      continue
  return (max(nums) + 1) if nums else 0


def save_clip(
  root: Path,
  label: str,
  radar1: torch.Tensor | np.ndarray,
  radar2: torch.Tensor | np.ndarray | None,
) -> Path:
  folder = class_dir(root, label)
  idx = next_idx(folder)
  r1 = radar1.detach().cpu().numpy() if isinstance(radar1, torch.Tensor) else np.asarray(radar1)
  if radar2 is None:
    r2 = r1
  else:
    r2 = radar2.detach().cpu().numpy() if isinstance(radar2, torch.Tensor) else np.asarray(radar2)
  lab_i = _label_id(label)
  path = folder / f"{idx:04d}.npz"
  np.savez_compressed(
    path,
    radar1=np.asarray(r1, dtype=np.float32),
    radar2=np.asarray(r2, dtype=np.float32),
    label=np.int64(lab_i),
  )
  man = Path(root) / "manifest.json"
  man.write_text(json.dumps(count_clips(root), indent=2), encoding="utf-8")
  return path


def _fresh_frame(session) -> tuple:
  t1, t2 = session.read_tensors()
  if t1 is None or int(session._miss_streak[0]) != 0:
    return None, None
  f1 = live_rd_to_frame(t1)
  if t2 is not None and int(session._miss_streak[1]) == 0:
    f2 = live_rd_to_frame(t2)
  else:
    f2 = f1
  return f1, f2


def _prompt_and_record(session, n: int, label: str):
  print(f"\n>>> {label}: get ready…", flush=True)
  for s in (3, 2, 1):
    print(f"  {s}", flush=True)
    time.sleep(1.0)
  print("  RECORD", flush=True)
  f1, f2 = [], []
  t0 = time.time()
  while len(f1) < n:
    a, b = _fresh_frame(session)
    if a is not None:
      f1.append(a)
      f2.append(b)
      print(f"  {len(f1)}/{n}", end="\r", flush=True)
    else:
      time.sleep(0.01)
    if time.time() - t0 > 60:
      raise TimeoutError(f"only got {len(f1)}/{n} frames for {label} (radar live?)")
  print(flush=True)
  r1 = torch.stack(f1[:n], 0)
  r2 = torch.stack(f2[:n], 0)
  return r1, r2


def parse_args():
  p = argparse.ArgumentParser(description="Collect BGT Push/Pull/Palm Hold/Palm Tilt clips")
  p.add_argument("--out", type=Path, default=Path("artifacts/gesture_edge_bgt"))
  p.add_argument("--class", dest="only_class", type=str, default=None, help="Push | Pull | Palm Hold | Palm Tilt")
  p.add_argument("--reps", type=int, default=15)
  p.add_argument("--window", type=int, default=40)
  p.add_argument("--num-rx", type=int, default=3)
  p.add_argument("--radar-profile", default="gesture")
  p.add_argument("--frame-rate", type=float, default=5.0)
  p.add_argument("--radar1-uuid", type=str, default=None)
  p.add_argument("--radar2-uuid", type=str, default=None)
  return p.parse_args()


def main():
  args = parse_args()
  from radar_utils import DualRadarSession

  labels = [args.only_class] if args.only_class else list(COLLECT_LABELS)
  resolved = []
  for lab in labels:
    hit = None
    for name in COLLECT_LABELS:
      if lab.lower() in (name.lower(), slug_for(name).lower(), slug_for(lab).lower()):
        hit = name
        break
    if hit is None:
      raise SystemExit(f"Unknown class {lab}; use {list(COLLECT_LABELS)}")
    resolved.append(hit)
  labels = resolved
  kwargs = dict(
    num_rx=args.num_rx,
    profile=args.radar_profile,
    frame_rate_hz=args.frame_rate,
    radar1_uuid=args.radar1_uuid,
    radar2_uuid=args.radar2_uuid,
    mirror_radar2=True,
  )
  import inspect

  sig = inspect.signature(DualRadarSession.__init__)
  kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
  args.out.mkdir(parents=True, exist_ok=True)
  print("counts", count_clips(args.out), flush=True)
  with DualRadarSession(**kwargs) as session:
    print(session.status_text, flush=True)
    if not session.slots[0].available:
      raise SystemExit("radar1 not available\n" + session.diagnose())
    for lab in labels:
      for i in range(args.reps):
        r1, r2 = _prompt_and_record(session, args.window, lab)
        path = save_clip(args.out, lab, r1, r2)
        print(f"  saved {path}  ({i+1}/{args.reps})  {count_clips(args.out)}", flush=True)
  print("done", count_clips(args.out), flush=True)


if __name__ == "__main__":
  main()
