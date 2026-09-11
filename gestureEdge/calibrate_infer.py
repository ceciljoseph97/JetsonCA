"""Fit Infer linear head: net logits + Doppler votes → Push/Pull/Hold.

  python gestureEdge/calibrate_infer.py --ckpt artifacts/gesture_edge_soli/best_gesture_edge.pt --bgt-data artifacts/gesture_edge_bgt --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from gestureEdge.ckpt import load_ckpt
from gestureEdge.drive_dsp import vote_vector
from gestureEdge.preprocess import DRIVE_LABELS, DRIVE_SLUGS


class InferCalibrator(nn.Module):
  def __init__(self, n_in: int = 9, n_out: int = 3):
    super().__init__()
    self.fc = nn.Linear(n_in, n_out)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(x)


def feat_from_clip(logits: torch.Tensor, radar1: np.ndarray) -> np.ndarray:
  v = vote_vector(radar1)
  return np.concatenate([logits.detach().float().cpu().numpy().reshape(-1), v], axis=0).astype(np.float32)


@torch.no_grad()
def collect_xy(model, root: Path, device: str, labels: list[str]):
  xs, ys = [], []
  name_to_i = {n: i for i, n in enumerate(labels)}
  for name, slug in zip(DRIVE_LABELS, DRIVE_SLUGS):
    yi = name_to_i.get(name)
    if yi is None:
      continue
    folder = Path(root) / slug
    for p in sorted(folder.glob("*.npz")):
      z = np.load(p)
      r1 = torch.from_numpy(np.asarray(z["radar1"], dtype=np.float32)).unsqueeze(0).to(device)
      r2 = torch.from_numpy(np.asarray(z["radar2"], dtype=np.float32)).unsqueeze(0).to(device)
      logits = model(r1, r2)["logits"][0]
      xs.append(feat_from_clip(logits, z["radar1"]))
      ys.append(yi)
  if not xs:
    raise SystemExit(f"No clips under {root}")
  return np.stack(xs, 0), np.asarray(ys, dtype=np.int64)


def train_calibrator(x: np.ndarray, y: np.ndarray, *, steps: int = 400, lr: float = 0.2):
  # Oversample Hold (class 2 if labels are Push/Pull/Palm Hold).
  xs, ys = [x], [y]
  for c in np.unique(y):
    m = y == c
    n = int(m.sum())
    target = int(max(np.bincount(y)) * (2 if c == int(y.max()) else 1))
    if n and target > n:
      idx = np.random.default_rng(0).choice(np.where(m)[0], size=target - n, replace=True)
      xs.append(x[idx])
      ys.append(y[idx])
  x = np.concatenate(xs, 0)
  y = np.concatenate(ys, 0)
  mu = x.mean(0)
  sd = x.std(0) + 1e-6
  xn = (x - mu) / sd
  net = InferCalibrator(n_in=x.shape[1], n_out=int(y.max()) + 1)
  opt = torch.optim.Adam(net.parameters(), lr=lr)
  xt = torch.from_numpy(xn)
  yt = torch.from_numpy(y)
  net.train()
  for _ in range(steps):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(net(xt), yt)
    loss.backward()
    opt.step()
  net.eval()
  with torch.no_grad():
    pred = net(xt).argmax(-1).numpy()
  acc = float((pred == y).mean())
  return net, mu.astype(np.float32), sd.astype(np.float32), acc


def save_calibrator(path: Path, net: InferCalibrator, mu, sd, labels, acc):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "state": net.state_dict(),
      "mu": mu,
      "sd": sd,
      "n_in": int(net.fc.in_features),
      "n_out": int(net.fc.out_features),
      "labels": list(labels),
      "train_acc": acc,
    },
    path,
  )


def load_calibrator(path: Path, device: str = "cpu"):
  try:
    blob = torch.load(path, map_location=device, weights_only=False)
  except TypeError:
    blob = torch.load(path, map_location=device)
  net = InferCalibrator(n_in=int(blob["n_in"]), n_out=int(blob["n_out"]))
  net.load_state_dict(blob["state"])
  net.eval()
  return net, np.asarray(blob["mu"], np.float32), np.asarray(blob["sd"], np.float32), blob


def apply_calibrator(net, mu, sd, feat: np.ndarray) -> np.ndarray:
  x = np.asarray(feat, np.float32).reshape(-1)
  mu = np.asarray(mu, np.float32).reshape(-1)
  sd = np.asarray(sd, np.float32).reshape(-1)
  if x.size != mu.size or x.size != sd.size:
    raise ValueError(f"calibrator feat {x.size} vs mu {mu.size}")
  x = (x - mu) / sd
  with torch.no_grad():
    logits = net(torch.from_numpy(x).unsqueeze(0))[0]
    return F.softmax(logits, -1).numpy().astype(np.float32)


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", type=Path, default=Path("artifacts/gesture_edge_soli/best_gesture_edge.pt"))
  p.add_argument("--bgt-data", type=Path, default=Path("artifacts/gesture_edge_bgt"))
  p.add_argument("--out", type=Path, default=Path("artifacts/gesture_edge_bgt/infer_calibrate.pt"))
  p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  return p.parse_args()


def main():
  args = parse_args()
  model, labels, _cfg = load_ckpt(args.ckpt, args.device)
  x, y = collect_xy(model, args.bgt_data, args.device, labels)
  net, mu, sd, acc = train_calibrator(x, y)
  xn = (x - mu) / sd
  with torch.no_grad():
    pred = net(torch.from_numpy(xn)).argmax(-1).numpy()
  print("labels", labels)
  print("true", y)
  print("pred", pred)
  print(f"train_acc={acc:.3f} n={len(y)}")
  save_calibrator(args.out, net, mu, sd, labels, acc)
  print("saved", args.out)


if __name__ == "__main__":
  main()
