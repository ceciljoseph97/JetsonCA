#!/usr/bin/env python3
"""Train gestureEdge (Soli CNN+LSTM + simple dual cross-attn).

  python gestureEdge/train.py --data ../../5_data/Gesture/SoliData.zip --epochs 30 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from gestureEdge.dataset import SoliGestureDataset, index_soli, split_by_session
from gestureEdge.model import GestureEdgeNet
from gestureEdge.preprocess import SOLI_LABELS, soli_name


def _acc(logits, labels):
  return float((logits.argmax(-1) == labels).float().mean().item())


def _gesture_counts(refs):
  return {soli_name(g): n for g, n in sorted(Counter(r.gesture for r in refs).items())}


@torch.no_grad()
def evaluate(model, loader, device):
  model.eval()
  loss_sum = acc_sum = n = 0.0
  for batch in loader:
    r1 = batch["radar1"].to(device)
    r2 = batch["radar2"].to(device)
    y = torch.as_tensor(batch["label"], device=device, dtype=torch.long)
    out = model(r1, r2)
    loss = F.cross_entropy(out["logits"], y)
    b = y.shape[0]
    loss_sum += float(loss.item()) * b
    acc_sum += _acc(out["logits"], y) * b
    n += b
  return {"loss": loss_sum / max(n, 1), "acc": acc_sum / max(n, 1)}


def train_epoch(model, loader, optim, device):
  model.train()
  loss_sum = acc_sum = n = 0.0
  for batch in loader:
    r1 = batch["radar1"].to(device)
    r2 = batch["radar2"].to(device)
    y = torch.as_tensor(batch["label"], device=device, dtype=torch.long)
    p1 = torch.as_tensor(batch["radar1_present"], device=device, dtype=torch.bool)
    p2 = torch.as_tensor(batch["radar2_present"], device=device, dtype=torch.bool)
    optim.zero_grad(set_to_none=True)
    out = model(r1, r2, radar1_present=p1, radar2_present=p2)
    loss = F.cross_entropy(out["logits"], y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    b = y.shape[0]
    loss_sum += float(loss.item()) * b
    acc_sum += _acc(out["logits"].detach(), y) * b
    n += b
  return {"loss": loss_sum / max(n, 1), "acc": acc_sum / max(n, 1)}


def parse_args():
  p = argparse.ArgumentParser(description="gestureEdge Soli train")
  p.add_argument(
    "--data",
    type=Path,
    default=Path(__file__).resolve().parents[3] / "5_data" / "Gesture" / "SoliData.zip",
  )
  p.add_argument("--out", type=Path, default=Path("artifacts/gesture_edge_soli"))
  p.add_argument("--epochs", type=int, default=30)
  p.add_argument("--batch-size", type=int, default=16)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--window", type=int, default=40)
  p.add_argument("--cnn-features", type=int, default=256)
  p.add_argument("--lstm-hidden", type=int, default=256)
  p.add_argument("--modality-dropout", type=float, default=0.2)
  p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--num-workers", type=int, default=0)
  p.add_argument("--val-ratio", type=float, default=0.2)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--max-train", type=int, default=None)
  p.add_argument("--max-val", type=int, default=None)
  p.add_argument("--patience", type=int, default=8)
  return p.parse_args()


def main():
  args = parse_args()
  torch.manual_seed(args.seed)
  args.out.mkdir(parents=True, exist_ok=True)

  refs = index_soli(args.data, max_label=len(SOLI_LABELS) - 1)
  train_refs, val_refs = split_by_session(refs, val_ratio=args.val_ratio, seed=args.seed)

  def _cap(refs_list, n, seed):
    if not n or n >= len(refs_list):
      return refs_list
    # Stratified cap so --max-train isn't all gesture 0 (sorted zip order).
    by_g: dict[int, list] = {}
    for r in refs_list:
      by_g.setdefault(r.gesture, []).append(r)
    rng = __import__("numpy").random.default_rng(seed)
    out = []
    gests = sorted(by_g)
    i = 0
    while len(out) < n and gests:
      g = gests[i % len(gests)]
      bucket = by_g[g]
      if not bucket:
        gests = [x for x in gests if by_g[x]]
        continue
      j = int(rng.integers(0, len(bucket)))
      out.append(bucket.pop(j))
      i += 1
    return out

  train_refs = _cap(train_refs, args.max_train, args.seed)
  val_refs = _cap(val_refs, args.max_val, args.seed + 1)

  train_ds = SoliGestureDataset(args.data, train_refs, window=args.window, train=True, seed=args.seed)
  val_ds = SoliGestureDataset(args.data, val_refs, window=args.window, train=False, seed=args.seed + 1)
  train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
  val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

  model = GestureEdgeNet(
    num_classes=len(SOLI_LABELS),
    cnn_features=args.cnn_features,
    lstm_hidden=args.lstm_hidden,
    modality_dropout=args.modality_dropout,
  ).to(args.device)
  optim = torch.optim.Adam(model.parameters(), lr=args.lr)
  sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)

  print(
    f"gestureEdge Soli: train={len(train_ds)} val={len(val_ds)} "
    f"classes={len(SOLI_LABELS)} window={args.window} device={args.device}"
  )
  print("  labels:", list(SOLI_LABELS))
  print("  train gest:", _gesture_counts(train_refs))
  print("  val gest:", _gesture_counts(val_refs))

  best = -1.0
  bad = 0
  history = []
  best_path = args.out / "best_gesture_edge.pt"

  for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    tr = train_epoch(model, train_loader, optim, args.device)
    va = evaluate(model, val_loader, args.device)
    sched.step()
    dt = time.time() - t0
    history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}})
    print(f"epoch={epoch:02d} loss={tr['loss']:.4f}/{va['loss']:.4f} acc={tr['acc']:.3f}/{va['acc']:.3f} ({dt:.1f}s)")
    if va["acc"] > best + 1e-4:
      best = va["acc"]
      bad = 0
      torch.save(
        {
          "model_state": model.state_dict(),
          "labels": list(SOLI_LABELS),
          "config": {
            "num_classes": len(SOLI_LABELS),
            "cnn_features": args.cnn_features,
            "lstm_hidden": args.lstm_hidden,
            "window": args.window,
            "in_channels": 3,
            "modality_dropout": args.modality_dropout,
            "dataset": "deepsoli_dsp",
            "arch": "cnn_lstm_cross_dual",
          },
          "val_acc": best,
          "epoch": epoch,
        },
        best_path,
      )
      print(f"  -> saved {best_path} (val_acc={best:.3f})")
    else:
      bad += 1
      if bad >= args.patience:
        print(f"Early stop @ {epoch} best={best:.3f}")
        break

  (args.out / "labels.json").write_text(json.dumps(list(SOLI_LABELS), indent=2), encoding="utf-8")
  (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
  train_ds.close()
  val_ds.close()
  print(f"Done best={best:.3f} -> {best_path}")


if __name__ == "__main__":
  main()
