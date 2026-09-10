#!/usr/bin/env python3
"""Train gestureEdge (Soli CNN+LSTM + simple dual cross-attn).

  # 3-class drive (Push / Pull / Palm Hold) — default
  python gestureEdge/train.py --data ../../5_data/Gesture/SoliData.zip --epochs 30 --device cuda

  # all 11 Soli gestures
  python gestureEdge/train.py --all-soli --epochs 30 --device cuda

  # finetune on live BGT clips (after collect_drive / GUI Collect tab)
  python gestureEdge/train.py --finetune artifacts/gesture_edge_soli/best_gesture_edge.pt \\
    --bgt-data artifacts/gesture_edge_bgt --epochs 20 --lr 1e-4 --freeze-cnn \\
    --out artifacts/gesture_edge_bgt --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from gestureEdge.dataset import (
  BgtDriveDataset,
  SoliGestureDataset,
  index_bgt_drive,
  index_soli,
  split_bgt_files,
  split_by_session,
)
from gestureEdge.model import GestureEdgeNet
from gestureEdge.preprocess import DRIVE_ID_TO_CLASS, DRIVE_LABELS, DRIVE_SOLI_IDS, SOLI_LABELS, soli_name


def _acc(logits, labels):
  return float((logits.argmax(-1) == labels).float().mean().item())


def _gesture_counts(refs, label_map=None):
  names = {}
  for r in refs:
    gid = int(label_map[r.gesture]) if label_map else int(r.gesture)
    key = soli_name(r.gesture) if not label_map else (
      DRIVE_LABELS[gid] if gid < len(DRIVE_LABELS) else str(gid)
    )
    names[key] = names.get(key, 0) + 1
  return dict(sorted(names.items()))


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
  p.add_argument(
    "--all-soli",
    action="store_true",
    help="Train all 11 Soli gestures instead of Push/Pull/Palm Hold",
  )
  p.add_argument(
    "--finetune",
    type=Path,
    default=None,
    help="Soli 3-class ckpt to finetune on live BGT clips",
  )
  p.add_argument(
    "--bgt-data",
    type=Path,
    default=Path("artifacts/gesture_edge_bgt"),
    help="Folder of Push/Pull/Palm_Hold *.npz from collect_drive / GUI Collect",
  )
  p.add_argument(
    "--freeze-cnn",
    action="store_true",
    help="Freeze FrameCNN (recommended for small BGT sets)",
  )
  p.add_argument("--no-freeze-cnn", action="store_true")
  return p.parse_args()


def _save_ckpt(path: Path, model, labels, cfg, val_acc, epoch):
  torch.save(
    {
      "model_state": model.state_dict(),
      "labels": labels,
      "config": cfg,
      "val_acc": val_acc,
      "epoch": epoch,
    },
    path,
  )


def _loop(model, train_loader, val_loader, optim, sched, args, labels, cfg, best_path):
  best = -1.0
  bad = 0
  history = []
  for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    tr = train_epoch(model, train_loader, optim, args.device)
    va = evaluate(model, val_loader, args.device)
    sched.step()
    dt = time.time() - t0
    history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}})
    print(f"epoch={epoch:02d} loss={tr['loss']:.4f}/{va['loss']:.4f} acc={tr['acc']:.3f}/{va['acc']:.3f} ({dt:.1f}s)", flush=True)
    if va["acc"] > best + 1e-4:
      best = va["acc"]
      bad = 0
      _save_ckpt(best_path, model, labels, cfg, best, epoch)
      print(f"  -> saved {best_path} (val_acc={best:.3f})")
    else:
      bad += 1
      if bad >= args.patience:
        print(f"Early stop @ {epoch} best={best:.3f}")
        break
  return best, history


def run_finetune(args):
  ckpt_path = Path(args.finetune)
  if not ckpt_path.exists():
    raise SystemExit(f"Missing finetune ckpt {ckpt_path}")
  try:
    blob = torch.load(ckpt_path, map_location=args.device, weights_only=False)
  except TypeError:
    blob = torch.load(ckpt_path, map_location=args.device)
  labels = list(blob.get("labels") or DRIVE_LABELS)
  src_cfg = dict(blob.get("config") or {})
  window = int(src_cfg.get("window", args.window) or args.window)
  files = index_bgt_drive(args.bgt_data)
  by: dict[int, int] = {}
  for _, lab in files:
    by[int(lab)] = by.get(int(lab), 0) + 1
  print(f"BGT clips: {len(files)}  per-class={ {DRIVE_LABELS[k]: v for k, v in sorted(by.items())} }", flush=True)
  missing = [DRIVE_LABELS[i] for i in range(len(DRIVE_LABELS)) if by.get(i, 0) == 0]
  if missing:
    raise SystemExit(f"Need clips for all three classes; missing {missing} under {args.bgt_data}")
  train_files, val_files = split_bgt_files(files, val_ratio=args.val_ratio, seed=args.seed)
  train_ds = BgtDriveDataset(train_files, window=window, train=True, seed=args.seed)
  val_ds = BgtDriveDataset(val_files, window=window, train=False, seed=args.seed + 1)
  train_loader = DataLoader(
    train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True, num_workers=args.num_workers
  )
  val_loader = DataLoader(
    val_ds, batch_size=min(args.batch_size, len(val_ds)), shuffle=False, num_workers=args.num_workers
  )

  model = GestureEdgeNet(
    num_classes=len(labels),
    cnn_features=int(src_cfg.get("cnn_features", args.cnn_features)),
    lstm_hidden=int(src_cfg.get("lstm_hidden", args.lstm_hidden)),
    modality_dropout=args.modality_dropout,
  ).to(args.device)
  model.load_state_dict(blob["model_state"], strict=True)
  freeze = not bool(args.no_freeze_cnn)
  if args.freeze_cnn:
    freeze = True
  if freeze:
    for p in model.cnn.parameters():
      p.requires_grad = False
    print("freeze-cnn=True (lstm/cross/head train)", flush=True)
  params = [p for p in model.parameters() if p.requires_grad]
  optim = torch.optim.Adam(params, lr=args.lr)
  sched = torch.optim.lr_scheduler.StepLR(optim, step_size=8, gamma=0.5)
  if "--out" not in sys.argv:
    args.out = Path("artifacts/gesture_edge_bgt")
  args.out.mkdir(parents=True, exist_ok=True)
  cfg = {
    "num_classes": len(labels),
    "cnn_features": int(src_cfg.get("cnn_features", args.cnn_features)),
    "lstm_hidden": int(src_cfg.get("lstm_hidden", args.lstm_hidden)),
    "window": window,
    "in_channels": 3,
    "modality_dropout": args.modality_dropout,
    "dataset": "bgt_drive",
    "task": "drive_bgt_finetune",
    "soli_ids": list(DRIVE_SOLI_IDS),
    "arch": "cnn_lstm_cross_dual",
    "backend": "gesture_edge",
    "use_dsp_gate": False,
    "finetune_from": str(ckpt_path),
    "freeze_cnn": freeze,
  }
  best_path = args.out / "best_gesture_edge.pt"
  print(
    f"gestureEdge drive_bgt_finetune: train={len(train_ds)} val={len(val_ds)} "
    f"window={window} lr={args.lr} device={args.device} -> {best_path}",
    flush=True,
  )
  best, history = _loop(model, train_loader, val_loader, optim, sched, args, labels, cfg, best_path)
  (args.out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
  (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
  print(f"Done best={best:.3f} -> {best_path}")
  print(f"Play with: python gui_app.py --gesture-edge --device cuda --gesture-edge-checkpoint {best_path}")


def main():
  args = parse_args()
  print(f"device={args.device} data={args.data} out={args.out}", flush=True)
  torch.manual_seed(args.seed)
  args.out.mkdir(parents=True, exist_ok=True)

  if args.finetune is not None:
    if "--lr" not in sys.argv:
      args.lr = 1e-4
    if "--patience" not in sys.argv:
      args.patience = 6
    run_finetune(args)
    return

  refs = index_soli(args.data, max_label=len(SOLI_LABELS) - 1)
  if args.all_soli:
    labels = list(SOLI_LABELS)
    label_map = None
    task = "soli_11"
  else:
    refs = [r for r in refs if r.gesture in DRIVE_SOLI_IDS]
    if not refs:
      raise SystemExit(f"No Push/Pull/Palm Hold clips in {args.data}")
    labels = list(DRIVE_LABELS)
    label_map = dict(DRIVE_ID_TO_CLASS)
    task = "drive_push_pull_hold"
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

  train_ds = SoliGestureDataset(
    args.data, train_refs, window=args.window, train=True, seed=args.seed, label_map=label_map
  )
  val_ds = SoliGestureDataset(
    args.data, val_refs, window=args.window, train=False, seed=args.seed + 1, label_map=label_map
  )
  train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
  val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

  model = GestureEdgeNet(
    num_classes=len(labels),
    cnn_features=args.cnn_features,
    lstm_hidden=args.lstm_hidden,
    modality_dropout=args.modality_dropout,
  ).to(args.device)
  optim = torch.optim.Adam(model.parameters(), lr=args.lr)
  sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)

  print(
    f"gestureEdge {task}: train={len(train_ds)} val={len(val_ds)} "
    f"classes={len(labels)} window={args.window} device={args.device}",
    flush=True,
  )
  print("  labels:", labels, flush=True)
  print("  train gest:", _gesture_counts(train_refs, label_map), flush=True)
  print("  val gest:", _gesture_counts(val_refs, label_map), flush=True)

  best_path = args.out / "best_gesture_edge.pt"
  cfg = {
    "num_classes": len(labels),
    "cnn_features": args.cnn_features,
    "lstm_hidden": args.lstm_hidden,
    "window": args.window,
    "in_channels": 3,
    "modality_dropout": args.modality_dropout,
    "dataset": "deepsoli_dsp",
    "task": task,
    "soli_ids": list(DRIVE_SOLI_IDS) if task == "drive_push_pull_hold" else list(range(len(SOLI_LABELS))),
    "arch": "cnn_lstm_cross_dual",
    "backend": "gesture_edge",
    "use_dsp_gate": True,
  }
  best, history = _loop(model, train_loader, val_loader, optim, sched, args, labels, cfg, best_path)

  (args.out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
  (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
  train_ds.close()
  val_ds.close()
  print(f"Done best={best:.3f} -> {best_path}")


if __name__ == "__main__":
  main()
