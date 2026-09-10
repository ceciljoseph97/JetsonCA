"""Checkpoint load for gestureEdge (GestureEdgeNet dict; optional 4uf04eG pickle)."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from .fmcw_upstream import FMCW_7_LABELS, UpstreamFmcwAdapter
from .model import GestureEdgeNet
from .preprocess import normalize_label_list

_MISSING = object()


@contextmanager
def _unpickle_aliases():
  import gestureEdge.fmcw_upstream as up

  saved_model = sys.modules.get("model")
  main = sys.modules["__main__"]
  names = ("FeatureExtractor", "GestureNet", "FinetuneGestureNet")
  prev_main = {n: getattr(main, n, _MISSING) for n in names}
  sys.modules["model"] = up
  for n in names:
    setattr(main, n, getattr(up, n))
  try:
    yield
  finally:
    if saved_model is None:
      sys.modules.pop("model", None)
    else:
      sys.modules["model"] = saved_model
    for n, old in prev_main.items():
      if old is _MISSING:
        try:
          delattr(main, n)
        except AttributeError:
          pass
      else:
        setattr(main, n, old)


def _torch_load(path: Path, device: str):
  kwargs = {"map_location": device}
  with _unpickle_aliases():
    try:
      return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
      return torch.load(path, **kwargs)


def load_ckpt(path: Path, device: str):
  path = Path(path)
  obj = _torch_load(path, device)
  if isinstance(obj, torch.nn.Module):
    obj = obj.to(device).eval()
    model = UpstreamFmcwAdapter(obj).to(device).eval()
    n = int(model.num_classes)
    labels = list(FMCW_7_LABELS) if n == len(FMCW_7_LABELS) else [str(i) for i in range(n)]
    cfg = {
      "backend": "fmcw_upstream",
      "arch": "4uf04eG_FinetuneGestureNet",
      "in_channels": 3,
      "window": 12,
    }
    return model, labels, cfg
  if not isinstance(obj, dict) or "model_state" not in obj:
    raise ValueError(f"Unrecognized gestureEdge checkpoint {path}")
  labels = normalize_label_list(obj.get("labels"))
  cfg = dict(obj.get("config") or {})
  cfg.setdefault("backend", "gesture_edge")
  cfg.setdefault("arch", "cnn_lstm_cross_dual")
  cfg.setdefault("use_dsp_gate", str(cfg.get("task") or "") != "drive_bgt_finetune")
  model = GestureEdgeNet(
    num_classes=len(labels),
    in_channels=int(cfg.get("in_channels", 3)),
    cnn_features=int(cfg.get("cnn_features", 256)),
    lstm_hidden=int(cfg.get("lstm_hidden", 256)),
    modality_dropout=0.0,
  ).to(device)
  model.load_state_dict(obj["model_state"], strict=True)
  model.eval()
  return model, labels, cfg
