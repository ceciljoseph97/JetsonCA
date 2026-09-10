"""Live dual-BGT gestureEdge GUI — radar perception + CNN-LSTM prediction."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from gestureEdge.ckpt import load_ckpt
from gestureEdge.gui_benchmark import mount_gesture_edge_profile_tab
from gestureEdge.preprocess import live_rd_to_frame
from radar_utils import DualRadarSession, combine_sensor_panels, render_radar_panel


def _dual_radar_kwargs(args: argparse.Namespace) -> dict[str, Any]:
  """Build DualRadarSession kwargs (Crossattention vs JetsonCA signatures)."""
  import inspect

  raw = {
    "num_rx": int(getattr(args, "num_rx", 3)),
    "profile": str(getattr(args, "radar_profile", "gesture")),
    "frame_rate_hz": float(getattr(args, "frame_rate", 5.0)),
    "radar1_uuid": getattr(args, "radar1_uuid", None),
    "radar2_uuid": getattr(args, "radar2_uuid", None),
    "radar1_port": getattr(args, "radar1_port", None),
    "radar2_port": getattr(args, "radar2_port", None),
    "mirror_radar2": True,
    "min_range_m": float(getattr(args, "min_range_m", 0.0) or 0.0),
    "max_range_m": getattr(args, "max_range_m", None),
    "prefer_port": bool(getattr(args, "prefer_port", False)),
  }
  allowed = set(inspect.signature(DualRadarSession.__init__).parameters)
  return {k: v for k, v in raw.items() if k in allowed}


def _ph(text: str, w=320, h=200):
  img = np.zeros((h, w, 3), dtype=np.uint8)
  img[:] = (40, 40, 40)
  return img


class Worker:
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.device = str(args.device)
    self.model, self.labels, self.config = load_ckpt(Path(args.gesture_edge_checkpoint), self.device)
    # Prefer train window from ckpt; gui_app --window defaults to 30 (HAR), not 40.
    ckpt_win = int(self.config.get("window", 40) or 40)
    arg_win = getattr(args, "window", None)
    self.window = int(arg_win) if arg_win not in (None, 30) else ckpt_win
    self.use_r1 = True
    self.use_r2 = True
    self.stop_event = threading.Event()
    self.lock = threading.Lock()
    self.buf1: deque[torch.Tensor] = deque(maxlen=self.window)
    self.buf2: deque[torch.Tensor] = deque(maxlen=self.window)
    self.state: dict[str, Any] = {
      "status": "idle",
      "prediction": "-",
      "confidence": 0.0,
      "probs": np.zeros(len(self.labels), dtype=np.float32),
      "radar_rgb": _ph("Radar"),
      "radar_status": "-",
      "dropout": "none",
      "latency_ms": 0.0,
      "rel1": 0.5,
      "rel2": 0.5,
    }

  def set_instances(self, r1: bool, r2: bool):
    if not r1 and not r2:
      r1 = True
    with self.lock:
      self.use_r1, self.use_r2 = bool(r1), bool(r2)
      parts = []
      if not self.use_r1:
        parts.append("radar1 off")
      if not self.use_r2:
        parts.append("radar2 off")
      self.state["dropout"] = "; ".join(parts) or "none"

  def get(self):
    with self.lock:
      st = dict(self.state)
      st["probs"] = np.asarray(self.state["probs"]).copy()
      st["radar_rgb"] = np.asarray(self.state["radar_rgb"]).copy()
      return st

  def stop(self):
    self.stop_event.set()

  @torch.no_grad()
  def _infer(self):
    with self.lock:
      u1, u2 = self.use_r1, self.use_r2
    if max(len(self.buf1), len(self.buf2)) < self.window:
      return "-", 0.0, np.zeros(len(self.labels), np.float32), 0.5, 0.5, 0.0

    def stack(buf):
      frames = list(buf)
      if len(frames) < self.window:
        frames = [frames[0]] * (self.window - len(frames)) + frames
      return torch.stack(frames[-self.window :], 0)

    r1 = stack(self.buf1 if self.buf1 else self.buf2).unsqueeze(0).to(self.device)
    r2 = stack(self.buf2 if self.buf2 else self.buf1).unsqueeze(0).to(self.device)
    p1 = torch.tensor([u1 and len(self.buf1) > 0], device=self.device)
    p2 = torch.tensor([u2 and len(self.buf2) > 0], device=self.device)
    if not bool(p1 | p2):
      p1 = torch.tensor([True], device=self.device)
    t0 = time.perf_counter()
    out = self.model(r1, r2, radar1_present=p1, radar2_present=p2)
    if str(self.device).startswith("cuda") and torch.cuda.is_available():
      torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    probs = F.softmax(out["logits"][0], -1).cpu().numpy().astype(np.float32)
    i = int(probs.argmax())
    rel = out["reliance"][0].cpu().numpy()
    return self.labels[i], float(probs[i]), probs, float(rel[0]), float(rel[1]), ms

  def run(self):
    with DualRadarSession(**_dual_radar_kwargs(self.args)) as session:
      while not self.stop_event.is_set():
        t1, t2 = session.read_tensors()
        p1 = render_radar_panel(t1.numpy()) if t1 is not None else None
        p2 = render_radar_panel(t2.numpy()) if t2 is not None else None
        if t1 is not None:
          self.buf1.append(live_rd_to_frame(t1))
        if t2 is not None:
          self.buf2.append(live_rd_to_frame(t2))
        elif t1 is not None:
          self.buf2.append(live_rd_to_frame(t1))

        if p1 is not None and p2 is not None:
          rgb = combine_sensor_panels(p1, p2, cross_sensor_mode="side_by_side")
        else:
          rgb = p1 if p1 is not None else (p2 if p2 is not None else _ph("Waiting…"))

        ready = max(len(self.buf1), len(self.buf2)) >= self.window
        if ready:
          pred, conf, probs, rel1, rel2, lat = self._infer()
          status = "running"
        else:
          pred, conf, probs, rel1, rel2, lat = "-", 0.0, np.zeros(len(self.labels), np.float32), 0.5, 0.5, 0.0
          status = f"warming {max(len(self.buf1), len(self.buf2))}/{self.window}"

        with self.lock:
          self.state.update(
            status=status,
            prediction=pred,
            confidence=conf,
            probs=probs,
            radar_rgb=rgb,
            radar_status=session.status_text,
            latency_ms=lat,
            rel1=rel1,
            rel2=rel2,
          )
        time.sleep(0.02)


class GestureEdgeApp:
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.root = tk.Tk()
    self.root.title("gestureEdge — dual radar CNN+LSTM")
    self.root.geometry("1080x700")
    self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    self.worker = Worker(args)
    self.thread = None
    self.use_r1 = tk.BooleanVar(value=True)
    self.use_r2 = tk.BooleanVar(value=True)
    self.pred = tk.StringVar(value="-")
    self.conf = tk.StringVar(value="0.00")
    self.status = tk.StringVar(value="idle")
    self.radar_st = tk.StringVar(value="-")
    self.drop = tk.StringVar(value="none")
    self.lat = tk.StringVar(value="-")
    self.rel1 = tk.StringVar(value="50%")
    self.rel2 = tk.StringVar(value="50%")
    self.prob_vars = []
    self.bars = []
    self._photo = None
    self._build()
    self.root.after(100, self._tick)

  def _build(self):
    main = ttk.Frame(self.root, padding=8)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=2)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(0, weight=1)

    left = ttk.LabelFrame(main, text="Radar perception (R1 | R2)")
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    left.rowconfigure(0, weight=1)
    left.columnconfigure(0, weight=1)
    self.radar_label = ttk.Label(left)
    self.radar_label.grid(sticky="nsew", padx=4, pady=4)

    right = ttk.Frame(main)
    right.grid(row=0, column=1, sticky="nsew")
    right.rowconfigure(1, weight=1)
    right.columnconfigure(0, weight=1)

    box = ttk.LabelFrame(right, text="Gesture", padding=8)
    box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(box, textvariable=self.pred, font=("Segoe UI", 22, "bold")).pack(anchor="w")
    ttk.Label(box, textvariable=self.conf).pack(anchor="w")

    notebook = ttk.Notebook(right)
    notebook.grid(row=1, column=0, sticky="nsew")
    realtime = ttk.Frame(notebook, padding=6)
    profile = ttk.Frame(notebook, padding=6)
    notebook.add(realtime, text="Realtime")
    notebook.add(profile, text="Profile")
    self._control_notebook = notebook
    self._profile_tab = profile

    run = ttk.LabelFrame(realtime, text="Run", padding=6)
    run.pack(fill="x", pady=(0, 8))
    ttk.Label(run, text=str(self.args.gesture_edge_checkpoint), wraplength=300).pack(anchor="w")
    row = ttk.Frame(run)
    row.pack(anchor="w", pady=4)
    ttk.Button(row, text="Start", command=self.start).pack(side="left", padx=(0, 6))
    ttk.Button(row, text="Stop", command=self.stop).pack(side="left", padx=(0, 6))
    ttk.Button(row, text="Benchmark", command=self._run_gui_benchmark).pack(side="left")

    abl = ttk.LabelFrame(realtime, text="Radar dropout", padding=6)
    abl.pack(fill="x", pady=(0, 8))
    ttk.Checkbutton(abl, text="Use Radar 1", variable=self.use_r1, command=self._ablate).pack(anchor="w")
    ttk.Checkbutton(abl, text="Use Radar 2", variable=self.use_r2, command=self._ablate).pack(anchor="w")
    ttk.Label(abl, textvariable=self.drop).pack(anchor="w")

    meta = ttk.LabelFrame(realtime, text="Status", padding=6)
    meta.pack(fill="x", pady=(0, 8))
    for lab, var in (
      ("Run", self.status),
      ("Radar", self.radar_st),
      ("Latency", self.lat),
      ("R1", self.rel1),
      ("R2", self.rel2),
    ):
      f = ttk.Frame(meta)
      f.pack(fill="x")
      ttk.Label(f, text=lab, width=8).pack(side="left")
      ttk.Label(f, textvariable=var, wraplength=240).pack(side="left")

    probs = ttk.LabelFrame(realtime, text="Class probs", padding=6)
    probs.pack(fill="both", expand=True)
    for name in self.worker.labels:
      var = tk.StringVar(value=f"{name}: 0.00")
      self.prob_vars.append(var)
      ttk.Label(probs, textvariable=var).pack(anchor="w")
      bar = ttk.Progressbar(probs, maximum=100, length=200)
      bar.pack(fill="x", pady=1)
      self.bars.append(bar)

    self._profile_run = mount_gesture_edge_profile_tab(
      profile,
      defaults_fn=self._benchmark_defaults,
      busy_fn=lambda: self.thread is not None and self.thread.is_alive(),
      ui_after=self.root.after,
    )

  def _benchmark_defaults(self) -> dict[str, Any]:
    return {
      "checkpoint": Path(self.args.gesture_edge_checkpoint),
      "device": str(self.args.device),
      "window": int(self.worker.window),
      "out": Path("artifacts/benchmark_gesture_edge.json"),
    }

  def _run_gui_benchmark(self):
    notebook = getattr(self, "_control_notebook", None)
    tab = getattr(self, "_profile_tab", None)
    if notebook is not None and tab is not None:
      notebook.select(tab)
    run = getattr(self, "_profile_run", None)
    if run is not None:
      run()

  def _ablate(self):
    self.worker.set_instances(self.use_r1.get(), self.use_r2.get())
    self.drop.set(self.worker.get()["dropout"])

  def start(self):
    if self.thread and self.thread.is_alive():
      return
    self.worker.stop_event.clear()
    self._ablate()
    self.thread = threading.Thread(target=self.worker.run, daemon=True)
    self.thread.start()

  def stop(self):
    self.worker.stop()

  def _tick(self):
    st = self.worker.get()
    self.pred.set(str(st["prediction"]))
    self.conf.set(f"{float(st['confidence']):.2f}")
    self.status.set(str(st["status"]))
    self.radar_st.set(str(st["radar_status"]))
    self.drop.set(str(st["dropout"]))
    self.lat.set(f"{float(st['latency_ms']):.1f} ms")
    self.rel1.set(f"{float(st['rel1'])*100:.0f}%")
    self.rel2.set(f"{float(st['rel2'])*100:.0f}%")
    probs = np.asarray(st["probs"], np.float32)
    for i, (var, bar) in enumerate(zip(self.prob_vars, self.bars)):
      p = float(probs[i]) if i < probs.size else 0.0
      var.set(f"{self.worker.labels[i]}: {p:.2f}")
      bar["value"] = p * 100
    rgb = np.asarray(st["radar_rgb"])
    if rgb.ndim == 3:
      w = max(320, self.radar_label.winfo_width())
      h = max(200, self.radar_label.winfo_height())
      self._photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((w, h)))
      self.radar_label.configure(image=self._photo)
    self.root.after(100, self._tick)

  def on_close(self):
    self.stop()
    self.root.destroy()

  def run(self):
    self.root.mainloop()


def run_gesture_edge_gui(args: argparse.Namespace):
  ckpt = Path(args.gesture_edge_checkpoint)
  if not ckpt.exists():
    raise SystemExit(
      f"Missing {ckpt}\nTrain: python gestureEdge/train.py --data ../../5_data/Gesture/SoliData.zip"
    )
  if getattr(args, "radar_profile", "safe") == "safe":
    args.radar_profile = "gesture"
  GestureEdgeApp(args).run()
