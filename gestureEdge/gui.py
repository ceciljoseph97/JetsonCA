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

from gestureEdge.calibrate_infer import apply_calibrator, feat_from_clip, load_calibrator
from gestureEdge.car_game import CarDriveGame
from gestureEdge.ckpt import load_ckpt
from gestureEdge.collect_drive import COLLECT_LABELS, count_clips, save_clip
from gestureEdge.drive_dsp import (
  _FLICK_LATCH,
  _HIST_N,
  _HOLD_VOTES,
  _PULL_FRAME,
  _PULL_VOTES,
  _PUSH_FRAME,
  _PUSH_VOTES,
  doppler_centroid,
  dual_imbalance,
  energy_ok,
  hand_present,
  honk_cue,
  range_peak,
)
from gestureEdge.fmcw_upstream import ActionDebouncer
from gestureEdge.gui_benchmark import mount_gesture_edge_profile_tab
from gestureEdge.preprocess import live_rd_to_frame
from radar_utils import DualRadarSession, combine_sensor_panels, render_radar_panel


def _norm_lab(name: str) -> str:
  return str(name).strip().lower().replace("-", " ").replace("_", " ")


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


def _apply_window_geometry(root: tk.Tk, args: argparse.Namespace):
  """Fit small / headless displays. Drag-resize always on."""
  root.resizable(True, True)
  root.minsize(400, 280)
  root.update_idletasks()
  sw = int(root.winfo_screenwidth() or 800)
  sh = int(root.winfo_screenheight() or 480)
  geo = str(getattr(args, "gui_geometry", None) or "").strip()
  if geo:
    root.geometry(geo)
    return
  w = max(400, min(960, sw - 16))
  h = max(280, min(720, sh - 48))
  root.geometry(f"{w}x{h}+0+0")


def _make_scrollable(parent: ttk.Frame) -> ttk.Frame:
  wrap = ttk.Frame(parent)
  wrap.pack(fill="both", expand=True)
  canvas = tk.Canvas(wrap, highlightthickness=0, bd=0)
  vsb = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
  inner = ttk.Frame(canvas, padding=6)
  inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")
  canvas.configure(yscrollcommand=vsb.set)

  def _inner_cfg(_e=None):
    canvas.configure(scrollregion=canvas.bbox("all"))

  def _canvas_cfg(e):
    canvas.itemconfigure(inner_id, width=e.width)

  inner.bind("<Configure>", _inner_cfg)
  canvas.bind("<Configure>", _canvas_cfg)

  def _on_wheel(e):
    delta = int(getattr(e, "delta", 0) or 0)
    if delta:
      canvas.yview_scroll(int(-1 * (delta / 120)), "units")
    elif getattr(e, "num", None) == 4:
      canvas.yview_scroll(-1, "units")
    elif getattr(e, "num", None) == 5:
      canvas.yview_scroll(1, "units")

  def _bind(_e=None):
    canvas.bind_all("<MouseWheel>", _on_wheel)
    canvas.bind_all("<Button-4>", _on_wheel)
    canvas.bind_all("<Button-5>", _on_wheel)

  def _unbind(_e=None):
    canvas.unbind_all("<MouseWheel>")
    canvas.unbind_all("<Button-4>")
    canvas.unbind_all("<Button-5>")

  canvas.bind("<Enter>", _bind)
  canvas.bind("<Leave>", _unbind)
  canvas.pack(side="left", fill="both", expand=True)
  vsb.pack(side="right", fill="y")
  return inner


class Worker:
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.device = str(args.device)
    self.model, self.labels, self.config = load_ckpt(Path(args.gesture_edge_checkpoint), self.device)
    # Live window = train window. Ignore HAR gui_app --window 30.
    self.window = max(1, int(self.config.get("window", 40) or 40))
    self.backend = str(self.config.get("backend") or "gesture_edge")
    self.debouncer = ActionDebouncer() if self.config.get("debounce") else None
    self.use_dsp_gate = bool(self.config.get("use_dsp_gate", True))
    idle = "no-action" if "no-action" in self.labels else "-"
    self._stable_pred = idle
    self._hold_idx = next((i for i, n in enumerate(self.labels) if "hold" in str(n).lower()), None)
    self._push_idx = next((i for i, n in enumerate(self.labels) if _norm_lab(n) == "push"), None)
    self._pull_idx = next((i for i, n in enumerate(self.labels) if _norm_lab(n) == "pull"), None)
    self._motion_ema = 0.0
    self._still_n = 0
    self._latch_idx: int | None = None
    self._cent_hist: deque[float] = deque(maxlen=_HIST_N)
    self._range_hist: deque[float] = deque(maxlen=_HIST_N)
    self._flick_idx: int | None = None
    self._flick_left = 0
    self.drive_need = 8 if (self._push_idx is not None and self._pull_idx is not None and self._hold_idx is not None) else self.window
    self.use_r1 = True
    self.use_r2 = True
    self.stop_event = threading.Event()
    self.lock = threading.Lock()
    self.buf1: deque[torch.Tensor] = deque(maxlen=self.window)
    self.buf2: deque[torch.Tensor] = deque(maxlen=self.window)
    self._rec_need = 0
    self._rec1: list[torch.Tensor] = []
    self._rec2: list[torch.Tensor] = []
    self._rec_ready: tuple[torch.Tensor, torch.Tensor] | None = None
    self._cal = None
    self._cal_mu = None
    self._cal_sd = None
    cal_path = Path(getattr(args, "gesture_edge_bgt_data", None) or "artifacts/gesture_edge_bgt") / "infer_calibrate.pt"
    if cal_path.is_file() and self._hold_idx is not None:
      try:
        self._cal, self._cal_mu, self._cal_sd, meta = load_calibrator(cal_path, "cpu")
        self.backend = f"{self.backend}+cal"
        print(f"infer calibrator {cal_path} acc={meta.get('train_acc')}", flush=True)
      except Exception as exc:
        print(f"infer calibrator skipped: {exc}", flush=True)
    self.state: dict[str, Any] = {
      "status": "idle",
      "prediction": "-",
      "confidence": 0.0,
      "probs": np.zeros(len(self.labels), dtype=np.float32),
      "raw_prediction": "-",
      "rel1": 0.5,
      "rel2": 0.5,
      "latency_ms": 0.0,
      "drive_pred": "-",
      "drive_conf": 0.0,
      "drive_probs": np.zeros(len(self.labels), dtype=np.float32),
      "radar_rgb": _ph("Radar"),
      "radar_status": "-",
      "dropout": "none",
      "backend": self.backend,
      "ready": False,
      "drive_ready": False,
      "window_fill": 0,
      "window": self.window,
      "moving": False,
      "doppler": 0.0,
      "honk": False,
      "record": "idle",
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
      st["drive_probs"] = np.asarray(self.state["drive_probs"]).copy()
      st["radar_rgb"] = np.asarray(self.state["radar_rgb"]).copy()
      return st

  def is_running(self) -> bool:
    return bool(self.thread_alive) if hasattr(self, "thread_alive") else (not self.stop_event.is_set() and self.state.get("status") not in ("idle",))

  def clear_window(self):
    with self.lock:
      self.buf1.clear()
      self.buf2.clear()
      self.state["ready"] = False
      self.state["window_fill"] = 0
      self.state["status"] = "calibrating 0/{0}".format(self.window)
      self.state["prediction"] = "-"
      self.state["raw_prediction"] = "-"
      self.state["probs"] = np.zeros(len(self.labels), dtype=np.float32)
      self.state["drive_pred"] = "-"
      self.state["drive_probs"] = np.zeros(len(self.labels), dtype=np.float32)
      self.state["drive_ready"] = False
      self.state["confidence"] = 0.0
      self.state["drive_conf"] = 0.0
      self.state["honk"] = False
    self._still_n = 0
    self._latch_idx = None
    self._motion_ema = 0.0
    self._cent_hist.clear()
    self._range_hist.clear()
    self._flick_idx = None
    self._flick_left = 0

  def begin_record(self, n: int):
    with self.lock:
      self._rec1 = []
      self._rec2 = []
      self._rec_need = max(1, int(n))
      self._rec_ready = None
      self.state["record"] = f"recording 0/{self._rec_need}"

  def cancel_record(self):
    with self.lock:
      self._rec_need = 0
      self._rec_ready = None
      self._rec1 = []
      self._rec2 = []
      self.state["record"] = "idle"

  def pop_record(self) -> tuple[torch.Tensor, torch.Tensor] | None:
    with self.lock:
      out = self._rec_ready
      if out is not None:
        self._rec_ready = None
        self.state["record"] = "idle"
      return out

  def _push_record(self, f1: torch.Tensor, f2: torch.Tensor):
    if self._rec_need <= 0:
      return
    self._rec1.append(f1.detach().cpu())
    self._rec2.append(f2.detach().cpu())
    n = len(self._rec1)
    self.state["record"] = f"recording {n}/{self._rec_need}"
    if n >= self._rec_need:
      self._rec_ready = (torch.stack(self._rec1[: self._rec_need], 0), torch.stack(self._rec2[: self._rec_need], 0))
      self._rec_need = 0
      self.state["record"] = "saved"

  def _stack_window(self, buf: deque[torch.Tensor]) -> torch.Tensor:
    frames = list(buf)
    if len(frames) < self.window:
      raise RuntimeError("window not full")
    return torch.stack(frames[-self.window :], 0)

  def stop(self):
    self.stop_event.set()

  @torch.no_grad()
  def _infer(self):
    with self.lock:
      u1, u2 = self.use_r1, self.use_r2
    n1, n2 = len(self.buf1), len(self.buf2)
    if max(n1, n2) < self.window:
      return "-", 0.0, np.zeros(len(self.labels), np.float32), 0.5, 0.5, 0.0, "-"

    src1 = self.buf1 if n1 >= self.window else self.buf2
    src2 = self.buf2 if n2 >= self.window else self.buf1
    r1 = self._stack_window(src1).unsqueeze(0).to(self.device)
    r2 = self._stack_window(src2).unsqueeze(0).to(self.device)
    p1 = torch.tensor([u1 and n1 > 0], device=self.device)
    p2 = torch.tensor([u2 and n2 > 0], device=self.device)
    if not bool(p1 | p2):
      p1 = torch.tensor([True], device=self.device)
    t0 = time.perf_counter()
    out = self.model(r1, r2, radar1_present=p1, radar2_present=p2)
    if str(self.device).startswith("cuda") and torch.cuda.is_available():
      torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    logits = out["logits"][0].detach().float().cpu()
    raw_probs = F.softmax(logits, -1).numpy().astype(np.float32)
    raw = self.labels[int(raw_probs.argmax())]
    probs = raw_probs
    if self._cal is not None:
      r1n = self._stack_window(src1).numpy()
      feat = feat_from_clip(logits, r1n)
      cal_p = apply_calibrator(self._cal, self._cal_mu, self._cal_sd, feat)
      if cal_p.size == probs.size:
        probs = cal_p
    i = int(probs.argmax())
    pred = self.labels[i]
    if self.debouncer is not None:
      hit = self.debouncer.update(probs)
      if hit is None:
        pred = self._stable_pred
      else:
        pred = self.labels[int(hit)]
        self._stable_pred = pred
    try:
      conf = float(probs[self.labels.index(pred)])
    except ValueError:
      conf = float(probs[i])
    rel = out["reliance"][0].cpu().numpy()
    return pred, conf, probs, float(rel[0]), float(rel[1]), ms, raw

  def _onehot(self, idx: int) -> np.ndarray:
    p = np.zeros(len(self.labels), dtype=np.float32)
    p[int(idx)] = 1.0
    return p

  def _apply_drive_dsp(self, frame_chw, frame_chw2=None) -> tuple:
    """Short-window votes. Default idle. Hold only if mid-band majority. Tilt→Honk."""
    c = doppler_centroid(np.asarray(frame_chw.detach().cpu() if hasattr(frame_chw, "detach") else frame_chw))
    self._cent_hist.append(c)
    self._range_hist.append(range_peak(frame_chw))
    cs = np.asarray(self._cent_hist, dtype=np.float32)
    n_push = int((cs <= _PUSH_FRAME).sum())
    n_pull = int((cs >= _PULL_FRAME).sum())
    n_mid = int(((cs > _PUSH_FRAME) & (cs < _PULL_FRAME)).sum())
    energy = energy_ok(frame_chw)
    present = hand_present(frame_chw)
    imb = dual_imbalance(frame_chw, frame_chw2) if frame_chw2 is not None else 0.0

    def emit(idx: int, moving_flag: bool):
      name = self.labels[int(idx)]
      return name, 1.0, self._onehot(idx), moving_flag, c, False

    def emit_idle():
      zeros = np.zeros(len(self.labels), dtype=np.float32)
      return "-", 0.0, zeros, False, c, False

    def emit_honk():
      zeros = np.zeros(len(self.labels), dtype=np.float32)
      return "Palm Tilting", 1.0, zeros, False, c, True

    if energy and n_push >= _PUSH_VOTES and n_push >= n_pull and self._push_idx is not None:
      self._flick_idx = self._push_idx
      self._flick_left = _FLICK_LATCH
      return emit(self._push_idx, True)
    if energy and n_pull >= _PULL_VOTES and self._pull_idx is not None:
      self._flick_idx = self._pull_idx
      self._flick_left = _FLICK_LATCH
      return emit(self._pull_idx, True)
    if self._flick_left > 0 and self._flick_idx is not None:
      self._flick_left -= 1
      return emit(self._flick_idx, True)

    if honk_cue(self._range_hist, n_mid=n_mid, present=present, imbalance=imb):
      self._flick_idx = None
      return emit_honk()

    if (
      present
      and self._hold_idx is not None
      and n_mid >= _HOLD_VOTES
      and len(cs) >= _HOLD_VOTES
    ):
      self._flick_idx = None
      return emit(self._hold_idx, False)

    self._flick_idx = None
    return emit_idle()

  def run(self):
    self.clear_window()
    with DualRadarSession(**_dual_radar_kwargs(self.args)) as session:
      while not self.stop_event.is_set():
        t1, t2 = session.read_tensors()
        fresh = t1 is not None and int(session._miss_streak[0]) == 0
        new_frame = fresh
        p1 = render_radar_panel(t1.numpy()) if t1 is not None else None
        p2 = render_radar_panel(t2.numpy()) if t2 is not None else None
        if fresh:
          f1 = live_rd_to_frame(t1)
          if t2 is not None and int(session._miss_streak[1]) == 0:
            f2 = live_rd_to_frame(t2)
          else:
            f2 = f1
          self.buf1.append(f1)
          self.buf2.append(f2)
          with self.lock:
            self._push_record(f1, f2)

        if p1 is not None and p2 is not None:
          rgb = combine_sensor_panels(p1, p2, cross_sensor_mode="side_by_side")
        elif p1 is not None or p2 is not None:
          rgb = p1 if p1 is not None else p2
        else:
          rgb = None

        nfill = max(len(self.buf1), len(self.buf2))
        net_ready = nfill >= self.window
        drive_ok = nfill >= 3
        drive_ready = nfill >= self.drive_need
        status = "running" if nfill else "idle"

        upd: dict[str, Any] = {
          "status": status,
          "radar_status": session.status_text,
          "ready": net_ready,
          "drive_ready": drive_ready,
          "window_fill": nfill,
          "window": self.window,
        }
        if rgb is not None:
          upd["radar_rgb"] = rgb

        if new_frame and drive_ok:
          d_pred, d_conf, d_probs, moving, doppler, honk = self._apply_drive_dsp(f1, f2)
          upd.update(
            drive_pred=d_pred,
            drive_conf=d_conf,
            drive_probs=d_probs,
            moving=moving,
            doppler=doppler,
            honk=honk,
            prediction=d_pred,
            confidence=d_conf,
            probs=d_probs,
          )
        if new_frame and net_ready:
          _pred, _conf, _probs, rel1, rel2, lat, raw = self._infer()
          upd.update(
            raw_prediction=raw,
            rel1=rel1,
            rel2=rel2,
            latency_ms=lat,
          )

        with self.lock:
          self.state.update(upd)
        if not new_frame:
          time.sleep(0.01)


class GestureEdgeApp:
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.worker = Worker(args)
    self.root = tk.Tk()
    backend = str(self.worker.config.get("backend") or "gesture_edge")
    profile = str(getattr(args, "radar_profile", "safe"))
    if backend == "fmcw_upstream":
      self.root.title(f"gestureEdge — 4uf04eG override · radar={profile}")
    else:
      self.root.title("gestureEdge — dual radar CNN+LSTM")
    _apply_window_geometry(self.root, args)
    self._compact = int(self.root.winfo_screenheight() or 800) < 720
    self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    self.thread = None
    self.use_r1 = tk.BooleanVar(value=True)
    self.use_r2 = tk.BooleanVar(value=True)
    self.swap_lr = tk.BooleanVar(value=False)
    self.pred = tk.StringVar(value="-")
    self.conf = tk.StringVar(value="0.00")
    self.net_raw = tk.StringVar(value="net: -")
    self.drive_pred = tk.StringVar(value="-")
    self.drive_conf = tk.StringVar(value="0.00")
    self.status = tk.StringVar(value="idle")
    self.radar_st = tk.StringVar(value="-")
    self.drop = tk.StringVar(value="none")
    self.lat = tk.StringVar(value="-")
    self.rel1 = tk.StringVar(value="50%")
    self.rel2 = tk.StringVar(value="50%")
    self.motion = tk.StringVar(value="static")
    self.backend = tk.StringVar(value=str(self.worker.backend))
    self.drive_hint = tk.StringVar(value="Push→1 lane left  Pull→1 lane right  Hold→stay  Tilt→HONK")
    self.collect_status = tk.StringVar(value="idle")
    self.collect_counts = tk.StringVar(value="")
    self.bgt_root = Path(getattr(args, "gesture_edge_bgt_data", None) or "artifacts/gesture_edge_bgt")
    self._collecting = False
    self._collect_label = ""
    self._countdown = 0
    self.prob_vars = []
    self.bars = []
    self._photo = None
    self._game = None
    self._pending_play = False
    self._last_game_t = time.perf_counter()
    self._build()
    self.root.bind("<Left>", lambda _e: self._game_key("left"))
    self.root.bind("<Right>", lambda _e: self._game_key("right"))
    self.root.bind("<Down>", lambda _e: self._game_key("center"))
    self.root.bind("<space>", lambda _e: self._game_key("center"))
    self.root.bind("<KeyRelease-Left>", lambda _e: self._game_key(None))
    self.root.bind("<KeyRelease-Right>", lambda _e: self._game_key(None))
    self.root.bind("<KeyRelease-Down>", lambda _e: self._game_key(None))
    self.root.bind("<KeyRelease-space>", lambda _e: self._game_key(None))
    self.root.bind("h", lambda _e: self._game_honk(True))
    self.root.bind("H", lambda _e: self._game_honk(True))
    self.root.bind("<KeyRelease-h>", lambda _e: self._game_honk(False))
    self.root.bind("<KeyRelease-H>", lambda _e: self._game_honk(False))
    self.root.bind("r", lambda _e: self._game_reset())
    self.root.bind("R", lambda _e: self._game_reset())
    self.root.after(100, self._tick)
    self.root.after(50, self._game_tick)

  def _build(self):
    main = ttk.Frame(self.root, padding=4)
    main.pack(fill="both", expand=True)
    split = ttk.Panedwindow(main, orient=tk.VERTICAL)
    split.pack(fill="both", expand=True)
    self._split = split

    perc = ttk.LabelFrame(split, text="Range-Doppler (R1 | R2)  — drag sash")
    perc.rowconfigure(0, weight=1)
    perc.columnconfigure(0, weight=1)
    self.radar_label = ttk.Label(perc)
    self.radar_label.grid(sticky="nsew", padx=2, pady=2)

    modes = ttk.Notebook(split)
    infer_host = ttk.Frame(modes)
    drive = ttk.Frame(modes, padding=4)
    collect_host = ttk.Frame(modes)
    profile_host = ttk.Frame(modes)
    modes.add(infer_host, text="Infer")
    modes.add(drive, text="Drive")
    modes.add(collect_host, text="Collect")
    modes.add(profile_host, text="Profile")
    self._control_notebook = modes
    self._modes = modes
    modes.bind("<<NotebookTabChanged>>", self._on_mode)

    split.add(perc, weight=1)
    split.add(modes, weight=3)
    self.root.after(80, self._place_sash)

    infer = _make_scrollable(infer_host)
    collect = _make_scrollable(collect_host)
    profile = _make_scrollable(profile_host)
    self._profile_tab = profile
    self._build_infer(infer)
    self._build_drive(drive)
    self._build_collect(collect)
    self._profile_run = mount_gesture_edge_profile_tab(
      self._profile_tab,
      defaults_fn=self._benchmark_defaults,
      busy_fn=lambda: self.thread is not None and self.thread.is_alive(),
      ui_after=self.root.after,
    )

  def _place_sash(self):
    try:
      h = int(self._split.winfo_height() or 0)
      if h > 80:
        self._split.sashpos(0, max(72, int(h * (0.22 if self._compact else 0.28))))
    except tk.TclError:
      pass

  def _on_mode(self, _evt=None):
    name = self._modes.tab(self._modes.select(), "text")
    if name != "Drive" and self._game is not None and self._game.playing:
      self._game.playing = False
      self._pending_play = False
      self._game.banner = "Play"

  def _build_infer(self, parent: ttk.Frame):
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(3, weight=1)
    box = ttk.LabelFrame(parent, text="Live gesture (Doppler votes)", padding=8)
    box.grid(row=0, column=0, sticky="ew")
    ttk.Label(box, textvariable=self.pred, font=("Segoe UI", 16 if self._compact else 22, "bold")).pack(anchor="w")
    ttk.Label(box, textvariable=self.conf).pack(anchor="w")
    ttk.Label(box, textvariable=self.net_raw).pack(anchor="w")
    ttk.Label(box, textvariable=self.backend).pack(anchor="w")

    run = ttk.LabelFrame(parent, text="Run", padding=6)
    run.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    ttk.Label(run, text=str(self.args.gesture_edge_checkpoint), wraplength=900).pack(anchor="w")
    radar_profile = str(getattr(self.args, "radar_profile", "safe"))
    ttk.Label(run, text=f"same Hold/Push/Pull votes as Drive · net softmax is the 'net:' line · radar={radar_profile}", wraplength=900).pack(anchor="w")
    row = ttk.Frame(run)
    row.pack(anchor="w", pady=4)
    ttk.Button(row, text="Start", command=self.start).pack(side="left", padx=(0, 6))
    ttk.Button(row, text="Stop", command=self.stop).pack(side="left", padx=(0, 6))
    ttk.Button(row, text="Benchmark", command=self._run_gui_benchmark).pack(side="left")

    meta = ttk.LabelFrame(parent, text="Reliance / status", padding=6)
    meta.grid(row=2, column=0, sticky="ew", pady=(8, 0))
    abl = ttk.Frame(meta)
    abl.pack(fill="x")
    ttk.Checkbutton(abl, text="Use Radar 1", variable=self.use_r1, command=self._ablate).pack(side="left", padx=(0, 12))
    ttk.Checkbutton(abl, text="Use Radar 2", variable=self.use_r2, command=self._ablate).pack(side="left")
    ttk.Label(abl, textvariable=self.drop).pack(side="left", padx=(12, 0))
    for lab, var in (
      ("Run", self.status),
      ("Radar", self.radar_st),
      ("Net", self.backend),
      ("Latency", self.lat),
      ("R1", self.rel1),
      ("R2", self.rel2),
    ):
      f = ttk.Frame(meta)
      f.pack(fill="x")
      ttk.Label(f, text=lab, width=8).pack(side="left")
      ttk.Label(f, textvariable=var, wraplength=800).pack(side="left")

    probs = ttk.LabelFrame(parent, text="Class probabilities", padding=6)
    probs.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
    for name in self.worker.labels:
      var = tk.StringVar(value=f"{name}: 0.00")
      self.prob_vars.append(var)
      ttk.Label(probs, textvariable=var).pack(anchor="w")
      bar = ttk.Progressbar(probs, maximum=100, length=280)
      bar.pack(fill="x", pady=1)
      self.bars.append(bar)

  def _build_drive(self, parent: ttk.Frame):
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)
    hud = ttk.Frame(parent)
    hud.grid(row=0, column=0, sticky="ew")
    ttk.Button(hud, text="Play", command=self._game_play).pack(side="left", padx=(0, 6))
    ttk.Button(hud, text="Reset", command=self._game_reset).pack(side="left", padx=(0, 10))
    ttk.Label(hud, textvariable=self.drive_pred, font=("Segoe UI", 16, "bold")).pack(side="left", padx=(0, 10))
    ttk.Label(hud, textvariable=self.drive_conf).pack(side="left", padx=(0, 10))
    ttk.Label(hud, textvariable=self.motion).pack(side="left", padx=(0, 10))
    ttk.Label(hud, textvariable=self.drive_hint, wraplength=420).pack(side="left")
    ttk.Checkbutton(hud, text="swap L/R", variable=self.swap_lr).pack(side="right")
    self.drive_canvas = tk.Canvas(parent, bg="#1c1d22", highlightthickness=0)
    self.drive_canvas.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
    self._game = CarDriveGame(self.drive_canvas)
    self._game.banner = "Drive tab · Play uses Doppler flicks; Palm Tilt → HONK"

  def _build_collect(self, parent: ttk.Frame):
    ttk.Label(parent, text="Live BGT clips (does not change Infer or Drive).", wraplength=900).pack(anchor="w")
    ttk.Label(parent, textvariable=self.collect_counts, font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(8, 8))
    row = ttk.Frame(parent)
    row.pack(anchor="w", pady=(0, 8))
    for lab in COLLECT_LABELS:
      ttk.Button(row, text=lab, command=lambda n=lab: self._collect_record(n)).pack(side="left", padx=(0, 8))
    ttk.Label(parent, textvariable=self.collect_status, font=("Segoe UI", 14)).pack(anchor="w")
    ttk.Label(
      parent,
      text=f"3-2-1 then do the gesture (~{self.worker.window} frames @ 5 Hz). Same chair/distance as Drive. ≥15 / class.",
      wraplength=900,
    ).pack(anchor="w", pady=(8, 0))
    ttk.Label(parent, text=str(self.bgt_root), wraplength=900).pack(anchor="w", pady=(4, 0))
    self._refresh_collect_counts()

  def _refresh_collect_counts(self):
    c = count_clips(self.bgt_root)
    self.collect_counts.set("  ".join(f"{k}: {v}" for k, v in c.items()))

  def _collect_record(self, label: str):
    if self._collecting:
      return
    self._collecting = True
    self._collect_label = label
    self._countdown = 3
    if not self._detect_running():
      self.start()
    self.collect_status.set(f"{label}: starting radar…")
    self._collect_tick()

  def _collect_tick(self):
    if not self._detect_running():
      self.collect_status.set(f"{self._collect_label}: starting radar…")
      self.root.after(200, self._collect_tick)
      return
    if self._countdown > 0:
      self.collect_status.set(f"{self._collect_label}: {self._countdown}…")
      self._countdown -= 1
      self.root.after(1000, self._collect_tick)
      return
    self.collect_status.set(f"{self._collect_label}: RECORD")
    self._collect_deadline = time.time() + 90.0
    self.worker.begin_record(self.worker.window)
    self._wait_record()

  def _wait_record(self):
    if time.time() > float(getattr(self, "_collect_deadline", 0) or 0):
      self.worker.cancel_record()
      self._collecting = False
      self.collect_status.set("timeout — no frames")
      return
    rec = self.worker.get().get("record", "")
    clip = self.worker.pop_record()
    if clip is None:
      self.collect_status.set(f"{self._collect_label}: {rec}")
      self.root.after(80, self._wait_record)
      return
    r1, r2 = clip
    path = save_clip(self.bgt_root, self._collect_label, r1, r2)
    self._collecting = False
    self._refresh_collect_counts()
    self.collect_status.set(f"saved {path.name}")

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

  def _game_key(self, cmd: str | None):
    if self._game is not None and self._game.playing:
      self._game.set_key(cmd)

  def _game_honk(self, on: bool):
    if self._game is not None:
      self._game.set_honk(on)

  def _game_reset(self):
    self._pending_play = False
    if self._game is not None:
      self._game.reset()
      self._game.playing = False
      self._game.banner = "Play"

  def _detect_running(self) -> bool:
    return self.thread is not None and self.thread.is_alive()

  def _game_play(self):
    if self._game is None:
      return
    self._game.reset()
    self._game.playing = False
    self._pending_play = True
    if not self._detect_running():
      self._game.banner = "starting radar…"
      self.start()
    else:
      st = self.worker.get()
      if st.get("drive_ready"):
        self._begin_game()
      else:
        fill, need = int(st.get("window_fill") or 0), int(self.worker.drive_need)
        self._game.banner = f"calibrating {fill}/{need}"

  def _begin_game(self):
    self._pending_play = False
    if self._game is None:
      return
    self._game.reset()
    self._game.playing = True
    self._game.alive = True
    self._game.banner = ""
    self._last_game_t = time.perf_counter()

  def _game_tick(self):
    game = self._game
    if game is not None:
      now = time.perf_counter()
      dt = min(0.08, max(0.001, now - self._last_game_t))
      self._last_game_t = now
      st = self.worker.get()
      if self._pending_play:
        fill = int(st.get("window_fill") or 0)
        need = int(self.worker.drive_need)
        if st.get("drive_ready"):
          self._begin_game()
        else:
          game.banner = f"calibrating {fill}/{need}"
          if not self._detect_running():
            game.banner = "starting radar…"
      game.swap_lr = bool(self.swap_lr.get())
      game.step(dt, self.worker.labels, st.get("drive_probs", st["probs"]), honk=bool(st.get("honk")))
      game.draw()
      self.drive_hint.set(game.hint())
    self.root.after(50, self._game_tick)

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
    self.net_raw.set(f"net: {st.get('raw_prediction') or '-'}")
    self.drive_pred.set(str(st.get("drive_pred") or "-"))
    self.drive_conf.set(f"{float(st.get('drive_conf') or 0.0):.2f}")
    self.status.set(str(st["status"]))
    self.radar_st.set(str(st["radar_status"]))
    self.drop.set(str(st["dropout"]))
    self.lat.set(f"{float(st['latency_ms']):.1f} ms")
    self.rel1.set(f"{float(st['rel1'])*100:.0f}%")
    self.rel2.set(f"{float(st['rel2'])*100:.0f}%")
    mv = bool(st.get("moving"))
    self.motion.set(f"{'move' if mv else 'static'}  c={float(st.get('doppler') or 0.0):.2f}")
    probs = np.asarray(st["probs"], np.float32)
    for i, (var, bar) in enumerate(zip(self.prob_vars, self.bars)):
      p = float(probs[i]) if i < probs.size else 0.0
      var.set(f"{self.worker.labels[i]}: {p:.2f}")
      bar["value"] = p * 100
    rgb = np.asarray(st["radar_rgb"])
    if rgb.ndim == 3:
      w = max(80, self.radar_label.winfo_width())
      h = max(48, self.radar_label.winfo_height())
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
      f"Missing {ckpt}\n"
      "Train 3-class drive: python gestureEdge/train.py --data ../../5_data/Gesture/SoliData.zip --device cuda"
    )
  if getattr(args, "radar_profile", "safe") == "safe":
    args.radar_profile = "gesture"
  # HAR default is 3 Hz; 40-frame window needs ~5 Hz or calib takes forever.
  if float(getattr(args, "frame_rate", 3.0) or 3.0) <= 3.0:
    args.frame_rate = 5.0
  GestureEdgeApp(args).run()
