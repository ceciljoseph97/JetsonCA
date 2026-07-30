#!/usr/bin/env python3
"""Lightweight Tk GUI for Jetson cam+radar inference (no alignment / dual-cam)."""

from __future__ import annotations

# Prefer conda libstdc++ BEFORE any native extension (cv2) imports.
# reexec=True is required: ld.so only reads LD_LIBRARY_PATH at process start.
from jetson_env import ensure_conda_lib_path

ensure_conda_lib_path(reexec=True)

import argparse
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk
from tkinter import ttk

from checkpoint import load_checkpoint, preprocess_camera_frame
from jetson_env import apply_jetson_runtime_tweaks, default_device
from label_hierarchy import format_hierarchy, inference_label
from radar_utils import (
  DualRadarSession,
  combine_sensor_panels,
  fuse_radar_streams_for_model,
  render_radar_panel,
)
from range_gating import estimate_peak_range_m, in_recognition_range, profile_metrics


def _import_camera_stream():
  """Import CameraStream after LD_LIBRARY_PATH is fixed; give a clear CXXABI hint."""
  try:
    from realtime_multimodal import CameraStream, get_version_full

    return CameraStream, get_version_full
  except ImportError as exc:
    msg = str(exc)
    if "CXXABI" in msg or "libstdc++" in msg:
      prefix = os.environ.get("CONDA_PREFIX", "$CONDA_PREFIX")
      raise SystemExit(
        "OpenCV failed to load (libstdc++ / CXXABI mismatch).\n"
        "On Jetson + conda, run:\n"
        f"  export LD_LIBRARY_PATH={prefix}/lib:${{LD_LIBRARY_PATH:-}}\n"
        "  python gui_app.py ...\n"
        "Or: conda install -c conda-forge 'libstdcxx-ng>=13'\n"
        f"Original error: {exc}"
      ) from exc
    raise


CameraStream, get_version_full = _import_camera_stream()


def _placeholder_rgb(message: str, width: int = 640, height: int = 360) -> np.ndarray:
  """PIL-only placeholder — avoids cv2 at GUI init."""
  img = Image.new("RGB", (width, height), (24, 24, 24))
  draw = ImageDraw.Draw(img)
  try:
    font = ImageFont.load_default()
  except Exception:
    font = None
  draw.text((16, max(16, height // 2 - 8)), message[:80], fill=(200, 200, 200), font=font)
  return np.asarray(img, dtype=np.uint8)


def _fit_frame(frame: np.ndarray, size: tuple[int, int], *, letterbox: bool) -> Image.Image:
  target_w, target_h = max(1, size[0]), max(1, size[1])
  img = Image.fromarray(frame)
  resample = getattr(Image, "Resampling", Image).BILINEAR
  if letterbox:
    fitted = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    copy = img.copy()
    copy.thumbnail((target_w, target_h), resample)
    x0 = (target_w - copy.width) // 2
    y0 = (target_h - copy.height) // 2
    fitted.paste(copy, (x0, y0))
    return fitted
  return img.resize((target_w, target_h), resample)

class InferenceWorker:
  def __init__(
    self,
    *,
    checkpoint: Path,
    device: str,
    camera_device: int,
    camera_width: int,
    camera_height: int,
    camera_fps: float,
    num_rx: int,
    radar_profile: str,
    frame_rate: float,
    radar1_port: str | None,
    radar2_port: str | None,
    mirror_radar2: bool,
    dual_radar_fuse: str,
    no_radar: bool,
    window_len: int,
    detect_threshold: float,
    human_threshold: float,
    min_range_m: float,
    max_range_m: float | None,
  ):
    self.model, self.labels, self.config = load_checkpoint(checkpoint, device)
    self.model.eval()
    self.device = device
    self.camera_device = camera_device
    self.camera_width = camera_width
    self.camera_height = camera_height
    self.camera_fps = camera_fps
    self.num_rx = num_rx
    self.radar_profile = radar_profile
    self.frame_rate = frame_rate
    self.radar1_port = radar1_port
    self.radar2_port = radar2_port
    self.mirror_radar2 = mirror_radar2
    self.dual_radar_fuse = dual_radar_fuse
    self.no_radar = no_radar
    self.window_len = window_len
    self.profile_metrics = profile_metrics(radar_profile)
    self.min_range_m = min_range_m
    self.max_range_m = float(max_range_m if max_range_m is not None else self.profile_metrics["max_range_m"])
    self.image_size = int(self.config["image_size"])

    self.stop_event = threading.Event()
    self.state_lock = threading.Lock()
    self.detect_threshold = detect_threshold
    self.human_threshold = human_threshold
    self.radar_enabled = not no_radar
    self.camera_enabled = True

    self.camera_buffer: deque[torch.Tensor] = deque(maxlen=window_len)
    self.radar_buffer: deque[torch.Tensor] = deque(maxlen=window_len)

    self.latest_state: dict[str, Any] = {
      "status": "idle",
      "prediction": "-",
      "hierarchy_text": "",
      "confidence": 0.0,
      "human_prob": 0.0,
      "latency_ms": 0.0,
      "fps": 0.0,
      "radar_status": "off" if no_radar else "not started",
      "target_range_m": 0.0,
      "in_range": False,
      "probs": np.zeros(len(self.labels), dtype=np.float32),
      "camera_rgb": _placeholder_rgb("Camera idle", camera_width, camera_height),
      "radar_rgb": np.zeros((64, 64, 3), dtype=np.uint8),
    }

  def set_threshold(self, value: float):
    with self.state_lock:
      self.detect_threshold = float(value)

  def set_human_threshold(self, value: float):
    with self.state_lock:
      self.human_threshold = float(value)

  def set_modalities(self, *, camera: bool, radar: bool):
    with self.state_lock:
      self.camera_enabled = bool(camera)
      self.radar_enabled = bool(radar) and not self.no_radar

  def get_state(self) -> dict[str, Any]:
    with self.state_lock:
      state = dict(self.latest_state)
      state["probs"] = np.asarray(self.latest_state["probs"], dtype=np.float32).copy()
      state["camera_rgb"] = np.asarray(self.latest_state["camera_rgb"]).copy()
      state["radar_rgb"] = np.asarray(self.latest_state["radar_rgb"]).copy()
      return state

  def stop(self):
    self.stop_event.set()

  def _predict(
    self,
    radar_tensor: torch.Tensor,
    camera_tensor: torch.Tensor,
    *,
    radar_present: bool,
    camera_present: bool,
  ) -> tuple[np.ndarray, str, float, float]:
    with torch.no_grad():
      out = self.model(
        radar_tensor,
        camera_tensor,
        radar_present=torch.tensor([radar_present], dtype=torch.bool, device=self.device),
        camera_present=torch.tensor([camera_present], dtype=torch.bool, device=self.device),
      )
      logits = out.get("activity_logits", out.get("logits"))
      probs = F.softmax(logits[0], dim=-1).detach().cpu().numpy()
      human_prob = 1.0
      if out.get("human_logits") is not None:
        human_prob = float(F.softmax(out["human_logits"][0], dim=-1)[1].item())
      label, conf = inference_label(
        self.labels,
        human_prob,
        probs,
        human_threshold=self.human_threshold,
      )
      return probs, label, conf, human_prob

  def _sync(self):
    if str(self.device).startswith("cuda") and torch.cuda.is_available():
      torch.cuda.synchronize()

  def run(self):
    camera: CameraStream | None = None
    try:
      camera = CameraStream(
        self.camera_device,
        self.camera_width,
        self.camera_height,
        self.camera_fps,
      ).start(warmup_s=2.0)
    except RuntimeError as exc:
      with self.state_lock:
        self.latest_state["status"] = f"camera open failed: {exc}"
        self.latest_state["camera_rgb"] = _placeholder_rgb(str(exc), self.camera_width, self.camera_height)

    t0 = time.time()
    n_infer = 0

    def _loop_body(radar_session: DualRadarSession | None):
      nonlocal n_infer

      with self.state_lock:
        radar_on = self.radar_enabled
        camera_on = self.camera_enabled
        threshold = self.detect_threshold

      fused_radar = None
      fuse_meta: dict[str, Any] = {"fusion": "none"}
      r1_panel = None
      r2_panel = None

      if radar_session is not None and radar_on:
        r1, r2 = radar_session.read_tensors()
        fused_radar, fuse_meta = fuse_radar_streams_for_model(
          r1,
          r2,
          mode=self.dual_radar_fuse,
          mirror_radar2=self.mirror_radar2,
        )
        if fused_radar is not None:
          self.radar_buffer.append(fused_radar.detach().cpu())
        if r1 is not None:
          r1_panel = render_radar_panel(r1.numpy())
        if r2 is not None:
          r2_panel = render_radar_panel(r2.numpy())
      elif not radar_on:
        fuse_meta = {"fusion": "disabled"}

      camera_rgb = _placeholder_rgb("Waiting for camera…", self.camera_width, self.camera_height)
      if camera is not None and camera_on:
        frame = camera.get_latest()
        if frame is not None:
          camera_rgb = frame
          self.camera_buffer.append(preprocess_camera_frame(frame, self.image_size).cpu())

      radar_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
      if radar_on:
        if r1_panel is not None and r2_panel is not None:
          radar_rgb = combine_sensor_panels(r1_panel, r2_panel, cross_sensor_mode="side_by_side")
        elif r1_panel is not None:
          radar_rgb = r1_panel
        elif r2_panel is not None:
          radar_rgb = r2_panel

      cam_ready = len(self.camera_buffer) >= self.window_len
      rad_ready = len(self.radar_buffer) >= self.window_len

      if camera_on and radar_on:
        can_predict = cam_ready and rad_ready
      elif camera_on:
        can_predict = cam_ready
      elif radar_on:
        can_predict = rad_ready
      else:
        can_predict = False

      prediction = "-"
      hierarchy_text = ""
      confidence = 0.0
      human_prob = 0.0
      probs = np.zeros(len(self.labels), dtype=np.float32)
      latency_ms = 0.0
      target_range_m = 0.0
      in_range = False
      status = "running"

      if not can_predict:
        if camera_on and radar_on:
          filled = min(len(self.camera_buffer), len(self.radar_buffer))
        elif camera_on:
          filled = len(self.camera_buffer)
        else:
          filled = len(self.radar_buffer)
        status = f"warming up {filled}/{self.window_len}"
      else:
        if radar_on and fused_radar is not None:
          peak = fused_radar[-1] if fused_radar.ndim == 4 else fused_radar
          target_range_m = estimate_peak_range_m(
            peak.numpy() if isinstance(peak, torch.Tensor) else peak,
            profile_max_range_m=self.profile_metrics["max_range_m"],
          )
          in_range = in_recognition_range(
            target_range_m,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
          )

        if camera_on and radar_on:
          radar_t = torch.stack(list(self.radar_buffer), dim=0).unsqueeze(0).to(self.device)
          camera_t = torch.stack(list(self.camera_buffer), dim=0).unsqueeze(0).to(self.device)
          radar_present = True
          camera_present = True
        elif camera_on:
          radar_t = torch.zeros(1, self.window_len, 3, 32, 32, device=self.device)
          camera_t = torch.stack(list(self.camera_buffer), dim=0).unsqueeze(0).to(self.device)
          radar_present = False
          camera_present = True
        else:
          radar_t = torch.stack(list(self.radar_buffer), dim=0).unsqueeze(0).to(self.device)
          camera_t = torch.zeros(1, self.window_len, 3, self.image_size, self.image_size, device=self.device)
          radar_present = True
          camera_present = False

        self._sync()
        t1 = time.perf_counter()
        probs, label, conf, human_prob = self._predict(
          radar_t,
          camera_t,
          radar_present=radar_present,
          camera_present=camera_present,
        )
        self._sync()
        latency_ms = (time.perf_counter() - t1) * 1000.0
        n_infer += 1

        prediction = label
        confidence = conf
        if conf < threshold:
          prediction = "none"
        hierarchy_text = format_hierarchy(label, conf)

      radar_status = "off"
      if radar_session is not None:
        radar_status = radar_session.status_text
      elif self.no_radar:
        radar_status = "disabled (--no-radar)"

      with self.state_lock:
        self.latest_state.update(
          {
            "status": status,
            "prediction": prediction,
            "hierarchy_text": hierarchy_text,
            "confidence": confidence,
            "human_prob": human_prob,
            "latency_ms": latency_ms,
            "fps": n_infer / max(time.time() - t0, 1e-6),
            "radar_status": radar_status,
            "fusion": str(fuse_meta.get("fusion", "none")),
            "target_range_m": target_range_m,
            "in_range": in_range,
            "probs": probs,
            "camera_rgb": camera_rgb,
            "radar_rgb": radar_rgb,
          }
        )

    try:
      if self.no_radar:
        with self.state_lock:
          self.latest_state["status"] = "running (camera-only)"
        while not self.stop_event.is_set():
          _loop_body(None)
          time.sleep(0.01)
      else:
        with DualRadarSession(
          num_rx=self.num_rx,
          profile=self.radar_profile,
          frame_rate_hz=self.frame_rate,
          radar1_port=self.radar1_port,
          radar2_port=self.radar2_port,
          mirror_radar2=self.mirror_radar2,
          min_range_m=self.min_range_m,
          max_range_m=self.max_range_m,
        ) as radar_session:
          with self.state_lock:
            self.latest_state["status"] = (
              f"running | sdk={get_version_full()} | {radar_session.status_text}"
            )
            self.latest_state["radar_status"] = radar_session.status_text
          while not self.stop_event.is_set():
            _loop_body(radar_session)
            time.sleep(0.01)
    finally:
      if camera is not None:
        camera.stop()
      with self.state_lock:
        self.latest_state["status"] = "stopped"


def _worker_from_args(args: argparse.Namespace, *, detect_threshold: float, human_threshold: float) -> InferenceWorker:
  return InferenceWorker(
    checkpoint=args.checkpoint,
    device=args.device,
    camera_device=args.camera_device,
    camera_width=args.camera_width,
    camera_height=args.camera_height,
    camera_fps=args.camera_fps,
    num_rx=args.num_rx,
    radar_profile=args.radar_profile,
    frame_rate=args.frame_rate,
    radar1_port=args.radar1_port,
    radar2_port=args.radar2_port,
    mirror_radar2=args.mirror_radar2,
    dual_radar_fuse=args.dual_radar_fuse,
    no_radar=args.no_radar,
    window_len=args.window,
    detect_threshold=detect_threshold,
    human_threshold=human_threshold,
    min_range_m=args.min_range_m,
    max_range_m=args.max_range_m,
  )


class JetsonGuiApp:
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.root = tk.Tk()
    self.root.title("JetsonCA — Activity Recognition")
    self.root.geometry("1100x640")
    self.root.minsize(900, 520)
    self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    self.worker = _worker_from_args(
      args,
      detect_threshold=args.detect_threshold,
      human_threshold=args.human_threshold,
    )
    self.worker_thread: threading.Thread | None = None

    self.status_var = tk.StringVar(value="idle")
    self.prediction_var = tk.StringVar(value="-")
    self.hierarchy_var = tk.StringVar(value="")
    self.conf_var = tk.StringVar(value="0.00")
    self.latency_var = tk.StringVar(value="-")
    self.range_var = tk.StringVar(value="-")
    self.radar_status_var = tk.StringVar(value="off")
    self.threshold_var = tk.StringVar(value=str(args.detect_threshold))
    self.human_threshold_var = tk.StringVar(value=str(args.human_threshold))
    self.camera_enabled_var = tk.BooleanVar(value=True)
    self.radar_enabled_var = tk.BooleanVar(value=not args.no_radar)

    self.camera_photo = None
    self.radar_photo = None
    self.prob_bars: list[ttk.Progressbar] = []
    self.prob_labels: list[tk.StringVar] = []

    self._build_ui()
    self.start()
    self.root.after(80, self._refresh_ui)

  def _build_ui(self):
    main = ttk.Frame(self.root, padding=8)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=3)
    main.columnconfigure(1, weight=2)
    main.rowconfigure(0, weight=1)

    left = ttk.Frame(main)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    left.rowconfigure(0, weight=3)
    left.rowconfigure(1, weight=2)
    left.columnconfigure(0, weight=1)

    cam_box = ttk.LabelFrame(left, text="Camera")
    cam_box.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
    cam_box.rowconfigure(0, weight=1)
    cam_box.columnconfigure(0, weight=1)
    self.camera_label = ttk.Label(cam_box, anchor="center")
    self.camera_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    radar_box = ttk.LabelFrame(left, text="Radar")
    radar_box.grid(row=1, column=0, sticky="nsew")
    radar_box.rowconfigure(0, weight=1)
    radar_box.columnconfigure(0, weight=1)
    self.radar_label = ttk.Label(radar_box, anchor="center")
    self.radar_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    right = ttk.Frame(main)
    right.grid(row=0, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)

    pred_box = ttk.LabelFrame(right, text="Prediction", padding=8)
    pred_box.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    pred_box.columnconfigure(0, weight=1)
    ttk.Label(pred_box, textvariable=self.prediction_var, font=("Segoe UI", 18, "bold")).grid(
      row=0, column=0, sticky="w"
    )
    ttk.Label(pred_box, textvariable=self.hierarchy_var, wraplength=320).grid(row=1, column=0, sticky="w")
    ttk.Label(pred_box, text=f"Confidence: ").grid(row=2, column=0, sticky="w", pady=(6, 0))
    ttk.Label(pred_box, textvariable=self.conf_var).grid(row=2, column=1, sticky="w", pady=(6, 0))

    ctrl = ttk.LabelFrame(right, text="Controls", padding=8)
    ctrl.grid(row=1, column=0, sticky="ew", pady=(0, 6))
    ctrl.columnconfigure(1, weight=1)

    ttk.Checkbutton(ctrl, text="Camera", variable=self.camera_enabled_var, command=self._apply_modalities).grid(
      row=0, column=0, sticky="w"
    )
    radar_cb = ttk.Checkbutton(
      ctrl,
      text="Radar",
      variable=self.radar_enabled_var,
      command=self._apply_modalities,
    )
    radar_cb.grid(row=0, column=1, sticky="w")
    if self.args.no_radar:
      radar_cb.state(["disabled"])

    ttk.Label(ctrl, text="Detect threshold").grid(row=1, column=0, sticky="w", pady=(6, 0))
    thr = ttk.Entry(ctrl, textvariable=self.threshold_var, width=8)
    thr.grid(row=1, column=1, sticky="w", pady=(6, 0))
    ttk.Button(ctrl, text="Apply", command=self._apply_thresholds).grid(row=1, column=2, padx=(6, 0), pady=(6, 0))

    ttk.Label(ctrl, text="Human threshold").grid(row=2, column=0, sticky="w", pady=(4, 0))
    ttk.Entry(ctrl, textvariable=self.human_threshold_var, width=8).grid(row=2, column=1, sticky="w", pady=(4, 0))

    btn_row = ttk.Frame(ctrl)
    btn_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
    ttk.Button(btn_row, text="Start", command=self.start).pack(side="left", padx=(0, 6))
    ttk.Button(btn_row, text="Stop", command=self.stop).pack(side="left")

    status_box = ttk.LabelFrame(right, text="Status", padding=8)
    status_box.grid(row=2, column=0, sticky="ew", pady=(0, 6))
    for row, (label, var) in enumerate(
      (
        ("Run", self.status_var),
        ("Latency", self.latency_var),
        ("Range", self.range_var),
        ("Radar", self.radar_status_var),
      )
    ):
      ttk.Label(status_box, text=f"{label}:").grid(row=row, column=0, sticky="w")
      ttk.Label(status_box, textvariable=var, wraplength=280).grid(row=row, column=1, sticky="w")

    prob_box = ttk.LabelFrame(right, text="Class probabilities", padding=8)
    prob_box.grid(row=3, column=0, sticky="nsew")
    prob_box.columnconfigure(1, weight=1)
    right.rowconfigure(3, weight=1)

    for idx, label in enumerate(self.worker.labels):
      var = tk.StringVar(value=f"{label}: 0.00")
      self.prob_labels.append(var)
      ttk.Label(prob_box, textvariable=var).grid(row=idx, column=0, sticky="w", pady=1)
      bar = ttk.Progressbar(prob_box, maximum=100.0, length=160)
      bar.grid(row=idx, column=1, sticky="ew", padx=(8, 0), pady=1)
      self.prob_bars.append(bar)

  def _apply_modalities(self):
    self.worker.set_modalities(
      camera=self.camera_enabled_var.get(),
      radar=self.radar_enabled_var.get(),
    )

  def _apply_thresholds(self):
    try:
      self.worker.set_threshold(float(self.threshold_var.get()))
      self.worker.set_human_threshold(float(self.human_threshold_var.get()))
    except ValueError:
      self.status_var.set("invalid threshold")

  def start(self):
    if self.worker_thread is not None and self.worker_thread.is_alive():
      return
    try:
      self.worker = _worker_from_args(
        self.args,
        detect_threshold=float(self.threshold_var.get()),
        human_threshold=float(self.human_threshold_var.get()),
      )
    except ValueError:
      self.status_var.set("invalid threshold")
      return
    self.worker.set_modalities(
      camera=self.camera_enabled_var.get(),
      radar=self.radar_enabled_var.get(),
    )
    self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
    self.worker_thread.start()
    self.status_var.set("starting…")

  def stop(self):
    if self.worker_thread is not None and self.worker_thread.is_alive():
      self.worker.stop()

  def _set_image(self, widget: ttk.Label, frame: np.ndarray, attr: str, size: tuple[int, int], *, letterbox: bool):
    img = _fit_frame(frame, size, letterbox=letterbox)
    photo = ImageTk.PhotoImage(image=img)
    setattr(self, attr, photo)
    widget.configure(image=photo)

  def _refresh_ui(self):
    state = self.worker.get_state()
    pred = str(state["prediction"])
    conf = float(state["confidence"])
    self.status_var.set(str(state["status"]))
    self.prediction_var.set(pred)
    self.hierarchy_var.set(str(state.get("hierarchy_text", "")))
    self.conf_var.set(f"{conf:.2f}")
    self.latency_var.set(f"{float(state['latency_ms']):.1f} ms  (~{float(state['fps']):.1f} infer/s)")
    target_range = float(state.get("target_range_m", 0.0))
    in_range = bool(state.get("in_range", False))
    self.range_var.set(f"{target_range:.2f} m ({'in gate' if in_range else 'out of gate'})")
    self.radar_status_var.set(str(state.get("radar_status", "-")))

    probs = np.asarray(state["probs"], dtype=np.float32)
    for idx, (bar, label_var) in enumerate(zip(self.prob_bars, self.prob_labels)):
      value = float(probs[idx]) if idx < len(probs) else 0.0
      bar["value"] = value * 100.0
      label_var.set(f"{self.worker.labels[idx]}: {value:.2f}")

    cam_size = (max(320, self.camera_label.winfo_width()), max(200, self.camera_label.winfo_height()))
    rad_size = (max(240, self.radar_label.winfo_width()), max(160, self.radar_label.winfo_height()))
    self._set_image(self.camera_label, state["camera_rgb"], "camera_photo", cam_size, letterbox=True)
    self._set_image(self.radar_label, state["radar_rgb"], "radar_photo", rad_size, letterbox=False)

    self.root.after(100, self._refresh_ui)

  def on_close(self):
    self.stop()
    self.root.destroy()

  def run(self):
    self.root.mainloop()


def parse_args():
  p = argparse.ArgumentParser(description="Lightweight Jetson cam+radar GUI")
  p.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_multimodal_crossattention.pt"))
  p.add_argument("--device", type=str, default=default_device())
  p.add_argument("--camera-device", type=int, default=0)
  p.add_argument("--camera-width", type=int, default=640)
  p.add_argument("--camera-height", type=int, default=480)
  p.add_argument("--camera-fps", type=float, default=15.0)
  p.add_argument("--num-rx", type=int, default=3)
  p.add_argument("--radar-profile", choices=("safe", "balanced", "gesture"), default="safe")
  p.add_argument("--frame-rate", type=float, default=5.0)
  p.add_argument("--radar1-port", type=str, default=None)
  p.add_argument("--radar2-port", type=str, default=None)
  p.add_argument("--mirror-radar2", action="store_true", default=True)
  p.add_argument("--no-mirror-radar2", action="store_false", dest="mirror_radar2")
  p.add_argument("--no-radar", action="store_true", help="Camera-only: skip radar SDK")
  p.add_argument("--dual-radar-fuse", choices=("mean", "max"), default="mean")
  p.add_argument("--window", type=int, default=30)
  p.add_argument("--detect-threshold", type=float, default=0.35)
  p.add_argument("--human-threshold", type=float, default=0.5)
  p.add_argument("--min-range-m", type=float, default=0.3)
  p.add_argument("--max-range-m", type=float, default=2.5)
  return p.parse_args()


def main():
  args = parse_args()
  apply_jetson_runtime_tweaks()
  JetsonGuiApp(args).run()


if __name__ == "__main__":
  main()
