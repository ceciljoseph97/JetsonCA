#!/usr/bin/env python3
"""Simplified Testing + Realtime GUI (Crossattention subset) for Jetson / desktop."""

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

from audio_features import mel_tensor_from_wave, render_audio_monitor_rgb
from checkpoint import load_checkpoint, preprocess_camera_frame
from device_select import prefer_microsoft
from jetson_env import apply_jetson_runtime_tweaks, default_device
from label_hierarchy import apply_logit_bias, combine_hierarchical_probs, format_hierarchy, inference_label
from live_audio import LiveAudioBuffer, list_audio_input_devices
from radar_utils import (
  DualRadarSession,
  combine_sensor_panels,
  fuse_dual_radar_tensors,
  fuse_radar_streams_for_model,
  list_radar_uuids,
  render_radar_panel,
)
from range_gating import estimate_peak_range_m, has_radar_motion, in_recognition_range, profile_metrics

try:
  from detector_pipeline import apply_camera_roi_crop_seq, apply_cfar_mask_to_radar_seq
except Exception:  # pragma: no cover - optional if cv2 missing
  apply_camera_roi_crop_seq = None  # type: ignore[assignment]
  apply_cfar_mask_to_radar_seq = None  # type: ignore[assignment]


def _import_camera_stream():
  """Import CameraStream after LD_LIBRARY_PATH is fixed; give a clear CXXABI hint."""
  try:
    from realtime_multimodal import CameraStream, get_version_full, probe_camera_devices

    return CameraStream, get_version_full, probe_camera_devices
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


CameraStream, get_version_full, probe_camera_devices = _import_camera_stream()


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
    radar1_uuid: str | None = None,
    radar2_uuid: str | None = None,
    mirror_radar2: bool,
    dual_radar_fuse: str | None,
    no_radar: bool,
    no_audio: bool = False,
    window_len: int,
    detect_threshold: float,
    human_threshold: float,
    min_margin: float = 0.12,
    smooth_n: int = 5,
    require_in_range: bool = True,
    towards_bias: float = 0.35,
    motion_threshold: float = 0.35,
    peak_ratio: float = 5.0,
    min_range_m: float,
    max_range_m: float | None,
    live_detector_preprocess: bool = False,
    use_motion_gate: bool = False,
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
    self.radar1_uuid = radar1_uuid
    self.radar2_uuid = radar2_uuid
    self.mirror_radar2 = mirror_radar2
    # Prefer CLI; fall back to checkpoint train config (Crossattention default: none).
    self.dual_radar_fuse = str(dual_radar_fuse or self.config.get("dual_radar_fuse", "none"))
    self.no_radar = no_radar
    self.no_audio = no_audio
    self.window_len = window_len
    self.profile_metrics = profile_metrics(radar_profile)
    self.min_range_m = float(min_range_m if min_range_m is not None else self.config.get("min_range_m", 0.3))
    self.max_range_m = float(
      max_range_m if max_range_m is not None else self.config.get("max_range_m", self.profile_metrics["max_range_m"])
    )
    self.image_size = int(self.config["image_size"])
    # Train-time CFAR/ROI is expensive on Jetson; off unless explicitly enabled live.
    self.detector_preprocess = bool(live_detector_preprocess)
    self.detector_cam_crop = bool(self.config.get("detector_cam_crop", True))
    self.detector_min_snr_db = float(self.config.get("detector_min_snr_db", 6.0))
    self.enable_detect_head = bool(self.config.get("has_detect_head") or self.config.get("enable_detect_head", False)) or (
      getattr(self.model, "detect_classifier", None) is not None
    )
    self.use_motion_gate = bool(use_motion_gate) and not self.enable_detect_head

    self.stop_event = threading.Event()
    self.state_lock = threading.Lock()
    self.detect_threshold = detect_threshold
    self.human_threshold = human_threshold
    self.min_margin = float(min_margin)
    self.smooth_n = max(1, int(smooth_n))
    self.require_in_range = bool(require_in_range)
    self.logit_bias = {"walking_towards": float(towards_bias)} if towards_bias else {}
    self.motion_threshold = float(motion_threshold)
    self.peak_ratio = float(peak_ratio)
    self.pred_hist: deque[str] = deque(maxlen=self.smooth_n)
    self.radar_enabled = not no_radar
    self.camera_enabled = True
    self.enable_audio = bool(self.config.get("enable_audio", False)) and not no_audio
    self.audio_sample_rate = int(self.config.get("audio_sample_rate", 16000) or 16000)
    self.audio_n_mels = int(self.config.get("audio_n_mels", 32))
    self.audio_mel_width = int(self.config.get("audio_mel_width", 32))
    self.audio_buffer = LiveAudioBuffer(sample_rate=self.audio_sample_rate)
    self.audio_device_index: int | None = None
    self.audio_input_enabled = True
    self.audio_enabled = bool(self.enable_audio)

    self.camera_buffer: deque[torch.Tensor] = deque(maxlen=window_len)
    self.camera_rgb_buffer: deque[np.ndarray] = deque(maxlen=window_len)
    self.radar_buffer: deque[torch.Tensor] = deque(maxlen=window_len)
    self.radar1_buffer: deque[torch.Tensor] = deque(maxlen=window_len)
    self.radar2_buffer: deque[torch.Tensor] = deque(maxlen=window_len)

    self.latest_state: dict[str, Any] = {
      "status": "idle",
      "prediction": "-",
      "raw_prediction": "-",
      "hierarchy_text": "",
      "confidence": 0.0,
      "human_prob": 0.0,
      "detect_prob": 0.0,
      "gate_open": False,
      "motion_ok": True,
      "latency_ms": 0.0,
      "fps": 0.0,
      "radar_status": "off" if no_radar else "not started",
      "target_range_m": 0.0,
      "in_range": False,
      "probs": np.zeros(len(self.labels), dtype=np.float32),
      "camera_rgb": _placeholder_rgb("Camera idle", camera_width, camera_height),
      "radar_rgb": np.zeros((64, 64, 3), dtype=np.uint8),
      "audio_rgb": _placeholder_rgb("Audio idle", 320, 180),
      "audio_verify_text": "off",
      "reliance": {"camera": 0.0, "radar": 0.0, "audio": 0.0},
    }

  def set_threshold(self, value: float):
    with self.state_lock:
      self.detect_threshold = float(value)

  def set_human_threshold(self, value: float):
    with self.state_lock:
      self.human_threshold = float(value)

  def set_modalities(self, *, camera: bool, radar: bool, audio: bool | None = None):
    with self.state_lock:
      self.camera_enabled = bool(camera)
      self.radar_enabled = bool(radar) and not self.no_radar
      if audio is not None:
        self.audio_enabled = bool(audio) and self.enable_audio

  def set_audio_device(self, device_index: int | None, *, input_enabled: bool = True):
    with self.state_lock:
      self.audio_device_index = device_index
      self.audio_input_enabled = bool(input_enabled)

  def get_state(self) -> dict[str, Any]:
    with self.state_lock:
      state = dict(self.latest_state)
      state["probs"] = np.asarray(self.latest_state["probs"], dtype=np.float32).copy()
      state["camera_rgb"] = np.asarray(self.latest_state["camera_rgb"]).copy()
      state["radar_rgb"] = np.asarray(self.latest_state["radar_rgb"]).copy()
      state["audio_rgb"] = np.asarray(self.latest_state["audio_rgb"]).copy()
      state["reliance"] = dict(self.latest_state.get("reliance") or {})
      return state

  def stop(self):
    self.stop_event.set()

  def _live_audio_wave(self) -> np.ndarray:
    wave = self.audio_buffer.snapshot()
    if wave.size == 0:
      return wave
    win_s = float(self.window_len) / max(float(self.frame_rate), 1e-6)
    n_keep = max(1, int(np.ceil((win_s + 0.35) * self.audio_sample_rate)))
    if wave.size > n_keep:
      return wave[-n_keep:]
    return wave

  def _build_audio_tensor(self) -> tuple[torch.Tensor | None, bool]:
    if not self.enable_audio:
      return None, False
    wave = self._live_audio_wave()
    min_samples = max(1, int(self.audio_sample_rate / max(self.frame_rate, 1.0)))
    audio_present = wave.size >= min_samples
    if not audio_present:
      wave = np.zeros(max(min_samples, 1), dtype=np.float32)
    patches = mel_tensor_from_wave(
      wave,
      self.audio_sample_rate,
      self.window_len,
      self.frame_rate,
      n_mels=self.audio_n_mels,
      mel_width=self.audio_mel_width,
    )
    if patches.shape[0] < self.window_len:
      pad_n = self.window_len - patches.shape[0]
      pad = patches[:1].expand(pad_n, *patches.shape[1:]).clone()
      patches = torch.cat([pad, patches], dim=0)
    elif patches.shape[0] > self.window_len:
      patches = patches[-self.window_len :]
    return patches.unsqueeze(0).to(self.device), audio_present

  def _audio_monitor_rgb(self) -> np.ndarray:
    wave = self._live_audio_wave()
    try:
      return render_audio_monitor_rgb(wave, self.audio_sample_rate)
    except Exception:
      return _placeholder_rgb("audio monitor unavailable", 320, 180)

  def _audio_verify_text(self) -> str:
    if not self.enable_audio:
      return "ckpt has no audio"
    if not self.audio_input_enabled:
      return "input off"
    if not self.audio_buffer.running:
      err = self.audio_buffer.last_error or "not started"
      return f"mic off ({err})"
    stats = self.audio_buffer.level_stats()
    if not stats.get("has_data"):
      return "warming…"
    ok = "ok" if stats.get("ok") else "dead/quiet"
    return f"{ok}  rms={float(stats['rms']):.3f}  peak={float(stats['peak']):.3f}"

  @staticmethod
  def _reliance_from_weights(weights: torch.Tensor | None) -> dict[str, float]:
    out = {"camera": 0.0, "radar": 0.0, "audio": 0.0}
    if weights is None:
      return out
    w = weights.detach().float().reshape(-1).cpu().numpy()
    if w.size >= 3:
      out["radar"] = float(w[0] + w[1])
      out["camera"] = float(w[2])
    if w.size >= 4:
      out["audio"] = float(w[3])
    return out

  @staticmethod
  def _presence_probs(outputs: dict) -> tuple[float, float]:
    human_prob = 1.0
    detect_prob = 1.0
    if outputs.get("human_logits") is not None:
      human_prob = float(F.softmax(outputs["human_logits"][0], dim=-1)[1].item())
      detect_prob = human_prob
    if outputs.get("detect_logits") is not None:
      detect_prob = float(F.softmax(outputs["detect_logits"][0], dim=-1)[1].item())
    return human_prob, detect_prob

  def _predict(
    self,
    radar_tensor: torch.Tensor,
    camera_tensor: torch.Tensor,
    *,
    radar_present: bool,
    camera_present: bool,
    radar2_tensor: torch.Tensor | None = None,
  ) -> tuple[np.ndarray, str, float, float, float]:
    with torch.no_grad():
      kwargs: dict[str, Any] = {
        "radar": radar_tensor,
        "camera": camera_tensor,
        "radar2": radar2_tensor,
        "radar_present": torch.tensor([radar_present], dtype=torch.bool, device=self.device),
        "camera_present": torch.tensor([camera_present], dtype=torch.bool, device=self.device),
      }
      if self.enable_audio:
        audio_tensor, audio_present = self._build_audio_tensor()
        if audio_tensor is not None:
          kwargs["audio"] = audio_tensor
          kwargs["audio_present"] = torch.tensor(
            [bool(audio_present and self.audio_enabled)],
            dtype=torch.bool,
            device=self.device,
          )
      out = self.model(**kwargs)
      self._cached_reliance = self._reliance_from_weights(out.get("quality_weights"))
      logits = out.get("activity_logits", out.get("logits"))
      probs = F.softmax(logits[0], dim=-1).detach().cpu().numpy()
      coarse_probs = None
      subaction_probs = None
      if self.config.get("use_hierarchical_fusion"):
        if out.get("coarse_logits") is not None:
          coarse_probs = F.softmax(out["coarse_logits"][0], dim=-1).detach().cpu().numpy()
        if out.get("subaction_logits") is not None:
          subaction_probs = F.softmax(out["subaction_logits"][0], dim=-1).detach().cpu().numpy()
        probs = combine_hierarchical_probs(
          self.labels,
          probs,
          coarse_probs,
          subaction_probs,
          hierarchy_labels=list(self.config.get("all_labels") or self.labels),
        )
      probs = apply_logit_bias(probs, self.labels, self.logit_bias)
      human_prob, detect_prob = self._presence_probs(out)
      # With learned DETECT, classical Doppler motion must not override class selection
      # (standing / slow motion still has a person → DETECT high, motion gate false).
      motion_ok = True
      if self.use_motion_gate and radar_present:
        motion_ok = bool(getattr(self, "_cached_radar_motion", True))
      presence = detect_prob if self.enable_detect_head else human_prob
      label, conf = inference_label(
        self.labels,
        presence,
        probs,
        human_threshold=self.human_threshold if not self.enable_detect_head else min(self.human_threshold, self.detect_threshold),
        min_margin=self.min_margin,
        motion_ok=motion_ok,
        in_range=(not self.require_in_range) or (not radar_present) or getattr(self, "_cached_in_range", True),
      )
      # Near-ties: still expose top activity so bars and prediction agree.
      if label in ("uncertain", "background") and self.enable_detect_head and detect_prob >= self.detect_threshold:
        activity_labels = [x for x in self.labels if x not in ("background", "no_human", "empty", "idle")]
        p = np.asarray(probs, dtype=np.float32).reshape(-1)
        if p.size and activity_labels:
          top = int(np.argmax(p[: len(activity_labels)]))
          label = activity_labels[top]
          conf = float(p[top])
      return probs, label, conf, human_prob, detect_prob

  def _smooth_label(self, label: str) -> str:
    if label in ("none", "uncertain", "background", "-"):
      self.pred_hist.clear()
      return label
    self.pred_hist.append(label)
    if len(self.pred_hist) < max(2, self.smooth_n // 2 + 1):
      return label
    # majority vote
    counts: dict[str, int] = {}
    for item in self.pred_hist:
      counts[item] = counts.get(item, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]
  def _sync(self):
    if str(self.device).startswith("cuda") and torch.cuda.is_available():
      torch.cuda.synchronize()

  def run(self):
    camera: CameraStream | None = None
    if self.camera_device is not None and int(self.camera_device) >= 0:
      try:
        camera = CameraStream(
          int(self.camera_device),
          self.camera_width,
          self.camera_height,
          self.camera_fps,
        ).start(warmup_s=2.0)
      except RuntimeError as exc:
        with self.state_lock:
          self.latest_state["status"] = f"camera open failed: {exc}"
          self.latest_state["camera_rgb"] = _placeholder_rgb(str(exc), self.camera_width, self.camera_height)
    elif self.camera_enabled:
      with self.state_lock:
        self.latest_state["status"] = "no camera selected"
        self.latest_state["camera_rgb"] = _placeholder_rgb("No camera selected", self.camera_width, self.camera_height)

    if self.enable_audio and self.audio_input_enabled:
      self.audio_buffer.start(device=self.audio_device_index)

    t0 = time.time()
    n_infer = 0

    def _loop_body(radar_session: DualRadarSession | None):
      nonlocal n_infer

      with self.state_lock:
        radar_on = self.radar_enabled
        camera_on = self.camera_enabled
        audio_on = self.audio_enabled and self.enable_audio
        threshold = self.detect_threshold

      fused_radar = None
      fuse_meta: dict[str, Any] = {"fusion": "none"}
      r1_panel = None
      r2_panel = None
      r1_t = None
      r2_t = None

      if radar_session is not None and radar_on:
        r1, r2 = radar_session.read_tensors()
        r1_t, r2_t = r1, r2
        if r2_t is None and r1_t is not None and self.mirror_radar2:
          r2_t = r1_t
        if r1_t is not None:
          self.radar1_buffer.append(r1_t.detach().cpu())
        if r2_t is not None:
          self.radar2_buffer.append(r2_t.detach().cpu())
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
          self.camera_rgb_buffer.append(np.asarray(frame))
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
      rad_ready = len(self.radar1_buffer) >= self.window_len or len(self.radar_buffer) >= self.window_len
      audio_ready = False
      if audio_on:
        wave = self._live_audio_wave()
        audio_ready = wave.size >= max(1, int(self.audio_sample_rate / max(self.frame_rate, 1.0)))

      if camera_on and radar_on:
        can_predict = cam_ready and rad_ready
      elif camera_on:
        can_predict = cam_ready
      elif radar_on:
        can_predict = rad_ready
      elif audio_on:
        can_predict = audio_ready
      else:
        can_predict = False

      prediction = "-"
      raw_prediction = "-"
      hierarchy_text = ""
      confidence = 0.0
      human_prob = 0.0
      detect_prob = 0.0
      gate_open = False
      probs = np.zeros(len(self.labels), dtype=np.float32)
      latency_ms = 0.0
      target_range_m = 0.0
      in_range = False
      status = "running"

      if not can_predict:
        if camera_on and radar_on:
          filled = min(len(self.camera_buffer), max(len(self.radar1_buffer), len(self.radar_buffer)))
        elif camera_on:
          filled = len(self.camera_buffer)
        else:
          filled = max(len(self.radar1_buffer), len(self.radar_buffer))
        status = f"warming up {filled}/{self.window_len}"
      else:
        peak_src = fused_radar
        if peak_src is None and len(self.radar1_buffer) > 0:
          peak_src = self.radar1_buffer[-1]
        if radar_on and peak_src is not None:
          peak = peak_src[-1] if getattr(peak_src, "ndim", 0) == 4 else peak_src
          peak_np = peak.numpy() if isinstance(peak, torch.Tensor) else peak
          target_range_m = estimate_peak_range_m(
            peak_np,
            profile_max_range_m=self.profile_metrics["max_range_m"],
          )
          in_range = in_recognition_range(
            target_range_m,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
          )
          radar_motion, _ = has_radar_motion(
            peak,
            motion_threshold=self.motion_threshold,
            peak_ratio=self.peak_ratio,
          )
          self._cached_in_range = in_range
          self._cached_radar_motion = radar_motion
        else:
          self._cached_in_range = True
          self._cached_radar_motion = True

        if len(self.radar1_buffer) >= self.window_len:
          radar_np = torch.stack(list(self.radar1_buffer), dim=0).numpy()
        elif len(self.radar_buffer) >= self.window_len:
          radar_np = torch.stack(list(self.radar_buffer), dim=0).numpy()
        else:
          radar_np = np.zeros((self.window_len, 3, 32, 32), dtype=np.float32)

        if len(self.radar2_buffer) >= self.window_len:
          radar2_np = torch.stack(list(self.radar2_buffer), dim=0).numpy()
        else:
          radar2_np = radar_np

        # Early-fuse only when config asks; dual-encoder still gets radar2 separately.
        radar_np = np.asarray(
          fuse_dual_radar_tensors(radar_np, radar2_np, mode=self.dual_radar_fuse),
          dtype=np.float32,
        )

        if self.detector_preprocess and apply_cfar_mask_to_radar_seq is not None:
          profile_max = float(self.profile_metrics["max_range_m"])
          radar_np = apply_cfar_mask_to_radar_seq(
            radar_np, profile_max_range_m=profile_max, min_snr_db=self.detector_min_snr_db
          )
          radar2_np = apply_cfar_mask_to_radar_seq(
            radar2_np, profile_max_range_m=profile_max, min_snr_db=self.detector_min_snr_db
          )

        if (
          self.detector_preprocess
          and self.detector_cam_crop
          and apply_camera_roi_crop_seq is not None
          and len(self.camera_rgb_buffer) >= self.window_len
        ):
          cam_rgb = apply_camera_roi_crop_seq(np.stack(list(self.camera_rgb_buffer), axis=0))
          cam_tensors = [preprocess_camera_frame(fr, self.image_size).cpu() for fr in cam_rgb]
          camera_t = torch.stack(cam_tensors, dim=0).unsqueeze(0).to(self.device)
        elif camera_on and len(self.camera_buffer) >= self.window_len:
          camera_t = torch.stack(list(self.camera_buffer), dim=0).unsqueeze(0).to(self.device)
        else:
          camera_t = torch.zeros(1, self.window_len, 3, self.image_size, self.image_size, device=self.device)

        if camera_on and radar_on:
          radar_present = True
          camera_present = True
        elif camera_on:
          radar_present = False
          camera_present = True
          radar_np = np.zeros((self.window_len, 3, 32, 32), dtype=np.float32)
          radar2_np = radar_np
        elif radar_on:
          radar_present = True
          camera_present = False
        else:
          radar_present = False
          camera_present = False
          radar_np = np.zeros((self.window_len, 3, 32, 32), dtype=np.float32)
          radar2_np = radar_np

        radar_t = torch.from_numpy(np.asarray(radar_np, dtype=np.float32)).unsqueeze(0).to(self.device)
        radar2_t = torch.from_numpy(np.asarray(radar2_np, dtype=np.float32)).unsqueeze(0).to(self.device)

        self._sync()
        t1 = time.perf_counter()
        probs, label, conf, human_prob, detect_prob = self._predict(
          radar_t,
          camera_t,
          radar_present=radar_present,
          camera_present=camera_present,
          radar2_tensor=radar2_t,
        )
        self._sync()
        latency_ms = (time.perf_counter() - t1) * 1000.0
        n_infer += 1

        gate_open = detect_prob >= threshold
        prediction = label
        confidence = conf
        raw_prediction = label
        suppress_reason = ""
        if not gate_open:
          prediction = "none"
          suppress_reason = f"gate closed (det={detect_prob:.2f})"
        elif label == "uncertain":
          prediction = "none"
          suppress_reason = "low margin"
        if self.require_in_range and radar_on and not in_range:
          prediction = "none"
          suppress_reason = f"out of range ({target_range_m:.2f}m)"
        prediction = self._smooth_label(prediction)
        hierarchy_text = format_hierarchy(raw_prediction, conf)
        if suppress_reason:
          hierarchy_text = f"{hierarchy_text}\n[{suppress_reason}]"

      radar_status = "off"
      if radar_session is not None:
        radar_status = radar_session.status_text
      elif self.no_radar:
        radar_status = "disabled (--no-radar)"

      audio_rgb = _placeholder_rgb("Audio idle", 320, 180)
      if self.enable_audio:
        audio_rgb = self._audio_monitor_rgb()
      audio_verify = self._audio_verify_text()

      with self.state_lock:
        self.latest_state.update(
          {
            "status": status,
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "hierarchy_text": hierarchy_text,
            "confidence": confidence,
            "human_prob": human_prob,
            "detect_prob": detect_prob,
            "gate_open": gate_open,
            "motion_ok": bool(getattr(self, "_cached_radar_motion", True)),
            "latency_ms": latency_ms,
            "fps": n_infer / max(time.time() - t0, 1e-6),
            "radar_status": radar_status,
            "fusion": str(fuse_meta.get("fusion", "none")),
            "target_range_m": target_range_m,
            "in_range": in_range,
            "probs": probs,
            "camera_rgb": camera_rgb,
            "radar_rgb": radar_rgb,
            "audio_rgb": audio_rgb,
            "audio_verify_text": audio_verify,
            "reliance": dict(getattr(self, "_cached_reliance", {"camera": 0.0, "radar": 0.0, "audio": 0.0})),
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
          radar1_uuid=self.radar1_uuid,
          radar2_uuid=self.radar2_uuid,
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
      if self.audio_buffer.running:
        self.audio_buffer.stop()
      with self.state_lock:
        self.latest_state["status"] = "stopped"


def _parse_radar_uuid_choice(value: str) -> str | None:
  v = (value or "").strip()
  if v in ("", "(auto first)", "(auto second)", "(auto)"):
    return None
  if v == "(none)":
    return "__none__"
  return v


def _worker_from_args(
  args: argparse.Namespace,
  *,
  detect_threshold: float,
  human_threshold: float,
  camera_device: int | None = None,
  audio_device: int | None = None,
  radar1_uuid: str | None = None,
  radar2_uuid: str | None = None,
) -> InferenceWorker:
  cam = args.camera_device if camera_device is None else camera_device
  if cam is None:
    cam = -1
  worker = InferenceWorker(
    checkpoint=args.checkpoint,
    device=args.device,
    camera_device=int(cam),
    camera_width=args.camera_width,
    camera_height=args.camera_height,
    camera_fps=args.camera_fps,
    num_rx=args.num_rx,
    radar_profile=args.radar_profile,
    frame_rate=args.frame_rate,
    radar1_port=args.radar1_port,
    radar2_port=args.radar2_port,
    radar1_uuid=radar1_uuid,
    radar2_uuid=radar2_uuid,
    mirror_radar2=args.mirror_radar2,
    dual_radar_fuse=None if args.dual_radar_fuse == "auto" else args.dual_radar_fuse,
    no_radar=args.no_radar,
    no_audio=bool(getattr(args, "no_audio", False)),
    window_len=args.window,
    detect_threshold=detect_threshold,
    human_threshold=human_threshold,
    min_margin=args.min_margin,
    smooth_n=args.smooth_n,
    require_in_range=not args.no_range_gate,
    towards_bias=args.towards_bias,
    motion_threshold=args.motion_threshold,
    peak_ratio=args.peak_ratio,
    min_range_m=args.min_range_m,
    max_range_m=args.max_range_m,
    live_detector_preprocess=bool(args.live_detector_preprocess),
    use_motion_gate=bool(args.use_motion_gate),
  )
  if getattr(args, "no_audio", False):
    worker.set_audio_device(None, input_enabled=False)
  else:
    worker.set_audio_device(audio_device, input_enabled=True)
  return worker


class JetsonGuiApp:
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.root = tk.Tk()
    self.root.title("JetsonCA — Testing / Realtime")
    self.root.geometry("1280x760")
    self.root.minsize(980, 600)
    self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    self.worker = _worker_from_args(
      args,
      detect_threshold=args.detect_threshold,
      human_threshold=args.human_threshold,
      camera_device=args.camera_device,
      audio_device=args.audio_device,
    )
    self.worker_thread: threading.Thread | None = None
    self._sensor_scan_thread: threading.Thread | None = None
    self._probed_once = False

    self.status_var = tk.StringVar(value="idle — pick devices, then Start")
    self.prediction_var = tk.StringVar(value="-")
    self.hierarchy_var = tk.StringVar(value="")
    self.conf_var = tk.StringVar(value="0.00")
    self.raw_var = tk.StringVar(value="-")
    self.latency_var = tk.StringVar(value="-")
    self.range_var = tk.StringVar(value="-")
    self.radar_status_var = tk.StringVar(value="off")
    self.threshold_var = tk.StringVar(value=str(args.detect_threshold))
    self.human_threshold_var = tk.StringVar(value=str(args.human_threshold))
    self.min_range_var = tk.StringVar(value=str(args.min_range_m))
    self.max_range_var = tk.StringVar(value=str(args.max_range_m if args.max_range_m is not None else ""))
    self.detect_var = tk.StringVar(value="detect=0.00")
    self.gate_var = tk.StringVar(value="GATE closed")
    self.audio_verify_var = tk.StringVar(value="off")
    self.discovered_var = tk.StringVar(value="Click Refresh devices")
    self.checkpoint_var = tk.StringVar(value=str(args.checkpoint))
    self.reliance_camera_var = tk.StringVar(value="0%")
    self.reliance_radar_var = tk.StringVar(value="0%")
    self.reliance_audio_var = tk.StringVar(value="0%")
    self.expected_label_var = tk.StringVar(value="(none)")
    self.camera_enabled_var = tk.BooleanVar(value=True)
    self.radar_enabled_var = tk.BooleanVar(value=not args.no_radar)
    self.audio_enabled_var = tk.BooleanVar(value=bool(self.worker.enable_audio))
    self.camera_device_var = tk.StringVar(value="(scanning…)")
    self.audio_device_var = tk.StringVar(value="(default)")
    self.radar1_uuid_var = tk.StringVar(value="(auto first)")
    self.radar2_uuid_var = tk.StringVar(value="(auto second)")
    self._camera_index_to_label: dict[str, int] = {}
    self._audio_index_to_label: dict[str, int] = {}

    self.camera_photo = None
    self.radar_photo = None
    self.audio_photo = None
    self.prob_bars: list[ttk.Progressbar] = []
    self.prob_labels: list[tk.StringVar] = []

    self._build_ui()
    self.root.after(50, self._refresh_sensor_lists)
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
    left.rowconfigure(2, weight=2)
    left.columnconfigure(0, weight=1)

    cam_box = ttk.LabelFrame(left, text="Camera")
    cam_box.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
    cam_box.rowconfigure(0, weight=1)
    cam_box.columnconfigure(0, weight=1)
    self.camera_label = ttk.Label(cam_box, anchor="center")
    self.camera_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    radar_box = ttk.LabelFrame(left, text="Radar")
    radar_box.grid(row=1, column=0, sticky="nsew", pady=(0, 6))
    radar_box.rowconfigure(0, weight=1)
    radar_box.columnconfigure(0, weight=1)
    self.radar_label = ttk.Label(radar_box, anchor="center")
    self.radar_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    audio_box = ttk.LabelFrame(left, text="Audio — mel + waveform")
    audio_box.grid(row=2, column=0, sticky="nsew")
    audio_box.rowconfigure(0, weight=1)
    audio_box.columnconfigure(0, weight=1)
    self.audio_label = ttk.Label(audio_box, anchor="center")
    self.audio_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
    self.audio_box = audio_box
    if not self.worker.enable_audio:
      audio_box.grid_remove()

    right = ttk.Frame(main)
    right.grid(row=0, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)
    right.rowconfigure(1, weight=1)

    pred_box = ttk.LabelFrame(right, text="Prediction", padding=8)
    pred_box.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    pred_box.columnconfigure(0, weight=1)
    ttk.Label(pred_box, textvariable=self.prediction_var, font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
    ttk.Label(pred_box, textvariable=self.hierarchy_var, wraplength=360).grid(row=1, column=0, sticky="w")
    ttk.Label(pred_box, textvariable=self.detect_var).grid(row=2, column=0, sticky="w", pady=(4, 0))
    ttk.Label(pred_box, textvariable=self.gate_var, font=("Segoe UI", 10, "bold")).grid(row=2, column=1, sticky="w", pady=(4, 0))

    notebook = ttk.Notebook(right)
    notebook.grid(row=1, column=0, sticky="nsew")
    testing = ttk.Frame(notebook, padding=6)
    realtime = ttk.Frame(notebook, padding=6)
    notebook.add(testing, text="Testing")
    notebook.add(realtime, text="Realtime")
    self._fill_testing_tab(testing)
    self._fill_realtime_tab(realtime)

  def _fill_testing_tab(self, parent: ttk.Frame):
    parent.columnconfigure(1, weight=1)
    devices = ttk.LabelFrame(parent, text="Devices (applied on Start)")
    devices.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
    devices.columnconfigure(1, weight=1)

    self.sensor_refresh_btn = ttk.Button(devices, text="Refresh devices", command=self._refresh_sensor_lists)
    self.sensor_refresh_btn.grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=4)

    ttk.Label(devices, text="Camera").grid(row=1, column=0, sticky="w", padx=6, pady=3)
    self.camera_combo = ttk.Combobox(devices, textvariable=self.camera_device_var, state="readonly", width=36)
    self.camera_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=3)

    ttk.Label(devices, text="Audio input").grid(row=2, column=0, sticky="w", padx=6, pady=3)
    self.audio_combo = ttk.Combobox(devices, textvariable=self.audio_device_var, state="readonly", width=36)
    self.audio_combo.grid(row=2, column=1, sticky="ew", padx=6, pady=3)
    self.audio_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_audio_selected())
    if not self.worker.enable_audio:
      self.audio_combo.state(["disabled"])

    ttk.Label(devices, text="Radar 1 UUID").grid(row=3, column=0, sticky="w", padx=6, pady=3)
    self.radar1_combo = ttk.Combobox(devices, textvariable=self.radar1_uuid_var, state="readonly", width=36)
    self.radar1_combo.grid(row=3, column=1, sticky="ew", padx=6, pady=3)

    ttk.Label(devices, text="Radar 2 UUID").grid(row=4, column=0, sticky="w", padx=6, pady=3)
    self.radar2_combo = ttk.Combobox(devices, textvariable=self.radar2_uuid_var, state="readonly", width=36)
    self.radar2_combo.grid(row=4, column=1, sticky="ew", padx=6, pady=3)

    ttk.Label(devices, textvariable=self.discovered_var, justify="left", wraplength=420).grid(
      row=5, column=0, columnspan=2, sticky="w", padx=6, pady=6
    )

    eval_box = ttk.LabelFrame(parent, text="Evaluation")
    eval_box.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))
    eval_box.columnconfigure(1, weight=1)
    ttk.Label(eval_box, text="Expected label").grid(row=0, column=0, sticky="w", padx=6, pady=3)
    ttk.Combobox(
      eval_box,
      textvariable=self.expected_label_var,
      values=["(none)", *list(self.worker.labels)],
      state="readonly",
      width=28,
    ).grid(row=0, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(eval_box, text="Classes").grid(row=1, column=0, sticky="nw", padx=6, pady=3)
    ttk.Label(eval_box, text=", ".join(self.worker.labels), wraplength=360, justify="left").grid(
      row=1, column=1, sticky="w", padx=6, pady=3
    )
    ttk.Label(eval_box, text="Detect threshold").grid(row=2, column=0, sticky="w", padx=6, pady=3)
    ttk.Entry(eval_box, textvariable=self.threshold_var, width=8).grid(row=2, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(eval_box, text="Human threshold").grid(row=3, column=0, sticky="w", padx=6, pady=3)
    ttk.Entry(eval_box, textvariable=self.human_threshold_var, width=8).grid(row=3, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(eval_box, text="Min range (m)").grid(row=4, column=0, sticky="w", padx=6, pady=3)
    ttk.Entry(eval_box, textvariable=self.min_range_var, width=8).grid(row=4, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(eval_box, text="Max range (m)").grid(row=5, column=0, sticky="w", padx=6, pady=3)
    ttk.Entry(eval_box, textvariable=self.max_range_var, width=8).grid(row=5, column=1, sticky="w", padx=6, pady=3)
    ttk.Button(eval_box, text="Apply gates", command=self._apply_thresholds).grid(row=6, column=0, sticky="w", padx=6, pady=6)

  def _fill_realtime_tab(self, parent: ttk.Frame):
    parent.columnconfigure(1, weight=1)
    box = ttk.LabelFrame(parent, text="Run")
    box.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
    box.columnconfigure(1, weight=1)
    ttk.Label(box, text="Checkpoint").grid(row=0, column=0, sticky="w", padx=6, pady=3)
    ttk.Label(box, textvariable=self.checkpoint_var, wraplength=340).grid(row=0, column=1, sticky="w", padx=6, pady=3)
    run_row = ttk.Frame(box)
    run_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
    ttk.Button(run_row, text="Start", command=self.start).pack(side="left", padx=(0, 6))
    ttk.Button(run_row, text="Stop", command=self.stop).pack(side="left")

    mod = ttk.LabelFrame(box, text="Modality (while running)")
    mod.grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
    ttk.Checkbutton(mod, text="Camera", variable=self.camera_enabled_var, command=self._apply_modalities).grid(
      row=0, column=0, sticky="w", padx=4, pady=2
    )
    radar_cb = ttk.Checkbutton(mod, text="Radar", variable=self.radar_enabled_var, command=self._apply_modalities)
    radar_cb.grid(row=0, column=1, sticky="w", padx=4, pady=2)
    if self.args.no_radar:
      radar_cb.state(["disabled"])
    if self.worker.enable_audio:
      ttk.Checkbutton(mod, text="Audio", variable=self.audio_enabled_var, command=self._apply_modalities).grid(
        row=0, column=2, sticky="w", padx=4, pady=2
      )

    ttk.Label(box, text="Audio verify").grid(row=3, column=0, sticky="w", padx=6, pady=3)
    ttk.Label(box, textvariable=self.audio_verify_var, wraplength=340).grid(row=3, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(box, text="Run status").grid(row=4, column=0, sticky="w", padx=6, pady=3)
    ttk.Label(box, textvariable=self.status_var, wraplength=340).grid(row=4, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(box, text="Latency").grid(row=5, column=0, sticky="w", padx=6, pady=3)
    ttk.Label(box, textvariable=self.latency_var).grid(row=5, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(box, text="Range").grid(row=6, column=0, sticky="w", padx=6, pady=3)
    ttk.Label(box, textvariable=self.range_var).grid(row=6, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(box, text="Radar").grid(row=7, column=0, sticky="w", padx=6, pady=3)
    ttk.Label(box, textvariable=self.radar_status_var, wraplength=340).grid(row=7, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(box, text="Raw / conf").grid(row=8, column=0, sticky="w", padx=6, pady=3)
    ttk.Label(box, textvariable=self.raw_var).grid(row=8, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(box, textvariable=self.conf_var).grid(row=8, column=2, sticky="w", padx=6, pady=3)

    probs_box = ttk.LabelFrame(parent, text="Fused class probs")
    probs_box.grid(row=1, column=0, columnspan=3, sticky="nsew")
    probs_box.columnconfigure(1, weight=1)
    parent.rowconfigure(1, weight=1)
    for idx, label in enumerate(self.worker.labels):
      var = tk.StringVar(value=f"{label}: 0.00")
      self.prob_labels.append(var)
      ttk.Label(probs_box, textvariable=var).grid(row=idx, column=0, sticky="w", pady=1)
      bar = ttk.Progressbar(probs_box, maximum=100.0, length=180)
      bar.grid(row=idx, column=1, sticky="ew", padx=(8, 0), pady=1)
      self.prob_bars.append(bar)
    rel = len(self.worker.labels)
    ttk.Label(probs_box, text="Camera reliance").grid(row=rel, column=0, sticky="w", padx=2, pady=2)
    ttk.Label(probs_box, textvariable=self.reliance_camera_var).grid(row=rel, column=1, sticky="w")
    self.camera_reliance_bar = ttk.Progressbar(probs_box, maximum=100, length=180)
    self.camera_reliance_bar.grid(row=rel + 1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
    ttk.Label(probs_box, text="Radar reliance").grid(row=rel + 2, column=0, sticky="w", padx=2, pady=2)
    ttk.Label(probs_box, textvariable=self.reliance_radar_var).grid(row=rel + 2, column=1, sticky="w")
    self.radar_reliance_bar = ttk.Progressbar(probs_box, maximum=100, length=180)
    self.radar_reliance_bar.grid(row=rel + 3, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
    if self.worker.enable_audio:
      ttk.Label(probs_box, text="Audio reliance").grid(row=rel + 4, column=0, sticky="w", padx=2, pady=2)
      ttk.Label(probs_box, textvariable=self.reliance_audio_var).grid(row=rel + 4, column=1, sticky="w")
      self.audio_reliance_bar = ttk.Progressbar(probs_box, maximum=100, length=180)
      self.audio_reliance_bar.grid(row=rel + 5, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
    else:
      self.audio_reliance_bar = None

  def _camera_index_from_var(self) -> int:
    label = self.camera_device_var.get().strip()
    if label in ("(none)", "", "(scanning…)"):
      return -1
    if label.isdigit():
      return int(label)
    return self._camera_index_to_label.get(label, -1)

  def _audio_index_from_var(self) -> int | None:
    label = self.audio_device_var.get().strip()
    if label in ("(none)", "", "(scanning…)"):
      return None
    if label == "(default)":
      return None
    if label.isdigit():
      return int(label)
    return self._audio_index_to_label.get(label)

  def _on_audio_selected(self):
    if self.audio_device_var.get().strip() == "(none)":
      self.audio_enabled_var.set(False)
      self.worker.set_audio_device(None, input_enabled=False)
    else:
      self.worker.set_audio_device(self._audio_index_from_var(), input_enabled=True)

  def _refresh_sensor_lists(self):
    if self._sensor_scan_thread is not None and self._sensor_scan_thread.is_alive():
      return
    if hasattr(self, "sensor_refresh_btn"):
      self.sensor_refresh_btn.configure(state="disabled")
    if hasattr(self, "camera_combo"):
      self.camera_combo["values"] = ["(scanning…)"]

    def work():
      cameras = probe_camera_devices()
      audio_devices = list_audio_input_devices()
      radar_uuids = list_radar_uuids()
      self.root.after(0, lambda: self._apply_sensor_probe_results(cameras, audio_devices, radar_uuids))

    self._sensor_scan_thread = threading.Thread(target=work, daemon=True)
    self._sensor_scan_thread.start()

  def _apply_sensor_probe_results(self, cameras, audio_devices, radar_uuids):
    cam_labels = ["(none)"]
    self._camera_index_to_label = {}
    for cam in cameras:
      label = str(cam["label"])
      cam_labels.append(label)
      self._camera_index_to_label[label] = int(cam["index"])
    if hasattr(self, "camera_combo"):
      self.camera_combo["values"] = cam_labels

    prev_cam = self.camera_device_var.get().strip()
    keep_cam = prev_cam in self._camera_index_to_label
    if keep_cam:
      pass
    elif self.args.camera_device is not None:
      match = next((c for c in cameras if int(c["index"]) == int(self.args.camera_device)), None)
      self.camera_device_var.set(str(match["label"]) if match else ("(none)" if not cameras else cameras[0]["label"]))
    else:
      chosen = prefer_microsoft(cameras)
      self.camera_device_var.set(str(chosen["label"]) if chosen else "(none)")

    audio_labels = ["(default)", "(none)"]
    self._audio_index_to_label = {}
    for dev in audio_devices:
      label = str(dev["label"])
      audio_labels.append(label)
      self._audio_index_to_label[label] = int(dev["index"])
    if hasattr(self, "audio_combo"):
      self.audio_combo["values"] = audio_labels

    prev_audio = self.audio_device_var.get().strip()
    if prev_audio not in audio_labels or prev_audio in ("(scanning…)", ""):
      if self.args.audio_device is not None:
        match = next((d for d in audio_devices if int(d["index"]) == int(self.args.audio_device)), None)
        self.audio_device_var.set(str(match["label"]) if match else "(default)")
      else:
        mic = prefer_microsoft(audio_devices)
        self.audio_device_var.set(str(mic["label"]) if mic else "(default)")

    radar_choices = ["(auto first)", "(auto second)", "(none)", *list(radar_uuids)]
    if hasattr(self, "radar1_combo"):
      self.radar1_combo["values"] = radar_choices
      self.radar2_combo["values"] = radar_choices

    cam_lines = [f"• {c['label']}" for c in cameras] or ["• (no cameras)"]
    audio_lines = [f"• {d['label']}" for d in audio_devices] or ["• (no mics / install sounddevice)"]
    radar_lines = [f"• {u}" for u in radar_uuids] or ["• (no BGT UUIDs)"]
    self.discovered_var.set(
      "Cameras:\n" + "\n".join(cam_lines) + "\n\nAudio:\n" + "\n".join(audio_lines) + "\n\nRadar:\n" + "\n".join(radar_lines)
    )
    if hasattr(self, "sensor_refresh_btn"):
      self.sensor_refresh_btn.configure(state="normal")
    self._probed_once = True
    self._on_audio_selected()

  def _apply_modalities(self):
    self.worker.set_modalities(
      camera=self.camera_enabled_var.get(),
      radar=self.radar_enabled_var.get(),
      audio=self.audio_enabled_var.get(),
    )

  def _apply_thresholds(self):
    try:
      self.worker.set_threshold(float(self.threshold_var.get()))
      self.worker.set_human_threshold(float(self.human_threshold_var.get()))
      self.worker.min_range_m = float(self.min_range_var.get())
      if self.max_range_var.get().strip():
        self.worker.max_range_m = float(self.max_range_var.get())
    except ValueError:
      self.status_var.set("invalid threshold / range")

  def start(self):
    if self.worker_thread is not None and self.worker_thread.is_alive():
      return
    try:
      detect_thr = float(self.threshold_var.get())
      human_thr = float(self.human_threshold_var.get())
    except ValueError:
      self.status_var.set("invalid threshold")
      return
    self.worker = _worker_from_args(
      self.args,
      detect_threshold=detect_thr,
      human_threshold=human_thr,
      camera_device=self._camera_index_from_var(),
      audio_device=self._audio_index_from_var(),
      radar1_uuid=_parse_radar_uuid_choice(self.radar1_uuid_var.get()),
      radar2_uuid=_parse_radar_uuid_choice(self.radar2_uuid_var.get()),
    )
    try:
      if self.min_range_var.get().strip():
        self.worker.min_range_m = float(self.min_range_var.get())
      if self.max_range_var.get().strip():
        self.worker.max_range_m = float(self.max_range_var.get())
    except ValueError:
      pass
    audio_on = self.audio_enabled_var.get() and self.audio_device_var.get().strip() != "(none)"
    self.worker.set_audio_device(self._audio_index_from_var(), input_enabled=audio_on)
    self.worker.set_modalities(
      camera=self.camera_enabled_var.get(),
      radar=self.radar_enabled_var.get(),
      audio=audio_on,
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
    expected = self.expected_label_var.get().strip()
    if expected not in ("", "(none)") and pred not in ("-", "none", "uncertain", "background"):
      mark = " ✓" if pred == expected else " ✗"
    else:
      mark = ""
    self.status_var.set(str(state["status"]))
    self.prediction_var.set(pred + mark)
    self.hierarchy_var.set(str(state.get("hierarchy_text", "")))
    self.conf_var.set(f"{float(state['confidence']):.2f}")
    self.raw_var.set(str(state.get("raw_prediction", "-")))
    detect_p = float(state.get("detect_prob", 0.0))
    self.detect_var.set(f"detect={detect_p:.2f}")
    self.gate_var.set("GATE OPEN" if bool(state.get("gate_open", False)) else "GATE closed")
    self.latency_var.set(f"{float(state['latency_ms']):.1f} ms  (~{float(state['fps']):.1f} infer/s)")
    target_range = float(state.get("target_range_m", 0.0))
    in_range = bool(state.get("in_range", False))
    self.range_var.set(f"{target_range:.2f} m ({'in gate' if in_range else 'out of gate'})")
    self.radar_status_var.set(str(state.get("radar_status", "-")))
    self.audio_verify_var.set(str(state.get("audio_verify_text", "off")))

    probs = np.asarray(state["probs"], dtype=np.float32)
    for idx, (bar, label_var) in enumerate(zip(self.prob_bars, self.prob_labels)):
      value = float(probs[idx]) if idx < len(probs) else 0.0
      bar["value"] = value * 100.0
      label_var.set(f"{self.worker.labels[idx]}: {value:.2f}")

    rel = state.get("reliance") or {}
    cam_r = float(rel.get("camera", 0.0))
    rad_r = float(rel.get("radar", 0.0))
    aud_r = float(rel.get("audio", 0.0))
    self.reliance_camera_var.set(f"{cam_r * 100:.0f}%")
    self.reliance_radar_var.set(f"{rad_r * 100:.0f}%")
    self.reliance_audio_var.set(f"{aud_r * 100:.0f}%")
    self.camera_reliance_bar["value"] = cam_r * 100.0
    self.radar_reliance_bar["value"] = rad_r * 100.0
    if self.audio_reliance_bar is not None:
      self.audio_reliance_bar["value"] = aud_r * 100.0

    cam_size = (max(320, self.camera_label.winfo_width()), max(200, self.camera_label.winfo_height()))
    rad_size = (max(240, self.radar_label.winfo_width()), max(140, self.radar_label.winfo_height()))
    self._set_image(self.camera_label, state["camera_rgb"], "camera_photo", cam_size, letterbox=True)
    self._set_image(self.radar_label, state["radar_rgb"], "radar_photo", rad_size, letterbox=False)
    if self.worker.enable_audio:
      aud_size = (max(240, self.audio_label.winfo_width()), max(120, self.audio_label.winfo_height()))
      self._set_image(self.audio_label, state["audio_rgb"], "audio_photo", aud_size, letterbox=False)

    self.root.after(100, self._refresh_ui)

  def on_close(self):
    self.stop()
    self.root.destroy()

  def run(self):
    self.root.mainloop()


def parse_args():
  p = argparse.ArgumentParser(description="JetsonCA Testing + Realtime GUI (Crossattention subset)")
  p.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_multimodal_crossattention.pt"))
  p.add_argument("--device", type=str, default=default_device())
  p.add_argument(
    "--camera-device",
    type=int,
    default=None,
    help="OpenCV index. Default: Microsoft cam if named, else first live camera (not hardcoded 0)",
  )
  p.add_argument(
    "--audio-device",
    type=int,
    default=None,
    help="sounddevice index. Default: Microsoft mic if named, else system default",
  )
  p.add_argument("--no-audio", action="store_true", help="Do not open a microphone even if the checkpoint has audio")
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
  p.add_argument("--dual-radar-fuse", choices=("auto", "none", "mean", "max"), default="auto",
                 help="auto = use checkpoint config (Crossattention train default: none)")
  p.add_argument("--window", type=int, default=30)
  p.add_argument("--detect-threshold", type=float, default=0.55,
                 help="GATE threshold on detect_prob (learned detect head, else human fallback)")
  p.add_argument("--human-threshold", type=float, default=0.55)
  p.add_argument("--min-margin", type=float, default=0.0,
                 help="Require top1-top2 prob margin; 0=always take argmax (default on Jetson)")
  p.add_argument("--smooth-n", type=int, default=5, help="Majority-vote window over recent labels")
  p.add_argument("--towards-bias", type=float, default=0.0,
                 help="Downweight walking_towards prior (0=off)")
  p.add_argument("--motion-threshold", type=float, default=0.35,
                 help="Only used with --use-motion-gate")
  p.add_argument("--peak-ratio", type=float, default=5.0,
                 help="Only used with --use-motion-gate")
  p.add_argument("--use-motion-gate", action="store_true",
                 help="Force classical Doppler motion gate (ignored when DETECT head is present)")
  p.add_argument("--live-detector-preprocess", action="store_true",
                 help="Apply train-time CFAR/ROI live (slow on Jetson; off by default)")
  p.add_argument("--no-range-gate", action="store_true",
                 help="Allow predictions even when radar peak is outside min/max range")
  p.add_argument("--min-range-m", type=float, default=0.3)
  p.add_argument("--max-range-m", type=float, default=2.5)
  return p.parse_args()


def main():
  args = parse_args()
  apply_jetson_runtime_tweaks()
  JetsonGuiApp(args).run()


if __name__ == "__main__":
  main()
