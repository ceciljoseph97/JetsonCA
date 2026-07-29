from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
  import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional recording dependency
  imageio = None  # type: ignore[assignment]

from label_hierarchy import inference_label
from model import MultiModalCrossAttentionNet
from preprocessing import do_inference_processing
from radar_bgt import configure_bgt_device
from range_doppler import DopplerAlgo

try:
  from ifxradarsdk import get_version_full
  from ifxradarsdk.common.exceptions import ErrorFrameAcquisitionFailed
  from ifxradarsdk.fmcw import DeviceFmcw
except ImportError:  # pragma: no cover - allows GUI import without hardware SDK
  def get_version_full() -> str:
    return "ifxradarsdk not installed"

  ErrorFrameAcquisitionFailed = Exception  # type: ignore[misc, assignment]
  DeviceFmcw = None  # type: ignore[misc, assignment]


@contextmanager
def _opencv_log_quiet():
  prev_level = None
  try:
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
      prev_level = cv2.utils.logging.getLogLevel()
      cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
  except Exception:
    prev_level = None
  try:
    yield
  finally:
    if prev_level is not None:
      try:
        cv2.utils.logging.setLogLevel(prev_level)
      except Exception:
        pass


@contextmanager
def _suppress_native_stdio():
  """Hide MSMF/DSHOW printf spam (e.g. 'error = 0') during VideoCapture."""
  devnull = os.open(os.devnull, os.O_WRONLY)
  saved_out = os.dup(1)
  saved_err = os.dup(2)
  try:
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    yield
  finally:
    os.dup2(saved_out, 1)
    os.dup2(saved_err, 2)
    os.close(devnull)
    os.close(saved_out)
    os.close(saved_err)


@contextmanager
def _camera_capture_quiet():
  with _opencv_log_quiet(), _suppress_native_stdio():
    yield


def _capture_backends_for_platform() -> list[int]:
  if sys.platform != "win32":
    return [0]
  order: list[int] = []
  if hasattr(cv2, "CAP_MSMF"):
    order.append(cv2.CAP_MSMF)
  if hasattr(cv2, "CAP_DSHOW"):
    order.append(cv2.CAP_DSHOW)
  if not order:
    order.append(0)
  return order


def open_video_capture(
  device_id: int,
  *,
  verify_frame: bool = True,
  read_retries: int = 5,
) -> cv2.VideoCapture | None:
  """Open a camera by index; tries MSMF then DSHOW on Windows."""
  with _camera_capture_quiet():
    for backend in _capture_backends_for_platform():
      cap = cv2.VideoCapture(device_id, backend)
      if not cap.isOpened():
        cap.release()
        continue
      if verify_frame:
        ok = False
        for _ in range(read_retries):
          ok, _ = cap.read()
          if ok:
            break
          time.sleep(0.04)
        if not ok:
          cap.release()
          continue
      return cap
  return None


class CameraStream:
  def __init__(self, device_id: int, width: int, height: int, fps: float):
    self.device_id = device_id
    self.width = width
    self.height = height
    self.fps = fps
    self._cap = None
    self._thread = None
    self._stop = threading.Event()
    self._lock = threading.Lock()
    self._latest_rgb = None

  def start(self, warmup_s: float = 5.0):
    self._cap = open_video_capture(self.device_id, verify_frame=True)
    if self._cap is None:
      raise RuntimeError(f"Could not open camera device {self.device_id}")

    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    self._cap.set(cv2.CAP_PROP_FPS, self.fps)
    self._thread = threading.Thread(target=self._reader_loop, daemon=True)
    self._thread.start()
    deadline = time.time() + warmup_s
    while time.time() < deadline and not self._stop.is_set():
      if self.get_latest() is not None:
        break
      time.sleep(0.05)
    return self

  def _reader_loop(self):
    while not self._stop.is_set():
      ok, frame = self._cap.read()
      if not ok:
        time.sleep(0.01)
        continue
      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      with self._lock:
        self._latest_rgb = rgb

  def get_latest(self) -> np.ndarray | None:
    with self._lock:
      if self._latest_rgb is None:
        return None
      return self._latest_rgb.copy()

  def stop(self):
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=2.0)
    if self._cap is not None:
      self._cap.release()


def _camera_frame_signature(frame: np.ndarray) -> np.ndarray:
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
  small = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
  return small.astype(np.float32)


def _camera_signatures_similar(sig_a: np.ndarray, sig_b: np.ndarray, *, threshold: float = 6.0) -> bool:
  return float(np.abs(sig_a - sig_b).mean()) < threshold


def _camera_frame_is_live(frame: np.ndarray, *, min_mean: float = 6.0, min_std: float = 2.5) -> bool:
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
  return float(gray.mean()) >= min_mean and float(gray.std()) >= min_std


def _read_probe_frames(cap: cv2.VideoCapture, *, count: int = 4) -> list[np.ndarray]:
  frames: list[np.ndarray] = []
  for _ in range(count):
    ok, frame = cap.read()
    if ok and frame is not None and frame.size > 0:
      frames.append(frame)
    time.sleep(0.06)
  return frames


def probe_camera_devices(max_index: int = 8, stop_after_misses: int = 2) -> list[dict[str, int | str | bool]]:
  """Return indices that OpenCV can open with a distinct live stream."""
  found: list[dict[str, int | str | bool]] = []
  signatures: list[np.ndarray] = []
  misses = 0
  with _camera_capture_quiet():
    for index in range(max_index):
      cap = open_video_capture(index, verify_frame=False, read_retries=1)
      if cap is None or not cap.isOpened():
        if cap is not None:
          cap.release()
        misses += 1
        if misses >= stop_after_misses:
          break
        continue

      frames = _read_probe_frames(cap, count=4)
      w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      cap.release()

      live_frames = [frame for frame in frames if _camera_frame_is_live(frame)]
      if not live_frames:
        misses += 1
        if misses >= stop_after_misses:
          break
        continue

      sig = _camera_frame_signature(live_frames[-1])
      if any(_camera_signatures_similar(sig, prev) for prev in signatures):
        continue

      signatures.append(sig)
      label = f"Camera {index} ({w}x{h})"
      found.append({"index": index, "label": label, "width": w, "height": h})
      misses = 0
  return found


def load_checkpoint(path: Path, device: str):
  checkpoint = torch.load(path, map_location=device)
  config = checkpoint["config"]
  labels = checkpoint.get("activity_labels", checkpoint["labels"])
  model = MultiModalCrossAttentionNet(
    num_classes=len(labels),
    num_activity_classes=int(config.get("num_activity_classes", len(labels))),
    model_dim=int(config["model_dim"]),
    num_heads=int(config["num_heads"]),
    num_layers=int(config["num_layers"]),
    dropout=float(config["dropout"]),
    modality_dropout=0.0,
    temporal_mode=str(config.get("temporal_mode", "none")),
    enable_human_head=bool(config.get("enable_human_head", False)),
  ).to(device)

  state = dict(checkpoint["model_state"])
  remapped = {}
  for key, value in state.items():
    if key.startswith("classifier."):
      remapped[key.replace("classifier.", "activity_classifier.", 1)] = value
    else:
      remapped[key] = value
  model.load_state_dict(remapped, strict=False)
  model.eval()
  return model, labels, config


def preprocess_camera_frame(frame_rgb: np.ndarray, image_size: int) -> torch.Tensor:
  resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
  tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
  tensor = (tensor - 0.5) / 0.5
  return tensor


def make_radar_montage(radar_tensor: np.ndarray) -> np.ndarray:
  channels = [np.asarray(radar_tensor[i], dtype=np.float32) for i in range(radar_tensor.shape[0])]
  channel_images = []
  for channel in channels:
    channel = channel - channel.min()
    denom = max(float(channel.max()), 1e-6)
    channel = np.uint8(np.clip(channel / denom, 0.0, 1.0) * 255.0)
    channel = cv2.applyColorMap(channel, cv2.COLORMAP_VIRIDIS)
    channel_images.append(channel)
  montage = np.concatenate(channel_images, axis=1)
  return cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)


class SessionRecorder:
  def __init__(self, output_path: Path | None, fps: float):
    self.output_path = output_path
    self.fps = fps
    self._writer = None

  def start(self):
    if self.output_path is None:
      return
    if imageio is None:
      raise ImportError("imageio is required for session recording; pip install imageio")
    self.output_path.parent.mkdir(parents=True, exist_ok=True)
    self._writer = imageio.get_writer(self.output_path, fps=self.fps)

  def add(self, rgb_frame: np.ndarray):
    if self._writer is not None:
      self._writer.append_data(rgb_frame)

  def close(self):
    if self._writer is not None:
      self._writer.close()
      self._writer = None


class LiveVisualizer:
  def __init__(self, labels: list[str], output_path: Path | None, record_fps: float):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import CheckButtons

    self.labels = labels
    self.output_path = output_path
    self.record_fps = record_fps
    self.on_close = None
    self.on_toggle = None
    self._lock = threading.Lock()
    self._state = {
      "camera_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
      "radar_rgb": np.zeros((32, 96, 3), dtype=np.uint8),
      "probs": np.zeros(len(labels), dtype=np.float32),
      "prediction_text": "warming up",
      "camera_enabled": True,
      "radar_enabled": True,
      "recording": output_path is not None,
    }
    self._recorder = SessionRecorder(output_path, record_fps)
    self._recorder.start()

    self.plt = plt
    self.fig = plt.figure("Crossattention Realtime", figsize=(12, 8))
    self.fig.canvas.mpl_connect("close_event", self._handle_close)
    self.fig.canvas.mpl_connect("key_press_event", self._handle_keypress)

    gs = self.fig.add_gridspec(2, 2, width_ratios=[3.0, 1.2], height_ratios=[1, 1], hspace=0.25, wspace=0.25)
    self.camera_ax = self.fig.add_subplot(gs[0, 0])
    self.radar_ax = self.fig.add_subplot(gs[1, 0])
    self.bar_ax = self.fig.add_subplot(gs[:, 1])

    self.camera_ax.set_title("Camera modality")
    self.radar_ax.set_title("Radar RD map modality")
    self.camera_ax.axis("off")
    self.radar_ax.axis("off")

    self.camera_im = self.camera_ax.imshow(self._state["camera_rgb"])
    self.radar_im = self.radar_ax.imshow(self._state["radar_rgb"])
    self.bar_plot = self.bar_ax.barh(np.arange(len(labels)), np.zeros(len(labels)))
    self.bar_ax.set_xlim(0, 1)
    self.bar_ax.set_yticks(np.arange(len(labels)))
    self.bar_ax.set_yticklabels(labels)
    self.bar_ax.invert_yaxis()
    self.bar_ax.set_xlabel("probability")
    self.status_text = self.fig.text(0.02, 0.965, "", fontsize=10, ha="left", va="top")

    toggle_ax = self.fig.add_axes([0.77, 0.02, 0.18, 0.12])
    self.check = CheckButtons(toggle_ax, ["camera", "radar"], [True, True])
    self.check.on_clicked(self._handle_check)
    toggle_ax.set_title("Enabled")

    self._anim = FuncAnimation(self.fig, self._refresh, interval=int(1000 / max(record_fps, 1.0)), blit=False)

  def _handle_close(self, _event):
    self._recorder.close()
    if self.on_close is not None:
      self.on_close()

  def _handle_check(self, _label):
    status = self.check.get_status()
    camera_enabled, radar_enabled = bool(status[0]), bool(status[1])
    if not camera_enabled and not radar_enabled:
      # Keep at least one modality active.
      self.check.set_active(0)
      camera_enabled = True
      radar_enabled = bool(self.check.get_status()[1])
    if self.on_toggle is not None:
      self.on_toggle(camera_enabled, radar_enabled)

  def _handle_keypress(self, event):
    if event.key == "c":
      self.check.set_active(0)
    elif event.key == "r":
      self.check.set_active(1)

  def update(
    self,
    camera_rgb: np.ndarray,
    radar_rgb: np.ndarray,
    probs: np.ndarray,
    prediction_text: str,
    camera_enabled: bool,
    radar_enabled: bool,
  ):
    with self._lock:
      self._state = {
        "camera_rgb": camera_rgb,
        "radar_rgb": radar_rgb,
        "probs": probs,
        "prediction_text": prediction_text,
        "camera_enabled": camera_enabled,
        "radar_enabled": radar_enabled,
        "recording": self.output_path is not None,
      }

  def _refresh(self, _frame):
    with self._lock:
      state = dict(self._state)

    camera_rgb = state["camera_rgb"].copy()
    radar_rgb = state["radar_rgb"].copy()

    if not state["camera_enabled"]:
      camera_rgb[:] = 20
      cv2.putText(camera_rgb, "CAMERA DISABLED", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2)
    if not state["radar_enabled"]:
      radar_rgb[:] = 20
      cv2.putText(radar_rgb, "RADAR DISABLED", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2)

    self.camera_im.set_data(camera_rgb)
    self.radar_im.set_data(radar_rgb)

    probs = np.asarray(state["probs"], dtype=np.float32)
    for bar, prob in zip(self.bar_plot, probs):
      bar.set_width(float(prob))

    status = (
      f"{state['prediction_text']} | "
      f"camera={'on' if state['camera_enabled'] else 'off'} "
      f"radar={'on' if state['radar_enabled'] else 'off'} "
      f"recording={'on' if state['recording'] else 'off'}"
    )
    self.status_text.set_text(status)

    self.fig.canvas.draw_idle()
    if self.output_path is not None:
      buffer = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
      self._recorder.add(buffer.copy())
    return [self.camera_im, self.radar_im, self.status_text, *self.bar_plot]

  def start(self):
    self.plt.show(block=True)


class LivePredictionPipeline:
  def __init__(
    self,
    model: torch.nn.Module,
    labels: list[str],
    image_size: int,
    device: str,
    num_rx: int,
    radar_profile: str,
    frame_rate_hz: float | None,
    camera_device: int,
    camera_width: int,
    camera_height: int,
    camera_fps: float,
    detect_threshold: float,
  ):
    self.model = model
    self.labels = labels
    self.image_size = image_size
    self.device = device
    self.num_rx = num_rx
    self.radar_profile = radar_profile
    self.frame_rate_hz = frame_rate_hz
    self.camera_device = camera_device
    self.camera_width = camera_width
    self.camera_height = camera_height
    self.camera_fps = camera_fps
    self.detect_threshold = detect_threshold
    self.stop_event = threading.Event()
    self.visualizer = None
    self._toggle_lock = threading.Lock()
    self.camera_enabled = True
    self.radar_enabled = True
    self.radar_buffer: deque[torch.Tensor] = deque(maxlen=30)
    self.camera_buffer: deque[torch.Tensor] = deque(maxlen=30)
    self.last_probs = np.zeros(len(labels), dtype=np.float32)

  def attach_visualizer(self, visualizer: LiveVisualizer):
    self.visualizer = visualizer
    visualizer.on_close = self.stop
    visualizer.on_toggle = self.set_modalities

  def stop(self):
    self.stop_event.set()

  def set_modalities(self, camera_enabled: bool, radar_enabled: bool):
    with self._toggle_lock:
      self.camera_enabled = camera_enabled
      self.radar_enabled = radar_enabled

  def _current_modalities(self) -> tuple[bool, bool]:
    with self._toggle_lock:
      return self.camera_enabled, self.radar_enabled

  def _predict(self) -> tuple[np.ndarray, str]:
    radar_tensor = torch.stack(list(self.radar_buffer), dim=0).unsqueeze(0).to(self.device)
    camera_tensor = torch.stack(list(self.camera_buffer), dim=0).unsqueeze(0).to(self.device)
    camera_enabled, radar_enabled = self._current_modalities()
    radar_present = torch.tensor([radar_enabled], dtype=torch.bool, device=self.device)
    camera_present = torch.tensor([camera_enabled], dtype=torch.bool, device=self.device)

    with torch.no_grad():
      outputs = self.model(radar_tensor, camera_tensor, radar_present=radar_present, camera_present=camera_present)
      probs = F.softmax(outputs["logits"][0], dim=-1).detach().cpu().numpy()

    top_idx = int(np.argmax(probs))
    top_label = self.labels[top_idx]
    prediction_text = f"pred={top_label} conf={probs[top_idx]:.2f}"
    return probs, prediction_text

  def run(self):
    print(f"Radar SDK: {get_version_full()}")
    print(f"Labels: {self.labels}")
    print(f"Radar profile: {self.radar_profile}")
    print("Shortcuts: press 'c' to toggle camera, 'r' to toggle radar")

    if DeviceFmcw is None:
      raise ImportError(
        "ifxradarsdk is required for live radar capture. "
        "Install editable from Exploration/radar_sdk/sdk/py/wrapper_radarsdk "
        "(or the Infineon wheel)."
      )

    camera_stream = CameraStream(
      self.camera_device,
      self.camera_width,
      self.camera_height,
      self.camera_fps,
    ).start()
    try:
      with DeviceFmcw() as device:
        cfg = configure_bgt_device(device, self.num_rx, profile=self.radar_profile, frame_rate_hz=self.frame_rate_hz)
        algo = DopplerAlgo(cfg, self.num_rx)
        print(
          f"Radar cfg: {cfg['num_chirps_per_frame']} chirps, "
          f"{cfg['num_samples_per_chirp']} samples/chirp, {cfg['frame_rate_hz']:.1f} Hz"
        )

        while not self.stop_event.is_set():
          try:
            raw = device.get_next_frame()[0]
          except ErrorFrameAcquisitionFailed:
            continue

          antenna_maps = [algo.compute_doppler_map(raw[i, :, :], i) for i in range(self.num_rx)]
          radar_tensor = do_inference_processing(antenna_maps).squeeze(0).cpu()
          radar_montage = make_radar_montage(radar_tensor.numpy())
          self.radar_buffer.append(radar_tensor)

          camera_rgb = camera_stream.get_latest()
          if camera_rgb is None:
            continue
          camera_tensor = preprocess_camera_frame(camera_rgb, self.image_size).cpu()
          self.camera_buffer.append(camera_tensor)

          probs = self.last_probs
          prediction_text = "warming up"
          if len(self.radar_buffer) == self.radar_buffer.maxlen and len(self.camera_buffer) == self.camera_buffer.maxlen:
            probs, prediction_text = self._predict()
            self.last_probs = probs
            if float(probs.max()) >= self.detect_threshold:
              prediction_text += " DETECT"

          if self.visualizer is not None:
            camera_enabled, radar_enabled = self._current_modalities()
            self.visualizer.update(
              camera_rgb=camera_rgb,
              radar_rgb=radar_montage,
              probs=probs,
              prediction_text=prediction_text,
              camera_enabled=camera_enabled,
              radar_enabled=radar_enabled,
            )
    finally:
      camera_stream.stop()
      print("Released camera and radar.")


def parse_args():
  parser = argparse.ArgumentParser(description="Realtime multimodal radar+camera walking demo")
  parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_multimodal_crossattention.pt"))
  parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--num-rx", type=int, default=3)
  parser.add_argument("--radar-profile", choices=("safe", "balanced", "gesture"), default="safe")
  parser.add_argument("--frame-rate", type=float, default=3.0)
  parser.add_argument("--camera-device", type=int, default=0)
  parser.add_argument("--camera-width", type=int, default=640)
  parser.add_argument("--camera-height", type=int, default=480)
  parser.add_argument("--camera-fps", type=float, default=15.0)
  parser.add_argument("--detect-threshold", type=float, default=0.6)
  parser.add_argument("--gui", action="store_true")
  parser.add_argument("--record", type=Path, default=None, help="Optional .mp4 or .gif path for GUI recording")
  parser.add_argument("--record-fps", type=float, default=8.0)
  return parser.parse_args()


def main():
  args = parse_args()
  if not args.checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

  model, labels, config = load_checkpoint(args.checkpoint, args.device)
  pipeline = LivePredictionPipeline(
    model=model,
    labels=labels,
    image_size=int(config["image_size"]),
    device=args.device,
    num_rx=args.num_rx,
    radar_profile=args.radar_profile,
    frame_rate_hz=args.frame_rate,
    camera_device=args.camera_device,
    camera_width=args.camera_width,
    camera_height=args.camera_height,
    camera_fps=args.camera_fps,
    detect_threshold=args.detect_threshold,
  )

  worker = None
  try:
    if args.gui:
      visualizer = LiveVisualizer(labels=labels, output_path=args.record, record_fps=args.record_fps)
      pipeline.attach_visualizer(visualizer)
      worker = threading.Thread(target=pipeline.run, daemon=False)
      worker.start()
      visualizer.start()
    else:
      pipeline.run()
  except KeyboardInterrupt:
    pipeline.stop()
  finally:
    pipeline.stop()
    if worker is not None and worker.is_alive():
      worker.join(timeout=3.0)


if __name__ == "__main__":
  main()
