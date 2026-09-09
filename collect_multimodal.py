#!/usr/bin/env python3
"""Collect dual BGT60 + camera + mic clips on Jetson into Crossattention-compatible layout.

Uses the same camera/audio discovery as gui_app.py:
  - jetson_env.ensure_conda_lib_path (cv2 CXXABI)
  - realtime_multimodal.open_video_capture / probe_camera_devices
  - live_audio.LiveAudioBuffer (sounddevice → arecord → ffmpeg; no PortAudio required)

Layout (same as Crossattention/train.py):
  data/{group}/{class}/{radar,radar1,radar2,camera,audio,meta}/{idx:02d}.npy|.json

Examples (on Jetson):
  python collect_multimodal.py --list-devices
  python collect_multimodal.py --radar-profile gesture --frames 40 --frame-rate 10 \\
    --group gestures --class push --count 20
  python collect_multimodal.py --radar1-port /dev/ttyACM0 --radar2-port /dev/ttyACM1 \\
    --class pinch_index --count 15
"""

from __future__ import annotations

# Prefer conda libstdc++ BEFORE cv2 (same as gui_app).
from jetson_env import ensure_conda_lib_path

ensure_conda_lib_path(reexec=True)

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from audio_features import audio_is_dead_microphone, audio_is_usable, audio_wave_stats
from device_select import attach_camera_names, match_audio_to_camera, prefer_microsoft, prefer_microsoft_index
from live_audio import LiveAudioBuffer, list_audio_input_devices
from radar_utils import DualRadarSession, list_radar_ports, list_radar_uuids
from range_gating import profile_metrics
from realtime_multimodal import open_video_capture, probe_camera_devices

try:
  from ifxradarsdk.common.exceptions import ErrorFrameAcquisitionFailed
except ImportError:  # pragma: no cover
  ErrorFrameAcquisitionFailed = RuntimeError  # type: ignore[misc, assignment]

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "data"


class CameraRecorder:
  """Frame grabber using the same open_video_capture() path as the GUI."""

  def __init__(self, device_id: int, width: int, height: int, fps: float):
    self.device_id = device_id
    self.width = width
    self.height = height
    self.fps = fps
    self._cap = None
    self._thread = None
    self._stop = threading.Event()
    self._frames: list[np.ndarray] = []
    self._timestamps: list[float] = []

  def start(self):
    import cv2

    self._cap = open_video_capture(self.device_id, verify_frame=True)
    if self._cap is None:
      raise RuntimeError(
        f"Could not open camera device {self.device_id}. "
        "Run --list-devices (same probe as GUI)."
      )
    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    self._cap.set(cv2.CAP_PROP_FPS, self.fps)
    self._thread = threading.Thread(target=self._reader_loop, daemon=True)
    self._thread.start()
    # brief warmup so first clip isn't empty
    deadline = time.time() + 1.5
    while time.time() < deadline and not self._frames:
      time.sleep(0.05)
    return self

  def _reader_loop(self):
    import cv2

    min_period_s = 1.0 / max(self.fps, 1.0)
    last_saved = 0.0
    while not self._stop.is_set():
      ok, frame = self._cap.read()
      now = time.perf_counter()
      if not ok:
        time.sleep(0.01)
        continue
      if (now - last_saved) < min_period_s:
        continue
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      if frame.shape[1] != self.width or frame.shape[0] != self.height:
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
      self._frames.append(frame)
      self._timestamps.append(now)
      last_saved = now

  def stop(self) -> tuple[np.ndarray, np.ndarray]:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=2.0)
    if self._cap is not None:
      self._cap.release()

    if not self._frames:
      return np.empty((0, self.height, self.width, 3), dtype=np.uint8), np.empty((0,), dtype=np.float64)

    t0 = self._timestamps[0]
    timestamps = np.asarray([ts - t0 for ts in self._timestamps], dtype=np.float64)
    frames = np.stack(self._frames, axis=0).astype(np.uint8)
    return frames, timestamps


def _audio_timestamps(wave: np.ndarray, sample_rate: int, block: int = 1024) -> np.ndarray:
  n = int(np.asarray(wave).reshape(-1).size)
  if n <= 0:
    return np.empty((0,), dtype=np.float64)
  step = max(1, int(block))
  starts = np.arange(0, n, step, dtype=np.float64)
  return starts / float(sample_rate)


def record_dual_radar_clip(session: DualRadarSession, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
  if not session.slots[0].available or not session.slots[1].available:
    raise RuntimeError(
      "Need two live BGT60 radars (mirroring disabled). "
      f"status={session.status_text}"
    )
  radar1_frames: list[np.ndarray] = []
  radar2_frames: list[np.ndarray] = []
  for _ in range(num_frames):
    radar1, radar2 = session.read_tensors()
    if radar1 is None:
      raise ErrorFrameAcquisitionFailed("radar1 frame drop")
    if radar2 is None:
      raise ErrorFrameAcquisitionFailed("radar2 frame drop")
    radar1_frames.append(radar1.numpy())
    radar2_frames.append(radar2.numpy())
  return (
    np.stack(radar1_frames, axis=0).astype(np.float32),
    np.stack(radar2_frames, axis=0).astype(np.float32),
  )


def record_multimodal_clip_dual(session: DualRadarSession, num_frames: int, args):
  camera = CameraRecorder(args.camera_device, args.camera_width, args.camera_height, args.camera_fps).start()
  audio_buf: LiveAudioBuffer | None = None
  if args.audio:
    audio_buf = LiveAudioBuffer(sample_rate=args.audio_sample_rate)
    audio_buf.start(device=args.audio_device)
    if not audio_buf.running:
      print(f"  WARNING: mic failed ({audio_buf.last_error}) — continuing without audio")
      audio_buf = None
    else:
      audio_buf.start_recording_sink()
  try:
    radar_clip, radar2_clip = record_dual_radar_clip(session, num_frames)
  finally:
    camera_frames, camera_timestamps = camera.stop()
    if audio_buf is None:
      audio_samples, audio_timestamps = None, None
    else:
      wave = audio_buf.stop_recording_sink()
      if wave.size == 0:
        wave = audio_buf.snapshot()
      audio_buf.stop()
      if wave.ndim == 1:
        audio_samples = wave[:, np.newaxis]
      else:
        audio_samples = wave
      audio_timestamps = _audio_timestamps(wave, args.audio_sample_rate, args.audio_blocksize)
  return radar_clip, radar2_clip, camera_frames, camera_timestamps, audio_samples, audio_timestamps


def next_index(class_dir: Path) -> int:
  radar_dir = class_dir / "radar"
  if not radar_dir.is_dir():
    radar_dir = class_dir / "radar1"
  existing = sorted(radar_dir.glob("*.npy"))
  if not existing:
    return 0
  return max(int(p.stem) for p in existing) + 1


def save_multimodal_clip(
  class_dir: Path,
  idx: int,
  radar_clip,
  radar2_clip,
  camera_frames,
  camera_timestamps,
  audio_samples,
  audio_timestamps,
  gesture: str,
  frame_rate_hz: float,
  radar_uuids: list[str],
  radar_ports: list[str],
  radar_profile: str,
  min_range_m: float,
  max_range_m: float | None,
  audio_sample_rate: int | None = None,
  audio_channels: int | None = None,
):
  radar_dir = class_dir / "radar"
  radar1_dir = class_dir / "radar1"
  radar2_dir = class_dir / "radar2"
  camera_dir = class_dir / "camera"
  audio_dir = class_dir / "audio"
  meta_dir = class_dir / "meta"
  for folder in (radar_dir, radar1_dir, radar2_dir, camera_dir, audio_dir, meta_dir):
    folder.mkdir(parents=True, exist_ok=True)

  stem = f"{idx:02d}"
  radar_path = radar_dir / f"{stem}.npy"
  radar1_path = radar1_dir / f"{stem}.npy"
  radar2_path = radar2_dir / f"{stem}.npy"
  camera_path = camera_dir / f"{stem}.npy"
  audio_path = audio_dir / f"{stem}.npy"
  meta_path = meta_dir / f"{stem}.json"

  np.save(radar_path, radar_clip)
  np.save(radar1_path, radar_clip)
  np.save(radar2_path, radar2_clip)
  np.save(camera_path, camera_frames)
  metrics = profile_metrics(radar_profile)
  meta_payload = {
    "gesture": gesture,
    "index": idx,
    "radar_shape": list(radar_clip.shape),
    "radar1_shape": list(radar_clip.shape),
    "radar2_shape": list(radar2_clip.shape),
    "camera_shape": list(camera_frames.shape),
    "radar_frame_rate_hz": frame_rate_hz,
    "camera_timestamps_s": camera_timestamps.tolist(),
    "radar_uuids": radar_uuids,
    "radar_ports": radar_ports,
    "radar2_mirrored": False,
    "radar_profile": radar_profile,
    "range_resolution_m": metrics["range_resolution_m"],
    "max_range_m": metrics["max_range_m"],
    "min_range_m": float(min_range_m),
    "max_recognition_range_m": float(max_range_m if max_range_m is not None else metrics["max_range_m"]),
    "host": "jetson",
  }
  if audio_samples is not None and audio_samples.size > 0:
    stats = audio_wave_stats(audio_samples)
    coarse_label = gesture.split("_")[0] if gesture else class_dir.parent.name
    if not audio_is_usable(audio_samples, label=coarse_label):
      reason = "dead mic" if audio_is_dead_microphone(audio_samples) else "too quiet for action class"
      print(
        f"  WARNING: audio unusable ({reason}: rms={stats['rms']:.2e}, "
        f"std={stats['std']:.2e}, peak={stats['peak']:.2e}) — check device/mute/gain"
      )
    np.save(audio_path, audio_samples)
    meta_payload["audio_shape"] = list(audio_samples.shape)
    meta_payload["audio_sample_rate_hz"] = int(audio_sample_rate or 0)
    meta_payload["audio_channels"] = int(audio_channels or audio_samples.shape[-1])
    meta_payload["audio_chunk_timestamps_s"] = audio_timestamps.tolist()
  meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
  return radar_path, radar1_path, radar2_path, camera_path, audio_path, meta_path


def _probe_cameras() -> list[dict]:
  try:
    return attach_camera_names(probe_camera_devices())
  except Exception as exc:
    print(f"  camera probe failed: {exc}")
    return []


def _default_camera_index() -> int:
  cams = _probe_cameras()
  idx = prefer_microsoft_index(cams)
  if idx is not None:
    return int(idx)
  if cams:
    return int(cams[0]["index"])
  return 0


def _default_audio_device(camera_label: str | None = None) -> int | str | None:
  """Same preference as GUI: match LifeCam mic, else Microsoft name, else first ALSA/open id."""
  devices = list_audio_input_devices()
  if not devices:
    return None
  if camera_label:
    mic = match_audio_to_camera(devices, camera_label)
    if mic is not None:
      return mic.get("open", mic.get("index"))
  chosen = prefer_microsoft(devices)
  if chosen is None:
    chosen = devices[0]
  return chosen.get("open", chosen.get("index"))


def list_devices() -> None:
  print("Radar UUIDs:", list_radar_uuids() or "(none)")
  print("Radar ports:", list_radar_ports() or "(none)")
  cams = _probe_cameras()
  print("Cameras (GUI probe):")
  if not cams:
    print("  (none)")
  for cam in cams:
    print(f"  [{cam.get('index')}] {cam.get('label') or cam.get('name')}")
  audio_devices = list_audio_input_devices()
  print("Audio inputs (GUI list: ALSA/arecord + optional sounddevice):")
  if not audio_devices:
    print("  (none — install alsa-utils / check arecord -l)")
  for dev in audio_devices:
    print(f"  open={dev.get('open')!r}  backend={dev.get('backend')}  {dev.get('label')}")



def _open_session(args, profile: str, frame_rate_hz: float) -> DualRadarSession:
  return DualRadarSession(
    num_rx=args.num_rx,
    profile=profile,
    frame_rate_hz=frame_rate_hz,
    radar1_uuid=args.radar1_uuid,
    radar2_uuid=args.radar2_uuid,
    radar1_port=args.radar1_port,
    radar2_port=args.radar2_port,
    mirror_radar2=False,
    min_range_m=args.min_range_m,
    max_range_m=args.max_range_m,
  )


def record_session(
  out_dir: Path,
  *,
  group: str,
  gesture: str,
  count: int,
  profile: str,
  frames: int,
  frame_rate_hz: float,
  args,
):
  class_dir = out_dir / group / gesture
  class_dir.mkdir(parents=True, exist_ok=True)
  uuids = list_radar_uuids()
  ports = list_radar_ports()
  print(f"\nRecording -> {class_dir}")
  print(f"Profile={profile}, {frames} frames/clip (~{frames / frame_rate_hz:.1f}s @ {frame_rate_hz} Hz)")
  print(f"Camera={args.camera_device} {args.camera_width}x{args.camera_height} @ {args.camera_fps} fps")
  if args.audio:
    dev = "default" if args.audio_device is None else args.audio_device
    print(
      f"Audio={args.audio_channels}ch @ {args.audio_sample_rate} Hz "
      f"(device={dev}, blocksize={args.audio_blocksize})"
    )
  else:
    print("Audio=disabled")
  print(f"Radar UUIDs: {uuids if uuids else 'none'}")
  print(f"Radar ports: {ports if ports else 'none'}\n")
  if len(uuids) < 2 and len(ports) < 2:
    raise SystemExit(
      f"Need two BGT60 radars (mirroring disabled); uuids={uuids} ports={ports}"
    )

  for n in range(count):
    if args.no_prompt:
      print(f"\nClip {n + 1}/{count} — perform '{gesture}' in 3s...")
    else:
      input(f"\nClip {n + 1}/{count} — perform '{gesture}', press Enter when ready...")
    for t in range(3, 0, -1):
      print(f"  {t}...")
      time.sleep(1)
    print("  RECORDING")
    try:
      idx = next_index(class_dir)
      with _open_session(args, profile, frame_rate_hz) as session:
        if not (session.slots[0].available and session.slots[1].available):
          raise RuntimeError(
            "Need two live BGT60 radars (mirroring disabled). "
            f"uuids={uuids} ports={ports} status={session.status_text}"
          )
        (
          radar_clip,
          radar2_clip,
          camera_frames,
          camera_timestamps,
          audio_samples,
          audio_timestamps,
        ) = record_multimodal_clip_dual(session, frames, args)
        active_uuids = [slot.uuid for slot in session.slots if slot.uuid]
        active_ports = [slot.port for slot in session.slots if getattr(slot, "port", None)]

        paths = save_multimodal_clip(
          class_dir,
          idx,
          radar_clip,
          radar2_clip,
          camera_frames,
          camera_timestamps,
          audio_samples,
          audio_timestamps,
          gesture,
          frame_rate_hz,
          active_uuids,
          active_ports,
          profile,
          args.min_range_m,
          args.max_range_m,
          audio_sample_rate=args.audio_sample_rate if args.audio else None,
          audio_channels=args.audio_channels if args.audio else None,
        )
        print(
          f"  saved radar={paths[0].name} radar2={paths[2].name} "
          f"camera={paths[3].name} audio={paths[4].name} meta={paths[5].name} "
          f"r1={radar_clip.shape} r2={radar2_clip.shape} cam={camera_frames.shape}"
        )
    except ErrorFrameAcquisitionFailed:
      print("  FRAME DROP — clip skipped. Lower --frame-rate or check USB power.")
    except RuntimeError as exc:
      print(f"  SENSOR ERROR — {exc}")
      break
    except ImportError as exc:
      print(f"  AUDIO ERROR — {exc}")
      break


def _parse_audio_device(value: str | None) -> int | str | None:
  if value is None:
    return None
  text = value.strip()
  if not text or text.lower() in ("default", "auto"):
    return None
  try:
    return int(text)
  except ValueError:
    return text  # plughw:1,0 / Pulse source name


def parse_args():
  p = argparse.ArgumentParser(description="Jetson: collect dual-radar + camera + audio clips")
  p.add_argument("--out", type=Path, default=DEFAULT_OUT)
  p.add_argument("--group", type=str, default="gestures", help="Top folder under --out")
  p.add_argument("--class", dest="class_name", type=str, default=None, help="Class folder (e.g. push)")
  p.add_argument("--count", type=int, default=None, help="Clips to record (skip interactive count)")
  p.add_argument("--no-prompt", action="store_true", help="No Enter between clips (3s countdown only)")
  p.add_argument("--frames", type=int, default=40)
  p.add_argument("--frame-rate", type=float, default=10.0)
  p.add_argument("--num-rx", type=int, default=3)
  p.add_argument("--radar-profile", choices=("safe", "balanced", "gesture"), default="gesture")
  p.add_argument("--min-range-m", type=float, default=0.0)
  p.add_argument("--max-range-m", type=float, default=None)
  p.add_argument("--radar1-uuid", type=str, default=None)
  p.add_argument("--radar2-uuid", type=str, default=None)
  p.add_argument("--radar1-port", type=str, default=None, help="e.g. /dev/ttyACM0")
  p.add_argument("--radar2-port", type=str, default=None, help="e.g. /dev/ttyACM1")
  p.add_argument("--camera-device", type=int, default=None, help="V4L/OpenCV index; default = GUI probe (prefer Microsoft)")
  p.add_argument("--camera-width", type=int, default=224)
  p.add_argument("--camera-height", type=int, default=224)
  p.add_argument("--camera-fps", type=float, default=15.0)
  p.add_argument("--no-audio", action="store_true")
  p.add_argument("--audio", action="store_true", help="Force mic on (default unless --no-audio)")
  p.add_argument(
    "--audio-device",
    type=str,
    default=None,
    help="sounddevice index OR ALSA open id e.g. plughw:1,0 (same as GUI). Default auto-match to camera.",
  )
  p.add_argument("--audio-sample-rate", type=int, default=16000)
  p.add_argument("--audio-channels", type=int, default=1)
  p.add_argument("--audio-blocksize", type=int, default=1024)
  p.add_argument("--list-devices", action="store_true")
  p.add_argument("--list-audio-devices", action="store_true")
  args = p.parse_args()
  if args.no_audio:
    args.audio = False
  else:
    args.audio = True
  args.audio_device = _parse_audio_device(args.audio_device)
  return args


def main():
  args = parse_args()
  if args.list_devices or args.list_audio_devices:
    list_devices()
    return

  cams = _probe_cameras()
  if args.camera_device is None:
    args.camera_device = _default_camera_index()
  cam_label = None
  for cam in cams:
    if int(cam.get("index", -1)) == int(args.camera_device):
      cam_label = str(cam.get("label") or cam.get("name") or "")
      break
  if args.audio and args.audio_device is None:
    args.audio_device = _default_audio_device(cam_label)
    print(f"Audio device auto: {args.audio_device!r}")
  print(f"Camera device: {args.camera_device} ({cam_label or 'unlabeled'})")
  if not cams:
    print("WARNING: GUI camera probe found nothing — try --camera-device 0/1 after ls /dev/video*")

  group = args.group
  gesture = args.class_name
  count = args.count
  if gesture is None:
    group = input(f"Group folder name [{group}]: ").strip() or group
    gesture = input("Class folder name (e.g. push): ").strip()
  if not gesture:
    raise SystemExit("Class name required (--class push).")
  if count is None:
    try:
      count = int(input("How many clips to record? [20]: ").strip() or "20")
    except ValueError:
      count = 20

  args.out.mkdir(parents=True, exist_ok=True)
  record_session(
    args.out,
    group=group,
    gesture=gesture,
    count=count,
    profile=args.radar_profile,
    frames=args.frames,
    frame_rate_hz=args.frame_rate,
    args=args,
  )
  print(f"\nDone. Data in {args.out.resolve()}")
  print("Copy to PC Crossattention, then:")
  print(
    f"  python train.py --data <path> --only-labels background push pull pinch_index palm_tilt swipe "
    f"--balance undersample --audio --out artifacts/gestures_dual"
  )


if __name__ == "__main__":
  main()
