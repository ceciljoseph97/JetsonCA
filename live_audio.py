"""Live microphone ring buffer for GUI inference."""

from __future__ import annotations

import sys
import threading
from typing import Any

import numpy as np

from device_select import ffmpeg_exe, is_usable_mic_label, linux_audio_names, windows_audio_names


def _sounddevice_inputs() -> list[dict[str, Any]]:
  try:
    import sounddevice as sd
  except Exception:
    return []
  devices: list[dict[str, Any]] = []
  try:
    listed = sd.query_devices()
  except Exception:
    return []
  for idx, dev in enumerate(listed):
    if int(dev.get("max_input_channels", 0)) <= 0:
      continue
    name = str(dev.get("name", f"device {idx}"))
    devices.append(
      {
        "index": int(idx),
        "name": name,
        "label": f"{idx}: {name}",
        "open": int(idx),
        "backend": "sounddevice",
      }
    )
  return devices


def _os_audio_inputs() -> list[dict[str, Any]]:
  names = windows_audio_names() if sys.platform == "win32" else linux_audio_names()
  devices: list[dict[str, Any]] = []
  for name in names:
    devices.append(
      {
        "index": None,
        "name": name,
        "label": name,
        "open": name,
        "backend": "dshow" if sys.platform == "win32" else "alsa",
      }
    )
  return devices


def _norm_name(name: str) -> str:
  return " ".join(str(name).lower().replace("®", "").replace("™", "").split())


def list_audio_input_devices() -> list[dict[str, Any]]:
  """PortAudio first, then OS/ffmpeg names (LifeCam mic) so listing works without sounddevice."""
  by_name: dict[str, dict[str, Any]] = {}
  for dev in _os_audio_inputs() + _sounddevice_inputs():
    key = _norm_name(str(dev.get("name") or ""))
    if not key or not is_usable_mic_label(str(dev.get("name") or "")):
      continue
    prev = by_name.get(key)
    if prev is None or (prev.get("backend") != "sounddevice" and dev.get("backend") == "sounddevice"):
      by_name[key] = dev
  devices = list(by_name.values())
  for i, dev in enumerate(devices):
    idx = dev.get("index")
    name = str(dev.get("name") or f"device {i}")
    if idx is not None:
      dev["label"] = f"{idx}: {name}"
    else:
      dev["label"] = name
  return devices


class LiveAudioBuffer:
  def __init__(self, sample_rate: int = 16000, max_seconds: float = 600.0):
    self.sample_rate = int(sample_rate)
    self.max_samples = max(1, int(self.sample_rate * max_seconds))
    self._buf = np.zeros(0, dtype=np.float32)
    self._lock = threading.Lock()
    self._stream = None
    self._ffmpeg_proc = None
    self._ffmpeg_thread: threading.Thread | None = None
    self.last_error: str | None = None
    self._record_chunks: list[np.ndarray] = []
    self._recording = False
    self._stop_ffmpeg = threading.Event()

  def start_recording_sink(self) -> None:
    with self._lock:
      self._record_chunks = []
      self._recording = True

  def stop_recording_sink(self) -> np.ndarray:
    with self._lock:
      self._recording = False
      if not self._record_chunks:
        return np.zeros(0, dtype=np.float32)
      out = np.concatenate(self._record_chunks).astype(np.float32)
      self._record_chunks = []
      return out

  def _push(self, chunk: np.ndarray) -> None:
    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if chunk.size == 0:
      return
    with self._lock:
      self._buf = np.concatenate([self._buf, chunk])
      if self._buf.shape[0] > self.max_samples:
        self._buf = self._buf[-self.max_samples :]
      if self._recording:
        self._record_chunks.append(chunk.copy())

  def start(self, device: int | str | None = None) -> None:
    keep_recording = False
    keep_chunks: list[np.ndarray] = []
    with self._lock:
      keep_recording = bool(self._recording)
      keep_chunks = list(self._record_chunks)
    self.stop()
    self.last_error = None
    opened = self._start_sounddevice(device)
    if not opened:
      opened = self._start_ffmpeg(device)
    with self._lock:
      self._recording = keep_recording
      if keep_recording:
        self._record_chunks = keep_chunks
    if not opened and self.last_error is None:
      self.last_error = "no audio backend (install sounddevice, or ffmpeg for LifeCam dshow)"

  def _start_sounddevice(self, device: int | str | None) -> bool:
    try:
      import sounddevice as sd
    except Exception as exc:
      self.last_error = f"sounddevice not available: {exc}"
      return False

    def callback(indata, frames, time_info, status):  # noqa: ARG001
      chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
      self._push(chunk.astype(np.float32, copy=False))

    try:
      kwargs: dict[str, Any] = dict(
        samplerate=self.sample_rate,
        channels=1,
        dtype="float32",
        blocksize=1024,
        callback=callback,
      )
      if device is not None:
        kwargs["device"] = device
      self._stream = sd.InputStream(**kwargs)
      self._stream.start()
      self.last_error = None
      return True
    except Exception as exc:
      self.last_error = str(exc)
      self._stream = None
      return False

  def _start_ffmpeg(self, device: int | str | None) -> bool:
    ffmpeg = ffmpeg_exe()
    if ffmpeg is None:
      if self.last_error is None:
        self.last_error = "ffmpeg not found for dshow/alsa audio fallback"
      return False
    name = str(device).strip() if isinstance(device, str) and device.strip() else None
    if name is None and sys.platform == "win32":
      names = windows_audio_names()
      name = names[0] if names else None
    if not name:
      return False
    if sys.platform == "win32":
      cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "dshow",
        "-i",
        f"audio={name}",
        "-ar",
        str(self.sample_rate),
        "-ac",
        "1",
        "-f",
        "f32le",
        "pipe:1",
      ]
    else:
      cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "alsa",
        "-i",
        "default",
        "-ar",
        str(self.sample_rate),
        "-ac",
        "1",
        "-f",
        "f32le",
        "pipe:1",
      ]
    try:
      import subprocess

      self._stop_ffmpeg.clear()
      self._ffmpeg_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
      )
    except Exception as exc:
      self.last_error = f"ffmpeg audio open failed: {exc}"
      self._ffmpeg_proc = None
      return False

    nbytes = 1024 * 4

    def reader():
      proc = self._ffmpeg_proc
      if proc is None or proc.stdout is None:
        return
      try:
        while not self._stop_ffmpeg.is_set():
          data = proc.stdout.read(nbytes)
          if not data:
            break
          self._push(np.frombuffer(data, dtype=np.float32))
      except Exception:
        pass

    self._ffmpeg_thread = threading.Thread(target=reader, daemon=True)
    self._ffmpeg_thread.start()
    self.last_error = None
    return True

  def stop(self) -> None:
    self._stop_ffmpeg.set()
    if self._stream is not None:
      try:
        self._stream.stop()
        self._stream.close()
      except Exception:
        pass
    self._stream = None
    proc = self._ffmpeg_proc
    self._ffmpeg_proc = None
    if proc is not None:
      try:
        if proc.stdout is not None:
          proc.stdout.close()
        proc.terminate()
        proc.wait(timeout=1.5)
      except Exception:
        try:
          proc.kill()
        except Exception:
          pass
    self._ffmpeg_thread = None

  @property
  def running(self) -> bool:
    if self._stream is not None:
      return True
    proc = self._ffmpeg_proc
    return proc is not None and proc.poll() is None

  def snapshot(self) -> np.ndarray:
    with self._lock:
      return self._buf.copy()

  def level_stats(self) -> dict[str, float | bool]:
    with self._lock:
      wave = self._buf.copy()
    if wave.size == 0:
      return {"rms": 0.0, "peak": 0.0, "std": 0.0, "ok": False, "has_data": False}
    from audio_features import audio_is_dead_microphone, audio_wave_stats

    stats = audio_wave_stats(wave)
    return {
      **stats,
      "ok": not audio_is_dead_microphone(wave),
      "has_data": True,
    }
