"""Live microphone ring buffer for GUI inference."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np


def list_audio_input_devices() -> list[dict[str, Any]]:
  try:
    import sounddevice as sd
  except ImportError:
    return []
  devices: list[dict[str, Any]] = []
  for idx, dev in enumerate(sd.query_devices()):
    if int(dev.get("max_input_channels", 0)) <= 0:
      continue
    name = str(dev.get("name", f"device {idx}"))
    devices.append({"index": int(idx), "name": name, "label": f"{idx}: {name}"})
  return devices


class LiveAudioBuffer:
  def __init__(self, sample_rate: int = 16000, max_seconds: float = 600.0):
    self.sample_rate = int(sample_rate)
    self.max_samples = max(1, int(self.sample_rate * max_seconds))
    self._buf = np.zeros(0, dtype=np.float32)
    self._lock = threading.Lock()
    self._stream = None
    self.last_error: str | None = None
    self._record_chunks: list[np.ndarray] = []
    self._recording = False

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

  def start(self, device: int | None = None) -> None:
    keep_recording = False
    keep_chunks: list[np.ndarray] = []
    with self._lock:
      keep_recording = bool(self._recording)
      keep_chunks = list(self._record_chunks)
    self.stop()
    self.last_error = None
    try:
      import sounddevice as sd
    except ImportError as exc:
      self.last_error = f"sounddevice not available: {exc}"
      return

    def callback(indata, frames, time_info, status):  # noqa: ARG001
      chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
      chunk = chunk.astype(np.float32, copy=False)
      with self._lock:
        self._buf = np.concatenate([self._buf, chunk])
        if self._buf.shape[0] > self.max_samples:
          self._buf = self._buf[-self.max_samples :]
        if self._recording:
          self._record_chunks.append(np.asarray(chunk, dtype=np.float32).copy())

    try:
      kwargs = dict(
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
      with self._lock:
        self._recording = keep_recording
        if keep_recording:
          self._record_chunks = keep_chunks
    except Exception as exc:
      self.last_error = str(exc)
      self._stream = None

  def stop(self) -> None:
    if self._stream is not None:
      try:
        self._stream.stop()
        self._stream.close()
      except Exception:
        pass
    self._stream = None

  @property
  def running(self) -> bool:
    return self._stream is not None

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
