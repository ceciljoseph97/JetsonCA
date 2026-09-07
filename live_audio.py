"""Live microphone ring buffer for GUI inference."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

import numpy as np

from device_select import ffmpeg_exe, is_usable_mic_label, linux_audio_devices, windows_audio_names


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
  devices: list[dict[str, Any]] = []
  if sys.platform == "win32":
    for name in windows_audio_names():
      devices.append(
        {
          "index": None,
          "name": name,
          "label": name,
          "open": name,
          "backend": "dshow",
        }
      )
    return devices
  for item in linux_audio_devices():
    name = str(item.get("name") or item.get("open") or "")
    open_id = str(item.get("open") or name)
    devices.append(
      {
        "index": None,
        "name": name,
        "label": f"{name} ({open_id})" if open_id != name else name,
        "open": open_id,
        "backend": "alsa",
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
    open_id = dev.get("open")
    if idx is not None:
      dev["label"] = f"{idx}: {name}"
    elif open_id and str(open_id) != name:
      dev["label"] = f"{name} ({open_id})"
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
    errors: list[str] = []
    for opener in (
      lambda: self._start_sounddevice(device),
      lambda: self._start_arecord(device),
      lambda: self._start_ffmpeg(device),
    ):
      try:
        if opener():
          break
      except Exception as exc:
        errors.append(str(exc))
      if self.last_error:
        errors.append(self.last_error)
    else:
      self.last_error = " | ".join(e for e in errors if e) or "no audio backend"
    with self._lock:
      self._recording = keep_recording
      if keep_recording:
        self._record_chunks = keep_chunks

  def _sounddevice_resolve(self, device: int | str | None) -> int | str | None:
    if device is None or isinstance(device, int):
      return device
    name = str(device).strip()
    if not name:
      return None
    try:
      import sounddevice as sd

      listed = list(sd.query_devices())
    except Exception:
      return name
    target = _norm_name(name)
    best_i, best = None, 0
    for idx, dev in enumerate(listed):
      if int(dev.get("max_input_channels", 0)) <= 0:
        continue
      other = _norm_name(str(dev.get("name") or ""))
      score = len(set(target.split()) & set(other.split()))
      if "lifecam" in target and "lifecam" in other:
        score += 5
      if score > best:
        best_i, best = idx, score
    return best_i if best >= 1 else name

  @staticmethod
  def _is_alsa_spec(device: int | str | None) -> bool:
    return isinstance(device, str) and device.strip().startswith(("hw:", "plughw:", "sysdefault"))

  def _start_sounddevice(self, device: int | str | None) -> bool:
    if self._is_alsa_spec(device):
      return False
    try:
      import sounddevice as sd
    except Exception as exc:
      self.last_error = f"sounddevice not available: {exc}"
      return False

    resolved = self._sounddevice_resolve(device)
    if isinstance(resolved, str) and self._is_alsa_spec(resolved):
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
      if resolved is not None:
        kwargs["device"] = resolved
      self._stream = sd.InputStream(**kwargs)
      self._stream.start()
      self.last_error = None
      return True
    except Exception as exc:
      self.last_error = str(exc)
      self._stream = None
      # Named / index pick failed: do NOT steal Pulse default (often HDMI / silent).
      # arecord/ffmpeg will open the LifeCam plughw next.
      return False

  def _alsa_device(self, device: int | str | None) -> str | None:
    if isinstance(device, str) and device.strip():
      raw = device.strip()
      if raw.startswith(("hw:", "plughw:", "default", "sysdefault")):
        return raw
      for item in linux_audio_devices():
        name = str(item.get("name") or "")
        open_id = str(item.get("open") or "")
        if _norm_name(raw) in (_norm_name(name), _norm_name(open_id)):
          return open_id
        if "lifecam" in _norm_name(raw) and "lifecam" in _norm_name(name):
          return open_id
      return raw
    for item in linux_audio_devices():
      name = str(item.get("name") or "")
      if "lifecam" in _norm_name(name) or "microsoft" in _norm_name(name):
        return str(item.get("open") or "")
    items = linux_audio_devices()
    return str(items[0]["open"]) if items else None

  def _start_pipe(self, cmd: list[str], *, s16: bool = False) -> bool:
    import subprocess

    self._stop_ffmpeg.clear()
    try:
      self._ffmpeg_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
      )
    except Exception as exc:
      self.last_error = f"{cmd[0]} open failed: {exc}"
      self._ffmpeg_proc = None
      return False
    time.sleep(0.25)
    proc = self._ffmpeg_proc
    if proc is None:
      return False
    if proc.poll() is not None:
      err = b""
      if proc.stderr is not None:
        err = proc.stderr.read() or b""
      self.last_error = (err.decode("utf-8", "replace") or f"{cmd[0]} exited").strip()[:240]
      self._ffmpeg_proc = None
      return False

    def _drain_err():
      try:
        if proc.stderr is not None:
          while proc.poll() is None and not self._stop_ffmpeg.is_set():
            if not proc.stderr.read(256):
              break
      except Exception:
        pass

    threading.Thread(target=_drain_err, daemon=True).start()

    nbytes = 2048 if s16 else 1024 * 4

    def reader():
      local = self._ffmpeg_proc
      if local is None or local.stdout is None:
        return
      try:
        while not self._stop_ffmpeg.is_set():
          data = local.stdout.read(nbytes)
          if not data:
            break
          if s16:
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            self._push(pcm)
          else:
            self._push(np.frombuffer(data, dtype=np.float32))
      except Exception as exc:
        self.last_error = str(exc)[:240]
      if local.poll() not in (None, 0) and not self._stop_ffmpeg.is_set():
        err = b""
        try:
          if local.stderr is not None:
            err = local.stderr.read() or b""
        except Exception:
          pass
        self.last_error = (err.decode("utf-8", "replace") or f"{cmd[0]} exited").strip()[:240]

    self._ffmpeg_thread = threading.Thread(target=reader, daemon=True)
    self._ffmpeg_thread.start()
    self.last_error = None
    return True

  def _start_arecord(self, device: int | str | None) -> bool:
    if sys.platform == "win32":
      return False
    import shutil

    arecord = shutil.which("arecord")
    alsa = self._alsa_device(device)
    if arecord is None:
      self.last_error = (self.last_error or "") + " | arecord not found"
      return False
    if not alsa:
      self.last_error = (self.last_error or "") + " | no ALSA capture device"
      return False
    cmd = [
      arecord,
      "-D",
      alsa,
      "-f",
      "S16_LE",
      "-r",
      str(self.sample_rate),
      "-c",
      "1",
      "-t",
      "raw",
      "-q",
    ]
    return self._start_pipe(cmd, s16=True)

  def _start_ffmpeg(self, device: int | str | None) -> bool:
    ffmpeg = ffmpeg_exe()
    if ffmpeg is None:
      self.last_error = (self.last_error or "") + " | ffmpeg not found"
      return False
    if sys.platform == "win32":
      name = str(device).strip() if isinstance(device, str) and device.strip() else None
      if name is None:
        names = windows_audio_names()
        name = names[0] if names else None
      if not name:
        return False
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
      return self._start_pipe(cmd, s16=False)

    alsa = self._alsa_device(device) or "default"
    for spec in (( "-f", "pulse", "-i", alsa), ("-f", "alsa", "-i", alsa), ("-f", "pulse", "-i", "default")):
      cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        *spec,
        "-ar",
        str(self.sample_rate),
        "-ac",
        "1",
        "-f",
        "f32le",
        "pipe:1",
      ]
      if self._start_pipe(cmd, s16=False):
        return True
    return False

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
      try:
        if hasattr(self._stream, "active") and not bool(self._stream.active):
          return False
      except Exception:
        return False
      return True
    proc = self._ffmpeg_proc
    if proc is None:
      return False
    if proc.poll() is None:
      return True
    if self.last_error is None:
      name = "audio capture"
      try:
        args = getattr(proc, "args", None)
        if isinstance(args, (list, tuple)) and args:
          name = str(args[0])
      except Exception:
        pass
      self.last_error = f"{name} exited"
    self._ffmpeg_proc = None
    return False

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
