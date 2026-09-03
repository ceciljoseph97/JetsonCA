"""Audio helpers for collection browsing and (later) model input."""

from __future__ import annotations

import numpy as np


def to_mono(audio: np.ndarray) -> np.ndarray:
  arr = np.asarray(audio, dtype=np.float32)
  if arr.ndim == 1:
    return arr
  if arr.shape[-1] == 1:
    return arr[:, 0]
  return arr.mean(axis=-1)


def audio_wave_stats(audio: np.ndarray) -> dict[str, float]:
  wave = to_mono(audio)
  if wave.size == 0:
    return {"rms": 0.0, "std": 0.0, "peak": 0.0}
  return {
    "rms": float(np.sqrt(np.mean(np.square(wave)))),
    "std": float(np.std(wave)),
    "peak": float(np.max(np.abs(wave))),
  }


def audio_is_dead_microphone(
  audio: np.ndarray,
  *,
  max_std: float = 2e-4,
  max_peak: float = 0.01,
) -> bool:
  """True when the mic stream is effectively silent (wrong device / muted / disconnected)."""
  stats = audio_wave_stats(audio)
  return stats["std"] < max_std and stats["peak"] < max_peak


def audio_is_usable(
  audio: np.ndarray,
  *,
  label: str | None = None,
  action_min_peak: float = 0.05,
) -> bool:
  """False for dead-mic captures and for action clips with no audible content."""
  if audio_is_dead_microphone(audio):
    return False
  if label and label != "background":
    if audio_wave_stats(audio)["peak"] < action_min_peak:
      return False
  return True


def _hann(n: int) -> np.ndarray:
  if n <= 1:
    return np.ones(max(n, 1), dtype=np.float32)
  i = np.arange(n, dtype=np.float32)
  return 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n - 1))


def log_spectrogram(
  audio: np.ndarray,
  sample_rate: int,
  *,
  n_fft: int = 512,
  hop_length: int = 160,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return log-power STFT (freq x time). Numpy-only (no scipy)."""
  wave = to_mono(audio)
  if wave.size == 0:
    return np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)

  n_fft = max(8, int(n_fft))
  hop_length = max(1, int(hop_length))
  window = _hann(n_fft)
  if wave.size < n_fft:
    wave = np.pad(wave, (0, n_fft - wave.size))

  n_frames = 1 + max(0, (wave.shape[0] - n_fft) // hop_length)
  spec = np.empty((n_fft // 2 + 1, n_frames), dtype=np.float32)
  for i in range(n_frames):
    start = i * hop_length
    frame = wave[start : start + n_fft]
    if frame.shape[0] < n_fft:
      frame = np.pad(frame, (0, n_fft - frame.shape[0]))
    frame = frame * window
    spec[:, i] = np.abs(np.fft.rfft(frame)).astype(np.float32) ** 2

  times = (np.arange(n_frames, dtype=np.float32) * hop_length) / float(sample_rate)
  freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate)).astype(np.float32)
  return np.log1p(spec), times, freqs


def radar_frame_time_s(frame_index: int, radar_hz: float) -> float:
  hz = max(float(radar_hz), 1e-6)
  return float(frame_index) / hz


def _pool_freq_bins(log_spec: np.ndarray, n_mels: int) -> np.ndarray:
  if log_spec.shape[0] == n_mels:
    return log_spec
  out = np.empty((n_mels, log_spec.shape[1]), dtype=np.float32)
  for i in range(n_mels):
    lo = int(i * log_spec.shape[0] / n_mels)
    hi = int((i + 1) * log_spec.shape[0] / n_mels)
    hi = max(hi, lo + 1)
    out[i] = log_spec[lo:hi].mean(axis=0)
  return out


def radar_aligned_mel_patches(
  audio: np.ndarray,
  sample_rate: int,
  radar_len: int,
  radar_hz: float,
  *,
  n_mels: int = 32,
  mel_width: int = 32,
  n_fft: int = 512,
  hop_length: int = 160,
) -> np.ndarray:
  """Return [T, 1, n_mels, mel_width] log-mel patches aligned to radar frames."""
  wave = to_mono(audio)
  if wave.size == 0 or radar_len <= 0:
    return np.zeros((radar_len, 1, n_mels, mel_width), dtype=np.float32)

  log_spec, times, _freqs = log_spectrogram(wave, sample_rate, n_fft=n_fft, hop_length=hop_length)
  spec = _pool_freq_bins(log_spec, n_mels)
  patches = np.zeros((radar_len, n_mels, mel_width), dtype=np.float32)
  duration = float(times[-1]) if times.size else 0.0
  half = mel_width // 2
  for fi in range(radar_len):
    if duration <= 0.0 or spec.shape[1] == 0:
      continue
    center = int(np.clip(np.searchsorted(times, radar_frame_time_s(fi, radar_hz)), 0, spec.shape[1] - 1))
    for j in range(mel_width):
      col = center - half + j
      if 0 <= col < spec.shape[1]:
        patches[fi, :, j] = spec[:, col]

  lo = float(patches.min())
  hi = float(patches.max())
  if hi > lo:
    patches = (patches - lo) / (hi - lo)
  return patches[:, np.newaxis, :, :].astype(np.float32)


def mel_tensor_from_wave(
  audio: np.ndarray,
  sample_rate: int,
  radar_len: int,
  radar_hz: float,
  *,
  n_mels: int = 32,
  mel_width: int = 32,
) -> "torch.Tensor":
  import torch

  patches = radar_aligned_mel_patches(
    audio,
    sample_rate,
    radar_len,
    radar_hz,
    n_mels=n_mels,
    mel_width=mel_width,
  )
  return torch.from_numpy(patches)


def render_audio_monitor_rgb(
  audio: np.ndarray,
  sample_rate: int,
  *,
  seconds: float = 2.5,
  width: int = 320,
  height: int = 180,
  n_mels: int = 64,
) -> np.ndarray:
  """Live monitor: log-mel spectrogram (model-like) + waveform strip."""
  import cv2

  wave = to_mono(audio)
  sr = max(1, int(sample_rate))
  n_keep = max(1, int(seconds * sr))
  if wave.size > n_keep:
    wave = wave[-n_keep:]
  canvas = np.full((height, width, 3), 24, dtype=np.uint8)
  if wave.size < 8:
    cv2.putText(canvas, "audio warming…", (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas

  wave_h = max(28, height // 4)
  mel_h = height - wave_h - 2

  # Waveform strip
  wave_img = np.full((wave_h, width, 3), 18, dtype=np.uint8)
  mid = wave_h // 2
  cv2.line(wave_img, (0, mid), (width - 1, mid), (40, 40, 40), 1)
  xs = np.linspace(0, wave.size - 1, num=width).astype(np.int32)
  ys = wave[xs]
  peak = float(np.max(np.abs(ys))) + 1e-6
  ys_px = (mid - (ys / peak) * (wave_h * 0.42)).astype(np.int32)
  ys_px = np.clip(ys_px, 0, wave_h - 1)
  for x in range(1, width):
    cv2.line(wave_img, (x - 1, int(ys_px[x - 1])), (x, int(ys_px[x])), (80, 220, 140), 1)
  canvas[:wave_h] = wave_img

  # Log-mel spectrogram (same frontend as model patches)
  log_spec, _times, _freqs = log_spectrogram(wave, sr, n_fft=512, hop_length=160)
  mel = _pool_freq_bins(log_spec, n_mels)
  if mel.size == 0:
    return canvas
  lo, hi = float(mel.min()), float(mel.max())
  if hi > lo:
    mel = (mel - lo) / (hi - lo)
  else:
    mel = np.zeros_like(mel)
  mel_u8 = np.clip(mel * 255.0, 0, 255).astype(np.uint8)
  # freq axis: low→bottom
  mel_u8 = np.flipud(mel_u8)
  mel_resized = cv2.resize(mel_u8, (width, mel_h), interpolation=cv2.INTER_NEAREST)
  mel_bgr = cv2.applyColorMap(mel_resized, cv2.COLORMAP_INFERNO)
  mel_rgb = cv2.cvtColor(mel_bgr, cv2.COLOR_BGR2RGB)
  canvas[wave_h + 2 :] = mel_rgb

  rms = float(np.sqrt(np.mean(np.square(wave))))
  cv2.putText(
    canvas,
    f"mel+wave  rms={rms:.3f}",
    (8, height - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (240, 240, 240),
    1,
    cv2.LINE_AA,
  )
  return canvas
