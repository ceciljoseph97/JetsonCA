"""BGT60TR13C configuration via ifxradarsdk."""

from __future__ import annotations


PROFILE_SAFE = {
  "range_resolution_m": 0.15,
  "max_range_m": 4.8,
  "max_speed_m_s": 2.45,
  "speed_resolution_m_s": 0.2,
  "sample_rate_Hz": 1_000_000,
  "if_gain_dB": 33,
  "tx_power_level": 28,
  "frame_rate_hz": 5.0,
}

PROFILE_BALANCED = {
  "range_resolution_m": 0.10,
  "max_range_m": 2.0,
  "max_speed_m_s": 2.5,
  "speed_resolution_m_s": 0.12,
  "sample_rate_Hz": 1_000_000,
  "if_gain_dB": 30,
  "tx_power_level": 28,
  "frame_rate_hz": 5.0,
}

PROFILE_GESTURE = {
  "range_resolution_m": 0.025,
  "max_range_m": 1.0,
  "max_speed_m_s": 3.0,
  "speed_resolution_m_s": 0.024,
  "sample_rate_Hz": 2_500_000,
  "if_gain_dB": 25,
  "tx_power_level": 31,
  "frame_rate_hz": 5.0,
}

PROFILES = {
  "safe": PROFILE_SAFE,
  "balanced": PROFILE_BALANCED,
  "gesture": PROFILE_GESTURE,
}


def configure_bgt_device(device, num_receivers=3, profile="safe", frame_rate_hz=None):
  try:
    from ifxradarsdk.fmcw.types import FmcwMetrics, FmcwSimpleSequenceConfig
  except ImportError as e:  # pragma: no cover - live capture only
    raise ImportError(
      "ifxradarsdk is required for live radar capture. "
      "Install editable from Exploration/radar_sdk/sdk/py/wrapper_radarsdk "
      "(or the Infineon wheel)."
    ) from e

  if profile not in PROFILES:
    raise ValueError(f"unknown profile {profile!r}; choose from {list(PROFILES)}")

  params = dict(PROFILES[profile])
  if frame_rate_hz is not None:
    params["frame_rate_hz"] = float(frame_rate_hz)

  sensor = device.get_sensor_information()
  min_rf = sensor["min_rf_frequency_Hz"]
  max_rf = sensor["max_rf_frequency_Hz"]

  metrics = FmcwMetrics(
    range_resolution_m=params["range_resolution_m"],
    max_range_m=params["max_range_m"],
    max_speed_m_s=params["max_speed_m_s"],
    speed_resolution_m_s=params["speed_resolution_m_s"],
    center_frequency_Hz=60_750_000_000,
  )

  sequence = device.create_simple_sequence(FmcwSimpleSequenceConfig())
  sequence.loop.repetition_time_s = 1.0 / params["frame_rate_hz"]

  chirp_loop = sequence.loop.sub_sequence.contents
  device.sequence_from_metrics(metrics, chirp_loop)

  chirp = chirp_loop.loop.sub_sequence.contents.chirp
  chirp.start_frequency_Hz = max(chirp.start_frequency_Hz, min_rf)
  chirp.end_frequency_Hz = min(chirp.end_frequency_Hz, max_rf)
  if chirp.end_frequency_Hz <= chirp.start_frequency_Hz:
    chirp.start_frequency_Hz = min_rf
    chirp.end_frequency_Hz = max_rf

  chirp.sample_rate_Hz = params["sample_rate_Hz"]
  chirp.rx_mask = (1 << num_receivers) - 1
  chirp.tx_mask = 1
  chirp.tx_power_level = params["tx_power_level"]
  chirp.if_gain_dB = params["if_gain_dB"]
  chirp.lp_cutoff_Hz = 500_000
  chirp.hp_cutoff_Hz = 80_000

  device.set_acquisition_sequence(sequence)

  return {
    "profile": profile,
    "num_chirps_per_frame": chirp_loop.loop.num_repetitions,
    "num_samples_per_chirp": chirp.num_samples,
    "frame_rate_hz": params["frame_rate_hz"],
    "metrics": metrics,
  }
