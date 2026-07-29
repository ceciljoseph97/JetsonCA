"""Range-doppler preprocessing used for multimodal training and inference."""

import numpy as np
import torch
from cv2 import INTER_AREA, resize

from range_gating import apply_range_gate_raw


def do_preprocessing(
  range_doppler,
  *,
  range_resolution_m: float | None = None,
  min_range_m: float = 0.0,
  max_range_m: float | None = None,
):
  maps = np.asarray(range_doppler)
  if np.iscomplexobj(maps):
    maps = np.abs(maps)
  maps = maps.astype(np.float32, copy=False)
  if maps.ndim == 3 and range_resolution_m is not None and max_range_m is not None:
    maps = apply_range_gate_raw(
      maps,
      range_resolution_m=range_resolution_m,
      min_range_m=min_range_m,
      max_range_m=max_range_m,
    )

  range_doppler = np.abs(maps)
  for index, channel in enumerate(range_doppler):
    channel_min = np.min(channel)
    channel_max = np.max(channel)
    if channel_max > channel_min:
      range_doppler[index] = (channel - channel_min) / (channel_max - channel_min)
    else:
      range_doppler[index] = channel - channel_min

  range_doppler = np.transpose(range_doppler, (2, 1, 0))
  range_doppler = resize(range_doppler, dsize=(32, 32), interpolation=INTER_AREA)
  range_doppler = np.transpose(range_doppler, (2, 1, 0))
  return range_doppler


def do_inference_processing(
  range_doppler,
  *,
  range_resolution_m: float | None = None,
  min_range_m: float = 0.0,
  max_range_m: float | None = None,
):
  range_doppler = do_preprocessing(
    range_doppler,
    range_resolution_m=range_resolution_m,
    min_range_m=min_range_m,
    max_range_m=max_range_m,
  )
  range_doppler = torch.from_numpy(range_doppler).float()
  return torch.unsqueeze(range_doppler, 0)
