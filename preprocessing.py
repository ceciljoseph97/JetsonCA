"""Range-doppler preprocessing used for multimodal training and inference."""

import numpy as np
import torch
import torch.nn.functional as F

from range_gating import apply_range_gate_raw


def _resize_hwc(maps: np.ndarray, size: tuple[int, int] = (32, 32)) -> np.ndarray:
  """maps: (H, W, C) float -> resized (size[1], size[0], C). Prefers cv2, falls back to torch."""
  try:
    from cv2 import INTER_AREA, resize

    return resize(maps, dsize=size, interpolation=INTER_AREA)
  except Exception:
    t = torch.from_numpy(np.ascontiguousarray(maps)).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(size[1], size[0]), mode="area")
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy()


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
  range_doppler = _resize_hwc(range_doppler, size=(32, 32))
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
