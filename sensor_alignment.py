"""Geometric facing / label remapping for multi-sensor realtime inference."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from sensor_registry import SensorAssignment, pair_key

FACING_CHOICES: tuple[int, ...] = (0, 90, 180, 270)
FACING_LABELS: dict[int, str] = {
  0: "0° left",
  90: "90° forward",
  180: "180° right",
  270: "270° back",
}

_TOWARDS = "walking_towards"
_AWAY = "walking_away"
_CROSS = "crossing"

CSV_FIELDS = (
  "timestamp",
  "frame",
  "sensor_id",
  "sensor_label",
  "facing_deg",
  "aligned",
  "local",
  "local_conf",
  "global",
  "fused",
  "fused_conf",
  "fused_mode",
  "pair_key",
  "consensus_global",
  "consensus_conf",
)


def normalize_facing(deg: int | float | str) -> int:
  try:
    value = int(float(str(deg).split("°")[0].strip().split()[0]))
  except (TypeError, ValueError):
    value = 90
  value = ((value % 360) + 360) % 360
  snapped = int(round(value / 90.0) * 90) % 360
  return snapped if snapped in FACING_CHOICES else 90


def relative_yaw(sensor_facing: int, ref_facing: int) -> int:
  return (normalize_facing(sensor_facing) - normalize_facing(ref_facing)) % 360


def is_cofacing(facing_a: int, facing_b: int) -> bool:
  return normalize_facing(facing_a) == normalize_facing(facing_b)


def remap_label_to_reference(local_label: str, sensor_facing: int, ref_facing: int) -> str:
  label = (local_label or "").strip()
  if not label:
    return ""
  delta = relative_yaw(sensor_facing, ref_facing)
  if delta == 0:
    return label
  if delta == 180:
    if label == _TOWARDS:
      return _AWAY
    if label == _AWAY:
      return _TOWARDS
    return label
  if delta == 90:
    if label in (_TOWARDS, _AWAY):
      return _CROSS
    if label == _CROSS:
      return _TOWARDS
    return label
  if delta == 270:
    if label in (_TOWARDS, _AWAY):
      return _CROSS
    if label == _CROSS:
      return _AWAY
    return label
  return label


@dataclass
class AlignmentConfig:
  enabled: bool = False
  reference: str = ""
  facings: dict[str, int] = field(default_factory=dict)
  multimodal_requires_cofacing: bool = True
  assignment: SensorAssignment | None = None

  def sensor_ids(self) -> list[str]:
    if self.assignment is not None:
      return self.assignment.ids()
    return list(self.facings.keys())

  def facing(self, key: str) -> int:
    return normalize_facing(self.facings.get(key, 90))

  def ref_facing(self) -> int:
    ref = self.reference or (self.assignment.default_reference() if self.assignment else "")
    return self.facing(ref) if ref else 90

  def aligned_with_ref(self, key: str) -> bool:
    ref = self.reference or (self.assignment.default_reference() if self.assignment else "")
    if not ref:
      return False
    return is_cofacing(self.facing(key), self.facing(ref))


def multimodal_pair_enabled(config: AlignmentConfig, cam_id: str, radar_id: str) -> bool:
  if not config.enabled or not config.multimodal_requires_cofacing:
    return True
  return is_cofacing(config.facing(cam_id), config.facing(radar_id))


def list_multimodal_pairs(config: AlignmentConfig) -> list[str]:
  if config.assignment is None:
    return []
  pairs: list[str] = []
  for cam in config.assignment.cameras():
    if not cam.infer:
      continue
    for radar in config.assignment.radars():
      if not radar.infer:
        continue
      if multimodal_pair_enabled(config, cam.sensor_id, radar.sensor_id):
        pairs.append(pair_key(cam.sensor_id, radar.sensor_id))
  return pairs


def build_local_and_global_views(
  *,
  config: AlignmentConfig,
  assignment: SensorAssignment,
  local_labels: dict[str, str],
  local_confs: dict[str, float],
  fused_by_sensor: dict[str, dict[str, object]],
  fused_pairs: dict[str, dict[str, object]],
) -> dict[str, object]:
  ref = config.reference or assignment.default_reference()
  if ref and not assignment.get(ref):
    ref = assignment.default_reference()

  per_sensor: dict[str, dict[str, object]] = {}
  votes: dict[str, float] = {}

  for spec in assignment.sensors:
    key = spec.sensor_id
    local = (local_labels.get(key) or "").strip()
    conf = float(local_confs.get(key, 0.0) or 0.0)
    face = config.facing(key)
    aligned = config.aligned_with_ref(key) if config.enabled else False
    global_lbl = ""
    if local and config.enabled and aligned and ref:
      global_lbl = remap_label_to_reference(local, face, config.facing(ref))
      weight = conf + (0.15 if key == ref else 0.0)
      votes[global_lbl] = votes.get(global_lbl, 0.0) + weight

    fused_row = fused_by_sensor.get(key, {})
    per_sensor[key] = {
      "local": local,
      "local_conf": conf,
      "global": global_lbl,
      "aligned": aligned,
      "facing": face,
      "label": spec.short_label,
      "fused": str(fused_row.get("label", "") or ""),
      "fused_conf": float(fused_row.get("confidence", 0.0) or 0.0),
      "fused_mode": str(fused_row.get("mode", "") or ""),
      "pair_key": str(fused_row.get("pair_key", "") or ""),
    }

  global_label = ""
  global_conf = 0.0
  if config.enabled and votes:
    global_label = max(votes.items(), key=lambda kv: kv[1])[0]
    total = sum(votes.values()) or 1.0
    global_conf = float(votes[global_label] / total)

  mm_pairs = list_multimodal_pairs(config)
  ref_label = assignment.label(ref) if ref else "-"

  return {
    "enabled": bool(config.enabled),
    "reference": ref,
    "ref_facing": config.ref_facing(),
    "per_sensor": per_sensor,
    "global_label": global_label,
    "global_confidence": global_conf,
    "multimodal_pairs": mm_pairs,
    "fused_pairs": fused_pairs,
    "summary": (
      f"ref={ref_label}@{config.ref_facing()}°"
      + (f" | global={global_label} ({global_conf:.2f})" if global_label else " | global=-")
      + (f" | mm={','.join(mm_pairs) if mm_pairs else 'none'}" if config.enabled else "")
    ),
  }


class PerceptionCsvLogger:
  def __init__(self, path: Path | None = None):
    self.path = path
    self._writer: csv.DictWriter | None = None
    self._file = None
    self.enabled = False

  def set_path(self, path: Path):
    self.close()
    self.path = path

  def start(self):
    if self.path is None:
      return
    self.path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not self.path.exists()
    self._file = self.path.open("a", newline="", encoding="utf-8")
    self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
    if new_file:
      self._writer.writeheader()
    self.enabled = True

  def close(self):
    if self._file is not None:
      self._file.close()
    self._file = None
    self._writer = None
    self.enabled = False

  def write_frame(
    self,
    *,
    frame: int,
    assignment: SensorAssignment,
    perception: dict[str, dict[str, object]],
    global_label: str,
    global_conf: float,
  ):
    if not self.enabled or self._writer is None:
      return
    ts = time.time()
    for spec in assignment.sensors:
      row = perception.get(spec.sensor_id, {})
      self._writer.writerow(
        {
          "timestamp": f"{ts:.3f}",
          "frame": frame,
          "sensor_id": spec.sensor_id,
          "sensor_label": spec.short_label,
          "facing_deg": row.get("facing", ""),
          "aligned": row.get("aligned", False),
          "local": row.get("local", ""),
          "local_conf": row.get("local_conf", 0.0),
          "global": row.get("global", ""),
          "fused": row.get("fused", ""),
          "fused_conf": row.get("fused_conf", 0.0),
          "fused_mode": row.get("fused_mode", ""),
          "pair_key": row.get("pair_key", ""),
          "consensus_global": global_label,
          "consensus_conf": global_conf,
        }
      )
    if self._file is not None:
      self._file.flush()


def facing_choice_strings() -> list[str]:
  return [FACING_LABELS[d] for d in FACING_CHOICES]


def parse_facing_choice(text: str) -> int:
  return normalize_facing(text)


def iter_sensor_ids(assignment: SensorAssignment | None) -> Iterable[str]:
  if assignment is None:
    return []
  return assignment.ids()
