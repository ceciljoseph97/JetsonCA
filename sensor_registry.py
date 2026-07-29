"""Dynamic sensor registry and assignment for realtime multimodal inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SensorKind = Literal["camera", "radar"]


@dataclass(frozen=True)
class SensorSpec:
  sensor_id: str
  kind: SensorKind
  slot: int
  label: str
  device_ref: str
  infer: bool
  stream: bool

  @property
  def short_label(self) -> str:
    return self.label or self.sensor_id


@dataclass
class SensorAssignment:
  sensors: list[SensorSpec] = field(default_factory=list)
  reference_id: str = ""

  def ids(self) -> list[str]:
    return [s.sensor_id for s in self.sensors]

  def cameras(self) -> list[SensorSpec]:
    return [s for s in self.sensors if s.kind == "camera"]

  def radars(self) -> list[SensorSpec]:
    return [s for s in self.sensors if s.kind == "radar"]

  def get(self, sensor_id: str) -> SensorSpec | None:
    for spec in self.sensors:
      if spec.sensor_id == sensor_id:
        return spec
    return None

  def label(self, sensor_id: str) -> str:
    spec = self.get(sensor_id)
    return spec.short_label if spec else sensor_id

  def inferring(self) -> list[SensorSpec]:
    return [s for s in self.sensors if s.infer]

  def default_reference(self) -> str:
    if self.reference_id and self.get(self.reference_id):
      return self.reference_id
    cams = self.cameras()
    if cams:
      return cams[0].sensor_id
    radars = self.radars()
    if radars:
      return radars[0].sensor_id
    return ""


def pair_key(cam_id: str, radar_id: str) -> str:
  return f"{cam_id}+{radar_id}"


def parse_pair_key(key: str) -> tuple[str, str] | None:
  if "+" not in key:
    return None
  cam_id, radar_id = key.split("+", 1)
  return cam_id, radar_id


def build_assignment(
  *,
  cameras: list[dict[str, object]],
  radars: list[dict[str, object]],
  reference_id: str = "",
) -> SensorAssignment:
  """Build registry from slot descriptors.

  cameras/radars items:
    slot, device_ref, label, infer, stream
  """
  sensors: list[SensorSpec] = []
  for item in cameras:
    slot = int(item["slot"])
    device_ref = str(item.get("device_ref", slot))
    label = str(item.get("label", f"Camera {device_ref}"))
    sensors.append(
      SensorSpec(
        sensor_id=f"cam:{slot}",
        kind="camera",
        slot=slot,
        label=label,
        device_ref=device_ref,
        infer=bool(item.get("infer", True)),
        stream=bool(item.get("stream", True)),
      )
    )
  for item in radars:
    slot = int(item["slot"])
    device_ref = str(item.get("device_ref", slot))
    label = str(item.get("label", f"Radar {slot + 1}"))
    sensors.append(
      SensorSpec(
        sensor_id=f"radar:{slot}",
        kind="radar",
        slot=slot,
        label=label,
        device_ref=device_ref,
        infer=bool(item.get("infer", True)),
        stream=bool(item.get("stream", True)),
      )
    )
  ref = reference_id
  assignment = SensorAssignment(sensors=sensors, reference_id=ref)
  if not ref or not assignment.get(ref):
    assignment.reference_id = assignment.default_reference()
  return assignment


def empty_perception_row() -> dict[str, object]:
  return {
    "local": "",
    "local_conf": 0.0,
    "global": "",
    "aligned": False,
    "facing": 90,
    "fused": "",
    "fused_conf": 0.0,
    "fused_mode": "",
    "label": "",
  }


def empty_perception(assignment: SensorAssignment) -> dict[str, dict[str, object]]:
  return {sid: empty_perception_row() for sid in assignment.ids()}


def empty_fused_pairs() -> dict[str, dict[str, object]]:
  return {}
