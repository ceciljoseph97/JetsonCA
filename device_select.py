"""Pick live camera / mic by name. Prefer Microsoft when present — never hardcode index 0."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from typing import Any

_MICROSOFT_RE = re.compile(r"microsoft", re.I)


def is_microsoft_label(label: str | None) -> bool:
  return bool(_MICROSOFT_RE.search(str(label or "")))


def prefer_microsoft(items: list[dict[str, Any]], *, label_key: str = "label") -> dict[str, Any] | None:
  """First Microsoft-named device, else first item, else None."""
  for item in items:
    if is_microsoft_label(str(item.get(label_key, ""))):
      return item
  return items[0] if items else None


def prefer_microsoft_index(items: list[dict[str, Any]], *, index_key: str = "index") -> int | None:
  chosen = prefer_microsoft(items)
  if chosen is None:
    return None
  try:
    return int(chosen[index_key])
  except (KeyError, TypeError, ValueError):
    return None


def _run(cmd: list[str], *, timeout: float = 8.0) -> str:
  try:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
  except (OSError, subprocess.TimeoutExpired):
    return ""
  return (proc.stdout or "") + "\n" + (proc.stderr or "")


def _dedupe(names: list[str]) -> list[str]:
  out: list[str] = []
  seen: set[str] = set()
  for name in names:
    key = name.strip()
    if not key or key.lower() in seen:
      continue
    seen.add(key.lower())
    out.append(key)
  return out


def windows_camera_names() -> list[str]:
  """DirectShow / PnP friendly names, discovery order ≈ OpenCV index order."""
  names = _ffmpeg_dshow_video_names()
  if names:
    return names
  ps = (
    "Get-PnpDevice -Class Camera,Image -Status OK -ErrorAction SilentlyContinue | "
    "ForEach-Object { $_.FriendlyName }"
  )
  raw = _run(["powershell", "-NoProfile", "-Command", ps])
  return _dedupe([ln.strip() for ln in raw.splitlines() if ln.strip()])


def _ffmpeg_dshow_video_names() -> list[str]:
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    try:
      import imageio_ffmpeg

      ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
      return []
  raw = _run([ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"], timeout=12.0)
  names: list[str] = []
  in_video = False
  for line in raw.splitlines():
    lower = line.lower()
    if "directshow video devices" in lower:
      in_video = True
      continue
    if "directshow audio devices" in lower:
      break
    if not in_video:
      continue
    match = re.search(r'"([^"]+)"', line)
    if match:
      names.append(match.group(1))
  return _dedupe(names)


def linux_camera_names_by_index() -> dict[int, str]:
  """Map /dev/videoN → driver/card name via v4l2-ctl when present."""
  v4l2 = shutil.which("v4l2-ctl")
  mapping: dict[int, str] = {}
  if v4l2:
    raw = _run([v4l2, "--list-devices"])
    current = ""
    for line in raw.splitlines():
      if line and not line.startswith("\t") and not line.startswith(" "):
        current = line.rstrip(":")
        continue
      m = re.search(r"/dev/video(\d+)", line)
      if m and current:
        mapping[int(m.group(1))] = current.strip()
    if mapping:
      return mapping
  # Fallback: sysfs card names
  for idx in range(16):
    path = f"/sys/class/video4linux/video{idx}/name"
    try:
      name = open(path, encoding="utf-8").read().strip()
    except OSError:
      continue
    if name:
      mapping[idx] = name
  return mapping


def camera_names_for_indices(indices: list[int]) -> dict[int, str]:
  if sys.platform == "win32":
    names = windows_camera_names()
    return {idx: names[i] for i, idx in enumerate(indices) if i < len(names)}
  linux = linux_camera_names_by_index()
  return {idx: linux[idx] for idx in indices if idx in linux}


def attach_camera_names(cameras: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Rewrite labels with OS names when we can map them. Keep OpenCV index."""
  indices = [int(cam["index"]) for cam in cameras]
  names = camera_names_for_indices(indices)
  out: list[dict[str, Any]] = []
  for cam in cameras:
    item = dict(cam)
    idx = int(item["index"])
    w = item.get("width", "?")
    h = item.get("height", "?")
    os_name = names.get(idx)
    if os_name:
      item["name"] = os_name
      item["label"] = f"{idx}: {os_name} ({w}x{h})"
    else:
      item["name"] = str(item.get("label", f"Camera {idx}"))
      item["label"] = f"{idx}: Camera ({w}x{h})"
    out.append(item)
  return out
