#!/usr/bin/env python3
"""Headless cam1+radar2 inference for Jetson (no Tk / Matplotlib GUI)."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import load_checkpoint
from jetson_env import apply_jetson_runtime_tweaks, default_device
from label_hierarchy import inference_label
from radar_utils import DualRadarSession, fuse_radar_streams_for_model
from checkpoint import preprocess_camera_frame


def _camera_stream_cls():
  from realtime_multimodal import CameraStream

  return CameraStream


def parse_args():
  p = argparse.ArgumentParser(description="JetsonCA headless multimodal inference")
  p.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_multimodal_crossattention.pt"))
  p.add_argument("--device", type=str, default=default_device())
  p.add_argument("--camera-device", type=int, default=0)
  p.add_argument("--camera-width", type=int, default=640)
  p.add_argument("--camera-height", type=int, default=480)
  p.add_argument("--camera-fps", type=float, default=15.0)
  p.add_argument("--num-rx", type=int, default=3)
  p.add_argument("--radar-profile", type=str, default="safe")
  p.add_argument("--frame-rate", type=float, default=5.0)
  p.add_argument("--radar1-uuid", type=str, default=None)
  p.add_argument("--radar2-uuid", type=str, default=None)
  p.add_argument("--mirror-radar2", action="store_true", default=True)
  p.add_argument("--no-mirror-radar2", action="store_false", dest="mirror_radar2")
  p.add_argument("--no-radar", action="store_true", help="Camera-only live: skip radar SDK, radar_present=False")
  p.add_argument("--dual-radar-fuse", choices=("mean", "max"), default="mean")
  p.add_argument("--window", type=int, default=30)
  p.add_argument("--detect-threshold", type=float, default=0.35)
  p.add_argument("--min-range-m", type=float, default=0.3)
  p.add_argument("--max-range-m", type=float, default=2.5)
  p.add_argument("--duration-s", type=float, default=0.0, help="0 = run until Ctrl+C")
  p.add_argument("--print-every", type=float, default=1.0)
  p.add_argument("--jsonl-out", type=Path, default=None)
  return p.parse_args()


def main():
  args = parse_args()
  tweaks = apply_jetson_runtime_tweaks()
  print(json.dumps({"runtime": tweaks, "device": args.device, "no_radar": args.no_radar}, indent=2))

  model, labels, config = load_checkpoint(args.checkpoint, args.device)
  model.eval()
  image_size = int(config["image_size"])
  window = int(args.window)

  radar_buf: deque[torch.Tensor] = deque(maxlen=window)
  camera_buf: deque[torch.Tensor] = deque(maxlen=window)

  camera = _camera_stream_cls()(
    args.camera_device,
    args.camera_width,
    args.camera_height,
    args.camera_fps,
  ).start(warmup_s=2.0)

  jsonl = None
  if args.jsonl_out is not None:
    args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.jsonl_out.open("w", encoding="utf-8")

  t0 = time.time()
  last_print = 0.0
  n_infer = 0

  def _infer_once(meta: dict):
    nonlocal n_infer, last_print
    if args.no_radar:
      if len(camera_buf) < window:
        return
      radar = torch.zeros(1, window, 3, 32, 32, device=args.device)
      radar_present = torch.zeros(1, dtype=torch.bool, device=args.device)
    else:
      if len(radar_buf) < window or len(camera_buf) < window:
        return
      radar = torch.stack(list(radar_buf), dim=0).unsqueeze(0).to(args.device)
      radar_present = torch.ones(1, dtype=torch.bool, device=args.device)

    cam = torch.stack(list(camera_buf), dim=0).unsqueeze(0).to(args.device)
    camera_present = torch.ones(1, dtype=torch.bool, device=args.device)

    with torch.no_grad():
      out = model(radar, cam, radar_present=radar_present, camera_present=camera_present)
      logits = out["activity_logits"] if "activity_logits" in out else out["logits"]
      probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    label = labels[pred_idx] if conf >= args.detect_threshold else "none"
    display = inference_label(label) if label != "none" else "none"
    n_infer += 1
    now = time.time()

    row = {
      "t": now,
      "label": display,
      "raw_label": label,
      "conf": conf,
      "probs": {labels[i]: float(probs[i]) for i in range(len(labels))},
      "radar_meta": meta,
      "radar_present": (not args.no_radar),
    }
    if jsonl is not None:
      jsonl.write(json.dumps(row) + "\n")
      jsonl.flush()

    if now - last_print >= args.print_every:
      last_print = now
      fps = n_infer / max(now - t0, 1e-6)
      print(
        f"[{now - t0:6.1f}s] {display:16s} conf={conf:.2f}  "
        f"infer_fps~{fps:.1f}  radar={'off' if args.no_radar else meta}"
      )

  try:
    if args.no_radar:
      print("camera-only live: radar SDK skipped")
      while True:
        if args.duration_s > 0 and (time.time() - t0) >= args.duration_s:
          break
        frame = camera.get_latest()
        if frame is not None:
          camera_buf.append(preprocess_camera_frame(frame, image_size).cpu())
        _infer_once({"radar": "disabled"})
        time.sleep(0.01)
    else:
      with DualRadarSession(
        num_rx=args.num_rx,
        profile=args.radar_profile,
        frame_rate_hz=args.frame_rate,
        radar1_uuid=args.radar1_uuid,
        radar2_uuid=args.radar2_uuid,
        mirror_radar2=args.mirror_radar2,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
      ) as radars:
        print(f"radar: {radars.status_text}")
        while True:
          if args.duration_s > 0 and (time.time() - t0) >= args.duration_s:
            break

          r1, r2 = radars.read_tensors()
          fused, meta = fuse_radar_streams_for_model(
            r1,
            r2,
            mode=args.dual_radar_fuse,
            mirror_radar2=args.mirror_radar2,
          )
          if fused is not None:
            radar_buf.append(fused.detach().cpu())

          frame = camera.get_latest()
          if frame is not None:
            camera_buf.append(preprocess_camera_frame(frame, image_size).cpu())

          _infer_once(meta)
          time.sleep(0.01)
  except KeyboardInterrupt:
    print("\nstopped")
  finally:
    camera.stop()
    if jsonl is not None:
      jsonl.close()


if __name__ == "__main__":
  main()
