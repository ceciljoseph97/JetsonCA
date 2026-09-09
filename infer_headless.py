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

from checkpoint import audio_off_kwargs, default_checkpoint, load_checkpoint, preprocess_camera_frame
from jetson_env import apply_jetson_runtime_tweaks, default_device
from label_hierarchy import combine_hierarchical_probs, inference_label
from radar_utils import DualRadarSession, fuse_dual_radar_tensors, fuse_radar_streams_for_model

try:
  from detector_pipeline import apply_cfar_mask_to_radar_seq
except Exception:  # pragma: no cover
  apply_cfar_mask_to_radar_seq = None  # type: ignore[assignment]


def _camera_stream_cls():
  from realtime_multimodal import CameraStream

  return CameraStream


def _presence_probs(outputs: dict) -> tuple[float, float]:
  human_prob = 1.0
  detect_prob = 1.0
  if outputs.get("human_logits") is not None:
    human_prob = float(F.softmax(outputs["human_logits"][0], dim=-1)[1].item())
    detect_prob = human_prob
  if outputs.get("detect_logits") is not None:
    detect_prob = float(F.softmax(outputs["detect_logits"][0], dim=-1)[1].item())
  return human_prob, detect_prob


def parse_args():
  p = argparse.ArgumentParser(description="JetsonCA headless multimodal inference")
  p.add_argument("--checkpoint", type=Path, default=default_checkpoint())
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
  p.add_argument("--radar1-port", type=str, default=None, help="Optional primary radar port, e.g. /dev/ttyACM0")
  p.add_argument("--radar2-port", type=str, default=None, help="Optional secondary radar port or __none__")
  p.add_argument("--mirror-radar2", action="store_true", default=False,
                 help="Opt-in: copy radar1 into radar2 when second HW unit is missing")
  p.add_argument("--no-mirror-radar2", action="store_false", dest="mirror_radar2")
  p.add_argument("--no-radar", action="store_true", help="Camera-only live: skip radar SDK, radar_present=False")
  p.add_argument("--dual-radar-fuse", choices=("auto", "none", "mean", "max"), default="auto")
  p.add_argument("--window", type=int, default=30)
  p.add_argument("--detect-threshold", type=float, default=0.35)
  p.add_argument("--human-threshold", type=float, default=0.5)
  p.add_argument("--live-detector-preprocess", action="store_true",
                 help="Apply train-time CFAR/ROI live (slow on Jetson)")
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
  fuse_mode = str(config.get("dual_radar_fuse", "none")) if args.dual_radar_fuse == "auto" else args.dual_radar_fuse
  detector_preprocess = False  # live CFAR is opt-in; train flag alone is too slow on Jetson
  detector_min_snr_db = float(config.get("detector_min_snr_db", 6.0))
  if bool(getattr(args, "live_detector_preprocess", False)):
    detector_preprocess = True

  radar1_buf: deque[torch.Tensor] = deque(maxlen=window)
  radar2_buf: deque[torch.Tensor] = deque(maxlen=window)
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
      radar_np = np.zeros((window, 3, 32, 32), dtype=np.float32)
      radar2_np = radar_np
      radar_present = torch.zeros(1, dtype=torch.bool, device=args.device)
    else:
      if len(radar1_buf) < window or len(camera_buf) < window:
        return
      radar_np = torch.stack(list(radar1_buf), dim=0).numpy()
      radar2_np = torch.stack(list(radar2_buf), dim=0).numpy() if len(radar2_buf) >= window else radar_np
      radar_np = np.asarray(fuse_dual_radar_tensors(radar_np, radar2_np, mode=fuse_mode), dtype=np.float32)
      radar_present = torch.ones(1, dtype=torch.bool, device=args.device)

    if detector_preprocess and apply_cfar_mask_to_radar_seq is not None:
      radar_np = apply_cfar_mask_to_radar_seq(
        radar_np, profile_max_range_m=float(args.max_range_m), min_snr_db=detector_min_snr_db
      )
      radar2_np = apply_cfar_mask_to_radar_seq(
        radar2_np, profile_max_range_m=float(args.max_range_m), min_snr_db=detector_min_snr_db
      )

    cam = torch.stack(list(camera_buf), dim=0).unsqueeze(0).to(args.device)
    camera_present = torch.ones(1, dtype=torch.bool, device=args.device)
    radar = torch.from_numpy(np.asarray(radar_np, dtype=np.float32)).unsqueeze(0).to(args.device)
    radar2 = torch.from_numpy(np.asarray(radar2_np, dtype=np.float32)).unsqueeze(0).to(args.device)

    with torch.no_grad():
      out = model(
        radar,
        cam,
        radar2=radar2,
        radar_present=radar_present,
        camera_present=camera_present,
        **audio_off_kwargs(model, args.device),
      )
      logits = out["activity_logits"] if "activity_logits" in out else out["logits"]
      probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
      if config.get("use_hierarchical_fusion"):
        coarse_probs = None
        subaction_probs = None
        if out.get("coarse_logits") is not None:
          coarse_probs = F.softmax(out["coarse_logits"][0], dim=-1).detach().cpu().numpy()
        if out.get("subaction_logits") is not None:
          subaction_probs = F.softmax(out["subaction_logits"][0], dim=-1).detach().cpu().numpy()
        probs = combine_hierarchical_probs(
          labels,
          probs,
          coarse_probs,
          subaction_probs,
          hierarchy_labels=list(config.get("all_labels") or labels),
        )
      human_prob, detect_prob = _presence_probs(out)

    display, conf = inference_label(labels, human_prob, probs, human_threshold=args.human_threshold)
    label = display
    gate_open = detect_prob >= args.detect_threshold
    if not gate_open:
      display = "none"
      label = "none"
    n_infer += 1
    now = time.time()

    row = {
      "t": now,
      "label": display,
      "raw_label": label,
      "conf": conf,
      "human_prob": human_prob,
      "detect_prob": detect_prob,
      "gate_open": gate_open,
      "probs": {labels[i]: float(probs[i]) for i in range(min(len(labels), len(probs)))},
      "radar_meta": meta,
      "radar_present": (not args.no_radar),
      "fuse_mode": fuse_mode,
    }
    if jsonl is not None:
      jsonl.write(json.dumps(row) + "\n")
      jsonl.flush()

    if now - last_print >= args.print_every:
      last_print = now
      fps = n_infer / max(now - t0, 1e-6)
      print(
        f"[{now - t0:6.1f}s] {display:16s} detect={detect_prob:.2f} gate={'OPEN' if gate_open else 'closed'}  "
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
        radar1_port=args.radar1_port,
        radar2_port=args.radar2_port,
        mirror_radar2=args.mirror_radar2,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
      ) as radars:
        print(f"radar: {radars.status_text} | fuse={fuse_mode}")
        while True:
          if args.duration_s > 0 and (time.time() - t0) >= args.duration_s:
            break

          r1, r2 = radars.read_tensors()
          _, meta = fuse_radar_streams_for_model(
            r1,
            r2,
            mode=fuse_mode,
            mirror_radar2=args.mirror_radar2,
          )
          if r1 is not None:
            radar1_buf.append(r1.detach().cpu())
          r2_use = r2
          if r2_use is None and r1 is not None and args.mirror_radar2:
            r2_use = r1
          if r2_use is not None:
            radar2_buf.append(r2_use.detach().cpu())

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
