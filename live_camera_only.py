#!/usr/bin/env python3
"""Live multimodal inference / KPI with 1 USB camera and NO radar.

Uses the full cross-attention model with:
  radar = zeros
  radar_present = False
  camera_present = True
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import load_checkpoint, preprocess_camera_frame
from jetson_env import apply_jetson_runtime_tweaks, default_device
from label_hierarchy import inference_label


def parse_args():
  p = argparse.ArgumentParser(description="Live camera-only multimodal run (radar unavailable)")
  p.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_multimodal_crossattention.pt"))
  p.add_argument("--device", type=str, default=default_device())
  p.add_argument("--camera-device", type=int, default=0)
  p.add_argument("--camera-width", type=int, default=640)
  p.add_argument("--camera-height", type=int, default=480)
  p.add_argument("--camera-fps", type=float, default=15.0)
  p.add_argument("--window", type=int, default=30)
  p.add_argument("--detect-threshold", type=float, default=0.35)
  p.add_argument("--warmup", type=int, default=5)
  p.add_argument("--runs", type=int, default=30, help="Timed inference samples for JSON KPI block")
  p.add_argument("--duration-s", type=float, default=0.0, help="Extra live print loop after timed runs; 0=skip")
  p.add_argument("--print-every", type=float, default=1.0)
  p.add_argument("--out", type=Path, default=Path("artifacts/live_camera_only_jetson.json"))
  p.add_argument("--jsonl-out", type=Path, default=None)
  return p.parse_args()


def _sync(device: str):
  if str(device).startswith("cuda") and torch.cuda.is_available():
    torch.cuda.synchronize()


def main():
  args = parse_args()
  runtime = apply_jetson_runtime_tweaks()
  print(json.dumps({"runtime": runtime, "device": args.device, "mode": "live_camera_only"}, indent=2))

  from realtime_multimodal import CameraStream

  model, labels, config = load_checkpoint(args.checkpoint, args.device)
  model.eval()
  image_size = int(config["image_size"])
  window = int(args.window)

  camera = CameraStream(
    args.camera_device,
    args.camera_width,
    args.camera_height,
    args.camera_fps,
  ).start(warmup_s=3.0)

  cam_buf: deque[torch.Tensor] = deque(maxlen=window)
  radar_present = torch.zeros(1, dtype=torch.bool, device=args.device)
  camera_present = torch.ones(1, dtype=torch.bool, device=args.device)

  jsonl = None
  if args.jsonl_out is not None:
    args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.jsonl_out.open("w", encoding="utf-8")

  times_ms: list[float] = []
  preds: list[dict] = []
  warmup_left = int(args.warmup)
  measured = 0
  t0 = time.time()
  last_print = 0.0
  n_infer = 0

  print(
    f"live camera_only: cam={args.camera_device} window={window} "
    f"radar=OFF (zeros, radar_present=False)"
  )
  try:
    while measured < args.runs:
      frame = camera.get_latest()
      if frame is None:
        time.sleep(0.01)
        continue
      cam_buf.append(preprocess_camera_frame(frame, image_size))
      if len(cam_buf) < window:
        continue

      camera_t = torch.stack(list(cam_buf), dim=0).unsqueeze(0).to(args.device)
      radar_t = torch.zeros(1, window, 3, 32, 32, device=args.device)

      if warmup_left > 0:
        with torch.no_grad():
          model(radar_t, camera_t, radar_present=radar_present, camera_present=camera_present)
        _sync(args.device)
        warmup_left -= 1
        continue

      _sync(args.device)
      t1 = time.perf_counter()
      with torch.no_grad():
        out = model(radar_t, camera_t, radar_present=radar_present, camera_present=camera_present)
      _sync(args.device)
      dt_ms = (time.perf_counter() - t1) * 1000.0
      times_ms.append(dt_ms)
      measured += 1
      n_infer += 1

      logits = out.get("activity_logits", out.get("logits"))
      probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
      idx = int(np.argmax(probs))
      conf = float(probs[idx])
      raw = labels[idx] if conf >= args.detect_threshold else "none"
      display = inference_label(raw) if raw != "none" else "none"
      row = {
        "t": time.time(),
        "label": display,
        "raw_label": raw,
        "conf": conf,
        "latency_ms": dt_ms,
        "probs": {labels[i]: float(probs[i]) for i in range(len(labels))},
        "radar_present": False,
      }
      preds.append(row)
      if jsonl is not None:
        jsonl.write(json.dumps(row) + "\n")
        jsonl.flush()

      now = time.time()
      if now - last_print >= args.print_every:
        last_print = now
        print(
          f"[timed {measured}/{args.runs}] {display:16s} conf={conf:.2f}  "
          f"infer={dt_ms:.1f} ms"
        )

    # optional continuous loop
    if args.duration_s > 0:
      print(f"continuing live print for {args.duration_s:.0f}s (Ctrl+C to stop early)…")
      end = time.time() + args.duration_s
      while time.time() < end:
        frame = camera.get_latest()
        if frame is None:
          time.sleep(0.01)
          continue
        cam_buf.append(preprocess_camera_frame(frame, image_size))
        if len(cam_buf) < window:
          continue
        camera_t = torch.stack(list(cam_buf), dim=0).unsqueeze(0).to(args.device)
        radar_t = torch.zeros(1, window, 3, 32, 32, device=args.device)
        _sync(args.device)
        t1 = time.perf_counter()
        with torch.no_grad():
          out = model(radar_t, camera_t, radar_present=radar_present, camera_present=camera_present)
        _sync(args.device)
        dt_ms = (time.perf_counter() - t1) * 1000.0
        n_infer += 1
        logits = out.get("activity_logits", out.get("logits"))
        probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        raw = labels[idx] if conf >= args.detect_threshold else "none"
        display = inference_label(raw) if raw != "none" else "none"
        now = time.time()
        if now - last_print >= args.print_every:
          last_print = now
          fps = n_infer / max(now - t0, 1e-6)
          print(f"[live] {display:16s} conf={conf:.2f}  infer={dt_ms:.1f} ms  ~{fps:.1f} FPS")
  except KeyboardInterrupt:
    print("\nstopped")
  finally:
    camera.stop()
    if jsonl is not None:
      jsonl.close()

  if not times_ms:
    raise SystemExit("No timed inferences collected — check camera index / OpenCV")

  arr = np.asarray(times_ms, dtype=np.float64)
  report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "mode": "live_camera_only",
    "device": args.device,
    "runtime": runtime,
    "camera_device": args.camera_device,
    "window": window,
    "radar_present": False,
    "note": "Full multimodal net; radar path gated off via radar_present=False + zero radar tensor.",
    "latency": {
      "runs": int(arr.size),
      "warmup": int(args.warmup),
      "latency_ms_mean": float(arr.mean()),
      "latency_ms_std": float(arr.std()),
      "latency_ms_p50": float(np.percentile(arr, 50)),
      "latency_ms_p95": float(np.percentile(arr, 95)),
      "throughput_fps": float(1000.0 / max(arr.mean(), 1e-9)),
    },
    "last_predictions": preds[-5:],
    "pass": {
      "latency_mean_lt_100ms": bool(arr.mean() < 100.0),
      "throughput_ge_8fps": bool((1000.0 / max(arr.mean(), 1e-9)) >= 8.0),
    },
  }
  report["pass_count"] = int(sum(1 for v in report["pass"].values() if v))
  report["check_count"] = int(len(report["pass"]))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
  lat = report["latency"]
  print(
    f"\nDONE camera_only live  mean={lat['latency_ms_mean']:.1f} ms  "
    f"p95={lat['latency_ms_p95']:.1f} ms  fps={lat['throughput_fps']:.1f}  "
    f"score={report['pass_count']}/{report['check_count']}"
  )
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
