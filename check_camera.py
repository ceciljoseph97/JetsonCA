#!/usr/bin/env python3
"""One-camera smoke checks for Jetson (no radar required).

Steps:
  1) OpenCV import
  2) Probe USB / CSI indices
  3) Grab N live frames from chosen device
  4) Optional: camera_only model forward on a filled window
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import load_checkpoint, preprocess_camera_frame
from jetson_env import apply_jetson_runtime_tweaks, default_device


def _try_import_cv2():
  try:
    import cv2

    return cv2, None
  except Exception as exc:
    return None, str(exc)


def parse_args():
  p = argparse.ArgumentParser(description="JetsonCA one-camera checks (no radar)")
  p.add_argument("--camera-device", type=int, default=None, help="Force index; default = first probed")
  p.add_argument("--width", type=int, default=640)
  p.add_argument("--height", type=int, default=480)
  p.add_argument("--fps", type=float, default=15.0)
  p.add_argument("--frames", type=int, default=30, help="Frames to capture for stream test")
  p.add_argument("--save-preview", type=Path, default=Path("artifacts/camera_preview.jpg"))
  p.add_argument("--infer", action="store_true", help="Also run camera_only model forwards")
  p.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_multimodal_crossattention.pt"))
  p.add_argument("--device", type=str, default=default_device())
  p.add_argument("--window", type=int, default=30)
  p.add_argument("--infer-runs", type=int, default=10)
  p.add_argument("--out", type=Path, default=Path("artifacts/camera_check_jetson.json"))
  return p.parse_args()


def main():
  args = parse_args()
  report: dict = {
    "runtime": apply_jetson_runtime_tweaks(),
    "device": args.device,
    "pass": {},
  }

  cv2, cv2_err = _try_import_cv2()
  report["opencv"] = {"ok": cv2 is not None, "error": cv2_err, "version": getattr(cv2, "__version__", None)}
  report["pass"]["opencv_import"] = cv2 is not None
  if cv2 is None:
    print("FAIL: OpenCV import")
    print(cv2_err)
    print(
      "\nFix hints:\n"
      "  conda install -y -c conda-forge 'libstdcxx-ng>=13'\n"
      "  # or\n"
      "  export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}\n"
      "  pip uninstall -y opencv-python opencv-python-headless\n"
      "  conda install -y -c conda-forge opencv\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    sys.exit(1)

  print(f"OpenCV {cv2.__version__} OK")

  from realtime_multimodal import CameraStream, probe_camera_devices

  print("Probing cameras…")
  found = probe_camera_devices()
  report["probed"] = found
  report["pass"]["camera_found"] = len(found) > 0
  print(json.dumps(found, indent=2))
  if not found:
    print("FAIL: no cameras probed — plug USB cam and retry")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    sys.exit(2)

  cam_idx = args.camera_device if args.camera_device is not None else int(found[0]["index"])
  report["selected_index"] = cam_idx
  print(f"Opening camera {cam_idx}…")

  stream = CameraStream(cam_idx, args.width, args.height, args.fps).start(warmup_s=3.0)
  grabbed = 0
  last_rgb = None
  t0 = time.time()
  try:
    while grabbed < args.frames:
      frame = stream.get_latest()
      if frame is None:
        time.sleep(0.02)
        continue
      last_rgb = frame
      grabbed += 1
      time.sleep(max(0.0, 1.0 / max(args.fps, 1.0)))
  finally:
    stream.stop()

  elapsed = time.time() - t0
  report["capture"] = {
    "frames": grabbed,
    "elapsed_s": elapsed,
    "approx_fps": grabbed / max(elapsed, 1e-6),
    "shape": list(last_rgb.shape) if last_rgb is not None else None,
    "mean": float(last_rgb.mean()) if last_rgb is not None else None,
    "std": float(last_rgb.std()) if last_rgb is not None else None,
  }
  report["pass"]["frames_captured"] = grabbed >= max(5, args.frames // 2)
  print(
    f"Captured {grabbed}/{args.frames} frames in {elapsed:.1f}s "
    f"(~{report['capture']['approx_fps']:.1f} FPS) shape={report['capture']['shape']}"
  )

  if last_rgb is not None and args.save_preview is not None:
    args.save_preview.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(args.save_preview), bgr)
    report["preview"] = str(args.save_preview)
    print(f"preview -> {args.save_preview}")

  if args.infer:
    print("Loading checkpoint for camera_only infer…")
    model, labels, config = load_checkpoint(args.checkpoint, args.device)
    image_size = int(config["image_size"])
    window = int(args.window)
    # refill short buffer by re-opening briefly
    stream = CameraStream(cam_idx, args.width, args.height, args.fps).start(warmup_s=2.0)
    buf: deque[torch.Tensor] = deque(maxlen=window)
    deadline = time.time() + 15.0
    try:
      while len(buf) < window and time.time() < deadline:
        frame = stream.get_latest()
        if frame is None:
          time.sleep(0.02)
          continue
        buf.append(preprocess_camera_frame(frame, image_size))
    finally:
      stream.stop()

    report["pass"]["window_filled"] = len(buf) >= window
    if len(buf) < window:
      print(f"FAIL: only filled {len(buf)}/{window} camera frames for infer")
    else:
      camera = torch.stack(list(buf), dim=0).unsqueeze(0).to(args.device)
      radar = torch.zeros(1, window, 3, 32, 32, device=args.device)
      radar_present = torch.zeros(1, dtype=torch.bool, device=args.device)
      camera_present = torch.ones(1, dtype=torch.bool, device=args.device)
      times_ms = []
      last_label = None
      last_conf = None
      with torch.no_grad():
        for _ in range(max(1, args.infer_runs)):
          if args.device.startswith("cuda"):
            torch.cuda.synchronize()
          t1 = time.perf_counter()
          out = model(
            radar,
            camera,
            radar_present=radar_present,
            camera_present=camera_present,
          )
          if args.device.startswith("cuda"):
            torch.cuda.synchronize()
          times_ms.append((time.perf_counter() - t1) * 1000.0)
          logits = out.get("activity_logits", out.get("logits"))
          probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
          idx = int(np.argmax(probs))
          last_label = labels[idx]
          last_conf = float(probs[idx])
      arr = np.asarray(times_ms, dtype=np.float64)
      report["infer_camera_only"] = {
        "runs": int(arr.size),
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "fps": float(1000.0 / max(arr.mean(), 1e-9)),
        "last_label": last_label,
        "last_conf": last_conf,
        "labels": labels,
      }
      report["pass"]["infer_lt_100ms"] = report["infer_camera_only"]["latency_ms_mean"] < 100.0
      print(
        f"camera_only infer: mean={arr.mean():.1f} ms  "
        f"p95={np.percentile(arr, 95):.1f} ms  "
        f"pred={last_label} ({last_conf:.2f})"
      )

  report["pass_count"] = int(sum(1 for v in report["pass"].values() if v))
  report["check_count"] = int(len(report["pass"]))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
  print(f"\nscore {report['pass_count']}/{report['check_count']}  wrote {args.out}")
  sys.exit(0 if report["pass_count"] == report["check_count"] else 3)


if __name__ == "__main__":
  main()
