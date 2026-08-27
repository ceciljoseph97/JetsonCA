#!/usr/bin/env python3
"""Export multimodal checkpoint to ONNX for TensorRT on Jetson."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from jetson_env import default_device
from checkpoint import load_checkpoint


class ActivityForward(torch.nn.Module):
  def __init__(self, model: torch.nn.Module):
    super().__init__()
    self.model = model

  def forward(
    self,
    radar: torch.Tensor,
    camera: torch.Tensor,
    radar2: torch.Tensor,
    radar_present: torch.Tensor,
    camera_present: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = self.model(
      radar,
      camera,
      radar2=radar2,
      radar_present=radar_present,
      camera_present=camera_present,
    )
    activity = out["activity_logits"] if "activity_logits" in out else out["logits"]
    detect = out["detect_logits"]
    if detect is None:
      detect = out["human_logits"]
    if detect is None:
      detect = torch.zeros(activity.shape[0], 2, device=activity.device, dtype=activity.dtype)
      detect[:, 1] = 1.0
    human = out["human_logits"]
    if human is None:
      human = detect
    return activity, detect, human


def parse_args():
  p = argparse.ArgumentParser(description="Export JetsonCA model to ONNX")
  p.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_multimodal_crossattention.pt"))
  p.add_argument("--out", type=Path, default=Path("artifacts/multimodal_crossattention.onnx"))
  p.add_argument("--device", type=str, default="cpu", help="Export usually on CPU; device for loading only")
  p.add_argument("--window", type=int, default=30)
  p.add_argument("--opset", type=int, default=17)
  return p.parse_args()


def main():
  args = parse_args()
  device = args.device or default_device()
  model, labels, config = load_checkpoint(args.checkpoint, device)
  model.eval()
  image_size = int(config["image_size"])
  window = int(args.window)

  wrapper = ActivityForward(model).to("cpu").eval()
  radar = torch.randn(1, window, 3, 32, 32)
  radar2 = torch.randn(1, window, 3, 32, 32)
  camera = torch.randn(1, window, 3, image_size, image_size)
  radar_present = torch.ones(1, dtype=torch.bool)
  camera_present = torch.ones(1, dtype=torch.bool)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    wrapper,
    (radar, camera, radar2, radar_present, camera_present),
    str(args.out),
    input_names=["radar", "camera", "radar2", "radar_present", "camera_present"],
    output_names=["activity_logits", "detect_logits", "human_logits"],
    opset_version=args.opset,
    dynamo=False,
  )
  meta = {
    "labels": labels,
    "config": config,
    "window": window,
    "onnx": str(args.out),
    "inputs": ["radar", "camera", "radar2", "radar_present", "camera_present"],
    "outputs": ["activity_logits", "detect_logits", "human_logits"],
  }
  meta_path = args.out.with_suffix(".json")
  meta_path.write_text(__import__("json").dumps(meta, indent=2, default=str), encoding="utf-8")
  print(f"wrote {args.out}")
  print(f"wrote {meta_path}")
  print("Next on Jetson: trtexec --onnx=... --saveEngine=artifacts/model.trt --fp16")


if __name__ == "__main__":
  main()
