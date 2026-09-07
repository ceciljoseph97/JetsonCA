"""Tk Profile tab: memory / resource / compute bench (no KPI gate)."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import scrolledtext, ttk

try:
  from checkpoint import default_checkpoint
except Exception:
  def default_checkpoint() -> Path:
    return Path("artifacts/walking_bg_audio_v1/best_multimodal_crossattention.pt")


def make_gui_bench_args(**overrides: Any) -> argparse.Namespace:
  ns = argparse.Namespace(
    checkpoint=default_checkpoint(),
    device="cpu",
    batch_size=1,
    window=30,
    warmup=10,
    runs=30,
    mode="both",
    all_modes=False,
    live=False,
    audio_device=None,
    n_cameras=1,
    n_radars=2,
    camera_device=0,
    camera_width=640,
    camera_height=480,
    camera_fps=15.0,
    num_rx=3,
    radar_profile="safe",
    frame_rate=5.0,
    radar1_uuid=None,
    radar2_uuid=None,
    radar1_port=None,
    radar2_port=None,
    no_mirror_radar2=False,
    min_range_m=0.0,
    max_range_m=None,
    system_monitor=True,
    system_monitor_interval_ms=500,
    out=Path("artifacts/benchmark_profile.json"),
    no_kpi=True,
    profile_memory=True,
    profile_resource=True,
    profile_compute=True,
  )
  for key, value in overrides.items():
    setattr(ns, key, value)
  return ns


def mount_profile_tab(
  parent: ttk.Frame,
  *,
  defaults_fn: Callable[[], dict[str, Any]],
  busy_fn: Callable[[], bool],
  ui_after: Callable[..., Any],
  mem_var: tk.BooleanVar | None = None,
  res_var: tk.BooleanVar | None = None,
  cmp_var: tk.BooleanVar | None = None,
) -> Callable[[], None]:
  parent.columnconfigure(0, weight=1)
  parent.rowconfigure(2, weight=1)

  opts = ttk.LabelFrame(parent, text="Profiling (no KPI checks)", padding=6)
  opts.grid(row=0, column=0, sticky="ew", pady=(0, 6))
  mem_var = mem_var if mem_var is not None else tk.BooleanVar(value=True)
  res_var = res_var if res_var is not None else tk.BooleanVar(value=True)
  cmp_var = cmp_var if cmp_var is not None else tk.BooleanVar(value=True)
  live_var = tk.BooleanVar(value=False)
  all_modes_var = tk.BooleanVar(value=False)
  mode_var = tk.StringVar(value="both")
  runs_var = tk.StringVar(value="30")
  ttk.Checkbutton(opts, text="Memory (weights, RSS, CUDA peak)", variable=mem_var).grid(
    row=0, column=0, sticky="w", padx=4, pady=2
  )
  ttk.Checkbutton(opts, text="Resource (CPU, RAM, tegrastats / GPU)", variable=res_var).grid(
    row=1, column=0, sticky="w", padx=4, pady=2
  )
  ttk.Checkbutton(opts, text="Computational (latency, FLOPs/MACs)", variable=cmp_var).grid(
    row=2, column=0, sticky="w", padx=4, pady=2
  )
  ttk.Checkbutton(opts, text="All modes (cam / radar / audio ablations)", variable=all_modes_var).grid(
    row=3, column=0, sticky="w", padx=4, pady=2
  )
  ttk.Checkbutton(opts, text="Live sensors (USB cam + radar + mic)", variable=live_var).grid(
    row=4, column=0, sticky="w", padx=4, pady=2
  )
  row = ttk.Frame(opts)
  row.grid(row=5, column=0, sticky="w", padx=4, pady=4)
  ttk.Label(row, text="Mode").pack(side="left")
  ttk.Combobox(
    row,
    textvariable=mode_var,
    values=("both", "radar_only", "camera_only", "audio_only", "audio_radar", "audio_camera"),
    state="readonly",
    width=16,
  ).pack(side="left", padx=(6, 12))
  ttk.Label(row, text="Runs").pack(side="left")
  ttk.Entry(row, textvariable=runs_var, width=6).pack(side="left", padx=(6, 0))

  run_row = ttk.Frame(parent)
  run_row.grid(row=1, column=0, sticky="ew", pady=(0, 6))
  status_var = tk.StringVar(value="idle")
  btn = ttk.Button(run_row, text="Run benchmark")
  btn.pack(side="left")
  ttk.Label(run_row, textvariable=status_var).pack(side="left", padx=8)

  log = scrolledtext.ScrolledText(parent, height=16, wrap="word", font=("Consolas", 9))
  log.grid(row=2, column=0, sticky="nsew")

  def _append(text: str):
    log.configure(state="normal")
    log.delete("1.0", "end")
    log.insert("1.0", text)
    log.configure(state="disabled")

  def _click():
    if busy_fn():
      status_var.set("stop live inference first")
      return
    if not (mem_var.get() or res_var.get() or cmp_var.get()):
      status_var.set("enable at least one profile")
      return
    try:
      runs = max(1, int(runs_var.get().strip()))
    except ValueError:
      status_var.set("runs must be an int")
      return
    btn.state(["disabled"])
    status_var.set("running…")
    _append("running…")

    def work():
      try:
        from benchmark import format_profile_report, run_benchmark

        kwargs = defaults_fn()
        kwargs.update(
          {
            "no_kpi": True,
            "profile_memory": bool(mem_var.get()),
            "profile_resource": bool(res_var.get()),
            "profile_compute": bool(cmp_var.get()),
            "system_monitor": bool(res_var.get()),
            "live": bool(live_var.get()),
            "all_modes": bool(all_modes_var.get()) and not live_var.get(),
            "mode": mode_var.get().strip() or "both",
            "runs": runs,
            "warmup": max(2, min(10, runs // 3)),
            "out": Path("artifacts/benchmark_profile.json"),
          }
        )
        report = run_benchmark(make_gui_bench_args(**kwargs))
        text = format_profile_report(report)
        text += f"\n\nwrote {kwargs['out']}"
        ui_after(0, lambda t=text: _done(t, "done"))
      except Exception as exc:
        ui_after(0, lambda e=exc: _done(f"benchmark failed:\n{e}", "failed"))

    def _done(text: str, status: str):
      _append(text)
      status_var.set(status)
      btn.state(["!disabled"])

    threading.Thread(target=work, daemon=True).start()

  btn.configure(command=_click)
  return _click
