"""Tk Profile tab for gestureEdge (radar-only modes)."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import scrolledtext, ttk


def make_gui_bench_args(**overrides: Any) -> argparse.Namespace:
  ns = argparse.Namespace(
    checkpoint=Path("artifacts/gesture_edge_soli/best_gesture_edge.pt"),
    device="cpu",
    window=40,
    warmup=10,
    runs=30,
    mode="both",
    all_modes=False,
    out=Path("artifacts/benchmark_gesture_edge.json"),
    profile_memory=True,
    profile_resource=False,
    profile_compute=True,
  )
  for key, value in overrides.items():
    setattr(ns, key, value)
  return ns


def mount_gesture_edge_profile_tab(
  parent: ttk.Frame,
  *,
  defaults_fn: Callable[[], dict[str, Any]],
  busy_fn: Callable[[], bool],
  ui_after: Callable[..., Any],
) -> Callable[[], None]:
  parent.columnconfigure(0, weight=1)
  parent.rowconfigure(2, weight=1)

  opts = ttk.LabelFrame(parent, text="Profiling (gestureEdge)", padding=6)
  opts.grid(row=0, column=0, sticky="ew", pady=(0, 6))
  mem_var = tk.BooleanVar(value=True)
  cmp_var = tk.BooleanVar(value=True)
  all_modes_var = tk.BooleanVar(value=True)
  mode_var = tk.StringVar(value="both")
  runs_var = tk.StringVar(value="30")
  ttk.Checkbutton(opts, text="Memory (weights, CUDA peak)", variable=mem_var).grid(
    row=0, column=0, sticky="w", padx=4, pady=2
  )
  ttk.Checkbutton(opts, text="Computational (latency, FLOPs/MACs)", variable=cmp_var).grid(
    row=1, column=0, sticky="w", padx=4, pady=2
  )
  ttk.Checkbutton(opts, text="All modes (both / r1 / r2)", variable=all_modes_var).grid(
    row=2, column=0, sticky="w", padx=4, pady=2
  )
  row = ttk.Frame(opts)
  row.grid(row=3, column=0, sticky="w", padx=4, pady=4)
  ttk.Label(row, text="Mode").pack(side="left")
  ttk.Combobox(
    row,
    textvariable=mode_var,
    values=("both", "radar1_only", "radar2_only"),
    state="readonly",
    width=14,
  ).pack(side="left", padx=(6, 12))
  ttk.Label(row, text="Runs").pack(side="left")
  ttk.Entry(row, textvariable=runs_var, width=6).pack(side="left", padx=(6, 0))

  run_row = ttk.Frame(parent)
  run_row.grid(row=1, column=0, sticky="ew", pady=(0, 6))
  status_var = tk.StringVar(value="idle")
  btn = ttk.Button(run_row, text="Run benchmark")
  btn.pack(side="left")
  ttk.Label(run_row, textvariable=status_var).pack(side="left", padx=8)

  log = scrolledtext.ScrolledText(parent, height=14, wrap="word", font=("Consolas", 9))
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
    if not (mem_var.get() or cmp_var.get()):
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
        from gestureEdge.benchmark import format_profile_report, run_benchmark

        kwargs = defaults_fn()
        kwargs.update(
          {
            "profile_memory": bool(mem_var.get()),
            "profile_compute": bool(cmp_var.get()),
            "profile_resource": False,
            "all_modes": bool(all_modes_var.get()),
            "mode": mode_var.get().strip() or "both",
            "runs": runs,
            "warmup": max(2, min(10, runs // 3)),
            "out": Path(kwargs.get("out") or "artifacts/benchmark_gesture_edge.json"),
          }
        )
        report = run_benchmark(make_gui_bench_args(**kwargs))
        text = format_profile_report(report)
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
