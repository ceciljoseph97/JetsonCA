"""Minimal 2D car: Push=left, Pull=right, Hold=stay, Palm Tilt=HONK overlay."""

from __future__ import annotations

import random

import numpy as np
import tkinter as tk


def _norm(name: str) -> str:
  return str(name).strip().lower().replace("-", " ").replace("_", " ")


def class_mass(labels: list[str], probs: np.ndarray, aliases: set[str]) -> float:
  p = np.asarray(probs, dtype=np.float32).reshape(-1)
  total = 0.0
  for i, lab in enumerate(labels):
    if i >= p.size:
      break
    if _norm(lab) in aliases:
      total += float(p[i])
  return total


def steer_from_probs(
  labels: list[str],
  probs: np.ndarray,
  *,
  swap_lr: bool = False,
  min_conf: float = 0.32,
) -> tuple[str, float]:
  """Push / Pull / Hold / idle. No analog x — caller steps one lane per onset."""
  push = class_mass(labels, probs, {"push"})
  pull = class_mass(labels, probs, {"pull"})
  hold = class_mass(labels, probs, {"palm hold"})
  if swap_lr:
    push, pull = pull, push
  mass = push + pull + hold
  if mass < 1e-6:
    return "neutral", 0.0
  push, pull, hold = push / mass, pull / mass, hold / mass
  conf = max(push, pull, hold)
  if conf < min_conf:
    return "neutral", conf
  if hold >= max(push, pull) or hold >= 0.40:
    return "hold", hold
  if push > pull:
    return "left", push
  return "right", pull


class CarDriveGame:
  """Top-down 3-lane dodge. x in [-1, 1]."""

  def __init__(self, canvas: tk.Canvas):
    self.canvas = canvas
    self.x = 0.0
    self.scroll = 0.0
    self.speed = 1.45
    self.score = 0
    self.hits = 0
    self.alive = True
    self.playing = False
    self.banner = "Play"
    self.control = "neutral"
    self.strength = 0.0
    self.swap_lr = False
    self.obstacles: list[dict[str, float]] = []
    self._spawn_cd = 0.0
    self._kb: str | None = None
    self.lane = 0.0
    self._gest: str | None = None
    self.honk = False
    self._honk_t = 0.0
    self._kb_honk = False

  def set_key(self, cmd: str | None):
    if cmd == "honk":
      self._kb_honk = True
      return
    if cmd == self._kb:
      return
    self._kb = cmd
    if cmd == "left":
      self._nudge(-1.0)
    elif cmd == "right":
      self._nudge(1.0)

  def set_honk(self, on: bool):
    self._kb_honk = bool(on)

  def _nudge(self, step: float):
    self.lane = float(np.clip(self.lane + step, -1.0, 1.0))

  def reset(self):
    self.x = 0.0
    self.lane = 0.0
    self._gest = None
    self._kb = None
    self.scroll = 0.0
    self.score = 0
    self.hits = 0
    self.alive = True
    self.obstacles.clear()
    self._spawn_cd = 1.8
    self.playing = False
    self.banner = "Play"
    self.honk = False
    self._honk_t = 0.0
    self._kb_honk = False

  def step(self, dt: float, labels: list[str], probs: np.ndarray, *, honk: bool = False):
    if honk or self._kb_honk:
      self.honk = True
      self._honk_t = 0.85
    else:
      self._honk_t = max(0.0, self._honk_t - dt)
      if self._honk_t <= 0.0:
        self.honk = False
    if (not self.playing) or (not self.alive):
      return
    if self._kb == "left":
      name, mag = "left", 1.0
    elif self._kb == "right":
      name, mag = "right", 1.0
    elif self._kb == "center":
      name, mag = "hold", 1.0
    else:
      name, mag = steer_from_probs(labels, probs, swap_lr=self.swap_lr)
      if name in ("left", "right") and name != self._gest:
        self._nudge(-1.0 if name == "left" else 1.0)
      if name in ("left", "right"):
        self._gest = name
      else:
        self._gest = None
    self.control, self.strength = name, mag
    self.x += (self.lane - self.x) * min(1.0, 10.0 * dt)
    self.x = float(np.clip(self.x, -1.0, 1.0))

    self.scroll += self.speed * 42.0 * dt
    self._spawn_cd -= dt
    if self._spawn_cd <= 0.0:
      lane = float(random.choice((-1.0, 0.0, 1.0)))
      self.obstacles.append({"y": -40.0, "lane": lane})
      self._spawn_cd = random.uniform(2.2, 3.6)

    h = max(int(self.canvas.winfo_height()), 1)
    car_lane = self._lane(self.x)
    kept: list[dict[str, float]] = []
    for ob in self.obstacles:
      ob["y"] += self.speed * 42.0 * dt
      if 0.62 * h < ob["y"] < 0.82 * h and abs(ob["lane"] - car_lane) < 0.5:
        self.hits += 1
        self.alive = False
        continue
      if ob["y"] < h + 40:
        kept.append(ob)
      else:
        self.score += 1
    self.obstacles = kept

  @staticmethod
  def _lane(x: float) -> float:
    if x < -0.4:
      return -1.0
    if x > 0.4:
      return 1.0
    return 0.0

  def draw(self):
    c = self.canvas
    w = max(c.winfo_width(), 8)
    h = max(c.winfo_height(), 8)
    c.delete("all")
    c.create_rectangle(0, 0, w, h, fill="#1c1d22", outline="")
    pad = int(w * 0.12)
    road_l, road_r = pad, w - pad
    c.create_rectangle(road_l, 0, road_r, h, fill="#2b2d33", outline="")
    mid = (road_l + road_r) / 2
    third = (road_r - road_l) / 3
    dash_off = int(self.scroll) % 36
    for k in (1, 2):
      x = road_l + k * third
      y = -36 + dash_off
      while y < h:
        c.create_line(x, y, x, y + 18, fill="#d8d8d8", width=2)
        y += 36
    c.create_line(road_l, 0, road_l, h, fill="#e6c35c", width=3)
    c.create_line(road_r, 0, road_r, h, fill="#e6c35c", width=3)

    def lane_x(lane: float) -> float:
      return mid + lane * third

    for ob in self.obstacles:
      ox = lane_x(ob["lane"])
      oy = ob["y"]
      c.create_polygon(
        ox, oy - 14, ox + 12, oy + 10, ox - 12, oy + 10,
        fill="#e67a22", outline="#111",
      )

    cx = lane_x(self.x)
    cy = 0.74 * h
    body = "#d94c4c" if self.alive else "#666"
    c.create_rectangle(cx - 16, cy - 22, cx + 16, cy + 22, fill=body, outline="#111", width=2)
    c.create_rectangle(cx - 10, cy - 12, cx + 10, cy + 4, fill="#89c4e8", outline="")

    hud = f"{self.control}  {self.strength:.2f}   score {self.score}  hits {self.hits}"
    if not self.alive:
      hud += "   CRASH — R to reset"
    c.create_text(10, 12, anchor="nw", fill="#f2f2f2", font=("Segoe UI", 11, "bold"), text=hud)
    c.create_text(
      10, 32, anchor="nw", fill="#9aa0a6", font=("Segoe UI", 9),
      text="Push→left 1 lane   Pull→right 1 lane   Hold→stay   Tilt→HONK",
    )
    if self.honk:
      box_y0, box_y1 = h * 0.06, h * 0.22
      c.create_rectangle(w * 0.22, box_y0, w * 0.78, box_y1, fill="#f5c518", outline="#111", width=3)
      c.create_text(w / 2, (box_y0 + box_y1) / 2, fill="#111", font=("Segoe UI", 32, "bold"), text="HONK")
    if (not self.playing) or self.banner or (not self.alive):
      msg = self.banner or ("CRASH — Reset / Play" if not self.alive else "Play")
      c.create_rectangle(w * 0.12, h * 0.38, w * 0.88, h * 0.58, fill="#111318", outline="#e6c35c")
      c.create_text(w / 2, h * 0.48, fill="#f2f2f2", font=("Segoe UI", 14, "bold"), text=msg)

  def hint(self) -> str:
    if self.banner:
      return str(self.banner)
    if not self.playing:
      return "paused · Play to start"
    if not self.alive:
      return f"crash · score {self.score} · Reset / Play"
    extra = " · HONK" if self.honk else ""
    return f"{self.control} · x={self.x:+.2f} · score {self.score}{extra}"
