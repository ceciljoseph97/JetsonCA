"""Hierarchical labels and training targets for human / activity / subaction."""

from __future__ import annotations

from dataclasses import dataclass

BACKGROUND_LABELS = frozenset({"background", "no_human", "empty", "idle"})

LABEL_HIERARCHY: dict[str, tuple[str, str, str]] = {
  "walking_towards": ("human", "walking", "towards"),
  "walking_away": ("human", "walking", "away"),
  "crossing": ("human", "walking", "crossing"),
  "background": ("background", "none", "none"),
  "no_human": ("background", "none", "none"),
  "empty": ("background", "none", "none"),
}


@dataclass(frozen=True)
class TrainingTargets:
  human_label: int
  activity_label: int
  is_background: bool
  flat_label: str


def is_background_label(label: str) -> bool:
  return label.lower() in BACKGROUND_LABELS


def label_hierarchy(label: str) -> tuple[str, str, str]:
  if label in LABEL_HIERARCHY:
    return LABEL_HIERARCHY[label]
  if is_background_label(label):
    return ("background", "none", "none")
  parts = label.split("_", 1)
  if len(parts) == 2:
    return ("human", parts[0], parts[1])
  return ("human", label, label)


def format_hierarchy(label: str, confidence: float | None = None) -> str:
  parent, activity, subaction = label_hierarchy(label)
  # Lead with flat class name so crossing / walking_away stay readable in overlays.
  head = label
  if confidence is not None:
    head += f" ({confidence:.2f})"
  return f"{head}\n{parent} > {activity} > {subaction}"


def hierarchy_dict(label: str, confidence: float | None = None) -> dict:
  parent, activity, subaction = label_hierarchy(label)
  payload = {
    "parent": parent,
    "activity": activity,
    "subaction": subaction,
    "flat_label": label,
  }
  if confidence is not None:
    payload["confidence"] = float(confidence)
  return payload


def build_activity_index(labels: list[str]) -> dict[str, int]:
  activity_labels = [label for label in labels if not is_background_label(label)]
  return {label: idx for idx, label in enumerate(activity_labels)}


def targets_for_label(label: str, activity_index: dict[str, int]) -> TrainingTargets:
  if is_background_label(label):
    return TrainingTargets(human_label=0, activity_label=-1, is_background=True, flat_label=label)
  if label not in activity_index:
    raise KeyError(f"Activity label {label!r} missing from activity_index")
  return TrainingTargets(
    human_label=1,
    activity_label=activity_index[label],
    is_background=False,
    flat_label=label,
  )


def inference_label(
  labels: list[str],
  human_prob: float,
  activity_probs,
  human_threshold: float = 0.5,
  *,
  min_margin: float = 0.0,
) -> tuple[str, float]:
  import numpy as np

  activity_labels = [label for label in labels if not is_background_label(label)]
  if human_prob < human_threshold:
    bg = next((label for label in labels if is_background_label(label)), "background")
    return bg, float(1.0 - human_prob)

  probs = np.asarray(activity_probs, dtype=np.float32).reshape(-1)
  if probs.size != len(activity_labels):
    # Full-label softmax (includes background): restrict to activity indices.
    idx_map = [i for i, label in enumerate(labels) if not is_background_label(label)]
    if len(idx_map) == len(activity_labels) and probs.size == len(labels):
      probs = probs[idx_map]
    else:
      probs = probs[: len(activity_labels)]

  order = np.argsort(probs)[::-1]
  top_idx = int(order[0])
  top_p = float(probs[top_idx])
  second_p = float(probs[order[1]]) if len(order) > 1 else 0.0
  if min_margin > 0.0 and (top_p - second_p) < min_margin:
    return "uncertain", top_p
  return activity_labels[top_idx], top_p


def apply_logit_bias(probs, labels: list[str], bias: dict[str, float] | None):
  """Re-normalize probs after multiplicative prior correction: p' ∝ p * exp(-bias)."""
  import numpy as np

  if not bias:
    return np.asarray(probs, dtype=np.float32)
  p = np.asarray(probs, dtype=np.float64).copy()
  activity_labels = [label for label in labels if not is_background_label(label)]
  for i, label in enumerate(activity_labels[: p.size]):
    if label in bias:
      p[i] *= float(np.exp(-float(bias[label])))
  s = p.sum()
  if s <= 1e-12:
    return np.asarray(probs, dtype=np.float32)
  return (p / s).astype(np.float32)