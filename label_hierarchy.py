"""Hierarchical labels and training targets for human / activity / subaction."""

from __future__ import annotations

from dataclasses import dataclass

BACKGROUND_LABELS = frozenset({"background", "no_human", "empty", "idle"})

LABEL_ALIASES: dict[str, str] = {
  "wavingleft": "waving",
  "wavingright": "waving",
  "waving_left": "waving",
  "waving_right": "waving",
  "standing_still": "standing_still",
}

LABEL_HIERARCHY: dict[str, tuple[str, str, str]] = {
  "clapping": ("human", "clapping", "still"),
  "jumping": ("human", "jumping", "still"),
  "walking": ("human", "walking", "still"),
  "walking_towards": ("human", "walking", "towards"),
  "walking_away": ("human", "walking", "away"),
  "crossing": ("human", "walking", "crossing"),
  "waving": ("human", "waving", "still"),
  "standing_still": ("human", "standing", "still"),
  "background": ("background", "none", "none"),
  "no_human": ("background", "none", "none"),
  "empty": ("background", "none", "none"),
}


@dataclass(frozen=True)
class TrainingTargets:
  human_label: int
  activity_label: int
  coarse_label: int
  subaction_label: int
  is_background: bool
  flat_label: str


def is_background_label(label: str) -> bool:
  return label.lower() in BACKGROUND_LABELS


def canonical_label_name(label: str) -> str:
  raw = label.strip()
  if raw in LABEL_HIERARCHY:
    return raw
  key = raw.lower()
  return LABEL_ALIASES.get(key, raw)


def label_hierarchy(label: str) -> tuple[str, str, str]:
  label = canonical_label_name(label)
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


def coarse_label_name(label: str) -> str:
  _, activity, _ = label_hierarchy(label)
  return activity


def subaction_label_name(label: str) -> str:
  _, _, subaction = label_hierarchy(label)
  return subaction


def build_coarse_index(labels: list[str]) -> dict[str, int]:
  coarse_labels = sorted({coarse_label_name(label) for label in labels})
  return {label: idx for idx, label in enumerate(coarse_labels)}


def build_subaction_index(labels: list[str]) -> dict[str, int]:
  sub_labels = sorted({subaction_label_name(label) for label in labels})
  return {label: idx for idx, label in enumerate(sub_labels)}


def combine_hierarchical_probs(
  labels: list[str],
  activity_probs,
  coarse_probs=None,
  subaction_probs=None,
  *,
  hierarchy_labels: list[str] | None = None,
):
  """Fuse activity × coarse × sub softmax.

  hierarchy_labels must match training label set (incl. background) so coarse/sub
  index order matches the classifier heads. Activity display labels can be a subset.
  """
  import numpy as np

  activity_labels = [label for label in labels if not is_background_label(label)]
  fused = np.asarray(activity_probs, dtype=np.float32).reshape(-1).copy()
  if coarse_probs is None and subaction_probs is None:
    return fused

  index_source = hierarchy_labels if hierarchy_labels else labels
  coarse_index = build_coarse_index(index_source)
  subaction_index = build_subaction_index(index_source)
  coarse_arr = None if coarse_probs is None else np.asarray(coarse_probs, dtype=np.float32).reshape(-1)
  sub_arr = None if subaction_probs is None else np.asarray(subaction_probs, dtype=np.float32).reshape(-1)

  for idx, label in enumerate(activity_labels[: fused.size]):
    coarse = coarse_label_name(label)
    subaction = subaction_label_name(label)
    score = fused[idx]
    if coarse_arr is not None and coarse in coarse_index and coarse_index[coarse] < coarse_arr.size:
      score *= float(coarse_arr[coarse_index[coarse]])
    if sub_arr is not None and subaction in subaction_index and subaction_index[subaction] < sub_arr.size:
      score *= float(sub_arr[subaction_index[subaction]])
    fused[idx] = score

  total = float(fused.sum())
  if total > 1e-12:
    fused /= total
  return fused


def targets_for_label(
  label: str,
  activity_index: dict[str, int],
  coarse_index: dict[str, int],
  subaction_index: dict[str, int],
) -> TrainingTargets:
  coarse = coarse_label_name(label)
  subaction = subaction_label_name(label)
  if is_background_label(label):
    return TrainingTargets(
      human_label=0,
      activity_label=-1,
      coarse_label=coarse_index[coarse],
      subaction_label=subaction_index[subaction],
      is_background=True,
      flat_label=label,
    )
  if label not in activity_index:
    raise KeyError(f"Activity label {label!r} missing from activity_index")
  return TrainingTargets(
    human_label=1,
    activity_label=activity_index[label],
    coarse_label=coarse_index[coarse],
    subaction_label=subaction_index[subaction],
    is_background=False,
    flat_label=label,
  )


def inference_label(
  labels: list[str],
  human_prob: float,
  activity_probs,
  human_threshold: float = 0.5,
  *,
  motion_ok: bool = True,
  in_range: bool = True,
  min_margin: float = 0.0,
) -> tuple[str, float]:
  import numpy as np

  activity_labels = [label for label in labels if not is_background_label(label)]
  bg = next((label for label in labels if is_background_label(label)), "background")
  if not motion_ok or not in_range or human_prob < human_threshold:
    return bg, float(1.0 - human_prob)

  probs = np.asarray(activity_probs, dtype=np.float32).reshape(-1)
  if probs.size != len(activity_labels):
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
    return bg, top_p
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
