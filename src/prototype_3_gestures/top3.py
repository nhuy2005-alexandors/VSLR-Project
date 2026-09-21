"""Fail-closed Top-3 prediction extraction, validation, and serialization.

Guarantees:
1. Strict confidence ordering: top1_confidence >= top2_confidence >= top3_confidence.
2. Label distinctness: no duplicate classes within top-3 predictions.
3. Exact margin invariant: top1_top2_margin == top1_confidence - top2_confidence (within tolerance).
4. Zero silent fallback: missing fields or invalid structures fail-closed with ValueError.
"""

from __future__ import annotations

from typing import Sequence
import torch

REQUIRED_TOP3_FIELDS = (
    "video",
    "person",
    "label",
    "predicted",
    "confidence",
    "correct",
    "top2_label",
    "top2_confidence",
    "top3_label",
    "top3_confidence",
    "top1_top2_margin",
)


def extract_topk(probs: torch.Tensor, k: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract top-k probabilities, class indices, and exact top1-top2 margin.

    Args:
        probs: Tensor of shape (..., num_classes) containing probabilities summing to 1.
        k: Number of top classes to extract (default 3, clamped to num_classes).

    Returns:
        (topk_conf, topk_idx, margin)
        where margin is topk_conf[..., 0] - topk_conf[..., 1] if k >= 2 else zeros.
    """
    num_classes = probs.shape[-1]
    k_effective = min(k, num_classes)
    topk_conf, topk_idx = torch.topk(probs, k=k_effective, dim=-1)

    if k_effective >= 2:
        margin = topk_conf[..., 0] - topk_conf[..., 1]
    else:
        margin = torch.zeros_like(topk_conf[..., 0])

    return topk_conf, topk_idx, margin


def validate_top3_row(row: dict, tolerance: float = 1e-4) -> None:
    """Validate that a single prediction record satisfies all fail-closed top-3 integrity rules.

    Raises:
        ValueError: If any field is missing, disordered, contains duplicate labels, or margin mismatches.
    """
    for field in REQUIRED_TOP3_FIELDS:
        if field not in row or row[field] is None:
            raise ValueError(f"Missing required top-3 field {field!r} in prediction row: {row}")

    top1_conf = float(row["confidence"])
    top2_conf = float(row["top2_confidence"])
    top3_conf = float(row["top3_confidence"])
    margin = float(row["top1_top2_margin"])

    # 1. Confidence ordering
    if not (top1_conf >= top2_conf >= top3_conf):
        raise ValueError(
            f"Confidence ordering violated: top1={top1_conf}, top2={top2_conf}, top3={top3_conf} "
            f"in video {row.get('video')}"
        )

    # 2. Distinct labels in top-3
    labels = [row["predicted"], row["top2_label"], row["top3_label"]]
    if len(set(labels)) != 3:
        raise ValueError(f"Duplicate labels in top-3: {labels} in video {row.get('video')}")

    # 3. Exact margin invariant
    expected_margin = top1_conf - top2_conf
    if abs(margin - expected_margin) > tolerance:
        raise ValueError(
            f"Margin mismatch: recorded {margin}, expected {expected_margin} "
            f"(diff={abs(margin - expected_margin):.6f} > {tolerance}) in video {row.get('video')}"
        )


def build_top3_record(
    video: str,
    person: str,
    target_idx: int,
    topk_indices: Sequence[int],
    topk_confidences: Sequence[float],
    margin: float,
    labels: Sequence[str],
) -> dict:
    """Build a validated prediction dictionary with top-1, top-2, top-3, and margin."""
    if len(topk_indices) < 3 or len(topk_confidences) < 3:
        raise ValueError(f"Expected at least 3 candidates, got indices={topk_indices}")

    true_label = labels[target_idx]
    pred_label = labels[topk_indices[0]]
    top2_label = labels[topk_indices[1]]
    top3_label = labels[topk_indices[2]]

    record = {
        "video": str(video),
        "person": str(person),
        "label": true_label,
        "predicted": pred_label,
        "confidence": float(topk_confidences[0]),
        "correct": bool(target_idx == topk_indices[0]),
        "top2_label": top2_label,
        "top2_confidence": float(topk_confidences[1]),
        "top3_label": top3_label,
        "top3_confidence": float(topk_confidences[2]),
        "top1_top2_margin": float(margin),
    }
    validate_top3_row(record)
    return record
