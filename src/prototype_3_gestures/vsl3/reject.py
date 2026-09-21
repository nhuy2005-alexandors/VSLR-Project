"""Closed-set rejection policy and calibration guardrails.

Softmax confidence is not an out-of-vocabulary detector by itself. A missing policy is
``uncalibrated`` and callers must opt in before accepting a segment. Calibration rows carry an
explicit split/source receipt so a held-out LOSO test cannot tune the threshold.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REJECT_POLICY_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CalibrationLeakageError(ValueError):
    """Calibration data contains a held-out/test row and cannot tune a reject threshold."""


@dataclass(frozen=True)
class RejectPolicy:
    calibrated: bool
    threshold: float | None = None
    margin: float | None = None
    source: str = "none"
    schema_version: int = REJECT_POLICY_SCHEMA_VERSION
    dataset_fingerprint: str | None = None
    source_fingerprint: str | None = None
    calibration_split: str | None = None
    feature_contract: str | None = None
    training_signature: str | None = None
    checkpoint_sha256: str | None = None
    rejection_contract: str | None = None
    categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != REJECT_POLICY_SCHEMA_VERSION:
            raise ValueError(f"unsupported reject policy schema_version={self.schema_version}")
        if self.calibrated:
            if self.threshold is None or not np.isfinite(self.threshold) or not 0.0 < self.threshold <= 1.0:
                raise ValueError("a calibrated reject policy needs threshold in (0, 1]")
            if self.margin is not None and (not np.isfinite(self.margin) or not 0.0 <= self.margin <= 1.0):
                raise ValueError("reject policy margin must be in [0, 1]")
        elif self.threshold is not None or self.margin is not None:
            raise ValueError("an uncalibrated reject policy cannot carry thresholds")

    @property
    def has_receipt(self) -> bool:
        return bool(
            self.dataset_fingerprint
            and self.source_fingerprint
            and self.calibration_split
            and self.feature_contract
            and self.training_signature
            and self.checkpoint_sha256
            and self.rejection_contract
            and {"idle_stationary", "oov_motion"}.issubset(self.categories)
        )


def uncalibrated_policy() -> RejectPolicy:
    return RejectPolicy(False, source="none")


def _is_test_split(value: object) -> bool:
    text = str(value).strip().casefold().replace("-", "_")
    return text in {"test", "loso_test", "heldout", "held_out", "validation_test"} or "loso" in text


def _parse_bool(value: object, field: str = "is_negative") -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"{field} must be a strict boolean (true/false or 1/0)")


def _fingerprint_rows(rows: list[dict]) -> str:
    canonical_rows = []
    for row in rows:
        canonical_rows.append(
            {
                "person": str(row["person"]),
                "clip": str(row["clip"]),
                "source_sha256": str(row["source_sha256"]),
                "confidence": float(row["confidence"]),
                "is_negative": _parse_bool(row["is_negative"]),
                "split": str(row["split"]).strip(),
                "source": str(row["source"]).strip(),
            }
        )
    payload = json.dumps(sorted(canonical_rows, key=lambda item: json.dumps(item, sort_keys=True)), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fit_reject_policy(
    rows: Iterable[dict],
    *,
    feature_contract: str | None = None,
    training_signature: str | None = None,
    checkpoint_sha256: str | None = None,
    rejection_contract: str | None = None,
    forbidden_identities: Iterable[tuple[str, str]] | None = None,
    forbidden_source_sha256: Iterable[str] | None = None,
) -> RejectPolicy:
    """Fit a confidence threshold on explicit calibration-only positive/negative rows.

    The feature/training bindings are required because a threshold fitted on one model/data
    contract must not silently be reused for another. ``source`` is a human-readable recording
    source identifier and is fingerprinted into the resulting receipt.
    """

    materialized = list(rows)
    if not materialized:
        raise ValueError("reject calibration needs at least one row")
    # Check leakage before any other row validation so a single LOSO/test row cannot be hidden
    # behind an unrelated missing-field error on an earlier calibration row.
    for row in materialized:
        if _is_test_split(row.get("split", "")) or _is_test_split(row.get("partition", "")):
            raise CalibrationLeakageError(
                "reject calibration cannot use LOSO/test rows; fit on a separate calibration split"
            )
    negative_scores: list[float] = []
    positive_scores: list[float] = []
    splits: set[str] = set()
    sources: set[str] = set()
    categories: set[str] = set()
    forbidden_ids = {(str(person), str(clip)) for person, clip in (forbidden_identities or ())}
    forbidden_hashes = {str(value).strip().casefold() for value in (forbidden_source_sha256 or ())}
    required_fields = {"person", "clip", "source_sha256", "split", "source", "confidence", "is_negative"}
    for row in materialized:
        split = str(row.get("split", "")).strip().casefold()
        if not split:
            raise ValueError("each reject calibration row needs explicit split=calibration")
        if _is_test_split(split):
            raise CalibrationLeakageError(
                "reject calibration cannot use LOSO/test rows; fit on a separate calibration split"
            )
        if split != "calibration":
            raise ValueError(f"reject calibration split must be 'calibration', got {split!r}")
        missing_fields = sorted(required_fields - set(row))
        if missing_fields:
            raise ValueError(f"each reject calibration row needs fields {missing_fields}")
        person = str(row["person"]).strip()
        clip = str(row["clip"]).strip()
        if not person or not clip or Path(clip).name != clip or any(part in {".", ".."} for part in Path(clip).parts):
            raise ValueError("reject calibration person/clip identity is invalid")
        source_sha256 = str(row["source_sha256"]).strip().casefold()
        if len(source_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_sha256):
            raise ValueError("reject calibration source_sha256 must be a 64-character hex digest")
        if (person, clip) in forbidden_ids or source_sha256 in forbidden_hashes:
            raise CalibrationLeakageError(
                f"reject calibration identity overlaps forbidden/test data: {person}/{clip}"
            )
        source = str(row.get("source", "")).strip()
        if not source:
            raise ValueError("each reject calibration row needs explicit source")
        category = str(row.get("category", "")).strip().casefold()
        if not category:
            raise ValueError("each reject calibration row needs explicit category")
        if "is_negative" not in row:
            raise ValueError("each reject calibration row needs is_negative")
        is_negative = _parse_bool(row["is_negative"])
        try:
            score = float(row["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each reject calibration row needs finite confidence") from exc
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid calibration confidence {score}")
        splits.add(split)
        sources.add(f"{source}|{source_sha256}")
        categories.add(category)
        (negative_scores if is_negative else positive_scores).append(score)
    if not feature_contract or not training_signature or not checkpoint_sha256 or not rejection_contract:
        raise ValueError(
            "reject calibration needs explicit feature_contract, rejection_contract, "
            "training_signature and checkpoint_sha256"
        )
    checkpoint_sha256 = checkpoint_sha256.strip().casefold()
    if len(checkpoint_sha256) != 64 or any(character not in "0123456789abcdef" for character in checkpoint_sha256):
        raise ValueError("reject calibration checkpoint_sha256 must be a 64-character hex digest")
    if not negative_scores:
        raise ValueError("reject calibration needs negative/idle/OOV rows")
    missing_categories = {"idle_stationary", "oov_motion"} - categories
    if missing_categories:
        raise ValueError(f"reject calibration missing required negative categories: {sorted(missing_categories)}")
    if not positive_scores:
        raise ValueError("reject calibration needs positive in-dictionary rows")

    negative_max = max(negative_scores)
    positive_min = min(positive_scores)
    if negative_max < positive_min:
        threshold = (negative_max + positive_min) / 2.0
    else:
        threshold = min(1.0, negative_max + 1e-6)
    threshold = max(float(threshold), np.finfo(np.float64).eps)
    source_fingerprint = hashlib.sha256("\n".join(sorted(sources)).encode("utf-8")).hexdigest()
    return RejectPolicy(
        True,
        threshold=threshold,
        source="calibration",
        dataset_fingerprint=_fingerprint_rows(materialized),
        source_fingerprint=source_fingerprint,
        calibration_split=next(iter(splits)),
        feature_contract=feature_contract,
        training_signature=training_signature,
        checkpoint_sha256=checkpoint_sha256,
        rejection_contract=rejection_contract,
        categories=tuple(sorted(categories)),
    )


def load_reject_policy(
    source: str | Path | dict | None,
    *,
    expected_feature_contract: str | None = None,
    expected_training_signature: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_rejection_contract: str | None = None,
) -> RejectPolicy:
    if source is None:
        return uncalibrated_policy()
    raw = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else source
    if not isinstance(raw, dict):
        raise ValueError("reject policy must be a JSON object")
    calibrated = _parse_bool(raw.get("calibrated", False), "calibrated")
    policy = RejectPolicy(
        calibrated=calibrated,
        threshold=float(raw["threshold"]) if raw.get("threshold") is not None else None,
        margin=float(raw["margin"]) if raw.get("margin") is not None else None,
        source=str(raw.get("source", "file")),
        schema_version=int(raw.get("schema_version", REJECT_POLICY_SCHEMA_VERSION)),
        dataset_fingerprint=str(raw["dataset_fingerprint"]) if raw.get("dataset_fingerprint") else None,
        source_fingerprint=str(raw["source_fingerprint"]) if raw.get("source_fingerprint") else None,
        calibration_split=str(raw["calibration_split"]) if raw.get("calibration_split") else None,
        feature_contract=str(raw["feature_contract"]) if raw.get("feature_contract") else None,
        training_signature=str(raw["training_signature"]) if raw.get("training_signature") else None,
        checkpoint_sha256=str(raw["checkpoint_sha256"]) if raw.get("checkpoint_sha256") else None,
        rejection_contract=str(raw["rejection_contract"]) if raw.get("rejection_contract") else None,
        categories=tuple(str(item) for item in raw.get("categories", ())),
    )
    if policy.calibrated and not policy.has_receipt:
        raise ValueError("calibrated reject policy is missing its dataset/source/contract receipt")
    if expected_feature_contract and policy.calibrated and policy.feature_contract != expected_feature_contract:
        raise ValueError("reject policy feature_contract does not match checkpoint")
    if expected_training_signature and policy.calibrated and policy.training_signature != expected_training_signature:
        raise ValueError("reject policy training_signature does not match checkpoint")
    if expected_checkpoint_sha256 and policy.calibrated and policy.checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("reject policy checkpoint_sha256 does not match checkpoint")
    if expected_rejection_contract and policy.calibrated and policy.rejection_contract != expected_rejection_contract:
        raise ValueError("reject policy rejection_contract does not match checkpoint")
    return policy


def reject_policy_dict(policy: RejectPolicy) -> dict:
    return {
        "schema_version": policy.schema_version,
        "calibrated": policy.calibrated,
        "threshold": policy.threshold,
        "margin": policy.margin,
        "source": policy.source,
        "dataset_fingerprint": policy.dataset_fingerprint,
        "source_fingerprint": policy.source_fingerprint,
        "calibration_split": policy.calibration_split,
        "feature_contract": policy.feature_contract,
        "training_signature": policy.training_signature,
        "checkpoint_sha256": policy.checkpoint_sha256,
        "rejection_contract": policy.rejection_contract,
        "categories": list(policy.categories),
    }


def save_reject_policy(path: str | Path, policy: RejectPolicy) -> None:
    """Persist a calibration receipt atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staging.write_text(json.dumps(reject_policy_dict(policy), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def reject_prediction(
    confidence: float,
    probabilities: np.ndarray | None,
    policy: RejectPolicy,
    *,
    manual_threshold: float,
    allow_uncalibrated: bool = False,
) -> tuple[bool, str]:
    """Return an explicit accept/reject decision and a user-facing reason."""

    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return False, "non-finite or out-of-range confidence"
    if not np.isfinite(manual_threshold) or not 0.0 < manual_threshold <= 1.0:
        return False, "invalid manual confidence threshold"
    if probabilities is None:
        return False, "missing probability vector"
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.isclose(float(values.sum()), 1.0, atol=1e-5, rtol=1e-5)
    ):
        return False, "invalid probability vector"
    if not np.isclose(float(values.max()), float(confidence), atol=1e-5, rtol=1e-5):
        return False, "confidence does not match probability vector"

    if not policy.calibrated and not allow_uncalibrated:
        return False, "reject policy uncalibrated (negative calibration required)"
    if policy.calibrated and not policy.has_receipt:
        return False, "calibrated reject policy missing receipt binding"
    threshold = policy.threshold if policy.calibrated else manual_threshold
    if confidence < float(threshold):
        return False, f"confidence {confidence:.1%} below threshold {float(threshold):.1%}"
    if policy.margin is not None:
        sorted_values = np.sort(values)
        if sorted_values.size >= 2 and float(sorted_values[-1] - sorted_values[-2]) < policy.margin:
            return False, f"top-class margin below calibrated margin {policy.margin:.1%}"
    return True, "accepted"
