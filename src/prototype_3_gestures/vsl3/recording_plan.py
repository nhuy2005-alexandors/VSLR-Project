"""Versioned recording contract shared by dataset init, QA and training.

The label manifest remains the single source of class names.  This file deliberately contains no
copy of that list: it only fixes which anonymous signers and how many recordings per pair belong
to the V1 recording session.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RECORDING_PLAN_FILE = "dataset/recording_plan.json"
RECORDING_PLAN_SCHEMA_VERSION = 1
V1_PEOPLE = ("P01", "P02", "P03", "P04")
RECORDING_PLAN_FIELDS = frozenset(
    {"schema_version", "dataset_version", "labels_file", "people", "clips_per_label"}
)


@dataclass(frozen=True)
class RecordingPlan:
    schema_version: int
    dataset_version: str
    labels_file: str
    people: tuple[str, ...]
    clips_per_label: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise ValueError("recording plan schema_version must be an integer JSON primitive")
        if self.schema_version != RECORDING_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported recording plan schema_version={self.schema_version}; "
                f"expected {RECORDING_PLAN_SCHEMA_VERSION}"
            )
        if type(self.dataset_version) is not str or not self.dataset_version.strip():
            raise ValueError("recording plan dataset_version must not be empty")
        if type(self.labels_file) is not str or not self.labels_file or Path(self.labels_file).is_absolute():
            raise ValueError("recording plan labels_file must be a non-empty relative path")
        if self.dataset_version == "recordings_v1":
            if self.people != V1_PEOPLE:
                raise ValueError(
                    f"V1 recording plan must contain exactly people {list(V1_PEOPLE)}, got {list(self.people)}"
                )
            if type(self.clips_per_label) is not int:
                raise ValueError("recording plan clips_per_label must be an integer JSON primitive")
            if self.clips_per_label != 6:
                raise ValueError(f"V1 recording plan clips_per_label must be exactly 6, got {self.clips_per_label}")
        else:
            if len(self.people) < 2:
                raise ValueError(
                    f"recording plan must contain at least 2 people for signer split, got {list(self.people)}"
                )
            if type(self.clips_per_label) is not int:
                raise ValueError("recording plan clips_per_label must be an integer JSON primitive")
            if self.clips_per_label < 1:
                raise ValueError(f"recording plan clips_per_label must be positive, got {self.clips_per_label}")

    @property
    def expected_clips(self) -> int:
        return len(self.people) * self.clips_per_label


def load_recording_plan(path: str | Path = DEFAULT_RECORDING_PLAN_FILE) -> RecordingPlan:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"recording plan not found: {path}. Create it before using directory mode."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"recording plan is not valid JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"recording plan must be a JSON object: {path}")
    keys = set(raw)
    missing = sorted(RECORDING_PLAN_FIELDS - keys)
    extra = sorted(keys - RECORDING_PLAN_FIELDS)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if extra:
            details.append(f"unknown fields {extra}")
        raise ValueError("invalid recording plan: " + "; ".join(details))
    if type(raw["schema_version"]) is not int:
        raise ValueError("recording plan schema_version must be an integer JSON primitive")
    if type(raw["dataset_version"]) is not str:
        raise ValueError("recording plan dataset_version must be a string JSON primitive")
    if type(raw["labels_file"]) is not str:
        raise ValueError("recording plan labels_file must be a string JSON primitive")
    people = raw["people"]
    if type(people) is not list or not all(type(item) is str for item in people):
        raise ValueError("recording plan people must be a JSON string array")
    if type(raw["clips_per_label"]) is not int:
        raise ValueError("recording plan clips_per_label must be an integer JSON primitive")
    try:
        return RecordingPlan(
            schema_version=raw["schema_version"],
            dataset_version=raw["dataset_version"],
            labels_file=raw["labels_file"],
            people=tuple(people),
            clips_per_label=raw["clips_per_label"],
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid recording plan values in {path}: {exc}") from exc


def save_recording_plan(path: str | Path, plan: RecordingPlan) -> None:
    """Write a plan atomically without ever creating a duplicate label manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": plan.schema_version,
        "dataset_version": plan.dataset_version,
        "labels_file": plan.labels_file,
        "people": list(plan.people),
        "clips_per_label": plan.clips_per_label,
    }
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def resolve_labels_file(plan_path: str | Path, plan: RecordingPlan) -> Path:
    """Resolve the plan's labels path relative to the plan, preventing cwd-dependent semantics."""
    return (Path(plan_path).parent / plan.labels_file).resolve()
