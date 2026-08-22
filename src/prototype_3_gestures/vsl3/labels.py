from __future__ import annotations

import unicodedata
from pathlib import Path


DEFAULT_LABELS_FILE = "dataset/labels.txt"


def normalise_label(text: str) -> str:
    """Return the canonical NFC spelling used by training, reports, and TTS."""
    return unicodedata.normalize("NFC", text.strip())


def read_expected_labels(path: str | Path) -> list[str]:
    """Read the authoritative gesture list; blank lines and full-line comments are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    labels: list[str] = []
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        label = normalise_label(raw)
        if label in labels:
            raise ValueError(f"Nhãn '{label}' xuất hiện hai lần trong {path}")
        labels.append(label)
    if not labels:
        raise ValueError(f"{path} không có nhãn nào")
    return labels
