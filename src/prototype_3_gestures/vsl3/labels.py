from __future__ import annotations

import unicodedata
from pathlib import Path


DEFAULT_LABELS_FILE = "dataset/labels.txt"


def normalise_label(text: str) -> str:
    """Return NFC text without forgiving spelling-significant whitespace."""
    return unicodedata.normalize("NFC", text)


def normalised_label_directories(parent: str | Path) -> list[tuple[Path, str]]:
    """Return label directories and refuse raw-name collisions after NFC.

    NFC and NFD are two encodings of the same visible Vietnamese label, so normalising one
    directory is useful. Two distinct raw directories for the same signer are instead ambiguous:
    silently merging them would inflate clip counts and hide a broken recording tree.
    """
    parent = Path(parent)
    directories: list[tuple[Path, str]] = []
    seen: dict[str, Path] = {}
    for directory in sorted(path for path in parent.iterdir() if path.is_dir()):
        label = normalise_label(directory.name)
        previous = seen.get(label)
        if previous is not None and previous.name != directory.name:
            raise ValueError(
                f"{parent} contains two gesture directories that become the same label after NFC: "
                f"{previous.name!r} and {directory.name!r} -> {label!r}. Merge them before checking or training."
            )
        seen[label] = directory
        directories.append((directory, label))
    return directories


def read_expected_labels(path: str | Path) -> list[str]:
    """Read the authoritative gesture list; blank lines and full-line comments are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    labels: list[str] = []
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        # Whitespace around a manifest line is formatting. Directory names are deliberately not
        # stripped, so an accidental leading/trailing space cannot pass the exact manifest gate.
        label = normalise_label(raw.strip())
        if label in labels:
            raise ValueError(f"Nhãn '{label}' xuất hiện hai lần trong {path}")
        labels.append(label)
    if not labels:
        raise ValueError(f"{path} không có nhãn nào")
    return labels
