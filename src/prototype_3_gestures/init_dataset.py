"""Create the signer/gesture directory tree used by check and training.

This command deliberately creates directories only. Empty video placeholders would be discovered
as broken clips, while moving or renaming recordings automatically could destroy source evidence.
The existing pipeline assigns person and label deterministically from the resulting path.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from .vsl3.console import configure_utf8_stdio
from .vsl3.labels import DEFAULT_LABELS_FILE, normalise_label, normalised_label_directories, read_expected_labels

configure_utf8_stdio()

PERSON_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
INVALID_WINDOWS_CHARS = set('<>:"/\\|?*')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


@dataclass(frozen=True)
class InitSummary:
    data_dir: Path
    people: tuple[str, ...]
    labels: tuple[str, ...]
    clips_per_label: int
    created_directories: int
    existing_directories: int
    existing_clips: int
    dry_run: bool

    @property
    def expected_clips(self) -> int:
        return len(self.people) * len(self.labels) * self.clips_per_label


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"phải >= 1, nhận {number}")
    return number


def build_people(count: int) -> list[str]:
    if count < 1:
        raise ValueError(f"Số người phải >= 1, nhận {count}")
    return [f"P{index:02d}" for index in range(1, count + 1)]


def validate_people(values: list[str]) -> list[str]:
    if not values:
        raise ValueError("Cần ít nhất một mã người ký")
    people = list(values)
    duplicates = sorted({person for person in people if people.count(person) > 1})
    if duplicates:
        raise ValueError(f"Mã người ký bị lặp: {duplicates}")
    folded_people: dict[str, str] = {}
    for person in people:
        previous = folded_people.get(person.casefold())
        if previous is not None and previous != person:
            raise ValueError(
                f"Mã người ký đụng nhau khi Windows bỏ qua hoa-thường: {previous!r}, {person!r}"
            )
        folded_people[person.casefold()] = person
    invalid = [person for person in people if not PERSON_ID_RE.fullmatch(person)]
    if invalid:
        raise ValueError(
            f"Mã người ký không hợp lệ: {invalid}. Dùng mã ẩn danh như P01, chỉ gồm chữ, số, '_' hoặc '-'."
        )
    reserved = [person for person in people if person.upper() in WINDOWS_RESERVED_NAMES]
    if reserved:
        raise ValueError(f"Mã người ký là tên thiết bị dành riêng của Windows: {reserved}")
    return people


def _validate_labels(values: list[str]) -> list[str]:
    if not values:
        raise ValueError("Cần ít nhất một nhãn cử chỉ")
    labels = [normalise_label(value) for value in values]
    if any(not label.strip() for label in labels):
        raise ValueError("Nhãn cử chỉ không được để trống")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Nhãn bị lặp: {duplicates}")
    folded_labels: dict[str, str] = {}
    for label in labels:
        previous = folded_labels.get(label.casefold())
        if previous is not None and previous != label:
            raise ValueError(f"Nhãn đụng nhau khi Windows bỏ qua hoa-thường: {previous!r}, {label!r}")
        folded_labels[label.casefold()] = label
    for label in labels:
        stem = label.split(".", 1)[0].upper()
        if (
            label != label.strip()
            or label in {".", ".."}
            or label.endswith(".")
            or any(character in INVALID_WINDOWS_CHARS or ord(character) < 32 for character in label)
            or stem in WINDOWS_RESERVED_NAMES
        ):
            raise ValueError(f"Nhãn không thể dùng làm tên thư mục an toàn: {label!r}")
    return labels


def _preflight(data_dir: Path, people: list[str], labels: list[str]) -> list[Path]:
    if data_dir.exists() and not data_dir.is_dir():
        raise NotADirectoryError(f"--data-dir không phải thư mục: {data_dir}")

    expected_people = set(people)
    expected_labels = set(labels)
    if data_dir.is_dir():
        unexpected_people = sorted(
            child.name for child in data_dir.iterdir() if child.is_dir() and child.name not in expected_people
        )
        if unexpected_people:
            raise ValueError(
                f"{data_dir} có thư mục cấp người ngoài kế hoạch: {unexpected_people}. "
                "Không trộn cây legacy/slug hoặc người chưa khai báo vào cùng bộ quay."
            )

        for person in people:
            person_dir = data_dir / person
            if person_dir.exists() and not person_dir.is_dir():
                raise NotADirectoryError(f"Đường dẫn người ký không phải thư mục: {person_dir}")
            if person_dir.is_dir():
                actual = normalised_label_directories(person_dir)
                unexpected_labels = sorted(path.name for path, label in actual if label not in expected_labels)
                if unexpected_labels:
                    raise ValueError(
                        f"{person_dir} có thư mục nhãn ngoài manifest: {unexpected_labels}. "
                        "Dọn cây trước khi khởi tạo để không gán nhãn sai."
                    )

    planned = [data_dir / person / label for person in people for label in labels]
    conflicts = [path for path in planned if path.exists() and not path.is_dir()]
    if conflicts:
        raise NotADirectoryError(f"Đường dẫn dự kiến đang là file, không phải thư mục: {conflicts[0]}")
    return planned


def initialise_dataset(
    data_dir: str | Path,
    *,
    people: list[str],
    labels: list[str],
    clips_per_label: int = 6,
    dry_run: bool = False,
) -> InitSummary:
    if clips_per_label < 1:
        raise ValueError(f"Số clip mỗi nhãn phải >= 1, nhận {clips_per_label}")
    data_dir = Path(data_dir)
    people = validate_people(people)
    labels = _validate_labels(labels)
    planned = _preflight(data_dir, people, labels)

    existing = sum(path.is_dir() for path in planned)
    created = len(planned) - existing
    if not dry_run:
        for path in planned:
            path.mkdir(parents=True, exist_ok=True)

    video_suffixes = {".mov", ".mp4"}
    existing_clips = sum(
        1
        for path in planned
        if path.is_dir()
        for clip in path.iterdir()
        if clip.is_file() and clip.suffix.lower() in video_suffixes
    )
    return InitSummary(
        data_dir=data_dir,
        people=tuple(people),
        labels=tuple(labels),
        clips_per_label=clips_per_label,
        created_directories=created,
        existing_directories=existing,
        existing_clips=existing_clips,
        dry_run=dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tạo cây quay DIR/<người>/<nhãn>/; không tạo, di chuyển hay ghi đè video."
    )
    parser.add_argument("--data-dir", required=True, help="Thư mục gốc của bộ quay mới")
    parser.add_argument("--labels-file", default=DEFAULT_LABELS_FILE, help="Một nhãn tiếng Việt mỗi dòng")
    people = parser.add_mutually_exclusive_group(required=True)
    people.add_argument("--people", nargs="+", help="Danh sách mã ẩn danh, ví dụ P01 P02 P03 P04")
    people.add_argument("--people-count", type=positive_int, help="Tự sinh P01..Pxx")
    parser.add_argument("--clips-per-label", type=positive_int, default=6)
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm và in kế hoạch, không tạo thư mục")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        labels = read_expected_labels(args.labels_file)
        people = args.people if args.people is not None else build_people(args.people_count)
        summary = initialise_dataset(
            args.data_dir,
            people=people,
            labels=labels,
            clips_per_label=args.clips_per_label,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    action = "SẼ TẠO" if summary.dry_run else "ĐÃ TẠO"
    print(
        f"{action} {summary.created_directories} thư mục nhãn; "
        f"giữ nguyên {summary.existing_directories} thư mục có sẵn."
    )
    print(
        f"Kế hoạch: {len(summary.people)} người × {len(summary.labels)} nhãn × "
        f"{summary.clips_per_label} clip = {summary.expected_clips} clip."
    )
    print(f"Hiện có {summary.existing_clips} clip .mov/.mp4 trong các thư mục dự kiến.")
    example = summary.data_dir / summary.people[0] / summary.labels[0] / "001.mov"
    print(f"Ví dụ: {example} -> person={summary.people[0]}, label={summary.labels[0]}")
    last_clip = f"{summary.clips_per_label:03d}.mov"
    print(
        f"Không tạo video rỗng. Lưu lần quay thành 001.mov ... {last_clip} "
        "(hoặc .mp4) trong đúng thư mục nhãn."
    )
    print(
        f"Kiểm sau khi quay: vslr-check --data-dir \"{summary.data_dir}\" "
        f"--labels-file \"{args.labels_file}\" --clips-per-label {summary.clips_per_label}"
    )


if __name__ == "__main__":
    main()
