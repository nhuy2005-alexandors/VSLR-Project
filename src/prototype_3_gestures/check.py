"""`vslr-check` — nghiệm thu clip ngay tại chỗ quay, không train gì.

Trả lời đúng ba câu hỏi mà người đứng ở chỗ quay cần: clip nào phải quay lại, còn thiếu
những cặp (người, nhãn) nào, và có thư mục nào đang bị pipeline bỏ qua âm thầm.

Cố ý KHÔNG gọi `validate_clips`: buổi quay đi theo vòng, nên sau vòng 1 mọi nhãn mới có
đúng một clip và mọi ràng buộc của bản train đều sẽ raise. Đây là công cụ báo cáo, không
phải cổng.
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

from .prepare_train import (
    VIDEO_SUFFIXES,
    Clip,
    discover_clips,
    label_order,
    non_negative_int,
)
from .vsl3.features import HolisticExtractor

DEFAULT_LABELS_FILE = "dataset/labels.txt"


def normalise_label(text: str) -> str:
    """NFC. Tên NFD (kiểu macOS) in ra giống hệt NFC nhưng là chuỗi khác."""
    return unicodedata.normalize("NFC", text.strip())


def read_expected_labels(path: str | Path) -> list[str]:
    """Danh sách nhãn mong đợi, một nhãn mỗi dòng; `#` là chú thích.

    Phải là file chứ không suy từ cây: nhãn chưa ai quay thì không thể xuất hiện trong cây,
    nên cây không bao giờ nói được là còn thiếu gì.
    """
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


def clip_status(stats: dict | None, error: Exception | None, min_hand_ratio: float) -> str:
    """Ba nguyên nhân lỗi tách riêng, vì ba hành động khác nhau.

    'KHONG MO DUOC' thường là lỗi truyền file, không phải lỗi quay.
    """
    if error is not None:
        message = str(error)
        if "Cannot open video" in message:
            return "KHONG MO DUOC"
        if "too few readable frames" in message:
            return "QUA NGAN"
        return "KHONG THAY TAY"
    assert stats is not None
    return "QUAY LAI" if stats["hand_frame_ratio"] < min_hand_ratio else "ok"


def coverage_gaps(clips: list[Clip], expected_labels: list[str], expected_per_label: int) -> dict:
    """Cặp (người, nhãn) chưa có clip nào, cặp còn thiếu clip, và nhãn ngoài danh sách."""
    people = sorted({clip.person for clip in clips})
    counts: dict[tuple[str, str], int] = {}
    for clip in clips:
        key = (clip.person, normalise_label(clip.label))
        counts[key] = counts.get(key, 0) + 1

    expected = [normalise_label(label) for label in expected_labels]
    missing = [(person, label) for person in people for label in expected if (person, label) not in counts]
    short = [
        (person, label, counts[(person, label)])
        for person in people
        for label in expected
        if 0 < counts.get((person, label), 0) < expected_per_label
    ]
    unexpected = [label for label in label_order(clips) if normalise_label(label) not in expected]
    return {"missing": sorted(missing), "short": sorted(short), "unexpected": sorted(unexpected)}


def find_skipped_dirs(data_dir: str | Path) -> list[Path]:
    """Thư mục cấp 1 chứa file video trực tiếp — `discover_clips` bỏ qua chúng không một lời."""
    data_dir = Path(data_dir)
    skipped = []
    for child in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if any(f.is_file() and f.suffix.lower() in VIDEO_SUFFIXES for f in child.iterdir()):
            skipped.append(child)
    return skipped


def discover_single_person(data_dir: str | Path, person: str) -> list[Clip]:
    """Cây `DIR/<nhãn>/*.mov` cho một người — dạng cây ở chỗ quay khi chỉ có P1."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"--data-dir không phải thư mục: {data_dir}")
    clips = [
        Clip(label=label_dir.name, person=person, path=path.resolve())
        for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir())
        for path in sorted(label_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    if not clips:
        raise ValueError(f"Không có clip .mov/.mp4 nào dưới {data_dir} theo dạng DIR/<nhãn>/*.mov")
    return clips


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nghiệm thu clip vừa quay: clip nào phải quay lại, còn thiếu cặp nào. Không train."
    )
    parser.add_argument("--data-dir", required=True, help="Cây DIR/<người>/<nhãn>/*.mov")
    parser.add_argument(
        "--person",
        help="Chế độ một người: coi --data-dir là DIR/<nhãn>/*.mov và gán mọi clip cho người này. "
        "Cờ tường minh, không đoán theo cấu trúc — đoán sai là biến tên nhãn thành tên người.",
    )
    parser.add_argument("--labels-file", default=DEFAULT_LABELS_FILE, help="Danh sách nhãn mong đợi")
    parser.add_argument("--clips-per-label", type=non_negative_int, default=4)
    parser.add_argument("--min-hand-ratio", type=float, default=0.5)
    parser.add_argument("--cache-dir", default="dataset/processed/landmark_cache", help="Dùng chung với vslr-train")
    return parser


def main() -> None:
    from .prepare_train import extract_with_cache  # cục bộ: giữ import torch ra khỏi đường QA

    args = build_parser().parse_args()
    if not 0.0 < args.min_hand_ratio <= 1.0:
        build_parser().error(f"--min-hand-ratio phải trong (0, 1], nhận {args.min_hand_ratio}")

    data_dir = Path(args.data_dir)
    try:
        clips = (
            discover_single_person(data_dir, args.person)
            if args.person
            else discover_clips(data_dir)
        )
    except (ValueError, NotADirectoryError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    if not args.person:
        for skipped in find_skipped_dirs(data_dir):
            print(
                f"CẢNH BÁO: {skipped} chứa video trực tiếp nên bị bỏ qua — cây phải là "
                f"<người>/<nhãn>/*.mov. Nếu đó là bộ clip cũ, dùng --person để kiểm riêng."
            )

    expected_labels: list[str] | None = None
    labels_path = Path(args.labels_file)
    if labels_path.exists():
        try:
            expected_labels = read_expected_labels(labels_path)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc
    else:
        print(
            f"CẢNH BÁO: không có {labels_path}, nên KHÔNG phát hiện được nhãn nào chưa ai quay — "
            "nhãn chưa quay thì không có trong cây. Tạo file đó (một nhãn mỗi dòng) rồi chạy lại."
        )

    print(f"\nKiểm {len(clips)} clip (cache: {args.cache_dir})\n")
    rows: list[tuple[str, Clip, dict | None]] = []
    cache_dir = Path(args.cache_dir)
    with HolisticExtractor() as extractor:
        for position, clip in enumerate(clips, start=1):
            try:
                _, stats = extract_with_cache(clip, extractor, cache_dir)
                status = clip_status(stats, None, args.min_hand_ratio)
            except Exception as exc:
                stats, status = None, clip_status(None, exc, args.min_hand_ratio)
            rows.append((status, clip, stats))
            print(f"  [{position}/{len(clips)}] {status:15s} {clip.person}/{clip.label}/{clip.path.name}")

    _print_report(rows, clips, expected_labels, args)


def _print_report(rows, clips: list[Clip], expected_labels: list[str] | None, args) -> None:
    ok = [r for r in rows if r[0] == "ok"]
    bad = [r for r in rows if r[0] != "ok"]

    print(f"\n{'trạng thái':16s} {'người/nhãn':34s} {'file':26s} {'sampled':>8s} {'trimmed':>8s} {'tay':>7s}")
    print("-" * 104)
    # Lỗi trước, rồi tăng dần theo hand_frame_ratio: dòng đầu tiên là clip đáng lo nhất.
    for status, clip, stats in sorted(bad, key=lambda r: (r[2] is not None, r[2]["hand_frame_ratio"] if r[2] else -1.0)):
        s = stats or {}
        ratio = f"{s['hand_frame_ratio']:.1%}" if stats else "-"
        print(
            f"{status:16s} {clip.person + '/' + clip.label:34.34s} {clip.path.name:26.26s} "
            f"{s.get('sampled_frames', '-'):>8} {s.get('trimmed_frames', '-'):>8} {ratio:>7s}"
        )
    for status, clip, stats in sorted(ok, key=lambda r: r[2]["hand_frame_ratio"])[:5]:
        print(
            f"{status:16s} {clip.person + '/' + clip.label:34.34s} {clip.path.name:26.26s} "
            f"{stats['sampled_frames']:>8} {stats['trimmed_frames']:>8} {stats['hand_frame_ratio']:>6.1%}"
        )
    if len(ok) > 5:
        print(f"{'':16s} ... và {len(ok) - 5} clip 'ok' nữa (tay thấy nhiều hơn) không in ra")

    problems = len(bad)
    print(f"\n{len(ok)} clip ok, {problems} clip cần xử lý.")
    if bad:
        print("  KHONG MO DUOC = lỗi file/truyền, không phải lỗi quay. QUA NGAN / KHONG THAY TAY / QUAY LAI = quay lại.")

    gaps = None
    if expected_labels is not None:
        gaps = coverage_gaps(clips, expected_labels, args.clips_per_label)
        people = sorted({c.person for c in clips})
        print(
            f"\nĐộ phủ: {len(people)} người × {len(expected_labels)} nhãn × {args.clips_per_label} clip "
            f"= {len(people) * len(expected_labels) * args.clips_per_label} clip mong đợi"
        )
        if gaps["unexpected"]:
            print(f"  Nhãn KHÔNG có trong {args.labels_file}: {gaps['unexpected']}")
        if gaps["missing"]:
            print(f"  Chưa có clip nào ({len(gaps['missing'])} cặp): {gaps['missing'][:12]}")
            if len(gaps["missing"]) > 12:
                print(f"    ... và {len(gaps['missing']) - 12} cặp nữa")
        if gaps["short"]:
            print(f"  Còn thiếu clip ({len(gaps['short'])} cặp, cần {args.clips_per_label}): {gaps['short'][:12]}")
        if not any(gaps.values()):
            print("  Đủ.")

    incomplete = bool(gaps and any(gaps.values()))
    if problems or incomplete:
        raise SystemExit(1)
    print("\nTất cả clip đạt và đủ độ phủ.")


if __name__ == "__main__":
    main()
