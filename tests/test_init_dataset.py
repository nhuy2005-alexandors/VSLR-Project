import io
import unittest
import tomllib
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from prototype_3_gestures.check import build_parser as build_check_parser, main as check_main
from prototype_3_gestures.init_dataset import (
    build_people,
    initialise_dataset,
    validate_people,
)
from prototype_3_gestures.vsl3.labels import read_expected_labels


EXPECTED_LABELS = [
    "Bạn cảm thấy thế nào",
    "Bạn có cần giúp đỡ không",
    "Bạn có vấn đề gì không",
    "Bạn đang làm gì",
    "Bạn có mệt không",
    "Bạn quê ở đâu",
    "Bạn tên gì",
    "Cảm ơn",
    "Chuyện gì",
    "Công viên",
    "Cứu tôi với",
    "Đi đâu",
    "Được không",
    "Gọi xe cứu thương",
    "Hẹn gặp lại",
    "Hôm nay bạn khỏe không",
    "Lâu rồi không gặp",
    "Mấy tuổi",
    "Nhà bạn có mấy người",
    "Nhà bạn ở đâu",
    "Như thế nào",
    "Rất vui được gặp bạn",
    "Sao thế",
    "Siêu thị",
    "Tạm biệt",
    "Tôi bình thường",
    "Tôi không khỏe",
    "Về nhà cẩn thận",
    "Xin chào",
    "Xin lỗi",
]


class InitDatasetTests(unittest.TestCase):
    def test_project_manifest_is_the_final_30_label_v1(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(read_expected_labels(root / "dataset" / "labels.txt"), EXPECTED_LABELS)

    def test_cli_is_registered_and_both_tools_default_to_six_clips(self):
        root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(
            pyproject["project"]["scripts"]["vslr-init-dataset"],
            "prototype_3_gestures.init_dataset:main",
        )
        args = build_check_parser().parse_args(["--data-dir", "recordings"])
        self.assertEqual(args.clips_per_label, 6)

    def test_people_count_generates_stable_anonymous_ids(self):
        self.assertEqual(build_people(4), ["P01", "P02", "P03", "P04"])

    def test_creates_every_person_label_directory_and_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            existing = root / "P01" / "Cảm ơn" / "001.mov"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"existing video")

            first = initialise_dataset(
                root,
                people=["P01", "P02"],
                labels=["Cảm ơn", "Xin chào"],
                clips_per_label=6,
            )
            second = initialise_dataset(
                root,
                people=["P01", "P02"],
                labels=["Cảm ơn", "Xin chào"],
                clips_per_label=6,
            )

            for person in ("P01", "P02"):
                for label in ("Cảm ơn", "Xin chào"):
                    self.assertTrue((root / person / label).is_dir())
            self.assertEqual(existing.read_bytes(), b"existing video")
            self.assertEqual(first.expected_clips, 24)
            self.assertEqual(first.created_directories, 3)
            self.assertEqual(second.created_directories, 0)
            self.assertEqual(second.existing_directories, 4)

    def test_dry_run_plans_without_writing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            summary = initialise_dataset(
                root,
                people=["P01"],
                labels=["Cảm ơn", "Xin chào"],
                clips_per_label=6,
                dry_run=True,
            )

            self.assertFalse(root.exists())
            self.assertEqual(summary.created_directories, 2)
            self.assertEqual(summary.expected_clips, 12)

    def test_invalid_or_duplicate_person_ids_are_refused(self):
        for people in (
            ["P01", "P01"],
            ["P01", "p01"],
            ["CON"],
            ["P01", "../P02"],
            ["P 01"],
            [" P01"],
        ):
            with self.subTest(people=people):
                with self.assertRaises(ValueError):
                    validate_people(people)

    def test_blank_label_is_refused(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Nhãn"):
                initialise_dataset(
                    Path(tmp) / "raw",
                    people=["P01"],
                    labels=[""],
                )

    def test_case_insensitive_label_collision_is_refused_before_writing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            with self.assertRaisesRegex(ValueError, "hoa-thường"):
                initialise_dataset(
                    root,
                    people=["P01"],
                    labels=["Cảm ơn", "cảm ơn"],
                )
            self.assertFalse(root.exists())

    def test_check_warns_about_flat_legacy_tree_before_discovery_aborts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = root / "cam_on"
            flat.mkdir()
            (flat / "old.mov").touch()
            labels = root / "labels.txt"
            labels.write_text("Cảm ơn\n", encoding="utf-8")
            output = io.StringIO()
            argv = [
                "vslr-check",
                "--data-dir",
                str(root),
                "--labels-file",
                str(labels),
            ]

            with mock.patch("sys.argv", argv), redirect_stdout(output), self.assertRaises(SystemExit):
                check_main()

            self.assertIn("CẢNH BÁO", output.getvalue())
            self.assertIn("cam_on", output.getvalue())

    def test_preflight_conflict_does_not_leave_a_partial_tree(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            conflict = root / "P02" / "Xin chào"
            conflict.parent.mkdir(parents=True)
            conflict.write_text("not a directory", encoding="utf-8")

            with self.assertRaises(NotADirectoryError):
                initialise_dataset(
                    root,
                    people=["P01", "P02"],
                    labels=["Cảm ơn", "Xin chào"],
                    clips_per_label=6,
                )

            self.assertFalse((root / "P01").exists())
            self.assertFalse((root / "P02" / "Cảm ơn").exists())

    def test_legacy_video_tree_is_refused_instead_of_mixing_layouts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            legacy = root / "cam_on"
            legacy.mkdir(parents=True)
            (legacy / "old.mov").touch()

            with self.assertRaisesRegex(ValueError, "cam_on"):
                initialise_dataset(
                    root,
                    people=["P01"],
                    labels=["Cảm ơn"],
                    clips_per_label=6,
                )

            self.assertFalse((root / "P01").exists())


if __name__ == "__main__":
    unittest.main()
