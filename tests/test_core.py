import json
import os
import unicodedata
import unittest
import warnings
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import cv2
import mediapipe as mp
import numpy as np
import torch

import prototype_3_gestures.prepare_train as prepare_train_module
from prototype_3_gestures.prepare_train import (
    SINGLE_SIGNER,
    Clip,
    GestureDataset,
    build_eval_loader,
    build_parser,
    cache_key,
    data_fingerprint,
    discover_clips,
    evaluate,
    extract_all,
    extract_with_cache,
    file_sha256,
    main as train_main,
    missing_labels_after_drop,
    parse_video_specs,
    signer_split,
    suspect_clips,
    training_signature,
    train_model,
    validate_clips,
    validate_expected_labels,
    worst_labels,
)
from prototype_3_gestures.check import (
    clip_status,
    coverage_gaps,
    find_skipped_dirs,
    main as check_main,
    read_expected_labels,
)
from prototype_3_gestures.vsl3.features import (
    FEATURE_DIM,
    FEATURES_VERSION,
    SEQUENCE_LENGTH,
    ClipExtractionError,
    HolisticExtractor,
    augment_sequence,
    resample_sequence,
    time_warp_sequence,
)
from prototype_3_gestures.vsl3.model import GestureLSTM, load_checkpoint, save_checkpoint
from prototype_3_gestures.realtime import Segment, SegmentTracker, main as realtime_main, should_accept_prediction


def _clip(label: str, person: str, name: str = "a.mov") -> Clip:
    return Clip(label=label, person=person, path=Path(f"/tmp/{person}/{label}/{name}"))


class FeatureTests(unittest.TestCase):
    def test_resample_shape_and_endpoints(self):
        sequence = np.zeros((12, FEATURE_DIM), dtype=np.float32)
        sequence[:, 0] = np.linspace(-1.0, 1.0, 12)
        output = resample_sequence(sequence, 60)
        self.assertEqual(output.shape, (60, FEATURE_DIM))
        self.assertAlmostEqual(float(output[0, 0]), -1.0, places=5)
        self.assertAlmostEqual(float(output[-1, 0]), 1.0, places=5)

    def test_augmentation_preserves_shape_and_missing_points(self):
        sequence = np.zeros((60, FEATURE_DIM), dtype=np.float32)
        sequence[:, :9] = 0.5
        output = augment_sequence(sequence, np.random.default_rng(7))
        self.assertEqual(output.shape, sequence.shape)
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(np.all(output[:, 9:] == 0.0))


    def test_time_warp_preserves_endpoints_and_bends_the_middle(self):
        """Absolute duration is already normalised away by extract_video's resample, so what is
        worth augmenting is INTERNAL timing — where in the gesture the signer lingers."""
        ramp = np.zeros((60, FEATURE_DIM), dtype=np.float32)
        ramp[:, :3] = np.linspace(0.0, 1.0, 60, dtype=np.float32)[:, None]

        midpoints = set()
        for seed in range(40):
            warped = time_warp_sequence(ramp, np.random.default_rng(seed))
            self.assertEqual(warped.shape, ramp.shape)
            self.assertAlmostEqual(float(warped[0, 0]), 0.0, places=4)
            self.assertAlmostEqual(float(warped[-1, 0]), 1.0, places=4)
            self.assertTrue(np.all(np.diff(warped[:, 0]) >= -1e-5), "a time warp must not run backwards")
            midpoints.add(round(float(warped[30, 0]), 3))

        self.assertGreater(len(midpoints), 5, "the middle of a time ramp must actually move")
        self.assertGreater(max(midpoints) - min(midpoints), 0.05, "warp too weak to matter")

    def test_time_warp_leaves_missing_landmarks_at_zero(self):
        sequence = np.zeros((60, FEATURE_DIM), dtype=np.float32)
        sequence[:, :9] = 0.5
        warped = time_warp_sequence(sequence, np.random.default_rng(3))
        self.assertTrue(np.all(warped[:, 9:] == 0.0))


class _FakeLandmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeLandmarkList:
    def __init__(self, points):
        self.landmark = [_FakeLandmark(*p) for p in points]


class _FakeResults:
    """Landmarks that drift with how many frames this instance has already seen.

    Stands in for MediaPipe's real temporal behaviour: `smooth_landmarks=True` with
    `static_image_mode=False` carries tracking state from frame to frame.
    """

    def __init__(self, drift):
        pose = [(0.5, 0.5, 0.0)] * 25
        pose[11] = (0.4 + drift, 0.5, 0.0)
        pose[12] = (0.6 + drift, 0.5, 0.0)
        self.pose_landmarks = _FakeLandmarkList(pose)
        self.left_hand_landmarks = _FakeLandmarkList([(0.45 + drift, 0.4, 0.0)] * 21)
        self.right_hand_landmarks = _FakeLandmarkList([(0.55 + drift, 0.4, 0.0)] * 21)


class _FakeHolistic:
    constructions = 0

    def __init__(self, **kwargs):
        type(self).constructions += 1
        self.frames_seen = 0

    def process(self, image):
        self.frames_seen += 1
        return _FakeResults(self.frames_seen * 0.002)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _write_dummy_video(path: Path, frames: int = 20) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64))
    for index in range(frames):
        writer.write(np.full((64, 64, 3), (index * 10) % 255, dtype=np.uint8))
    writer.release()
    return path


class ExtractorIsolationTests(unittest.TestCase):
    def test_extract_video_is_not_contaminated_by_a_previous_video(self):
        """One clip's landmarks must not depend on which clip was extracted before it.

        Sharing one MediaPipe instance across clips made every clip after the first a function of
        the tree-walk order, which neither the cache key nor the data fingerprint records.
        """
        with TemporaryDirectory() as tmp:
            video = _write_dummy_video(Path(tmp) / "a.mp4")
            _FakeHolistic.constructions = 0
            with mock.patch.object(mp.solutions.holistic, "Holistic", _FakeHolistic):
                with HolisticExtractor() as extractor:
                    first, _ = extractor.extract_video(video, SEQUENCE_LENGTH)
                    second, _ = extractor.extract_video(video, SEQUENCE_LENGTH)

        np.testing.assert_array_equal(first, second)

    def test_live_camera_path_still_carries_state_between_frames(self):
        """process_frame is the camera stream: smoothing across frames there is the point."""
        _FakeHolistic.constructions = 0
        with mock.patch.object(mp.solutions.holistic, "Holistic", _FakeHolistic):
            with HolisticExtractor() as extractor:
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                one = extractor.process_frame(frame).features
                two = extractor.process_frame(frame).features

        self.assertFalse(np.array_equal(one, two), "the live path must keep temporal state")


class ModelTests(unittest.TestCase):
    def test_forward_shape(self):
        model = GestureLSTM(FEATURE_DIM, 3)
        logits = model(torch.zeros((2, 60, FEATURE_DIM), dtype=torch.float32))
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_pool_takes_forward_last_and_backward_first(self):
        """The backward half at t=-1 has only seen one frame; it must come from t=0.

        The ramp deliberately starts at 1, not 0: with a zero at t=0 an implementation that just
        returned zeros for the backward half would pass.
        """
        model = GestureLSTM(FEATURE_DIM, 3, hidden_size=4)
        out = torch.zeros(1, 5, 8)
        ramp = torch.arange(1, 6, dtype=torch.float32).unsqueeze(1)
        out[0, :, :4] = ramp
        out[0, :, 4:] = ramp * 10.0

        pooled = model.pool_sequence(out)

        self.assertEqual(tuple(pooled.shape), (1, 8))
        self.assertTrue(torch.allclose(pooled[0, :4], torch.full((4,), 5.0)))
        self.assertTrue(torch.allclose(pooled[0, 4:], torch.full((4,), 10.0)))

    def test_pool_matches_lstm_final_hidden_states(self):
        """Against a real nn.LSTM: pooling must equal cat(h_n[forward], h_n[backward])."""
        torch.manual_seed(0)
        model = GestureLSTM(9, 3, hidden_size=5)
        lstm = model.lstm
        x = torch.randn(2, 7, 9)
        out, (h_n, _) = lstm(x)

        pooled = model.pool_sequence(out)

        self.assertTrue(torch.allclose(pooled, torch.cat([h_n[0], h_n[1]], dim=1), atol=1e-6))
        self.assertFalse(torch.allclose(out[:, -1, 5:], h_n[1], atol=1e-6))

    def test_checkpoint_without_features_version_is_treated_as_version_one(self):
        """A missing key means "written before the key existed", i.e. v1 — not "no need to check".

        Skipping the check for keyless checkpoints is exactly the silent landmark mismatch that
        FEATURES_VERSION exists to catch.
        """
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            model = GestureLSTM(FEATURE_DIM, 3, pooling="legacy_last_step")
            save_checkpoint(path, model, ["a", "b", "c"], {"input_dim": FEATURE_DIM, "pooling": "legacy_last_step"})

            with self.assertRaisesRegex(ValueError, "FEATURES_VERSION"):
                load_checkpoint(path)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded, _, _ = load_checkpoint(path, allow_incompatible_features=True)

            messages = " ".join(str(w.message) for w in caught)
            self.assertEqual(loaded.pooling, "legacy_last_step")
            self.assertIn("FEATURES_VERSION", messages)
            self.assertIn("retrain", messages.lower())

    def test_checkpoint_with_unknown_model_architecture_version_is_refused(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "future-model.pt"
            model = GestureLSTM(FEATURE_DIM, 3)
            save_checkpoint(
                path,
                model,
                ["a", "b", "c"],
                {
                    "input_dim": FEATURE_DIM,
                    "features_version": FEATURES_VERSION,
                    "pooling": "fwd_last_bwd_first",
                    "model_architecture_version": 999,
                },
            )

            with self.assertRaisesRegex(ValueError, "model architecture"):
                load_checkpoint(path)

    def test_legacy_pooling_stays_reproducible_for_old_checkpoints(self):
        model = GestureLSTM(FEATURE_DIM, 3, hidden_size=4, pooling="legacy_last_step")
        out = torch.zeros(1, 5, 8)
        ramp = torch.arange(1, 6, dtype=torch.float32).unsqueeze(1)
        out[0, :, :4] = ramp
        out[0, :, 4:] = ramp * 10.0

        pooled = model.pool_sequence(out)

        self.assertTrue(torch.allclose(pooled[0, 4:], torch.full((4,), 50.0)))


class DiscoveryTests(unittest.TestCase):
    def test_discover_clips_reads_person_and_vietnamese_label_from_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for person in ("P1", "P2"):
                for label in ("Cảm ơn", "Xin chào"):
                    folder = root / person / label
                    folder.mkdir(parents=True)
                    (folder / f"{label} 01.mov").touch()
                    (folder / f"{label} 02.MP4").touch()
            (root / "P1" / "Cảm ơn" / "notes.txt").touch()

            clips = discover_clips(root)

        self.assertEqual(len(clips), 8)
        self.assertEqual({c.person for c in clips}, {"P1", "P2"})
        self.assertEqual({c.label for c in clips}, {"Cảm ơn", "Xin chào"})
        self.assertTrue(all(c.path.suffix.lower() in {".mov", ".mp4"} for c in clips))

    def test_discover_normalises_macos_style_nfd_labels(self):
        nfd = unicodedata.normalize("NFD", "Cảm ơn")
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "P1" / nfd
            folder.mkdir(parents=True)
            (folder / "01.mov").touch()

            clips = discover_clips(tmp)

        self.assertEqual(clips[0].label, "Cảm ơn")

    def test_discover_refuses_two_raw_directories_that_collapse_to_one_nfc_label(self):
        nfc = "Cảm ơn"
        nfd = unicodedata.normalize("NFD", nfc)
        self.assertNotEqual(nfc, nfd)
        with TemporaryDirectory() as tmp:
            for raw_label in (nfc, nfd):
                folder = Path(tmp) / "P1" / raw_label
                folder.mkdir(parents=True)
                (folder / f"{len(raw_label)}.mov").touch()

            with self.assertRaises(ValueError) as ctx:
                discover_clips(tmp)

        self.assertIn("P1", str(ctx.exception))
        self.assertIn("Cảm ơn", str(ctx.exception))

    def test_directory_whitespace_is_not_silently_stripped_into_a_manifest_label(self):
        with TemporaryDirectory() as tmp:
            # Windows strips trailing dots/spaces from directory paths, so use the leading-space
            # form of the same malformed label on every platform.
            folder = Path(tmp) / "P1" / " Cảm ơn"
            folder.mkdir(parents=True)
            (folder / "01.mov").touch()
            clips = discover_clips(tmp)

        with self.assertRaises(ValueError):
            validate_expected_labels(clips, ["Cảm ơn"])

    def test_parse_video_specs_marks_every_clip_single_signer(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = []
            for label in ("Cảm ơn", "Chuyện gì", "Xin chào", "Tạm biệt"):
                for idx in range(2):
                    path = root / f"{label}-{idx}.mov"
                    path.touch()
                    values.append(f"{label}={path}")

            clips = parse_video_specs(values)

        self.assertEqual(len(clips), 8)
        self.assertEqual({c.person for c in clips}, {SINGLE_SIGNER})
        self.assertEqual(len({c.label for c in clips}), 4)


class ValidationTests(unittest.TestCase):
    def test_signer_mode_needs_two_people_per_label(self):
        clips = [_clip("Cảm ơn", "P1", "1.mov"), _clip("Cảm ơn", "P1", "2.mov")]

        validate_clips(clips, require_signer_split=False)
        with self.assertRaises(ValueError) as ctx:
            validate_clips(clips, require_signer_split=True)
        self.assertIn("Cảm ơn", str(ctx.exception))

    def test_loso_raises_when_a_label_is_missing_for_one_person(self):
        clips = [
            _clip("Cảm ơn", "P1"),
            _clip("Cảm ơn", "P2"),
            _clip("Xin chào", "P1"),
            _clip("Xin chào", "P2"),
            _clip("Tạm biệt", "P1"),
            _clip("Tạm biệt", "P2"),
            _clip("Mèo", "P1"),
        ]

        with self.assertRaises(ValueError) as ctx:
            validate_clips(clips, require_signer_split=True)

        message = str(ctx.exception)
        self.assertIn("Mèo", message)
        self.assertIn("P2", message)

    def test_missing_labels_after_drop_catches_a_label_that_lost_every_clip(self):
        expected = ["Cảm ơn", "Xin chào", "Mèo"]
        survivors = [_clip("Cảm ơn", "P1"), _clip("Cảm ơn", "P2"), _clip("Xin chào", "P1")]

        self.assertEqual(missing_labels_after_drop(expected, survivors), ["Mèo"])
        self.assertEqual(missing_labels_after_drop(expected, survivors + [_clip("Mèo", "P1")]), [])

    def test_manifest_catches_a_label_missing_for_every_person(self):
        clips = [_clip("Cảm ơn", "P1"), _clip("Cảm ơn", "P2")]

        with self.assertRaises(ValueError) as ctx:
            validate_expected_labels(clips, ["Cảm ơn", "Xin chào"])

        self.assertIn("Xin chào", str(ctx.exception))

    def test_manifest_rejects_unexpected_slug_directories(self):
        clips = [_clip("cam_on", "P1"), _clip("cam_on", "P2")]

        with self.assertRaises(ValueError) as ctx:
            validate_expected_labels(clips, ["Cảm ơn"])

        message = str(ctx.exception)
        self.assertIn("Cảm ơn", message)
        self.assertIn("cam_on", message)

    def test_video_mode_refuses_loso(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = []
            for label in ("Cảm ơn", "Xin chào"):
                for idx in range(2):
                    path = root / f"{label}-{idx}.mov"
                    path.touch()
                    values.append(f"{label}={path}")
            clips = parse_video_specs(values)

        validate_clips(clips, require_signer_split=False)
        with self.assertRaises(ValueError):
            validate_clips(clips, require_signer_split=True)


class SplitTests(unittest.TestCase):
    def test_signer_split_never_puts_a_person_in_both_sides(self):
        clips = [
            _clip(label, person, f"{idx}.mov")
            for person in ("P1", "P2", "P3")
            for label in ("Cảm ơn", "Xin chào", "Tạm biệt")
            for idx in range(2)
        ]

        train_idx, val_idx = signer_split(clips, val_person="P2")

        train_people = {clips[i].person for i in train_idx}
        val_people = {clips[i].person for i in val_idx}
        self.assertEqual(val_people, {"P2"})
        self.assertTrue(train_people.isdisjoint(val_people))
        self.assertEqual(len(train_idx) + len(val_idx), len(clips))
        self.assertEqual({clips[i].label for i in val_idx}, {"Cảm ơn", "Xin chào", "Tạm biệt"})
        self.assertEqual({clips[i].label for i in train_idx}, {"Cảm ơn", "Xin chào", "Tạm biệt"})

    def test_signer_split_rejects_unknown_person(self):
        clips = [_clip("Cảm ơn", "P1"), _clip("Cảm ơn", "P2")]
        with self.assertRaises(ValueError):
            signer_split(clips, val_person="P9")


class DatasetTests(unittest.TestCase):
    def _sequences(self, count: int) -> list[np.ndarray]:
        return [
            np.full((SEQUENCE_LENGTH, FEATURE_DIM), float(idx) + 1.0, dtype=np.float32)
            for idx in range(count)
        ]

    def test_dataset_len_and_first_index_is_unaugmented(self):
        sequences = self._sequences(3)
        dataset = GestureDataset(sequences, [0, 1, 2], augmentations_per_clip=4, seed=42)

        self.assertEqual(len(dataset), 3 * 5)
        for clip_idx in range(3):
            features, target = dataset[clip_idx * 5]
            self.assertEqual(int(target), clip_idx)
            np.testing.assert_array_equal(features.numpy(), sequences[clip_idx])

        augmented, target = dataset[1]
        self.assertEqual(int(target), 0)
        self.assertFalse(np.array_equal(augmented.numpy(), sequences[0]))

    def test_dataset_without_augmentation_is_one_sample_per_clip(self):
        sequences = self._sequences(3)
        dataset = GestureDataset(sequences, [0, 1, 2], augmentations_per_clip=0, seed=42)

        self.assertEqual(len(dataset), 3)
        np.testing.assert_array_equal(dataset[2][0].numpy(), sequences[2])

    def test_augment_is_reproducible_for_same_seed_and_index(self):
        sequences = self._sequences(2)
        first = GestureDataset(sequences, [0, 1], augmentations_per_clip=3, seed=42)
        again = GestureDataset(sequences, [0, 1], augmentations_per_clip=3, seed=42)
        other = GestureDataset(sequences, [0, 1], augmentations_per_clip=3, seed=43)

        np.testing.assert_array_equal(first[5][0].numpy(), again[5][0].numpy())
        self.assertFalse(np.array_equal(first[5][0].numpy(), other[5][0].numpy()))
        self.assertFalse(np.array_equal(first[5][0].numpy(), first[6][0].numpy()))


class CacheTests(unittest.TestCase):
    def test_cache_key_changes_when_any_component_changes(self):
        path = Path("/data/P1/Cảm ơn/Cảm ơn 01.mov")
        base = cache_key(path, mtime=1.0, features_version=FEATURES_VERSION)

        self.assertEqual(base, cache_key(path, mtime=1.0, features_version=FEATURES_VERSION))
        self.assertNotEqual(base, cache_key(path, mtime=1.0, features_version=FEATURES_VERSION + 1))
        self.assertNotEqual(base, cache_key(path, mtime=2.0, features_version=FEATURES_VERSION))
        self.assertNotEqual(
            base, cache_key(Path("/data/P2/Cảm ơn/Cảm ơn 01.mov"), mtime=1.0, features_version=FEATURES_VERSION)
        )
        self.assertNotEqual(base, cache_key(path, mtime=1.0, feature_dim=FEATURE_DIM + 1))
        self.assertNotEqual(base, cache_key(path, mtime=1.0, sequence_length=SEQUENCE_LENGTH + 1))

    def test_video_byte_change_invalidates_cache_and_fingerprint_when_mtime_is_restored(self):
        with TemporaryDirectory() as tmp:
            video = Path(tmp) / "01.mov"
            video.write_bytes(b"AAAA")
            original = video.stat()
            clip = Clip(label="Cảm ơn", person="P1", path=video.resolve())
            first_key = cache_key(video, original.st_mtime)
            first_fingerprint = data_fingerprint([clip])

            video.write_bytes(b"BBBB")  # same path and size, different bytes
            os.utime(video, ns=(original.st_atime_ns, original.st_mtime_ns))

            self.assertNotEqual(cache_key(video, video.stat().st_mtime), first_key)
            self.assertNotEqual(data_fingerprint([clip]), first_fingerprint)

    def test_corrupt_cache_entry_is_discarded_and_reextracted(self):
        """A poisoned cache entry must not be reported as a bad clip, nor poison it forever."""

        class StubExtractor:
            def __init__(self):
                self.calls = 0

            def extract_video(self, path, target_len):
                self.calls += 1
                return (
                    np.zeros((target_len, FEATURE_DIM), dtype=np.float32),
                    {"video": str(path), "sampled_frames": 50, "trimmed_frames": 40, "hand_frame_ratio": 0.8},
                )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "P1" / "Cảm ơn" / "Cảm ơn 01.mov"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"not really a video")
            clip = Clip(label="Cảm ơn", person="P1", path=video)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            key = cache_key(video, video.stat().st_mtime)
            (cache_dir / f"{key}.npz").write_bytes(b"garbage, not a zip")

            extractor = StubExtractor()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sequence, stats = extract_with_cache(clip, extractor, cache_dir)

            self.assertEqual(extractor.calls, 1, "must fall through to a real extraction")
            self.assertFalse(stats["from_cache"])
            self.assertEqual(sequence.shape, (SEQUENCE_LENGTH, FEATURE_DIM))

            # The poisoned entry is replaced, so the next run is a clean hit rather than a re-extract.
            again = StubExtractor()
            _, cached_stats = extract_with_cache(clip, again, cache_dir)
            self.assertEqual(again.calls, 0)
            self.assertTrue(cached_stats["from_cache"])
            self.assertEqual(cached_stats["hand_frame_ratio"], 0.8)
            self.assertEqual(list(cache_dir.glob("*.tmp")), [])

    def test_readable_but_invalid_cache_is_reextracted(self):
        class StubExtractor:
            def __init__(self):
                self.calls = 0

            def extract_video(self, path, target_len):
                self.calls += 1
                return (
                    np.zeros((target_len, FEATURE_DIM), dtype=np.float32),
                    {"video": str(path), "sampled_frames": 50, "trimmed_frames": 40, "hand_frame_ratio": 0.8},
                )

        invalid_sequences = (
            np.zeros((1, FEATURE_DIM), dtype=np.float32),
            np.full((SEQUENCE_LENGTH, FEATURE_DIM), np.nan, dtype=np.float32),
        )
        for invalid in invalid_sequences:
            with self.subTest(shape=invalid.shape, finite=bool(np.isfinite(invalid).all())):
                with TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    video = root / "P1" / "Cảm ơn" / "01.mov"
                    video.parent.mkdir(parents=True)
                    video.touch()
                    clip = Clip(label="Cảm ơn", person="P1", path=video)
                    cache_dir = root / "cache"
                    cache_dir.mkdir()
                    cached = cache_dir / f"{cache_key(video, video.stat().st_mtime)}.npz"
                    np.savez(
                        cached,
                        sequence=invalid,
                        sampled_frames=50,
                        trimmed_frames=40,
                        hand_frame_ratio=0.8,
                    )

                    extractor = StubExtractor()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sequence, stats = extract_with_cache(clip, extractor, cache_dir)

                self.assertEqual(extractor.calls, 1)
                self.assertFalse(stats["from_cache"])
                self.assertEqual(sequence.shape, (SEQUENCE_LENGTH, FEATURE_DIM))


class ExtractionFailureTests(unittest.TestCase):
    class DummyExtractor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

    def test_system_failure_is_not_tolerated_as_one_bad_clip(self):
        clip = _clip("Cảm ơn", "P1")
        with (
            mock.patch(
                "prototype_3_gestures.prepare_train.HolisticExtractor",
                return_value=self.DummyExtractor(),
            ),
            mock.patch(
                "prototype_3_gestures.prepare_train.extract_with_cache",
                side_effect=PermissionError("Access is denied"),
            ),
        ):
            with self.assertRaises(PermissionError):
                extract_all([clip], Path("cache"))

    def test_expected_bad_recording_can_still_be_isolated(self):
        clip = _clip("Cảm ơn", "P1")
        with (
            mock.patch(
                "prototype_3_gestures.prepare_train.HolisticExtractor",
                return_value=self.DummyExtractor(),
            ),
            mock.patch(
                "prototype_3_gestures.prepare_train.extract_with_cache",
                side_effect=ClipExtractionError("detected hands in only 2.0%"),
            ),
        ):
            sequences, kept, stats, failures = extract_all([clip], Path("cache"))

        self.assertEqual((sequences, kept, stats), ([], [], []))
        self.assertEqual(len(failures), 1)
        self.assertIn("ClipExtractionError", failures[0]["error"])


class TrainingSignatureTests(unittest.TestCase):
    def _metadata(self):
        return {
            "data_fingerprint": "abc",
            "labels": ["Cảm ơn", "Xin chào"],
            "epochs": 40,
            "augmentations_per_clip": 120,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "seed": 42,
            "pooling": "fwd_last_bwd_first",
            "features_version": FEATURES_VERSION,
            "augmentation_version": 1,
            "training_recipe_version": 1,
            "sequence_length": SEQUENCE_LENGTH,
            "feature_dim": FEATURE_DIM,
            "model": {"architecture_version": 1, "hidden_size": 96},
            "optimizer": {"name": "AdamW", "weight_decay": 1e-4},
            "loss": {"name": "CrossEntropyLoss", "train_label_smoothing": 0.03},
            "device": "cpu",
            "runtime": {"torch": str(torch.__version__), "numpy": np.__version__},
            "num_workers": 0,
        }

    def test_every_training_input_changes_the_signature(self):
        metadata = self._metadata()
        baseline = training_signature(metadata)

        for field in (
            "data_fingerprint",
            "labels",
            "epochs",
            "augmentations_per_clip",
            "batch_size",
            "learning_rate",
            "seed",
            "pooling",
            "features_version",
            "augmentation_version",
            "training_recipe_version",
            "sequence_length",
            "feature_dim",
            "model",
            "optimizer",
            "loss",
            "device",
            "runtime",
        ):
            changed = dict(metadata)
            changed[field] = f"different-{field}"
            with self.subTest(field=field):
                self.assertNotEqual(training_signature(changed), baseline)

        worker_only = dict(metadata, num_workers=4)
        self.assertEqual(training_signature(worker_only), baseline)

    def test_device_changes_the_signature(self):
        metadata = self._metadata()

        self.assertNotEqual(training_signature(metadata), training_signature(dict(metadata, device="cuda")))

    def test_ship_embeds_the_metrics_training_signature_in_checkpoint_config(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = ["Cảm ơn", "Xin chào"]
            clips = []
            for label in labels:
                folder = root / "P1" / label
                folder.mkdir(parents=True)
                for index in range(2):
                    path = folder / f"{index}.mov"
                    path.touch()
                    clips.append(Clip(label=label, person="P1", path=path.resolve()))
            sequences = [np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32) for _ in clips]
            extraction = [
                {
                    "video": str(clip.path),
                    "label": clip.label,
                    "person": clip.person,
                    "sampled_frames": 20,
                    "trimmed_frames": 20,
                    "hand_frame_ratio": 1.0,
                    "from_cache": True,
                }
                for clip in clips
            ]
            model = GestureLSTM(FEATURE_DIM, len(labels))
            model_dir = root / "models"
            argv = [
                "vslr-train",
                "--data-dir",
                str(root),
                "--model-dir",
                str(model_dir),
                "--cache-dir",
                str(root / "cache"),
                "--epochs",
                "1",
                "--augment",
                "0",
            ]
            with (
                mock.patch("sys.argv", argv),
                mock.patch("prototype_3_gestures.prepare_train.discover_clips", return_value=clips),
                mock.patch("prototype_3_gestures.prepare_train.read_expected_labels", return_value=labels),
                mock.patch(
                    "prototype_3_gestures.prepare_train.extract_all",
                    return_value=(sequences, clips, extraction, []),
                ),
                mock.patch(
                    "prototype_3_gestures.prepare_train.train_model",
                    return_value=(model, []),
                ),
            ):
                train_main()

            checkpoint = torch.load(model_dir / "gesture_lstm.pt", map_location="cpu")
            checkpoint_config = checkpoint["config"]
            metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint_config["training_signature"], metrics["training_signature"])
            self.assertEqual(metrics["checkpoint_sha256"], file_sha256(model_dir / "gesture_lstm.pt"))

    def test_pair_decision_rechecks_the_current_checkpoint_hash(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "gesture_lstm.pt"
            labels = ["Cảm ơn", "Xin chào"]
            signature = "same-training-signature"
            config = {
                "input_dim": FEATURE_DIM,
                "sequence_length": SEQUENCE_LENGTH,
                "features_version": FEATURES_VERSION,
                "pooling": "fwd_last_bwd_first",
                "training_signature": signature,
            }
            save_checkpoint(path, GestureLSTM(FEATURE_DIM, len(labels)), labels, config)
            metrics = {
                "labels": labels,
                "training_signature": signature,
                "checkpoint_sha256": file_sha256(path),
            }

            self.assertTrue(prepare_train_module.artifact_pair_matches(signature, metrics, path))

            # Same claimed recipe, different random weights: the embedded signature alone is not
            # enough. The decision-time SHA must reject the replacement.
            save_checkpoint(path, GestureLSTM(FEATURE_DIM, len(labels)), labels, config)
            self.assertFalse(prepare_train_module.artifact_pair_matches(signature, metrics, path))


class TrainLoopTests(unittest.TestCase):
    """Locks the history-row contract that the LOSO fold summary reads."""

    def _args(self, **overrides):
        base = dict(augment=0, seed=42, batch_size=2, learning_rate=1e-3, epochs=1, num_workers=0)
        base.update(overrides)
        return Namespace(**base)

    def _data(self):
        rng = np.random.default_rng(0)
        sequences = [rng.normal(size=(SEQUENCE_LENGTH, FEATURE_DIM)).astype(np.float32) for _ in range(4)]
        return sequences, [0, 1, 0, 1]

    def test_history_keys_distinguish_smoothed_train_loss_from_plain_val_loss(self):
        sequences, targets = self._data()
        cpu = torch.device("cpu")

        _, history = train_model(sequences, targets, 2, self._args(), cpu)
        self.assertEqual(set(history[0]), {"epoch", "train_loss_smoothed", "train_accuracy"})

        _, history = train_model(
            sequences, targets, 2, self._args(), cpu, val_sequences=sequences[:2], val_targets=targets[:2]
        )
        self.assertEqual(
            set(history[0]),
            {"epoch", "train_loss_smoothed", "train_accuracy", "val_loss_plain_ce", "val_accuracy"},
        )

    def test_trains_exactly_the_requested_epochs_with_no_early_stopping(self):
        sequences, targets = self._data()
        _, history = train_model(
            sequences,
            targets,
            2,
            self._args(epochs=4),
            torch.device("cpu"),
            val_sequences=sequences[:2],
            val_targets=targets[:2],
        )
        self.assertEqual([row["epoch"] for row in history], [1, 2, 3, 4])


class ArgumentTests(unittest.TestCase):
    def test_rejects_values_that_would_produce_a_useless_run(self):
        parser = build_parser()
        for bad in (
            ["--data-dir", "x", "--epochs", "0"],  # would ship a random-init checkpoint
            ["--data-dir", "x", "--batch-size", "0"],
            ["--data-dir", "x", "--augment", "-1"],
            ["--data-dir", "x", "--seed", "-1"],
            ["--data-dir", "x", "--num-workers", "-1"],
            ["--data-dir", "x", "--min-hand-ratio", "0"],
            ["--data-dir", "x", "--min-hand-ratio", "1.1"],
            ["--data-dir", "x", "--learning-rate", "0"],
            ["--data-dir", "x", "--learning-rate", "nan"],
        ):
            with self.assertRaises(SystemExit, msg=f"{bad} should be rejected"):
                parser.parse_args(bad)

    def test_requires_exactly_one_clip_source(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--data-dir", "x", "--video", "Cảm ơn=a.mov"])

        args = build_parser().parse_args(["--data-dir", "x", "--epochs", "1", "--augment", "0"])
        self.assertEqual((args.epochs, args.augment, args.num_workers, args.loso), (1, 0, 0, False))
        self.assertEqual(args.labels_file, "dataset/labels.txt")
        self.assertEqual(args.min_hand_ratio, 0.5)


class EvaluationTests(unittest.TestCase):
    def _model_and_loader(self, n_clips=6, n_classes=3):
        sequences = [
            np.full((SEQUENCE_LENGTH, FEATURE_DIM), float(i) + 1.0, dtype=np.float32) for i in range(n_clips)
        ]
        targets = [i % n_classes for i in range(n_clips)]
        args = Namespace(batch_size=4, seed=42)
        return GestureLSTM(FEATURE_DIM, n_classes), build_eval_loader(sequences, targets, args), targets

    def test_evaluate_returns_one_row_per_clip_in_loader_order(self):
        """The join that names a failing clip rests on this: row i is clip i.

        It holds only because the eval loader is unshuffled AND unaugmented. Nothing else in the
        pipeline enforces that, so it is locked here.
        """
        model, loader, targets = self._model_and_loader(n_clips=7)

        loss, accuracy, rows = evaluate(model, loader, torch.device("cpu"))

        self.assertEqual(len(rows), len(targets))
        self.assertEqual([row["target"] for row in rows], targets)
        self.assertGreaterEqual(loss, 0.0)
        self.assertAlmostEqual(accuracy, sum(r["target"] == r["predicted"] for r in rows) / len(rows))

    def test_evaluate_confidence_is_softmax_max_like_the_demo(self):
        """`--confidence` in realtime compares against softmax-max, so the report must too."""
        model, loader, _ = self._model_and_loader()

        _, _, rows = evaluate(model, loader, torch.device("cpu"))

        for row in rows:
            self.assertGreater(row["confidence"], 1.0 / 3.0 - 1e-6)
            self.assertLessEqual(row["confidence"], 1.0)

        features, _ = next(iter(loader))
        with torch.no_grad():
            expected = torch.softmax(model(features), dim=1).max(dim=1).values
        for row, value in zip(rows, expected.tolist()):
            self.assertAlmostEqual(row["confidence"], value, places=6)

    def test_worst_labels_ranks_by_accuracy_and_names_the_confusion(self):
        predictions = [
            {"label": "Mèo", "predicted": "Cảm ơn", "correct": False},
            {"label": "Mèo", "predicted": "Cảm ơn", "correct": False},
            {"label": "Mèo", "predicted": "Mèo", "correct": True},
            {"label": "Cảm ơn", "predicted": "Cảm ơn", "correct": True},
            {"label": "Cảm ơn", "predicted": "Cảm ơn", "correct": True},
            {"label": "Xin chào", "predicted": "Mèo", "correct": False},
        ]

        worst = worst_labels(predictions, limit=2)

        self.assertEqual(worst[0]["label"], "Xin chào")
        self.assertEqual(worst[0]["accuracy"], 0.0)
        self.assertEqual(worst[0]["confused_with"], "Mèo")
        self.assertEqual(worst[1]["label"], "Mèo")
        self.assertAlmostEqual(worst[1]["accuracy"], 1 / 3)
        self.assertEqual(worst[1]["confused_with"], "Cảm ơn")
        self.assertNotIn("Cảm ơn", [row["label"] for row in worst], "a perfect label is not 'worst'")

    def test_suspect_clips_needs_both_a_wrong_prediction_and_a_low_hand_ratio(self):
        """"Below the median" would be self-fulfilling: half of any set is below its own median."""
        predictions = [
            {"video": "a.mov", "label": "Mèo", "predicted": "Cảm ơn", "correct": False},
            {"video": "b.mov", "label": "Mèo", "predicted": "Cảm ơn", "correct": False},
            {"video": "c.mov", "label": "Mèo", "predicted": "Mèo", "correct": True},
        ]
        extraction = [
            {"video": "a.mov", "hand_frame_ratio": 0.31},
            {"video": "b.mov", "hand_frame_ratio": 0.88},
            {"video": "c.mov", "hand_frame_ratio": 0.12},
        ]

        suspects = suspect_clips(predictions, extraction, min_hand_ratio=0.5)

        self.assertEqual([row["video"] for row in suspects], ["a.mov"])
        self.assertAlmostEqual(suspects[0]["hand_frame_ratio"], 0.31)

    def test_suspect_clips_is_empty_when_every_clip_was_recorded_well(self):
        predictions = [{"video": "a.mov", "label": "Mèo", "predicted": "Cảm ơn", "correct": False}]
        extraction = [{"video": "a.mov", "hand_frame_ratio": 0.91}]

        self.assertEqual(suspect_clips(predictions, extraction, min_hand_ratio=0.5), [])

    def test_suspect_clips_refuses_a_broken_prediction_extraction_join(self):
        predictions = [{"video": "missing.mov", "label": "Mèo", "predicted": "Cảm ơn", "correct": False}]

        with self.assertRaisesRegex(ValueError, "missing.mov"):
            suspect_clips(predictions, [], min_hand_ratio=0.5)


class SegmentTrackerTests(unittest.TestCase):
    """The boundary rules used to live inline in realtime.main(), untestable without a camera.

    Thresholds are DURATIONS, not frame counts: the same physical gesture reaches the tracker at
    ~60 fps from a video file and ~20 fps from the live camera on this machine, so a frame-count cap
    encoded two different real-world limits.
    """

    def _feed(self, tracker, script):
        """script: list of (hands_present, now). Returns every segment the tracker completed."""
        segments = []
        for index, (hands_present, now) in enumerate(script):
            done = tracker.feed(hands_present, np.full(3, float(index), dtype=np.float32), now)
            if done is not None:
                segments.append(done)
        return segments

    def _hands_up(self, seconds, fps, start=0.0):
        dt = 1.0 / fps
        return [(True, start + i * dt) for i in range(int(round(seconds * fps)))]

    def _hands_down(self, seconds, fps, start):
        dt = 1.0 / fps
        return [(False, start + i * dt) for i in range(int(round(seconds * fps)))]

    def test_cap_is_the_same_duration_at_any_frame_rate(self):
        """A frame-count cap made an 8 s gesture get cut at 60 fps and left whole at 20 fps."""
        for fps in (60.0, 20.0):
            tracker = SegmentTracker(word_gap=0.45, max_seconds=5.0)
            script = self._hands_up(8.0, fps) + self._hands_down(1.0, fps, start=8.0)

            segments = self._feed(tracker, script)

            self.assertEqual(len(segments), 1, f"fps={fps}")
            self.assertTrue(segments[0].forced, f"fps={fps}: the cap must have closed it")
            self.assertAlmostEqual(segments[0].duration, 5.0, delta=2.0 / fps, msg=f"fps={fps}")

    def test_gesture_shorter_than_the_cap_is_never_forced(self):
        for fps in (60.0, 20.0):
            tracker = SegmentTracker(word_gap=0.45, max_seconds=5.0)
            script = self._hands_up(2.5, fps) + self._hands_down(1.0, fps, start=2.5)

            segments = self._feed(tracker, script)

            self.assertEqual(len(segments), 1, f"fps={fps}")
            self.assertFalse(segments[0].forced, f"fps={fps}")
            self.assertAlmostEqual(segments[0].duration, 2.5, delta=0.5, msg=f"fps={fps}")

    def test_active_duration_excludes_the_word_gap_tail(self):
        tracker = SegmentTracker(word_gap=0.45, max_seconds=5.0)

        self.assertIsNone(tracker.feed(True, np.zeros(3), now=0.0))
        self.assertIsNone(tracker.feed(False, np.zeros(3), now=0.1))
        segment = tracker.feed(False, np.zeros(3), now=0.46)

        self.assertIsNotNone(segment)
        self.assertEqual(segment.end_time, 0.46)
        self.assertEqual(segment.duration, 0.0, "one detected hand frame has zero active duration")
        self.assertLess(segment.duration, 0.35, "the default minimum must reject this blip")

    def test_slow_gesture_does_not_become_two_words(self):
        """After a forced cut the rest of the gesture is discarded, not turned into a second word."""
        tracker = SegmentTracker(word_gap=0.45, max_seconds=1.0)
        script = self._hands_up(2.5, 30.0) + self._hands_down(1.0, 30.0, start=2.5)

        segments = self._feed(tracker, script)

        self.assertEqual(len(segments), 1)
        self.assertTrue(segments[0].forced)

    def test_new_gesture_starts_only_after_hands_drop(self):
        tracker = SegmentTracker(word_gap=0.45, max_seconds=1.0)
        script = self._hands_up(1.5, 30.0)
        script += self._hands_down(1.0, 30.0, start=1.5)
        script += self._hands_up(0.5, 30.0, start=2.6)
        script += self._hands_down(1.0, 30.0, start=3.2)

        segments = self._feed(tracker, script)

        self.assertEqual(len(segments), 2)
        self.assertTrue(segments[0].forced)
        self.assertFalse(segments[1].forced)

    def test_one_dropped_detection_frame_does_not_reopen_the_segment(self):
        """A single no-hand frame is noise everywhere else in this machine, so it must not clear
        awaiting_hand_drop either — one frame of motion blur used to start a second word."""
        tracker = SegmentTracker(word_gap=0.45, max_seconds=0.5)
        dt = 1.0 / 30.0
        script = self._hands_up(0.7, 30.0)
        script += [(False, 21 * dt)]
        script += [(True, (22 + i) * dt) for i in range(20)]

        segments = self._feed(tracker, script)

        self.assertEqual(len(segments), 1, "a detection blink is not the hands coming down")

    def test_word_gap_closes_a_segment_and_short_blips_do_not(self):
        tracker = SegmentTracker(word_gap=0.45, max_seconds=60.0)
        blip = [(True, 0.0), (True, 0.1), (True, 0.2), (False, 0.3), (True, 0.4), (True, 0.5)]
        self.assertEqual(self._feed(tracker, blip), [], "a gap under word_gap must not split")

        segments = self._feed(tracker, [(False, 0.6), (False, 1.2)])
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0].features), 7, "frames inside the sub-gap stay in the segment")
        self.assertFalse(segments[0].forced)

    def test_cap_reached_after_hands_are_down_does_not_swallow_the_next_word(self):
        """The cap can be crossed inside the no-hand tail, where the hands are already down, so the
        next gesture must be allowed to start immediately."""
        tracker = SegmentTracker(word_gap=3.0, max_seconds=0.2)
        script = [(True, 0.0), (True, 0.05), (True, 0.10)]
        script += [(False, 0.15), (False, 0.25)]
        script += [(True, 0.30 + i * 0.05) for i in range(6)]

        segments = self._feed(tracker, script)

        self.assertEqual(len(segments), 2)
        self.assertFalse(segments[0].forced, "hands were already down when the cap was crossed")

    def test_force_boundary_and_reset(self):
        tracker = SegmentTracker(word_gap=0.45, max_seconds=60.0)
        self._feed(tracker, self._hands_up(0.5, 10.0))
        forced = tracker.force_boundary(now=0.5)
        self.assertEqual(len(forced.features), 5)
        self.assertIsNone(tracker.force_boundary(now=0.6), "nothing open, nothing to force")

        self._feed(tracker, [(True, 1.0), (True, 1.1)])
        tracker.reset()
        self.assertIsNone(tracker.force_boundary(now=1.2))
        self.assertFalse(tracker.in_segment)


class CheckToolTests(unittest.TestCase):
    def test_missing_manifest_fails_before_mediapipe_and_cannot_report_success(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "P1" / "Cảm ơn"
            folder.mkdir(parents=True)
            (folder / "01.mov").touch()
            missing_manifest = root / "missing-labels.txt"
            argv = [
                "vslr-check",
                "--data-dir",
                str(root),
                "--labels-file",
                str(missing_manifest),
            ]
            with (
                mock.patch("sys.argv", argv),
                mock.patch(
                    "prototype_3_gestures.check.HolisticExtractor",
                    side_effect=AssertionError("MediaPipe must not start without a manifest"),
                ),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    check_main()

        self.assertNotEqual(ctx.exception.code, 0)
        self.assertIn("missing-labels.txt", str(ctx.exception))

    def test_reads_expected_labels_skipping_comments_and_blanks(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.txt"
            path.write_text(
                "# danh sach nhan\nCảm ơn\n\n  Xin chào  \n# ghi chu\nTạm biệt\n",
                encoding="utf-8",
            )
            self.assertEqual(read_expected_labels(path), ["Cảm ơn", "Xin chào", "Tạm biệt"])

    def test_duplicate_label_in_the_file_is_refused(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.txt"
            path.write_text("Cảm ơn\nXin chào\nCảm ơn\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                read_expected_labels(path)
            self.assertIn("Cảm ơn", str(ctx.exception))

    def test_labels_are_normalised_so_nfd_and_nfc_are_the_same_label(self):
        """macOS-style NFD filenames print identically to NFC but are a different string."""
        nfd = unicodedata.normalize("NFD", "Cảm ơn")
        self.assertNotEqual(nfd, "Cảm ơn")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.txt"
            path.write_text(nfd + "\n" + "Xin chào\n", encoding="utf-8")
            self.assertEqual(read_expected_labels(path)[0], "Cảm ơn")

    def test_clip_status_separates_the_three_failure_causes(self):
        self.assertEqual(clip_status(None, ValueError("Cannot open video: a.mov"), 0.5), "KHONG MO DUOC")
        self.assertEqual(clip_status(None, ValueError("has too few readable frames (3)"), 0.5), "QUA NGAN")
        self.assertEqual(clip_status(None, ValueError("detected hands in only 4.0%"), 0.5), "KHONG THAY TAY")
        self.assertEqual(clip_status(None, PermissionError("Access is denied"), 0.5), "LOI HE THONG")
        self.assertEqual(clip_status({"hand_frame_ratio": 0.41}, None, 0.5), "QUAY LAI")
        self.assertEqual(clip_status({"hand_frame_ratio": 0.5}, None, 0.5), "ok")

    def test_coverage_gaps_reports_missing_labels_and_short_counts(self):
        clips = [
            _clip("Cảm ơn", "P1", "1.mov"),
            _clip("Cảm ơn", "P1", "2.mov"),
            _clip("Xin chào", "P1", "1.mov"),
        ]
        gaps = coverage_gaps(clips, ["Cảm ơn", "Xin chào", "Mèo"], expected_per_label=2)

        self.assertEqual(gaps["missing"], [("P1", "Mèo")])
        self.assertEqual(gaps["short"], [("P1", "Xin chào", 1)])
        self.assertEqual(gaps["unexpected"], [])

    def test_coverage_gaps_flags_a_label_not_in_the_expected_list(self):
        clips = [_clip("Mèo", "P1", "1.mov"), _clip("Mèo", "P1", "2.mov")]
        gaps = coverage_gaps(clips, ["Cảm ơn"], expected_per_label=2)

        self.assertEqual(gaps["unexpected"], ["Mèo"])
        self.assertEqual(gaps["missing"], [("P1", "Cảm ơn")])

    def test_flat_person_directory_holding_videos_is_reported_as_skipped(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "P1" / "Cảm ơn").mkdir(parents=True)
            (root / "P1" / "Cảm ơn" / "Cảm ơn 01.mov").touch()
            (root / "cam_on").mkdir()
            (root / "cam_on" / "Cảm ơn 1.mov").touch()

            skipped = find_skipped_dirs(root)

        self.assertEqual([p.name for p in skipped], ["cam_on"])


class RealtimeLogicTests(unittest.TestCase):
    def test_non_finite_duration_arguments_are_refused_before_model_load(self):
        for option, value in (
            ("--word-gap", "nan"),
            ("--sentence-gap", "nan"),
            ("--min-seconds", "nan"),
            ("--max-seconds", "nan"),
            ("--max-seconds", "inf"),
        ):
            with self.subTest(option=option, value=value):
                with (
                    mock.patch("sys.argv", ["vslr-camera", option, value]),
                    mock.patch(
                        "prototype_3_gestures.realtime.load_checkpoint",
                        side_effect=AssertionError("invalid arguments must stop before model load"),
                    ),
                ):
                    with self.assertRaises(SystemExit):
                        realtime_main()

    def test_prediction_must_meet_confidence_threshold(self):
        self.assertFalse(should_accept_prediction(0.719, 0.72))
        self.assertTrue(should_accept_prediction(0.72, 0.72))


if __name__ == "__main__":
    unittest.main()
