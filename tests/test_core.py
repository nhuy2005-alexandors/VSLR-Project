import unittest
import warnings
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from prototype_3_gestures.prepare_train import (
    SINGLE_SIGNER,
    Clip,
    GestureDataset,
    build_parser,
    cache_key,
    discover_clips,
    extract_with_cache,
    missing_labels_after_drop,
    parse_video_specs,
    signer_split,
    train_model,
    validate_clips,
)
from prototype_3_gestures.vsl3.features import (
    FEATURE_DIM,
    FEATURES_VERSION,
    SEQUENCE_LENGTH,
    augment_sequence,
    resample_sequence,
    time_warp_sequence,
)
from prototype_3_gestures.vsl3.model import GestureLSTM
from prototype_3_gestures.realtime import SegmentTracker, should_accept_prediction


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


class SegmentTrackerTests(unittest.TestCase):
    """The boundary rules used to live inline in realtime.main(), untestable without a camera."""

    def _feed(self, tracker, script):
        """script: list of (hands_present, now). Returns every segment the tracker completed."""
        segments = []
        for index, (hands_present, now) in enumerate(script):
            done = tracker.feed(hands_present, np.full(3, float(index), dtype=np.float32), now)
            if done is not None:
                segments.append(done)
        return segments

    def test_slow_gesture_does_not_become_two_words(self):
        """Hitting --max-frames must not reopen a segment while the hands are still up.

        The old loop set in_segment = False on the forced commit, so the very next frame — hands
        still raised, mid-gesture — started a new word. One slow gesture became two.
        """
        tracker = SegmentTracker(word_gap=0.45, max_frames=10)
        script = [(True, i * 0.1) for i in range(25)] + [(False, 2.5 + i * 0.1) for i in range(10)]

        segments = self._feed(tracker, script)

        self.assertEqual(len(segments), 1, "one continuous gesture must yield exactly one segment")
        self.assertEqual(len(segments[0]), 10)

    def test_new_gesture_starts_only_after_hands_drop(self):
        tracker = SegmentTracker(word_gap=0.45, max_frames=10)
        script = (
            [(True, i * 0.1) for i in range(15)]          # forced cut at frame 10, rest ignored
            + [(False, 1.5 + i * 0.1) for i in range(10)]  # hands down past word_gap
            + [(True, 2.6 + i * 0.1) for i in range(10)]   # a genuinely new gesture
            + [(False, 3.7 + i * 0.1) for i in range(10)]
        )

        segments = self._feed(tracker, script)

        self.assertEqual(len(segments), 2)

    def test_word_gap_closes_a_segment_and_short_blips_do_not(self):
        tracker = SegmentTracker(word_gap=0.45, max_frames=150)
        blip = [(True, 0.0), (True, 0.1), (True, 0.2), (False, 0.3), (True, 0.4), (True, 0.5)]
        self.assertEqual(self._feed(tracker, blip), [], "a gap under word_gap must not split")

        segments = self._feed(tracker, [(False, 0.6), (False, 1.2)])
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]), 7, "frames inside the sub-gap stay in the segment")

    def test_force_boundary_and_reset(self):
        tracker = SegmentTracker(word_gap=0.45, max_frames=150)
        self._feed(tracker, [(True, i * 0.1) for i in range(5)])
        forced = tracker.force_boundary()
        self.assertEqual(len(forced), 5)
        self.assertIsNone(tracker.force_boundary(), "nothing open, nothing to force")

        self._feed(tracker, [(True, 1.0), (True, 1.1)])
        tracker.reset()
        self.assertIsNone(tracker.force_boundary())
        self.assertFalse(tracker.in_segment)


class RealtimeLogicTests(unittest.TestCase):
    def test_prediction_must_meet_confidence_threshold(self):
        self.assertFalse(should_accept_prediction(0.719, 0.72))
        self.assertTrue(should_accept_prediction(0.72, 0.72))


if __name__ == "__main__":
    unittest.main()
