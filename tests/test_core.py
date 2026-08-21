import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from prototype_3_gestures.prepare_train import build_dataset, parse_video_specs, source_holdout_split
from prototype_3_gestures.vsl3.features import FEATURE_DIM, augment_sequence, resample_sequence
from prototype_3_gestures.vsl3.model import GestureLSTM
from prototype_3_gestures.realtime import should_accept_prediction


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


class ModelTests(unittest.TestCase):
    def test_forward_shape(self):
        model = GestureLSTM(FEATURE_DIM, 3)
        logits = model(torch.zeros((2, 60, FEATURE_DIM), dtype=torch.float32))
        self.assertEqual(tuple(logits.shape), (2, 3))


class TrainingDataTests(unittest.TestCase):
    def test_parse_video_specs_accepts_three_sources_per_three_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = []
            for label in ("cam on", "chuyen gi", "xin chao"):
                for idx in range(3):
                    path = root / f"{label}-{idx}.mov"
                    path.touch()
                    values.append(f"{label}={path}")

            specs = parse_video_specs(values)

        self.assertEqual(len(specs), 9)
        self.assertEqual({label for label, _ in specs}, {"cam on", "chuyen gi", "xin chao"})

    def test_build_dataset_tracks_source_video_and_class(self):
        sequences = [np.full((60, FEATURE_DIM), idx, dtype=np.float32) for idx in range(6)]
        base_targets = [0, 0, 1, 1, 2, 2]

        X, y, synthetic, source_ids = build_dataset(sequences, base_targets, augmentations_per_video=2, seed=7)

        self.assertEqual(X.shape[0], 18)
        self.assertEqual(y.tolist(), [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        self.assertEqual(source_ids.tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
        self.assertEqual(int(synthetic.sum()), 12)

    def test_source_holdout_split_never_leaks_a_video_between_train_and_val(self):
        y = np.asarray([0] * 6 + [1] * 6 + [2] * 6, dtype=np.int64)
        source_ids = np.asarray([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 3)

        train_idx, val_idx = source_holdout_split(y, source_ids, seed=42)

        train_sources = set(source_ids[train_idx].tolist())
        val_sources = set(source_ids[val_idx].tolist())
        self.assertTrue(train_sources.isdisjoint(val_sources))
        self.assertEqual(set(y[train_idx].tolist()), {0, 1, 2})
        self.assertEqual(set(y[val_idx].tolist()), {0, 1, 2})
        self.assertEqual(len(val_sources), 3)


class RealtimeLogicTests(unittest.TestCase):
    def test_prediction_must_meet_confidence_threshold(self):
        self.assertFalse(should_accept_prediction(0.719, 0.72))
        self.assertTrue(should_accept_prediction(0.72, 0.72))


if __name__ == "__main__":
    unittest.main()
