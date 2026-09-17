"""Unit tests for Top-3 prediction extraction, ordering, distinct labels, and exact margin."""

from __future__ import annotations

import unittest
import numpy as np
import torch

from prototype_3_gestures.top3 import (
    extract_topk,
    validate_top3_row,
    build_top3_record,
)


class TestTop3(unittest.TestCase):
    def test_extract_topk_valid(self):
        # 4 classes, distinct probabilities
        probs = torch.tensor([[0.70, 0.10, 0.05, 0.15]])
        topk_conf, topk_idx, margin = extract_topk(probs, k=3)

        self.assertEqual(topk_idx.tolist(), [[0, 3, 1]])
        self.assertAlmostEqual(topk_conf[0, 0].item(), 0.70, places=5)
        self.assertAlmostEqual(topk_conf[0, 1].item(), 0.15, places=5)
        self.assertAlmostEqual(topk_conf[0, 2].item(), 0.10, places=5)
        self.assertAlmostEqual(margin[0].item(), 0.55, places=5)

    def test_extract_topk_ordering_enforced(self):
        probs = torch.tensor([[0.6, 0.3, 0.1]])
        topk_conf, _, _ = extract_topk(probs, k=3)
        conf = topk_conf[0].tolist()
        self.assertGreaterEqual(conf[0], conf[1])
        self.assertGreaterEqual(conf[1], conf[2])

    def test_extract_topk_distinct_indices(self):
        probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
        _, topk_idx, _ = extract_topk(probs, k=3)
        indices = topk_idx[0].tolist()
        self.assertEqual(len(indices), len(set(indices)))

    def test_extract_topk_exact_margin(self):
        probs = torch.tensor([[0.6234, 0.2511, 0.0821, 0.0434]])
        topk_conf, _, margin = extract_topk(probs, k=3)
        expected_margin = topk_conf[0, 0].item() - topk_conf[0, 1].item()
        self.assertAlmostEqual(margin[0].item(), expected_margin, places=6)

    def test_validate_top3_row_success(self):
        row = {
            "video": "path/to/001.mov",
            "person": "P01",
            "label": "Cảm ơn",
            "predicted": "Cảm ơn",
            "confidence": 0.85,
            "correct": True,
            "top2_label": "Xin chào",
            "top2_confidence": 0.10,
            "top3_label": "Tạm biệt",
            "top3_confidence": 0.05,
            "top1_top2_margin": 0.75,
        }
        # Must not raise
        validate_top3_row(row)

    def test_validate_top3_row_missing_key_fail_closed(self):
        row = {
            "video": "path/to/001.mov",
            "person": "P01",
            "label": "Cảm ơn",
            "predicted": "Cảm ơn",
            "confidence": 0.85,
            "correct": True,
            # missing top2_label
            "top2_confidence": 0.10,
            "top3_label": "Tạm biệt",
            "top3_confidence": 0.05,
            "top1_top2_margin": 0.75,
        }
        with self.assertRaises(ValueError) as ctx:
            validate_top3_row(row)
        self.assertIn("Missing required top-3 field", str(ctx.exception))

    def test_validate_top3_row_disordered_confidence_fail_closed(self):
        row = {
            "video": "path/to/001.mov",
            "person": "P01",
            "label": "Cảm ơn",
            "predicted": "Cảm ơn",
            "confidence": 0.40,
            "correct": True,
            "top2_label": "Xin chào",
            "top2_confidence": 0.50,  # Invalid: top2 > top1
            "top3_label": "Tạm biệt",
            "top3_confidence": 0.10,
            "top1_top2_margin": -0.10,
        }
        with self.assertRaises(ValueError) as ctx:
            validate_top3_row(row)
        self.assertIn("Confidence ordering violated", str(ctx.exception))

    def test_validate_top3_row_duplicate_labels_fail_closed(self):
        row = {
            "video": "path/to/001.mov",
            "person": "P01",
            "label": "Cảm ơn",
            "predicted": "Cảm ơn",
            "confidence": 0.70,
            "correct": True,
            "top2_label": "Cảm ơn",  # Duplicate with top1
            "top2_confidence": 0.20,
            "top3_label": "Tạm biệt",
            "top3_confidence": 0.10,
            "top1_top2_margin": 0.50,
        }
        with self.assertRaises(ValueError) as ctx:
            validate_top3_row(row)
        self.assertIn("Duplicate labels in top-3", str(ctx.exception))

    def test_validate_top3_row_inaccurate_margin_fail_closed(self):
        row = {
            "video": "path/to/001.mov",
            "person": "P01",
            "label": "Cảm ơn",
            "predicted": "Cảm ơn",
            "confidence": 0.70,
            "correct": True,
            "top2_label": "Xin chào",
            "top2_confidence": 0.20,
            "top3_label": "Tạm biệt",
            "top3_confidence": 0.10,
            "top1_top2_margin": 0.99,  # Invalid: 0.70 - 0.20 != 0.99
        }
        with self.assertRaises(ValueError) as ctx:
            validate_top3_row(row)
        self.assertIn("Margin mismatch", str(ctx.exception))

    def test_build_top3_record_success(self):
        labels = ["A", "B", "C", "D"]
        record = build_top3_record(
            video="test.mov",
            person="P01",
            target_idx=0,
            topk_indices=[0, 2, 1],
            topk_confidences=[0.75, 0.15, 0.10],
            margin=0.60,
            labels=labels,
        )
        self.assertEqual(record["predicted"], "A")
        self.assertEqual(record["top2_label"], "C")
        self.assertEqual(record["top3_label"], "B")
        self.assertAlmostEqual(record["top1_top2_margin"], 0.60)
        self.assertTrue(record["correct"])


if __name__ == "__main__":
    unittest.main()
