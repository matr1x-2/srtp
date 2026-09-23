import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import run_visual_attention_tracker as tracker


class VisualAttentionTrackerTests(unittest.TestCase):
    def test_visual_region_mask_matches_flattened_grid(self):
        trusted, pattern, shape, rows = tracker.visual_region_masks([1, 22, 34], 2, 1 / 7)
        self.assertEqual(shape, (1, 11, 17))
        self.assertEqual(rows, 2)
        self.assertEqual(int(pattern.sum()), 34)
        self.assertEqual(int(trusted.sum()), 153)
        self.assertTrue(np.all(~(trusted & pattern)))

    def test_candidate_equation_and_selection(self):
        statistics = {
            "normal_mean": np.asarray([[0.8, 0.6]]),
            "normal_std": np.asarray([[0.02, 0.1]]),
            "attack_mean": np.asarray([[0.2, 0.5]]),
            "attack_std": np.asarray([[0.03, 0.1]]),
        }
        candidate = tracker.candidate_map(statistics, k=4)
        np.testing.assert_allclose(candidate, [[0.4, -0.7]])
        self.assertEqual(tracker.select_heads(candidate), [(0, 0)])

    def test_perfect_focus_separation(self):
        curves = tracker.binary_curves(
            np.asarray([0.8, 0.9, 0.7]), np.asarray([0.1, 0.2, 0.3])
        )
        self.assertAlmostEqual(curves["auroc"], 1.0)
        self.assertAlmostEqual(curves["auprc"], 1.0)
        metrics = tracker.classification_metrics(
            np.asarray([0.8, 0.9]), np.asarray([0.1, 0.2]), threshold=0.5
        )
        self.assertEqual((metrics["tp"], metrics["tn"]), (2, 2))
        self.assertEqual((metrics["fp"], metrics["fn"]), (0, 0))

    def test_banner_keeps_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            target = Path(directory) / "attack.png"
            Image.new("RGB", (640, 480), (70, 100, 130)).save(source)
            tracker.render_banner(source, target, tracker.ATTACK_TEXT, 1 / 7)
            with Image.open(target) as rendered:
                self.assertEqual(rendered.size, (640, 480))


if __name__ == "__main__":
    unittest.main()
