from __future__ import annotations

import unittest

import numpy as np

from event_analyzer.data.downsample import downsample_indices, min_max_downsample


class DownsampleTests(unittest.TestCase):
    def test_min_max_downsample_preserves_narrow_spike(self) -> None:
        x = np.arange(10_000, dtype=float)
        y = np.zeros_like(x)
        y[4_321] = 100.0

        x_plot, y_plot = min_max_downsample(x, y, max_points=500)

        self.assertLessEqual(x_plot.size, 500)
        self.assertIn(100.0, y_plot)
        self.assertEqual(float(x_plot[np.argmax(y_plot)]), 4_321.0)

    def test_downsample_indices_preserve_threshold_crossing_neighbors(self) -> None:
        x = np.arange(1_000, dtype=float)
        y = np.zeros_like(x)
        y[500:505] = 10.0

        indices = downsample_indices(x, y, max_points=100, threshold=5.0, region=(450.0, 550.0))

        self.assertLessEqual(indices.size, 100)
        self.assertIn(499, indices)
        self.assertIn(500, indices)
        self.assertIn(504, indices)
        self.assertIn(505, indices)

    def test_downsample_indices_keep_region_boundaries_nearby(self) -> None:
        x = np.arange(1_000, dtype=float)
        y = np.sin(x)

        indices = downsample_indices(x, y, max_points=80, region=(123.2, 789.8))
        selected_x = x[indices]

        self.assertTrue(np.any(np.abs(selected_x - 123.2) <= 1.0))
        self.assertTrue(np.any(np.abs(selected_x - 789.8) <= 1.0))

    def test_downsample_zero_budget_returns_empty_arrays(self) -> None:
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)

        x_plot, y_plot = min_max_downsample(x, y, max_points=0)

        self.assertEqual(x_plot.size, 0)
        self.assertEqual(y_plot.size, 0)


if __name__ == "__main__":
    unittest.main()
