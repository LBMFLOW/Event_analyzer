from __future__ import annotations

import importlib.util
from pathlib import Path
import os
import unittest

import numpy as np

from event_analyzer.analysis.exceedance_count_curve import compute_exceedance_count_curve

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PYQT_AVAILABLE = importlib.util.find_spec("PyQt6") is not None

if PYQT_AVAILABLE:
    from PyQt6.QtWidgets import QApplication

    from event_analyzer.plotting.exceedance_count_curve import ExceedanceCountCurveWidget


class ExceedanceCountCurveAlgorithmTests(unittest.TestCase):
    def test_counts_cases_whose_region_maximum_exceeds_threshold(self) -> None:
        result = compute_exceedance_count_curve(
            [0.0, 1.0, 2.0, 3.0],
            {
                "case_a": [0.0, 5.0, 4.0, 1.0],
                "case_b": [0.0, 2.0, 1.0, 0.0],
                "case_c": [np.nan, np.nan, np.nan, np.nan],
            },
            threshold_min=0.0,
            threshold_max=5.0,
            levels=3,
        )

        np.testing.assert_allclose(result.thresholds, [0.0, 2.5, 5.0])
        np.testing.assert_array_equal(result.counts, [2, 1, 0])
        self.assertEqual(result.case_maxima["case_a"], 5.0)
        self.assertTrue(np.isnan(result.case_maxima["case_c"]))

    def test_region_limits_are_applied(self) -> None:
        result = compute_exceedance_count_curve(
            [0.0, 1.0, 2.0, 3.0],
            {
                "case_a": [10.0, 1.0, 1.0, 1.0],
                "case_b": [0.0, 4.0, 5.0, 0.0],
            },
            threshold_min=0.0,
            threshold_max=6.0,
            levels=4,
            region_start=1.0,
            region_end=2.0,
        )

        np.testing.assert_array_equal(result.counts, [2, 1, 1, 0])
        self.assertEqual(result.case_maxima["case_a"], 1.0)
        self.assertEqual(result.case_maxima["case_b"], 5.0)

    def test_ragged_case_arrays_are_accepted(self) -> None:
        result = compute_exceedance_count_curve(
            [0.0, 1.0, 2.0, 3.0],
            {"case_a": [1.0, 2.0], "case_b": [3.0, 4.0, 5.0, 6.0, 7.0]},
            threshold_min=2.0,
            threshold_max=6.0,
            levels=3,
        )

        np.testing.assert_array_equal(result.counts, [1, 1, 0])


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required")
class ExceedanceCountCurveWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        for path in Path.cwd().glob("test_count_curve_*.svg"):
            if path.is_file():
                path.unlink()

    def test_widget_plots_and_exports_svg(self) -> None:
        widget = ExceedanceCountCurveWidget()
        widget.set_source_data(
            [0.0, 1.0, 2.0],
            {"case_a": [0.0, 5.0, 1.0], "case_b": [0.0, 2.0, 3.0]},
            region=(0.0, 2.0),
            region_name="Region A",
            default_x_axis_title="Voltage (V)",
        )
        widget.threshold_min_edit.setText("0")
        widget.threshold_max_edit.setText("5")
        widget.level_count_spin.setValue(25)
        widget.x_axis_title_edit.setText("Candidate threshold")
        widget.y_axis_title_edit.setText("Case count")

        result = widget.plot_curve()
        path = Path.cwd() / "test_count_curve_export.svg"
        widget.export_svg(path)

        self.assertEqual(result.thresholds.size, 25)
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("<svg", text.lower())
        widget.close()


if __name__ == "__main__":
    unittest.main()
