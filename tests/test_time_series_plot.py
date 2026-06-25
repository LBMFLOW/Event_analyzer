from __future__ import annotations

import importlib.util
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PYQT_AVAILABLE = importlib.util.find_spec("PyQt6") is not None

if PYQT_AVAILABLE:
    from PyQt6.QtWidgets import QApplication

    from event_analyzer.plotting.time_series_plot import TimeSeriesPlotWidget


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required")
class TimeSeriesPlotWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_auxiliary_shorter_than_time_is_padded_with_nan_gaps(self) -> None:
        widget = TimeSeriesPlotWidget()
        widget.set_data([0, 1, 2, 3], {"case_a": [1, 2, 3, 4]})

        widget.set_auxiliary_data({"aux_ends_early": [10, 11]}, {"aux_ends_early": "y2"})

        aligned = widget._auxiliary_values["aux_ends_early"]
        self.assertEqual(aligned.size, 4)
        self.assertEqual(aligned[0], 10.0)
        self.assertEqual(aligned[1], 11.0)
        self.assertTrue(np.isnan(aligned[2]))
        self.assertTrue(np.isnan(aligned[3]))
        widget.close()

    def test_auxiliary_longer_than_time_is_trimmed(self) -> None:
        widget = TimeSeriesPlotWidget()
        widget.set_data([0, 1, 2], {"case_a": [1, 2, 3]})

        widget.set_auxiliary_data({"aux_extra_tail": [10, 11, 12, 13]}, {"aux_extra_tail": "y2"})

        np.testing.assert_allclose(widget._auxiliary_values["aux_extra_tail"], [10, 11, 12])
        widget.close()

    def test_trace_boxes_can_be_hidden_without_disabling_marker(self) -> None:
        widget = TimeSeriesPlotWidget()
        widget.set_data([0, 1, 2], {"case_a": [1, 2, 3]})
        widget.set_active_case("case_a")
        widget.set_slider_time(1.0)

        self.assertTrue(widget._trace_marker.isVisible())
        self.assertTrue(widget._trace_label.isVisible())
        self.assertTrue(widget._trace_x_axis_label.isVisible())
        self.assertTrue(widget._trace_y_axis_label.isVisible())

        widget.set_trace_boxes_visible(False)

        self.assertTrue(widget._trace_marker.isVisible())
        self.assertFalse(widget._trace_label.isVisible())
        self.assertFalse(widget._trace_x_axis_label.isVisible())
        self.assertFalse(widget._trace_y_axis_label.isVisible())

        widget.set_trace_boxes_visible(True)

        self.assertTrue(widget._trace_label.isVisible())
        self.assertTrue(widget._trace_x_axis_label.isVisible())
        self.assertTrue(widget._trace_y_axis_label.isVisible())
        widget.close()


if __name__ == "__main__":
    unittest.main()
