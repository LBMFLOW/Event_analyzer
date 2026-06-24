from __future__ import annotations

import importlib.util
from pathlib import Path
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PYQT_AVAILABLE = importlib.util.find_spec("PyQt6") is not None

if PYQT_AVAILABLE:
    from PyQt6.QtWidgets import QApplication

    from event_analyzer.analysis.exceedance import ExceedanceEvent
    from event_analyzer.plotting.exceedance_chart import ExceedanceBarChartWidget


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required")
class ExceedanceBarChartWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        for path in Path.cwd().glob("test_exceedance_chart_*"):
            if path.is_file():
                path.unlink()

    def test_set_events_and_export_csv(self) -> None:
        widget = ExceedanceBarChartWidget()
        events = [
            ExceedanceEvent("case_a", 1, 0.5, 2.5, 2.0, 3.0, 1.0, 1.0, 0.0, 4.0),
            ExceedanceEvent("case_a", 2, 3.0, 3.5, 0.5, 2.0, 3.2, 1.0, 0.0, 4.0),
            ExceedanceEvent("case_b", 1, 1.0, 1.5, 0.5, 4.0, 1.2, 1.0, 0.0, 4.0),
        ]

        widget.set_events(events)

        self.assertEqual(len(widget.events), 3)
        self.assertEqual(len(widget._patch_events), 3)
        path = Path.cwd() / "test_exceedance_chart_events.csv"
        widget.export_csv(path)
        text = path.read_text(encoding="utf-8")

        self.assertIn("case_name,event_index,start_time", text)
        self.assertIn("case_a,1,0.5,2.5,2.0,3.0,1.0,1.0,0.0,4.0", text)

    def test_export_svg_without_crashing(self) -> None:
        widget = ExceedanceBarChartWidget()
        widget.set_events([ExceedanceEvent("case_a", 1, 0.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.0, 2.0)])

        path = Path.cwd() / "test_exceedance_chart.svg"
        widget.export_svg(path)

        self.assertTrue(path.exists())
        self.assertIn("<svg", path.read_text(encoding="utf-8", errors="ignore").lower())

    def test_long_case_labels_render_without_crashing(self) -> None:
        widget = ExceedanceBarChartWidget()
        events = [
            ExceedanceEvent(
                f"Cell voltage - delta_OCV 2 very long case label {index}",
                1,
                0.0,
                1.0,
                float(index + 1),
                float(index + 2),
                0.5,
                1.0,
                0.0,
                2.0,
            )
            for index in range(18)
        ]

        widget.set_events(events)

        self.assertEqual(len(widget.events), 18)
        self.assertIn("events across", widget.status_label.text())


if __name__ == "__main__":
    unittest.main()
