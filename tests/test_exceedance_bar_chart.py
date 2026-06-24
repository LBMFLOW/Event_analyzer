from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None
PYQT_AVAILABLE = importlib.util.find_spec("PyQt6") is not None

if MATPLOTLIB_AVAILABLE and PYQT_AVAILABLE:
    from PyQt6.QtWidgets import QApplication

    from event_analyzer.analysis.exceedance import ExceedanceEvent
    from event_analyzer.plotting.exceedance_chart import ExceedanceBarChartWidget


@unittest.skipUnless(MATPLOTLIB_AVAILABLE and PYQT_AVAILABLE, "PyQt6 and matplotlib are required")
class ExceedanceBarChartWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

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
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.csv"
            widget.export_csv(path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("case_name,event_index,start_time", text)
        self.assertIn("case_a,1,0.5,2.5,2.0,3.0,1.0,1.0,0.0,4.0", text)


if __name__ == "__main__":
    unittest.main()

