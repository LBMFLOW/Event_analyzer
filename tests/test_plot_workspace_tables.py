from __future__ import annotations

import importlib.util
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PYQT_AVAILABLE = importlib.util.find_spec("PyQt6") is not None

if PYQT_AVAILABLE:
    from PyQt6.QtWidgets import QApplication, QHeaderView

    from event_analyzer.ui.plot_workspace import CsvPreviewTable, ExceedanceSummaryTable, RegionStatisticsTable


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required")
class PlotWorkspaceTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_workspace_tables_have_resizable_columns(self) -> None:
        preview = CsvPreviewTable()
        preview.set_preview(["time", "value"], [{"time": 0, "value": 1}])
        tables = [preview, ExceedanceSummaryTable(), RegionStatisticsTable()]

        for table in tables:
            self.assertEqual(
                table.horizontalHeader().sectionResizeMode(0),
                QHeaderView.ResizeMode.Interactive,
            )
            table.close()


if __name__ == "__main__":
    unittest.main()
