from __future__ import annotations

import importlib.util
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PYQT_AVAILABLE = importlib.util.find_spec("PyQt6") is not None

if PYQT_AVAILABLE:
    from PyQt6.QtWidgets import QApplication, QHeaderView, QTableWidgetSelectionRange

    from event_analyzer.analysis.exceedance import ExceedanceEvent
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

    def test_exceedance_summary_copies_table_for_excel(self) -> None:
        table = ExceedanceSummaryTable()
        table.set_events([ExceedanceEvent("case_a", 1, 0.0, 2.0, 2.0, 5.0, 1.0, 3.0, 0.0, 4.0)])

        text = table.copy_table_to_clipboard()

        self.assertEqual(
            text,
            "Case\tEvent\tStart\tEnd\tDuration\tPeak\tPeak Time\ncase_a\t1\t0\t2\t2\t5\t1",
        )
        self.assertEqual(QApplication.clipboard().text(), text)
        table.close()

    def test_region_statistics_copies_selected_cells_for_excel(self) -> None:
        table = RegionStatisticsTable()
        table.set_statistics(
            [
                {
                    "case": "case_a",
                    "unit": "V",
                    "min": 1.0,
                    "max": 3.0,
                    "mean": 2.0,
                    "median": 2.0,
                    "std": 0.5,
                    "time_above_threshold": 4.0,
                    "event_count": 2,
                    "samples": 10,
                }
            ]
        )
        table.setRangeSelected(QTableWidgetSelectionRange(0, 0, 0, 2), True)

        text = table.copy_selection_to_clipboard()

        self.assertEqual(text, "case_a\tV\t1")
        self.assertEqual(QApplication.clipboard().text(), text)
        table.close()


if __name__ == "__main__":
    unittest.main()
