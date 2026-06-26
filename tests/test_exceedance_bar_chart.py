from __future__ import annotations

import importlib.util
from pathlib import Path
import os
import re
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PYQT_AVAILABLE = importlib.util.find_spec("PyQt6") is not None

if PYQT_AVAILABLE:
    from PyQt6.QtWidgets import QApplication

    from event_analyzer.analysis.exceedance import ExceedanceEvent
    from event_analyzer.plotting.exceedance_chart import ExceedanceBarChartWidget, _export_fallback_svg


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

    def test_chart_y_range_and_font_sizes_apply_to_matplotlib_axis(self) -> None:
        widget = ExceedanceBarChartWidget()
        widget.set_axis_titles(x_axis_title="Selected cases", y_axis_title="Seconds above limit")
        widget.set_y_range((0.0, 10.0))
        widget.set_font_sizes(axis_title_font_size=20, tick_label_font_size=16)
        widget.set_events([ExceedanceEvent("case_a", 1, 0.0, 2.0, 4.0, 5.0, 1.0, 3.0, 0.0, 4.0)])

        if widget.figure is not None:
            axis = widget.figure.axes[0]
            self.assertEqual(axis.get_ylim(), (0.0, 10.0))
            self.assertEqual(axis.get_xlabel(), "Selected cases")
            self.assertEqual(axis.get_ylabel(), "Seconds above limit")
            self.assertEqual(axis.xaxis.label.get_size(), 20)
            self.assertEqual(axis.yaxis.label.get_size(), 20)
            self.assertTrue(all(label.get_size() == 16 for label in axis.get_yticklabels()))
        widget.close()

    def test_tab_local_controls_apply_and_export_svg_button_uses_save_helpers(self) -> None:
        widget = ExceedanceBarChartWidget()
        widget.set_events([ExceedanceEvent("case_a", 1, 0.0, 2.0, 4.0, 5.0, 1.0, 3.0, 0.0, 4.0)])
        widget.x_axis_title_edit.setText("Selected cases")
        widget.y_axis_title_edit.setText("Seconds above threshold")
        widget.y_min_edit.setText("0")
        widget.y_max_edit.setText("20")
        widget.axis_title_font_spin.setValue(22)
        widget.tick_label_font_spin.setValue(18)

        self.assertTrue(widget.apply_control_settings())
        self.assertEqual(widget.chart_y_range_texts(), ("0", "20"))
        self.assertEqual(widget.chart_axis_titles(), ("Selected cases", "Seconds above threshold"))
        self.assertEqual(widget.chart_font_sizes(), (22, 18))
        self.assertEqual(widget.export_button.text(), "Export SVG")
        if widget.figure is not None:
            axis = widget.figure.axes[0]
            self.assertEqual(axis.get_ylim(), (0.0, 20.0))
            self.assertEqual(axis.get_xlabel(), "Selected cases")
            self.assertEqual(axis.get_ylabel(), "Seconds above threshold")

        path = Path.cwd() / "test_exceedance_chart_tab_button.svg"
        selected_paths: list[str] = []
        widget.set_save_dialog_helpers(lambda _default_name: str(path), selected_paths.append)
        with patch(
            "event_analyzer.plotting.exceedance_chart.QFileDialog.getSaveFileName",
            return_value=(str(path), "SVG files (*.svg)"),
        ):
            widget.export_button.click()

        self.assertTrue(path.exists())
        self.assertEqual(selected_paths, [str(path)])
        self.assertIn("Exported SVG", widget.status_label.text())
        widget.close()

    def test_fallback_case_labels_are_rotated_below_axis(self) -> None:
        events = [
            ExceedanceEvent("Cell voltage - delta_OCV 2 001", 1, 0.0, 2.0, 2.0, 5.0, 1.0, 3.0, 0.0, 4.0),
            ExceedanceEvent("Cell voltage - delta_OCV 2 002", 1, 0.0, 1.0, 1.0, 4.0, 0.5, 3.0, 0.0, 4.0),
        ]

        with patch("event_analyzer.plotting.exceedance_chart.MATPLOTLIB_AVAILABLE", False):
            widget = ExceedanceBarChartWidget()
            widget.set_events(events)

        self.assertEqual(len(widget._fallback_case_labels), 2)
        for label in widget._fallback_case_labels:
            self.assertGreater(label.angle, 0)
            self.assertLess(label.pos().y(), 0)
            self.assertEqual((float(label.anchor.x()), float(label.anchor.y())), (1.0, 0.0))
        widget.close()

    def test_case_display_labels_are_used_in_fallback_chart_and_svg(self) -> None:
        events = [
            ExceedanceEvent("Cell voltage", 1, 0.0, 2.0, 2.0, 5.0, 1.0, 3.0, 0.0, 4.0),
            ExceedanceEvent("Cell voltage 2", 1, 0.0, 1.0, 1.0, 4.0, 0.5, 3.0, 0.0, 4.0),
        ]
        labels = {"Cell voltage": "Case# 1", "Cell voltage 2": "Case# 2"}

        with patch("event_analyzer.plotting.exceedance_chart.MATPLOTLIB_AVAILABLE", False):
            widget = ExceedanceBarChartWidget()
            widget.set_case_display_labels(labels)
            widget.set_events(events)

            visible_labels = [label.textItem.toPlainText() for label in widget._fallback_case_labels]
            self.assertEqual(visible_labels, ["Case# 1", "Case# 2"])

            path = Path.cwd() / "test_exceedance_chart_case_labels.svg"
            widget.export_svg(path)

        text = path.read_text(encoding="utf-8")
        self.assertIn("Case# 1", text)
        self.assertIn("Case# 2", text)
        self.assertNotIn(">Cell voltage<", text)
        widget.close()

    def test_region_name_is_written_to_fallback_svg(self) -> None:
        events = [ExceedanceEvent("case_a", 1, 0.0, 2.0, 2.0, 5.0, 1.0, 3.0, 0.0, 4.0)]

        with patch("event_analyzer.plotting.exceedance_chart.MATPLOTLIB_AVAILABLE", False):
            widget = ExceedanceBarChartWidget()
            widget.set_region_name("Discharge & hold")
            widget.set_events(events)
            path = Path.cwd() / "test_exceedance_chart_region_name.svg"
            widget.export_svg(path)

        text = path.read_text(encoding="utf-8")
        self.assertIn("Region: Discharge &amp; hold", text)
        region_match = re.search(r'<text x="28" y="([\d.]+)"[^>]*>Region:', text)
        legend_match = re.search(r'<rect x="128" y="([\d.]+)" width="16" height="16"', text)
        plot_match = re.search(r'<rect x="128" y="([\d.]+)" width="[^"]+" height="440"', text)
        x_label_match = re.search(
            r'<text x="[\d.]+" y="([\d.]+)" font-family="Arial" font-size="\d+" text-anchor="middle">Case</text>',
            text,
        )
        self.assertIsNotNone(region_match)
        self.assertIsNotNone(legend_match)
        self.assertIsNotNone(plot_match)
        self.assertIsNotNone(x_label_match)
        self.assertGreater(float(legend_match.group(1)), float(region_match.group(1)) + 10)
        self.assertLess(float(x_label_match.group(1)) - (float(plot_match.group(1)) + 440), 160)
        widget.close()

    def test_fallback_svg_uses_full_labels_and_duration_values(self) -> None:
        events = [
            ExceedanceEvent(
                "Cell voltage - delta_OCV 2 very long case label 001",
                1,
                0.0,
                12.34,
                12.34,
                5.0,
                6.0,
                1.0,
                0.0,
                20.0,
            ),
            ExceedanceEvent(
                "Cell voltage - delta_OCV 2 very long case label 002",
                1,
                1.0,
                8.75,
                7.75,
                4.0,
                5.0,
                1.0,
                0.0,
                20.0,
            ),
        ]
        path = Path.cwd() / "test_exceedance_chart_fallback.svg"

        _export_fallback_svg(events, path)
        text = path.read_text(encoding="utf-8")

        self.assertIn("Cell voltage - delta_OCV 2 very long case label 001", text)
        self.assertIn("Cell voltage - delta_OCV 2 very long case label 002", text)
        self.assertIn(">12.34<", text)
        self.assertNotIn("...", text)
        self.assertIn('font-size="28"', text)
        width = int(re.search(r'width="(\d+)"', text).group(1))
        self.assertGreater(width, 900)
        self.assertLessEqual(width, 2200)

    def test_fallback_svg_respects_y_range_and_axis_font_sizes(self) -> None:
        events = [ExceedanceEvent("case_a", 1, 0.0, 2.0, 4.0, 5.0, 1.0, 3.0, 0.0, 4.0)]
        path = Path.cwd() / "test_exceedance_chart_y_range_fonts.svg"

        _export_fallback_svg(
            events,
            path,
            x_axis_title="Selected cases",
            y_axis_title="Seconds above threshold",
            y_range=(0.0, 20.0),
            axis_title_font_size=24,
            tick_label_font_size=16,
        )
        text = path.read_text(encoding="utf-8")

        self.assertIn('font-size="24" text-anchor="middle">Selected cases</text>', text)
        self.assertIn('font-size="24" transform=', text)
        self.assertIn(">Seconds above threshold</text>", text)
        self.assertIn('font-size="16" text-anchor="end">20</text>', text)


if __name__ == "__main__":
    unittest.main()
