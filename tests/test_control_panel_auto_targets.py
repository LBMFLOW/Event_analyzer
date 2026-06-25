from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from event_analyzer.ui.control_panel import ControlPanel
from event_analyzer.ui.main_window_controller import _case_display_labels


def test_target_columns_are_derived_from_time_and_auxiliary_selection() -> None:
    app = QApplication.instance() or QApplication([])
    panel = ControlPanel()
    panel.set_columns(
        ["time_s", "case_a", "case_b", "aux_temperature", "operator_note"],
        numeric_columns=["time_s", "case_a", "case_b", "aux_temperature"],
        likely_time_columns=["time_s"],
    )

    assert panel.selected_time_column() == "time_s"
    assert panel.selected_target_columns() == ["case_a", "case_b", "aux_temperature", "operator_note"]

    panel.auxiliary_selector.set_selected_columns(["aux_temperature"])

    assert panel.selected_auxiliary_columns() == ["aux_temperature"]
    assert panel.selected_target_columns() == ["case_a", "case_b", "operator_note"]
    assert panel.active_case_combo.count() == 3
    assert panel.active_case_combo.currentText() == "case_a"
    assert panel.trace_boxes_visible() is True

    panel.set_trace_boxes_visible(False)

    assert panel.trace_boxes_visible() is False
    panel.main_time_min_edit.setText("1")
    panel.main_time_max_edit.setText("2")
    panel.main_target_min_edit.setText("3")
    panel.main_target_max_edit.setText("4")
    panel.chart_y_min_edit.setText("0")
    panel.chart_y_max_edit.setText("10")
    panel.chart_x_axis_title_edit.setText("Selected cases")
    panel.chart_y_axis_title_edit.setText("Duration above limit")
    panel.set_chart_font_sizes(axis_title_font_size=18, tick_label_font_size=16)

    assert panel.main_plot_range_texts() == ("1", "2", "3", "4")
    assert panel.chart_y_range_texts() == ("0", "10")
    assert panel.chart_axis_titles() == ("Selected cases", "Duration above limit")
    assert panel.chart_font_sizes() == (18, 16)

    panel.close()
    app.processEvents()


def test_duplicate_source_target_headers_get_case_number_labels() -> None:
    labels = _case_display_labels(
        ["Cell voltage", "Cell voltage 2", "Cell voltage 3", "aux_temperature"],
        {
            "Cell voltage": "Cell voltage",
            "Cell voltage 2": "Cell voltage",
            "Cell voltage 3": "Cell voltage",
            "aux_temperature": "aux_temperature",
        },
    )

    assert labels == {
        "Cell voltage": "Case# 1",
        "Cell voltage 2": "Case# 2",
        "Cell voltage 3": "Case# 3",
    }
