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
