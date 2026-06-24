from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from event_analyzer.ui.control_panel import ControlPanel


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
