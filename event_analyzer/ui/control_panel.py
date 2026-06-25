from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from event_analyzer.ui.column_selector import SearchableColumnSelector
from event_analyzer.controllers.threshold_manager import ThresholdSummary


AXIS_CHOICES = ["y1", "y2", "y3", "y4", "y5", "y6"]


class AuxiliaryAxisTable(QTableWidget):
    """Selected auxiliary columns with a per-column y-axis assignment."""

    assignments_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Auxiliary column", "Axis"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setVisible(False)

    def set_auxiliary_columns(self, columns: list[str]) -> None:
        previous = self.assignments()
        self.setRowCount(len(columns))
        for row, column in enumerate(columns):
            item = QTableWidgetItem(column)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, 0, item)

            combo = QComboBox()
            combo.addItems(AXIS_CHOICES)
            combo.setCurrentText(previous.get(column, "y2"))
            combo.currentTextChanged.connect(lambda _text: self.assignments_changed.emit())
            self.setCellWidget(row, 1, combo)

    def assignments(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            combo = self.cellWidget(row, 1)
            if item is not None and isinstance(combo, QComboBox):
                result[item.text()] = combo.currentText()
        return result


class CaseStyleTable(QTableWidget):
    """Target case display options: visibility, unit label, and color."""

    visibility_changed = pyqtSignal(str, bool)
    color_changed = pyqtSignal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(["Case", "Unit", "Visible", "Color"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self._colors: dict[str, str] = {}

    def set_cases(
        self,
        cases: list[str],
        *,
        colors: dict[str, str] | None = None,
        visibility: dict[str, bool] | None = None,
        units: dict[str, str] | None = None,
    ) -> None:
        colors = colors or {}
        visibility = visibility or {}
        units = units or {}
        with QSignalBlocker(self):
            self.setRowCount(len(cases))
            for row, case_name in enumerate(cases):
                case_item = QTableWidgetItem(case_name)
                case_item.setFlags(case_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.setItem(row, 0, case_item)

                unit_item = QTableWidgetItem(units.get(case_name, ""))
                unit_item.setFlags(unit_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.setItem(row, 1, unit_item)

                visible_item = QTableWidgetItem()
                visible_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable
                )
                visible_item.setCheckState(
                    Qt.CheckState.Checked if visibility.get(case_name, True) else Qt.CheckState.Unchecked
                )
                self.setItem(row, 2, visible_item)

                color = colors.get(case_name, self._colors.get(case_name, ""))
                if color:
                    self._colors[case_name] = color
                button = QPushButton(color or "Auto")
                button.clicked.connect(lambda _checked=False, name=case_name: self._choose_color(name))
                self._style_color_button(button, color)
                self.setCellWidget(row, 3, button)

    def colors(self) -> dict[str, str]:
        return dict(self._colors)

    def visibility(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for row in range(self.rowCount()):
            case_item = self.item(row, 0)
            visible_item = self.item(row, 2)
            if case_item is None or visible_item is None:
                continue
            result[case_item.text()] = visible_item.checkState() == Qt.CheckState.Checked
        return result

    def set_case_color(self, case_name: str, color: str) -> None:
        self._colors[case_name] = color
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item is not None and item.text() == case_name:
                button = self.cellWidget(row, 3)
                if isinstance(button, QPushButton):
                    button.setText(color)
                    self._style_color_button(button, color)
                break

    def _choose_color(self, case_name: str) -> None:
        initial = QColor(self._colors.get(case_name, "#2563eb"))
        color = QColorDialog.getColor(initial, self, f"Choose color for {case_name}")
        if not color.isValid():
            return
        value = color.name()
        self.set_case_color(case_name, value)
        self.color_changed.emit(case_name, value)

    def _style_color_button(self, button: QPushButton, color: str) -> None:
        if color:
            button.setStyleSheet(f"background-color: {color}; color: {_contrast_text(color)};")
        else:
            button.setStyleSheet("")


class ControlPanel(QWidget):
    """Left-side control panel with no data or plotting logic."""

    open_csv_requested = pyqtSignal()
    update_plot_requested = pyqtSignal()
    active_case_changed = pyqtSignal(str)
    threshold_changed = pyqtSignal(float)
    threshold_disabled_requested = pyqtSignal()
    divider_add_requested = pyqtSignal()
    divider_edit_requested = pyqtSignal()
    divider_delete_requested = pyqtSignal()
    column_selection_changed = pyqtSignal()
    cancel_task_requested = pyqtSignal()
    case_visibility_changed = pyqtSignal(str, bool)
    case_color_changed = pyqtSignal(str, str)
    legend_visibility_changed = pyqtSignal(bool)
    trace_boxes_visibility_changed = pyqtSignal(bool)
    csv_layout_apply_requested = pyqtSignal()
    plot_settings_changed = pyqtSignal()
    region_name_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._all_columns: list[str] = []
        self._numeric_columns: list[str] = []
        self.file_label = QLabel("No CSV loaded")
        self.file_label.setWordWrap(True)
        self.row_count_label = QLabel("Rows: -")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.open_button = QPushButton("Open CSV")
        self.cancel_task_button = QPushButton("Cancel")
        self.cancel_task_button.setVisible(False)
        self.header_row_spin = QSpinBox()
        self.header_row_spin.setRange(1, 1_000_000)
        self.header_row_spin.setValue(1)
        self.units_row_spin = QSpinBox()
        self.units_row_spin.setRange(0, 1_000_000)
        self.units_row_spin.setSpecialValueText("None")
        self.units_row_spin.setValue(0)
        self.data_start_row_spin = QSpinBox()
        self.data_start_row_spin.setRange(1, 1_000_000)
        self.data_start_row_spin.setValue(2)
        self.apply_csv_layout_button = QPushButton("Apply CSV rows")
        self.time_column_combo = QComboBox()
        self.target_selector = SearchableColumnSelector("Auto target case columns")
        self.auxiliary_selector = SearchableColumnSelector("Auxiliary columns")
        self.auxiliary_axis_table = AuxiliaryAxisTable()
        self.case_style_table = CaseStyleTable()
        self.active_case_combo = QComboBox()
        self.show_trace_boxes_checkbox = QCheckBox("Show yellow trace boxes")
        self.show_trace_boxes_checkbox.setChecked(True)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(8)
        self.threshold_spin.setRange(-1e18, 1e18)
        self.threshold_spin.setKeyboardTracking(False)
        self.disable_threshold_button = QPushButton("Disable threshold")
        self.region_name_edit = QLineEdit("Full time range")
        self.region_label = QLabel("No region selected")
        self.region_label.setWordWrap(True)
        self.exceeding_case_count_label = QLabel("Exceeding cases: 0")
        self.event_count_label = QLabel("Events: 0")
        self.max_exceedance_label = QLabel("Max exceedance value: -")
        self.longest_duration_label = QLabel("Longest duration: -")
        self.summary_region_label = QLabel("Active region: -")
        self.exceeding_cases_list = QListWidget()
        self.exceeding_cases_list.setMaximumHeight(90)

        self.divider_table = QTableWidget(0, 2)
        self.divider_table.setHorizontalHeaderLabels(["Label", "Time"])
        self.divider_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.divider_table.verticalHeader().setVisible(False)
        self.add_divider_button = QPushButton("Add")
        self.edit_divider_button = QPushButton("Edit")
        self.delete_divider_button = QPushButton("Delete")
        self.update_plot_button = QPushButton("Plot / Update")
        self.show_legend_checkbox = QCheckBox("Show legend")
        self.show_legend_checkbox.setChecked(True)
        self.plot_title_edit = QLineEdit("Time-series plot")
        self.x_axis_title_edit = QLineEdit()
        self.x_axis_title_edit.setPlaceholderText("Auto")
        self.y_axis_title_edit = QLineEdit()
        self.y_axis_title_edit.setPlaceholderText("Auto")
        self.main_time_min_edit = QLineEdit()
        self.main_time_min_edit.setPlaceholderText("Auto")
        self.main_time_max_edit = QLineEdit()
        self.main_time_max_edit.setPlaceholderText("Auto")
        self.main_target_min_edit = QLineEdit()
        self.main_target_min_edit.setPlaceholderText("Auto")
        self.main_target_max_edit = QLineEdit()
        self.main_target_max_edit.setPlaceholderText("Auto")
        self.chart_y_min_edit = QLineEdit()
        self.chart_y_min_edit.setPlaceholderText("Auto")
        self.chart_y_max_edit = QLineEdit()
        self.chart_y_max_edit.setPlaceholderText("Auto")
        self.chart_axis_title_font_spin = QSpinBox()
        self.chart_axis_title_font_spin.setRange(6, 48)
        self.chart_axis_title_font_spin.setValue(14)
        self.chart_tick_label_font_spin = QSpinBox()
        self.chart_tick_label_font_spin.setRange(6, 48)
        self.chart_tick_label_font_spin.setValue(12)

        self._build_layout()
        self._connect_signals()

    def set_busy(self, busy: bool) -> None:
        self.progress_bar.setVisible(busy)
        self.progress_bar.setRange(0, 0 if busy else 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%" if busy else "")
        self.cancel_task_button.setVisible(busy)
        self.cancel_task_button.setEnabled(busy)
        self.open_button.setEnabled(not busy)
        self.update_plot_button.setEnabled(not busy)

    def set_progress(self, current: int, total: int, message: str = "") -> None:
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(max(0, min(total, current)))
        else:
            self.progress_bar.setRange(0, 0)
        if message:
            self.progress_bar.setFormat(message)

    def set_file_info(self, path: str, row_count: int | None = None) -> None:
        self.file_label.setText(path)
        self.row_count_label.setText("Rows: -" if row_count is None else f"Rows: {row_count:,}")

    def csv_layout_rows(self) -> tuple[int, int, int]:
        return self.header_row_spin.value(), self.units_row_spin.value(), self.data_start_row_spin.value()

    def set_csv_layout(self, *, header_row: int, units_row: int | None, data_start_row: int) -> None:
        with QSignalBlocker(self.header_row_spin), QSignalBlocker(self.units_row_spin), QSignalBlocker(self.data_start_row_spin):
            self.header_row_spin.setValue(max(1, int(header_row)))
            self.units_row_spin.setValue(0 if units_row is None else max(1, int(units_row)))
            self.data_start_row_spin.setValue(max(1, int(data_start_row)))

    def plot_settings(self) -> tuple[str, str, str]:
        return (
            self.plot_title_edit.text().strip(),
            self.x_axis_title_edit.text().strip(),
            self.y_axis_title_edit.text().strip(),
        )

    def main_plot_range_texts(self) -> tuple[str, str, str, str]:
        return (
            self.main_time_min_edit.text().strip(),
            self.main_time_max_edit.text().strip(),
            self.main_target_min_edit.text().strip(),
            self.main_target_max_edit.text().strip(),
        )

    def chart_y_range_texts(self) -> tuple[str, str]:
        return self.chart_y_min_edit.text().strip(), self.chart_y_max_edit.text().strip()

    def chart_font_sizes(self) -> tuple[int, int]:
        return self.chart_axis_title_font_spin.value(), self.chart_tick_label_font_spin.value()

    def set_main_plot_ranges(
        self,
        *,
        time_range: tuple[float, float] | None,
        target_range: tuple[float, float] | None,
    ) -> None:
        with (
            QSignalBlocker(self.main_time_min_edit),
            QSignalBlocker(self.main_time_max_edit),
            QSignalBlocker(self.main_target_min_edit),
            QSignalBlocker(self.main_target_max_edit),
        ):
            self.main_time_min_edit.setText("" if time_range is None else f"{time_range[0]:.12g}")
            self.main_time_max_edit.setText("" if time_range is None else f"{time_range[1]:.12g}")
            self.main_target_min_edit.setText("" if target_range is None else f"{target_range[0]:.12g}")
            self.main_target_max_edit.setText("" if target_range is None else f"{target_range[1]:.12g}")

    def set_chart_y_range(self, value_range: tuple[float, float] | None) -> None:
        with QSignalBlocker(self.chart_y_min_edit), QSignalBlocker(self.chart_y_max_edit):
            self.chart_y_min_edit.setText("" if value_range is None else f"{value_range[0]:.12g}")
            self.chart_y_max_edit.setText("" if value_range is None else f"{value_range[1]:.12g}")

    def set_chart_font_sizes(self, *, axis_title_font_size: int, tick_label_font_size: int) -> None:
        with QSignalBlocker(self.chart_axis_title_font_spin), QSignalBlocker(self.chart_tick_label_font_spin):
            self.chart_axis_title_font_spin.setValue(int(axis_title_font_size))
            self.chart_tick_label_font_spin.setValue(int(tick_label_font_size))

    def set_plot_axis_placeholders(self, *, x_axis: str, y_axis: str) -> None:
        self.x_axis_title_edit.setPlaceholderText(x_axis or "Auto")
        self.y_axis_title_edit.setPlaceholderText(y_axis or "Auto")

    def set_columns(
        self,
        columns: list[str],
        *,
        numeric_columns: list[str],
        likely_time_columns: list[str],
    ) -> None:
        numeric = set(numeric_columns)
        self._all_columns = list(columns)
        self._numeric_columns = list(numeric_columns)
        disabled = set(columns) - numeric
        with QSignalBlocker(self.time_column_combo):
            self.time_column_combo.clear()
            self.time_column_combo.addItems(columns)
            if likely_time_columns:
                self.time_column_combo.setCurrentText(likely_time_columns[0])

        self.target_selector.set_columns(columns, disabled_columns=disabled)
        self.auxiliary_selector.set_columns(columns, disabled_columns=disabled)
        self.set_active_cases([])
        self.auxiliary_axis_table.set_auxiliary_columns([])
        self.case_style_table.set_cases([])
        self._sync_auto_targets()

    def selected_time_column(self) -> str:
        return self.time_column_combo.currentText()

    def set_time_column(self, column: str) -> None:
        index = self.time_column_combo.findText(column)
        if index >= 0:
            self.time_column_combo.setCurrentIndex(index)
            self._sync_auto_targets()

    def selected_target_columns(self) -> list[str]:
        return self._auto_target_columns()

    def selected_auxiliary_columns(self) -> list[str]:
        return self.auxiliary_selector.selected_columns()

    def auxiliary_axis_assignments(self) -> dict[str, str]:
        return self.auxiliary_axis_table.assignments()

    def set_selected_columns(
        self,
        *,
        target_columns: list[str],
        auxiliary_columns: list[str],
        auxiliary_axes: dict[str, str] | None = None,
    ) -> None:
        self.target_selector.set_selected_columns(target_columns)
        self.auxiliary_selector.set_selected_columns(auxiliary_columns)
        self.auxiliary_axis_table.set_auxiliary_columns(auxiliary_columns)
        axes = auxiliary_axes or {}
        for row in range(self.auxiliary_axis_table.rowCount()):
            item = self.auxiliary_axis_table.item(row, 0)
            combo = self.auxiliary_axis_table.cellWidget(row, 1)
            if item is not None and isinstance(combo, QComboBox) and item.text() in axes:
                combo.setCurrentText(axes[item.text()])
        self._sync_auto_targets()

    def selected_divider_row(self) -> int:
        return self.divider_table.currentRow()

    def set_active_cases(self, cases: list[str]) -> None:
        previous = self.active_case_combo.currentText()
        with QSignalBlocker(self.active_case_combo):
            self.active_case_combo.clear()
            self.active_case_combo.addItems(cases)
            if previous in cases:
                self.active_case_combo.setCurrentText(previous)
            elif cases:
                self.active_case_combo.setCurrentIndex(0)

    def trace_boxes_visible(self) -> bool:
        return self.show_trace_boxes_checkbox.isChecked()

    def set_trace_boxes_visible(self, visible: bool) -> None:
        with QSignalBlocker(self.show_trace_boxes_checkbox):
            self.show_trace_boxes_checkbox.setChecked(bool(visible))

    def set_case_styles(
        self,
        cases: list[str],
        *,
        colors: dict[str, str] | None = None,
        visibility: dict[str, bool] | None = None,
        units: dict[str, str] | None = None,
    ) -> None:
        self.case_style_table.set_cases(cases, colors=colors, visibility=visibility, units=units)

    def case_colors(self) -> dict[str, str]:
        return self.case_style_table.colors()

    def case_visibility(self) -> dict[str, bool]:
        return self.case_style_table.visibility()

    def set_region_text(self, text: str) -> None:
        self.region_label.setText(text)

    def region_name(self) -> str:
        return self.region_name_edit.text().strip()

    def set_region_name(self, name: str) -> None:
        with QSignalBlocker(self.region_name_edit):
            self.region_name_edit.setText(name)

    def set_threshold_value(self, value: float | None) -> None:
        if value is None:
            return
        with QSignalBlocker(self.threshold_spin):
            self.threshold_spin.setValue(float(value))

    def set_threshold_summary(self, summary: ThresholdSummary) -> None:
        self.exceeding_case_count_label.setText(f"Exceeding cases: {summary.exceeding_case_count}")
        self.event_count_label.setText(f"Events: {summary.event_count}")
        self.max_exceedance_label.setText(f"Max exceedance value: {_format_optional(summary.max_exceedance_value)}")
        self.longest_duration_label.setText(f"Longest duration: {_format_optional(summary.longest_exceedance_duration)}")
        self.summary_region_label.setText(
            f"Active region: {_format_region(summary.region_start, summary.region_end)}"
        )
        self.exceeding_cases_list.clear()
        self.exceeding_cases_list.addItems(list(summary.exceeding_cases))

    def set_dividers(self, rows: list[tuple[str, str]]) -> None:
        self.divider_table.setRowCount(len(rows))
        for row, (label, time_text) in enumerate(rows):
            self.divider_table.setItem(row, 0, QTableWidgetItem(label))
            self.divider_table.setItem(row, 1, QTableWidgetItem(time_text))

    def _build_layout(self) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)

        file_group = QGroupBox("CSV file")
        file_layout = QVBoxLayout(file_group)
        file_buttons = QHBoxLayout()
        file_buttons.addWidget(self.open_button)
        file_buttons.addWidget(self.cancel_task_button)
        file_layout.addLayout(file_buttons)
        file_form = QFormLayout()
        file_form.addRow("Header row", self.header_row_spin)
        file_form.addRow("Units row", self.units_row_spin)
        file_form.addRow("Data start row", self.data_start_row_spin)
        file_layout.addLayout(file_form)
        file_layout.addWidget(self.apply_csv_layout_button)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.row_count_label)
        file_layout.addWidget(self.progress_bar)

        column_group = QGroupBox("Columns")
        column_layout = QVBoxLayout(column_group)
        form = QFormLayout()
        form.addRow("Time", self.time_column_combo)
        column_layout.addLayout(form)
        column_layout.addWidget(self.target_selector, stretch=2)
        column_layout.addWidget(self.auxiliary_selector, stretch=2)
        column_layout.addWidget(QLabel("Auxiliary axis assignment"))
        column_layout.addWidget(self.auxiliary_axis_table)

        trace_group = QGroupBox("Tracer")
        trace_layout = QFormLayout(trace_group)
        trace_layout.addRow("Active case", self.active_case_combo)
        trace_layout.addRow(self.show_trace_boxes_checkbox)

        case_style_group = QGroupBox("Cases")
        case_style_layout = QVBoxLayout(case_style_group)
        case_style_layout.addWidget(self.case_style_table)

        threshold_group = QGroupBox("Threshold")
        threshold_layout = QVBoxLayout(threshold_group)
        threshold_form = QFormLayout()
        threshold_form.addRow("Value", self.threshold_spin)
        threshold_layout.addLayout(threshold_form)
        threshold_layout.addWidget(self.disable_threshold_button)

        threshold_summary_group = QGroupBox("Threshold summary")
        threshold_summary_layout = QVBoxLayout(threshold_summary_group)
        threshold_summary_layout.addWidget(self.exceeding_case_count_label)
        threshold_summary_layout.addWidget(self.event_count_label)
        threshold_summary_layout.addWidget(self.max_exceedance_label)
        threshold_summary_layout.addWidget(self.longest_duration_label)
        threshold_summary_layout.addWidget(self.summary_region_label)
        threshold_summary_layout.addWidget(QLabel("Exceeding cases"))
        threshold_summary_layout.addWidget(self.exceeding_cases_list)

        region_group = QGroupBox("Selected region")
        region_layout = QVBoxLayout(region_group)
        region_form = QFormLayout()
        region_form.addRow("Name", self.region_name_edit)
        region_layout.addLayout(region_form)
        region_layout.addWidget(self.region_label)

        divider_group = QGroupBox("Dividers")
        divider_layout = QVBoxLayout(divider_group)
        divider_layout.addWidget(self.divider_table)
        buttons = QHBoxLayout()
        buttons.addWidget(self.add_divider_button)
        buttons.addWidget(self.edit_divider_button)
        buttons.addWidget(self.delete_divider_button)
        divider_layout.addLayout(buttons)

        plot_group = QGroupBox("Plot")
        plot_layout = QVBoxLayout(plot_group)
        plot_form = QFormLayout()
        plot_form.addRow("Plot title", self.plot_title_edit)
        plot_form.addRow("X-axis title", self.x_axis_title_edit)
        plot_form.addRow("Y-axis title", self.y_axis_title_edit)
        plot_form.addRow(QLabel("Main plot range (blank = auto)"))
        plot_form.addRow("Time min", self.main_time_min_edit)
        plot_form.addRow("Time max", self.main_time_max_edit)
        plot_form.addRow("Target min", self.main_target_min_edit)
        plot_form.addRow("Target max", self.main_target_max_edit)
        plot_form.addRow(QLabel("Exceedance chart"))
        plot_form.addRow("Y min", self.chart_y_min_edit)
        plot_form.addRow("Y max", self.chart_y_max_edit)
        plot_form.addRow("Axis title font", self.chart_axis_title_font_spin)
        plot_form.addRow("Tick label font", self.chart_tick_label_font_spin)
        plot_layout.addLayout(plot_form)
        plot_layout.addWidget(self.show_legend_checkbox)
        plot_layout.addWidget(self.update_plot_button)

        layout.addWidget(file_group)
        layout.addWidget(column_group)
        layout.addWidget(trace_group)
        layout.addWidget(case_style_group)
        layout.addWidget(threshold_group)
        layout.addWidget(threshold_summary_group)
        layout.addWidget(region_group)
        layout.addWidget(divider_group)
        layout.addWidget(plot_group)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll_layout = QVBoxLayout(self)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(scroll)

    def _connect_signals(self) -> None:
        self.open_button.clicked.connect(self.open_csv_requested)
        self.cancel_task_button.clicked.connect(self.cancel_task_requested)
        self.apply_csv_layout_button.clicked.connect(self.csv_layout_apply_requested)
        self.update_plot_button.clicked.connect(self.update_plot_requested)
        self.time_column_combo.currentTextChanged.connect(self._time_column_changed)
        self.active_case_combo.currentTextChanged.connect(self.active_case_changed)
        self.threshold_spin.valueChanged.connect(self.threshold_changed)
        self.disable_threshold_button.clicked.connect(self.threshold_disabled_requested)
        self.add_divider_button.clicked.connect(self.divider_add_requested)
        self.edit_divider_button.clicked.connect(self.divider_edit_requested)
        self.delete_divider_button.clicked.connect(self.divider_delete_requested)
        self.target_selector.selection_changed.connect(self._target_selection_changed)
        self.auxiliary_selector.selection_changed.connect(self._auxiliary_selection_changed)
        self.auxiliary_axis_table.assignments_changed.connect(self.column_selection_changed)
        self.case_style_table.color_changed.connect(self.case_color_changed)
        self.case_style_table.itemChanged.connect(self._case_style_item_changed)
        self.show_legend_checkbox.toggled.connect(self.legend_visibility_changed)
        self.show_trace_boxes_checkbox.toggled.connect(self.trace_boxes_visibility_changed)
        self.plot_title_edit.editingFinished.connect(self.plot_settings_changed)
        self.x_axis_title_edit.editingFinished.connect(self.plot_settings_changed)
        self.y_axis_title_edit.editingFinished.connect(self.plot_settings_changed)
        self.main_time_min_edit.editingFinished.connect(self.plot_settings_changed)
        self.main_time_max_edit.editingFinished.connect(self.plot_settings_changed)
        self.main_target_min_edit.editingFinished.connect(self.plot_settings_changed)
        self.main_target_max_edit.editingFinished.connect(self.plot_settings_changed)
        self.chart_y_min_edit.editingFinished.connect(self.plot_settings_changed)
        self.chart_y_max_edit.editingFinished.connect(self.plot_settings_changed)
        self.chart_axis_title_font_spin.valueChanged.connect(self.plot_settings_changed)
        self.chart_tick_label_font_spin.valueChanged.connect(self.plot_settings_changed)
        self.region_name_edit.editingFinished.connect(
            lambda: self.region_name_changed.emit(self.region_name())
        )

    def _target_selection_changed(self) -> None:
        cases = self.selected_target_columns()
        self.set_active_cases(cases)
        self.column_selection_changed.emit()

    def _auxiliary_selection_changed(self) -> None:
        self.auxiliary_axis_table.set_auxiliary_columns(self.selected_auxiliary_columns())
        self._sync_auto_targets()
        self.column_selection_changed.emit()

    def _time_column_changed(self, _column: str) -> None:
        self._sync_auto_targets()
        self.column_selection_changed.emit()

    def _case_style_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 2:
            return
        row = item.row()
        case_item = self.case_style_table.item(row, 0)
        if case_item is None:
            return
        self.case_visibility_changed.emit(case_item.text(), item.checkState() == Qt.CheckState.Checked)

    def _auto_target_columns(self) -> list[str]:
        time_column = self.selected_time_column()
        auxiliaries = set(self.selected_auxiliary_columns())
        return [column for column in self._all_columns if column != time_column and column not in auxiliaries]

    def _sync_auto_targets(self) -> None:
        targets = self._auto_target_columns()
        self.target_selector.set_selected_columns(targets)
        self.set_active_cases(targets)
        self.set_case_styles(
            targets,
            colors=self.case_style_table.colors(),
            visibility=self.case_style_table.visibility(),
            units={},
        )


def _format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def _format_region(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return "full range"
    return f"{start:.6g} to {end:.6g}"


def _contrast_text(color: str) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return "#111827"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    return "#ffffff" if luminance < 0.55 else "#111827"
