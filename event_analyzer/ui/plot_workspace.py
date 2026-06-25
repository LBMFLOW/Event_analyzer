from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QMenu,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from event_analyzer.analysis.exceedance import ExceedanceEvent
from event_analyzer.plotting.exceedance_chart import ExceedanceBarChartWidget
from event_analyzer.plotting.exceedance_count_curve import ExceedanceCountCurveWidget
from event_analyzer.plotting.time_series_plot import TimeSeriesPlotWidget


class MainPlotPanel(QWidget):
    """PyQtGraph plot placeholder for the interactive time-series view."""

    plot_clicked = pyqtSignal(float)
    plot_right_clicked = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plot_widget = TimeSeriesPlotWidget()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)

        # Plug-in point: the app controller calls TimeSeriesPlotWidget.set_data,
        # set_auxiliary_data, set_threshold, set_selected_region, and divider
        # methods from this panel.


class BarChartPanel(ExceedanceBarChartWidget):
    """Compatibility wrapper around the real exceedance bar chart widget."""


class CountCurvePanel(ExceedanceCountCurveWidget):
    """Count of cases exceeding candidate target thresholds."""


class CsvPreviewTable(QTableWidget):
    """Read-only table for displaying a small CSV preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 0, parent)
        _configure_resizable_table(self)

    def set_preview(self, headers: list[str], rows: list[dict[str, object]]) -> None:
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, header in enumerate(headers):
                value = row.get(header, "")
                self.setItem(row_index, column_index, QTableWidgetItem("" if value is None else str(value)))


class CopyableTableWidget(QTableWidget):
    """Read-only table with Excel-friendly clipboard export."""

    def __init__(self, rows: int, columns: int, parent: QWidget | None = None) -> None:
        super().__init__(rows, columns, parent)
        _configure_resizable_table(self)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_copy_menu)

    def copy_selection_to_clipboard(self) -> str:
        """Copy selected cells as tab-separated text and return the copied text."""
        indexes = self.selectedIndexes()
        if not indexes:
            return self.copy_table_to_clipboard()
        rows = sorted({index.row() for index in indexes})
        columns = sorted({index.column() for index in indexes})
        text = self._tsv_for_cells(rows, columns, include_headers=False)
        QApplication.clipboard().setText(text)
        return text

    def copy_table_to_clipboard(self) -> str:
        """Copy the whole table, including column headers, as tab-separated text."""
        rows = list(range(self.rowCount()))
        columns = list(range(self.columnCount()))
        text = self._tsv_for_cells(rows, columns, include_headers=True)
        QApplication.clipboard().setText(text)
        return text

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection_to_clipboard()
            event.accept()
            return
        super().keyPressEvent(event)

    def _show_copy_menu(self, position: QPoint) -> None:
        menu = QMenu(self)
        copy_selection = menu.addAction("Copy")
        copy_selection.setEnabled(bool(self.selectedIndexes()) or self.rowCount() > 0)
        copy_all = menu.addAction("Copy table with headers")
        copy_all.setEnabled(self.columnCount() > 0)
        action = menu.exec(self.viewport().mapToGlobal(position))
        if action == copy_selection:
            self.copy_selection_to_clipboard()
        elif action == copy_all:
            self.copy_table_to_clipboard()

    def _tsv_for_cells(self, rows: list[int], columns: list[int], *, include_headers: bool) -> str:
        lines: list[str] = []
        if include_headers:
            lines.append("\t".join(_clean_tsv_cell(self._header_text(column)) for column in columns))
        for row in rows:
            values = [self._cell_text(row, column) for column in columns]
            lines.append("\t".join(_clean_tsv_cell(value) for value in values))
        return "\n".join(lines)

    def _cell_text(self, row: int, column: int) -> str:
        item = self.item(row, column)
        return "" if item is None else item.text()

    def _header_text(self, column: int) -> str:
        item = self.horizontalHeaderItem(column)
        return "" if item is None else item.text()


class ExceedanceSummaryTable(CopyableTableWidget):
    """Read-only table for event summaries exported to CSV."""

    HEADERS = ["Case", "Event", "Start", "End", "Duration", "Peak", "Peak Time"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)

    def set_events(self, events: list[ExceedanceEvent]) -> None:
        self.setRowCount(len(events))
        for row, event in enumerate(events):
            values = [
                event.case_name,
                str(event.event_index),
                _format_float(event.start_time),
                _format_float(event.end_time),
                _format_float(event.duration),
                _format_float(event.peak_value),
                _format_float(event.peak_time),
            ]
            for column, value in enumerate(values):
                self.setItem(row, column, QTableWidgetItem(value))


class RegionStatisticsTable(CopyableTableWidget):
    """Read-only selected-region statistics table."""

    HEADERS = [
        "Case",
        "Unit",
        "Min",
        "Max",
        "Mean",
        "Median",
        "Std Dev",
        "Time > Threshold",
        "Events",
        "Samples",
    ]
    FIELDS = [
        "case",
        "unit",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "time_above_threshold",
        "event_count",
        "samples",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)

    def set_statistics(self, rows: list[dict[str, object]]) -> None:
        self.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, field in enumerate(self.FIELDS):
                self.setItem(row_index, column_index, QTableWidgetItem(_format_value(row.get(field))))


class PlotWorkspace(QWidget):
    """Right-side workspace containing the plot, chart, preview, and summary."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.main_plot = MainPlotPanel()
        self.bar_chart = BarChartPanel(time_unit="time")
        self.bar_chart.set_plot_adapter(self.main_plot.plot_widget)
        self.count_curve = CountCurvePanel()
        self.preview_table = CsvPreviewTable()
        self.summary_table = ExceedanceSummaryTable()
        self.statistics_table = RegionStatisticsTable()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.bar_chart, "Exceedance durations")
        self.tabs.addTab(self.count_curve, "Exceedance counts")
        self.tabs.addTab(self.preview_table, "CSV preview")
        self.tabs.addTab(self.summary_table, "Exceedance summary")
        self.tabs.addTab(self.statistics_table, "Region statistics")

        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Vertical)
        splitter.addWidget(self.main_plot)
        splitter.addWidget(self.tabs)
        splitter.setSizes([650, 260])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def _configure_resizable_table(table: QTableWidget) -> None:
    table.setAlternatingRowColors(True)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    header.setStretchLastSection(False)
    header.setMinimumSectionSize(45)
    header.setDefaultSectionSize(145)


def _clean_tsv_cell(value: str) -> str:
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")
