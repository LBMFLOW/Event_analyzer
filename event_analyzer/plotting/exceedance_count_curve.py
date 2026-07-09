from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from event_analyzer.analysis.exceedance_count_curve import ExceedanceCountCurve, compute_exceedance_count_curve
from event_analyzer.data.units import split_column_unit

try:  # Matplotlib is a normal dependency, but keep import failures graceful.
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - only used in partial installations.
    FigureCanvasQTAgg = None
    Figure = None
    MATPLOTLIB_AVAILABLE = False


class ExceedanceCountCurveWidget(QWidget):
    """Curve of case count versus candidate target threshold."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8.8, 4.2), tight_layout=False) if MATPLOTLIB_AVAILABLE else None
        self.canvas = FigureCanvasQTAgg(self.figure) if MATPLOTLIB_AVAILABLE else pg.PlotWidget()
        self.status_label = QLabel("Load target data, choose a target range, then click Plot curve.")
        self.status_label.setWordWrap(True)

        self.threshold_min_edit = QLineEdit()
        self.threshold_min_edit.setPlaceholderText("Auto")
        self.threshold_max_edit = QLineEdit()
        self.threshold_max_edit.setPlaceholderText("Auto")
        self.level_count_spin = QSpinBox()
        self.level_count_spin.setRange(20, 10_000)
        self.level_count_spin.setValue(500)

        self.title_edit = QLineEdit("Cases exceeding candidate threshold")
        self.x_axis_title_edit = QLineEdit()
        self.x_axis_title_edit.setPlaceholderText("Auto")
        self.y_axis_title_edit = QLineEdit("Number of exceeding cases")
        self.x_axis_min_edit = QLineEdit()
        self.x_axis_min_edit.setPlaceholderText("Auto")
        self.x_axis_max_edit = QLineEdit()
        self.x_axis_max_edit.setPlaceholderText("Auto")
        self.x_tick_increment_edit = QLineEdit()
        self.x_tick_increment_edit.setPlaceholderText("Auto")
        self.y_axis_min_edit = QLineEdit()
        self.y_axis_min_edit.setPlaceholderText("Auto")
        self.y_axis_max_edit = QLineEdit()
        self.y_axis_max_edit.setPlaceholderText("Auto")
        self.y_tick_increment_edit = QLineEdit()
        self.y_tick_increment_edit.setPlaceholderText("Auto")
        self.axis_title_font_spin = QSpinBox()
        self.axis_title_font_spin.setRange(6, 48)
        self.axis_title_font_spin.setValue(14)
        self.tick_label_font_spin = QSpinBox()
        self.tick_label_font_spin.setRange(6, 48)
        self.tick_label_font_spin.setValue(12)

        self.plot_button = QPushButton("Plot curve")
        self.export_button = QPushButton("Export SVG")
        self.export_csv_button = QPushButton("Export CSV")

        self._time: np.ndarray | None = None
        self._values_by_case: dict[str, np.ndarray] = {}
        self._region: tuple[float | None, float | None] | None = None
        self._region_name = ""
        self._default_x_axis_title = "Target parameter"
        self._last_result: ExceedanceCountCurve | None = None
        self._save_dialog_initial_path: Callable[[str], str] | None = None
        self._save_dialog_path_selected: Callable[[str], None] | None = None

        self._build_layout()
        if isinstance(self.canvas, pg.PlotWidget):
            self._configure_fallback_plot()
        self._connect_signals()
        self._render_empty_message("No count curve has been plotted yet.")

    def set_save_dialog_helpers(
        self,
        initial_path: Callable[[str], str] | None,
        path_selected: Callable[[str], None] | None,
    ) -> None:
        self._save_dialog_initial_path = initial_path
        self._save_dialog_path_selected = path_selected

    def set_source_data(
        self,
        time: Sequence[float],
        values_by_case: Mapping[str, Sequence[float]],
        *,
        region: tuple[float | None, float | None] | None = None,
        region_name: str = "",
        default_x_axis_title: str = "Target parameter",
    ) -> None:
        """Set source target data used when the user clicks Plot curve."""
        self._time = np.asarray(time, dtype=float)
        self._values_by_case = {
            str(case): np.asarray(values, dtype=float)
            for case, values in values_by_case.items()
        }
        self._region = region
        self._region_name = str(region_name or "").strip()
        self._default_x_axis_title = default_x_axis_title or "Target parameter"
        self._fill_candidate_range_if_blank()
        self._last_result = None
        self._render_empty_message("Source data changed. Click Plot curve to update.")
        self.status_label.setText(f"Ready: {len(self._values_by_case)} target case(s).")

    def set_region_name(self, name: str) -> None:
        """Update the selected-region name shown in the curve and SVG export."""
        self._region_name = str(name or "").strip()
        if self._last_result is not None:
            self._render_result(self._last_result)

    def settings(self) -> dict[str, object]:
        """Return serializable user settings for project/session save."""
        return {
            "threshold_range": _optional_range_from_edits(
                self.threshold_min_edit,
                self.threshold_max_edit,
                "count-curve target range",
            ),
            "levels": self.level_count_spin.value(),
            "title": self.title_edit.text().strip(),
            "x_axis_title": self.x_axis_title_edit.text().strip(),
            "y_axis_title": self.y_axis_title_edit.text().strip(),
            "x_range": _optional_range_from_edits(self.x_axis_min_edit, self.x_axis_max_edit, "count-curve x-axis range"),
            "x_tick_increment": _optional_positive_float_from_edit(
                self.x_tick_increment_edit,
                "count-curve x-axis tick increment",
            ),
            "y_range": _optional_range_from_edits(self.y_axis_min_edit, self.y_axis_max_edit, "count-curve y-axis range"),
            "y_tick_increment": _optional_positive_float_from_edit(
                self.y_tick_increment_edit,
                "count-curve y-axis tick increment",
            ),
            "axis_title_font_size": self.axis_title_font_spin.value(),
            "tick_label_font_size": self.tick_label_font_spin.value(),
        }

    def apply_settings(self, settings: Mapping[str, object] | None) -> None:
        """Apply serialized settings from a saved project/session."""
        if not settings:
            return
        _set_range_edits(self.threshold_min_edit, self.threshold_max_edit, _coerce_range(settings.get("threshold_range")))
        _set_range_edits(self.x_axis_min_edit, self.x_axis_max_edit, _coerce_range(settings.get("x_range")))
        _set_range_edits(self.y_axis_min_edit, self.y_axis_max_edit, _coerce_range(settings.get("y_range")))
        self.x_tick_increment_edit.setText(_format_setting_number(_coerce_positive_float(settings.get("x_tick_increment"))))
        self.y_tick_increment_edit.setText(_format_setting_number(_coerce_positive_float(settings.get("y_tick_increment"))))
        self.level_count_spin.setValue(int(settings.get("levels", self.level_count_spin.value()) or self.level_count_spin.value()))
        self.title_edit.setText(str(settings.get("title", self.title_edit.text()) or ""))
        self.x_axis_title_edit.setText(str(settings.get("x_axis_title", "") or ""))
        self.y_axis_title_edit.setText(str(settings.get("y_axis_title", self.y_axis_title_edit.text()) or ""))
        self.axis_title_font_spin.setValue(int(settings.get("axis_title_font_size", self.axis_title_font_spin.value()) or 14))
        self.tick_label_font_spin.setValue(int(settings.get("tick_label_font_size", self.tick_label_font_spin.value()) or 12))

    def plot_curve(self) -> ExceedanceCountCurve:
        """Compute and render the count curve."""
        if self._time is None or not self._values_by_case:
            raise ValueError("No target data is available for the count curve.")
        suggested = self._suggest_candidate_range()
        threshold_range = _optional_range_from_edits(
            self.threshold_min_edit,
            self.threshold_max_edit,
            "count-curve target range",
            fallback=suggested,
        )
        if threshold_range is None:
            raise ValueError("Enter a target parameter range before plotting the count curve.")
        _set_range_edits(self.threshold_min_edit, self.threshold_max_edit, threshold_range)
        region_start, region_end = self._region or (None, None)
        result = compute_exceedance_count_curve(
            self._time,
            self._values_by_case,
            threshold_min=threshold_range[0],
            threshold_max=threshold_range[1],
            levels=self.level_count_spin.value(),
            region_start=region_start,
            region_end=region_end,
        )
        self._last_result = result
        self._render_result(result)
        return result

    def export_svg(self, path: str | Path) -> None:
        """Export the current count curve to SVG."""
        if self._last_result is None:
            self.plot_curve()
        elif self.figure is not None and MATPLOTLIB_AVAILABLE:
            self._render_result(self._last_result)
        if self.figure is None or not MATPLOTLIB_AVAILABLE:
            if self._last_result is None:
                raise ValueError("No count curve has been plotted.")
            _export_fallback_svg(
                self._last_result,
                path,
                title=self._plot_title(),
                x_axis_title=self._x_axis_title(),
                y_axis_title=self._y_axis_title(),
                x_range=self._x_axis_range(self._last_result),
                y_range=self._y_axis_range(self._last_result),
                axis_title_font_size=self.axis_title_font_spin.value(),
                tick_label_font_size=self.tick_label_font_spin.value(),
                region_name=self._region_name,
                x_tick_increment=self._x_tick_increment(self._last_result),
                y_tick_increment=self._y_tick_increment(self._last_result),
            )
            return
        self.figure.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.12)

    save_svg = export_svg

    def _build_layout(self) -> None:
        control = QWidget()
        form = QFormLayout(control)
        form.addRow("Target min", self.threshold_min_edit)
        form.addRow("Target max", self.threshold_max_edit)
        form.addRow("Levels", self.level_count_spin)
        form.addRow("Plot title", self.title_edit)
        form.addRow("X-axis title", self.x_axis_title_edit)
        form.addRow("Y-axis title", self.y_axis_title_edit)
        form.addRow("X min", self.x_axis_min_edit)
        form.addRow("X max", self.x_axis_max_edit)
        form.addRow("X tick increment", self.x_tick_increment_edit)
        form.addRow("Y min", self.y_axis_min_edit)
        form.addRow("Y max", self.y_axis_max_edit)
        form.addRow("Y tick increment", self.y_tick_increment_edit)
        form.addRow("Axis title font", self.axis_title_font_spin)
        form.addRow("Tick label font", self.tick_label_font_spin)
        buttons = QHBoxLayout()
        buttons.addWidget(self.plot_button)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.export_csv_button)
        form.addRow(buttons)
        form.addRow(self.status_label)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setWidget(control)
        control_scroll.setMinimumWidth(270)
        control_scroll.setMaximumWidth(360)

        chart_area = QWidget()
        chart_layout = QVBoxLayout(chart_area)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.addWidget(self.canvas)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(control_scroll)
        splitter.addWidget(chart_area)
        splitter.setSizes([300, 900])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        self.plot_button.clicked.connect(self._plot_button_clicked)
        self.export_button.clicked.connect(self._export_button_clicked)
        self.export_csv_button.clicked.connect(self._export_csv_button_clicked)

    def _plot_button_clicked(self) -> None:
        try:
            self.plot_curve()
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _export_button_clicked(self) -> None:
        default_path = (
            self._save_dialog_initial_path("exceedance_count_curve.svg")
            if self._save_dialog_initial_path is not None
            else "exceedance_count_curve.svg"
        )
        path, _ = QFileDialog.getSaveFileName(self, "Export count curve SVG", default_path, "SVG files (*.svg)")
        if not path:
            return
        try:
            self.export_svg(path)
        except Exception as exc:
            self.status_label.setText(f"Export failed: {exc}")
            return
        if self._save_dialog_path_selected is not None:
            self._save_dialog_path_selected(path)
        self.status_label.setText(f"Exported SVG: {path}")

    def _export_csv_button_clicked(self) -> None:
        default_path = (
            self._save_dialog_initial_path("exceedance_count_curve.csv")
            if self._save_dialog_initial_path is not None
            else "exceedance_count_curve.csv"
        )
        path, _ = QFileDialog.getSaveFileName(self, "Export count curve CSV", default_path, "CSV files (*.csv)")
        if not path:
            return
        try:
            self.export_csv(path)
        except Exception as exc:
            self.status_label.setText(f"CSV export failed: {exc}")
            return
        if self._save_dialog_path_selected is not None:
            self._save_dialog_path_selected(path)
        self.status_label.setText(f"Exported CSV: {path}")

    def export_csv(self, path: str | Path) -> None:
        """Export the plotted count curve as column-oriented CSV with a units row."""
        if self._last_result is None:
            self.plot_curve()
        if self._last_result is None:
            raise ValueError("No count curve has been plotted.")
        x_label, x_unit = split_column_unit(self._x_axis_title())
        y_label, y_unit = split_column_unit(self._y_axis_title())
        headers = [x_label or "threshold", y_label or "exceeding_case_count"]
        units = [x_unit, y_unit or "cases"]
        if self._region_name:
            headers = ["region_name", *headers]
            units = ["", *units]
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerow(units)
            for threshold, count in zip(self._last_result.thresholds, self._last_result.counts):
                row = [float(threshold), int(count)]
                writer.writerow([self._region_name, *row] if self._region_name else row)

    def _render_result(self, result: ExceedanceCountCurve) -> None:
        if self.figure is None or not MATPLOTLIB_AVAILABLE:
            self._render_fallback_result(result)
            return
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.plot(result.thresholds, result.counts, color="#2563eb", linewidth=2.0, drawstyle="steps-post")
        axis.set_title(self._plot_title(), fontsize=max(12, self.axis_title_font_spin.value() + 2))
        axis.set_xlabel(self._x_axis_title(), fontsize=self.axis_title_font_spin.value(), labelpad=1)
        axis.set_ylabel(self._y_axis_title(), fontsize=self.axis_title_font_spin.value(), labelpad=2)
        axis.set_xlim(*self._x_axis_range(result))
        axis.set_ylim(*self._y_axis_range(result))
        axis.set_xticks(_axis_ticks(*self._x_axis_range(result), increment=self._x_tick_increment(result)))
        axis.set_yticks(_axis_ticks(*self._y_axis_range(result), increment=self._y_tick_increment(result)))
        axis.tick_params(axis="both", labelsize=self.tick_label_font_spin.value())
        axis.grid(True, alpha=0.25)
        if self._region_name:
            self.figure.text(0.10, 0.92, f"Region: {self._region_name}", fontsize=10, color="#111827")
        self.figure.subplots_adjust(left=0.10, right=0.985, top=0.84 if self._region_name else 0.88, bottom=0.16)
        self.canvas.draw_idle()
        self.status_label.setText(
            f"Plotted {result.thresholds.size} level(s); "
            f"{int(np.nanmax(result.counts)) if result.counts.size else 0} max exceeding case(s)."
        )

    def _render_empty_message(self, message: str) -> None:
        if self.figure is None or not MATPLOTLIB_AVAILABLE:
            self._render_fallback_empty_message(message)
            return
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center", fontsize=12)
        axis.set_axis_off()
        self.canvas.draw_idle()

    def _configure_fallback_plot(self) -> None:
        if not isinstance(self.canvas, pg.PlotWidget):
            return
        self.canvas.setBackground("w")
        self.canvas.showGrid(x=True, y=True, alpha=0.25)
        self.canvas.plotItem.setMenuEnabled(False)

    def _render_fallback_result(self, result: ExceedanceCountCurve) -> None:
        if not isinstance(self.canvas, pg.PlotWidget):
            return
        plot_item = self.canvas.plotItem
        plot_item.clear()
        self._configure_fallback_plot()
        self._apply_fallback_labels(plot_item)
        x_min, x_max = self._x_axis_range(result)
        y_min, y_max = self._y_axis_range(result)

        thresholds = np.asarray(result.thresholds, dtype=float)
        counts = np.asarray(result.counts, dtype=float)
        if thresholds.size == 1:
            plot_item.plot(
                thresholds,
                counts,
                pen=pg.mkPen("#2563eb", width=2.4),
                symbol="o",
                symbolBrush=pg.mkBrush("#2563eb"),
                symbolPen=pg.mkPen("#2563eb"),
            )
        else:
            step_x = np.repeat(thresholds, 2)[1:]
            step_y = np.repeat(counts, 2)[:-1]
            plot_item.plot(step_x, step_y, pen=pg.mkPen("#2563eb", width=2.4))

        plot_item.setXRange(x_min, x_max, padding=0)
        plot_item.setYRange(y_min, y_max, padding=0)
        plot_item.getAxis("bottom").setTicks([_pyqtgraph_ticks(x_min, x_max, self._x_tick_increment(result))])
        plot_item.getAxis("left").setTicks([_pyqtgraph_ticks(y_min, y_max, self._y_tick_increment(result))])
        self.status_label.setText(
            f"Plotted {result.thresholds.size} level(s); "
            f"{int(np.nanmax(result.counts)) if result.counts.size else 0} max exceeding case(s)."
        )

    def _render_fallback_empty_message(self, message: str) -> None:
        if not isinstance(self.canvas, pg.PlotWidget):
            return
        plot_item = self.canvas.plotItem
        plot_item.clear()
        self._configure_fallback_plot()
        self._apply_fallback_labels(plot_item)
        plot_item.setXRange(0.0, 1.0, padding=0)
        plot_item.setYRange(0.0, 1.0, padding=0)
        label = pg.TextItem(message, color="#52525b", anchor=(0.5, 0.5))
        plot_item.addItem(label)
        label.setPos(0.5, 0.5)

    def _apply_fallback_labels(self, plot_item: pg.PlotItem) -> None:
        title = _html_escape(self._plot_title())
        if self._region_name:
            title = (
                f"<span>{title}</span><br>"
                f"<span style='font-size:10pt'>Region: {_html_escape(self._region_name)}</span>"
            )
        plot_item.setTitle(title)
        label_style = {"font-size": f"{self.axis_title_font_spin.value()}pt"}
        tick_font = QFont()
        tick_font.setPointSize(self.tick_label_font_spin.value())
        plot_item.setLabel("bottom", self._x_axis_title(), **label_style)
        plot_item.setLabel("left", self._y_axis_title(), **label_style)
        plot_item.getAxis("bottom").setStyle(tickFont=tick_font)
        plot_item.getAxis("left").setStyle(tickFont=tick_font)

    def _fill_candidate_range_if_blank(self) -> None:
        if self.threshold_min_edit.text().strip() or self.threshold_max_edit.text().strip():
            return
        suggested = self._suggest_candidate_range()
        if suggested is not None:
            _set_range_edits(self.threshold_min_edit, self.threshold_max_edit, suggested)

    def _suggest_candidate_range(self) -> tuple[float, float] | None:
        if self._time is None or not self._values_by_case:
            return None
        region_start, region_end = self._region or (None, None)
        mask = np.isfinite(self._time)
        if region_start is not None:
            mask &= self._time >= float(region_start)
        if region_end is not None:
            mask &= self._time <= float(region_end)
        values: list[np.ndarray] = []
        for array in self._values_by_case.values():
            sample_count = min(self._time.size, array.size)
            if sample_count == 0:
                continue
            local = array[:sample_count]
            local_mask = mask[:sample_count] & np.isfinite(local)
            if local_mask.any():
                values.append(local[local_mask])
        if not values:
            return None
        combined = np.concatenate(values)
        start = float(np.nanmin(combined))
        end = float(np.nanmax(combined))
        if not np.isfinite(start) or not np.isfinite(end):
            return None
        if start == end:
            delta = max(abs(start) * 0.05, 1.0)
            return start - delta, end + delta
        return start, end

    def _x_axis_title(self) -> str:
        return self.x_axis_title_edit.text().strip() or self._default_x_axis_title

    def _y_axis_title(self) -> str:
        return self.y_axis_title_edit.text().strip() or "Number of exceeding cases"

    def _plot_title(self) -> str:
        return self.title_edit.text().strip() or "Cases exceeding candidate threshold"

    def _x_axis_range(self, result: ExceedanceCountCurve) -> tuple[float, float]:
        value = _optional_range_from_edits(self.x_axis_min_edit, self.x_axis_max_edit, "count-curve x-axis range")
        return value or (float(result.thresholds[0]), float(result.thresholds[-1]))

    def _x_tick_increment(self, result: ExceedanceCountCurve) -> float:
        return _optional_positive_float_from_edit(
            self.x_tick_increment_edit,
            "count-curve x-axis tick increment",
        ) or _nice_tick_increment(*self._x_axis_range(result))

    def _y_axis_range(self, result: ExceedanceCountCurve) -> tuple[float, float]:
        value = _optional_range_from_edits(self.y_axis_min_edit, self.y_axis_max_edit, "count-curve y-axis range")
        if value is not None:
            return value
        max_count = max(1.0, float(np.nanmax(result.counts)) if result.counts.size else 1.0)
        return 0.0, max_count * 1.08

    def _y_tick_increment(self, result: ExceedanceCountCurve) -> float:
        return _optional_positive_float_from_edit(
            self.y_tick_increment_edit,
            "count-curve y-axis tick increment",
        ) or _nice_tick_increment(*self._y_axis_range(result))


def _optional_range_from_edits(
    min_edit: QLineEdit,
    max_edit: QLineEdit,
    label: str,
    *,
    fallback: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    minimum = min_edit.text().strip()
    maximum = max_edit.text().strip()
    if not minimum and not maximum:
        return fallback
    if not minimum or not maximum:
        raise ValueError(f"Enter both minimum and maximum for {label}, or leave both blank.")
    try:
        start = float(minimum)
        end = float(maximum)
    except ValueError as exc:
        raise ValueError(f"{label} must use numeric values.") from exc
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError(f"{label} minimum must be less than maximum.")
    return start, end


def _set_range_edits(min_edit: QLineEdit, max_edit: QLineEdit, value_range: tuple[float, float] | None) -> None:
    min_edit.setText("" if value_range is None else f"{value_range[0]:.12g}")
    max_edit.setText("" if value_range is None else f"{value_range[1]:.12g}")


def _optional_positive_float_from_edit(edit: QLineEdit, label: str) -> float | None:
    value = edit.text().strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must use a positive numeric value, or be left blank for auto.") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} must be greater than zero, or be left blank for auto.")
    return parsed


def _coerce_range(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start, end = float(value[0]), float(value[1])
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        return None
    return start, end


def _coerce_positive_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _format_setting_number(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def _nice_tick_increment(start: float, end: float, *, target_ticks: int = 8) -> float:
    span = abs(float(end) - float(start))
    if not np.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(1, target_ticks - 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized <= 1:
        factor = 1
    elif normalized <= 2:
        factor = 2
    elif normalized <= 2.5:
        factor = 2.5
    elif normalized <= 5:
        factor = 5
    else:
        factor = 10
    return float(factor * magnitude)


def _axis_ticks(start: float, end: float, *, increment: float | None = None) -> list[float]:
    start = float(start)
    end = float(end)
    if not np.isfinite(start) or not np.isfinite(end):
        return []
    if start == end:
        return [start]
    if start > end:
        start, end = end, start
    step = float(increment) if increment is not None else _nice_tick_increment(start, end)
    if not np.isfinite(step) or step <= 0:
        step = _nice_tick_increment(start, end)
    first = math.ceil((start / step) - 1e-10) * step
    last = math.floor((end / step) + 1e-10) * step
    if first > last:
        return [start, end]
    count = int(round((last - first) / step)) + 1
    if count <= 0:
        return [start, end]
    if count > 200:
        return _axis_ticks(start, end, increment=_nice_tick_increment(start, end))
    ticks = [first + index * step for index in range(count)]
    return [0.0 if abs(value) < step * 1e-10 else float(value) for value in ticks]


def _pyqtgraph_ticks(start: float, end: float, increment: float | None) -> list[tuple[float, str]]:
    return [(tick, _format_number(tick)) for tick in _axis_ticks(start, end, increment=increment)]


def _export_fallback_svg(
    result: ExceedanceCountCurve,
    path: str | Path,
    *,
    title: str,
    x_axis_title: str,
    y_axis_title: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    axis_title_font_size: int,
    tick_label_font_size: int,
    region_name: str = "",
    x_tick_increment: float | None = None,
    y_tick_increment: float | None = None,
) -> None:
    width = 1200
    height = 650
    plot_x = 108
    plot_y = 118
    plot_width = 1030
    plot_height = 430
    x_tick_label_y = plot_y + plot_height + 28
    x_axis_title_y = plot_y + plot_height + 64
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_span = max(abs(x_max - x_min), 1.0)
    y_span = max(abs(y_max - y_min), 1.0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="28" y="42" font-family="Arial" font-size="24">{_xml_escape(title)}</text>',
    ]
    if region_name:
        lines.append(f'<text x="28" y="72" font-family="Arial" font-size="14">Region: {_xml_escape(region_name)}</text>')
    lines.extend(
        [
            f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#d4d4d8"/>',
            f'<line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height}" stroke="#111827"/>',
            f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_height}" stroke="#111827"/>',
            f'<text x="{plot_x + plot_width / 2:.1f}" y="{x_axis_title_y:.1f}" font-family="Arial" font-size="{axis_title_font_size}" text-anchor="middle">{_xml_escape(x_axis_title)}</text>',
            f'<text x="34" y="{plot_y + plot_height / 2:.1f}" font-family="Arial" font-size="{axis_title_font_size}" transform="rotate(-90 34 {plot_y + plot_height / 2:.1f})" text-anchor="middle">{_xml_escape(y_axis_title)}</text>',
        ]
    )
    for value in _axis_ticks(y_min, y_max, increment=y_tick_increment):
        y = plot_y + plot_height - ((value - y_min) / y_span) * plot_height
        lines.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_width}" y2="{y:.1f}" stroke="#e4e4e7"/>')
        lines.append(
            f'<text x="{plot_x - 10}" y="{y + 5:.1f}" font-family="Arial" font-size="{tick_label_font_size}" text-anchor="end">{_format_number(value)}</text>'
        )
    for value in _axis_ticks(x_min, x_max, increment=x_tick_increment):
        x = plot_x + ((value - x_min) / x_span) * plot_width
        lines.append(f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_y + plot_height}" stroke="#f1f5f9"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{x_tick_label_y:.1f}" font-family="Arial" font-size="{tick_label_font_size}" text-anchor="middle">{_format_number(value)}</text>'
        )
    points: list[str] = []
    thresholds = np.asarray(result.thresholds, dtype=float)
    counts = np.asarray(result.counts, dtype=float)
    if thresholds.size > 1:
        thresholds = np.repeat(thresholds, 2)[1:]
        counts = np.repeat(counts, 2)[:-1]
    for x_value, y_value in zip(thresholds, counts):
        if x_min <= x_value <= x_max:
            x = plot_x + ((float(x_value) - x_min) / x_span) * plot_width
            y = plot_y + plot_height - ((float(y_value) - y_min) / y_span) * plot_height
            points.append(f"{x:.1f},{y:.1f}")
    if points:
        lines.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{" ".join(points)}"/>')
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.5g}"


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html_escape(value: str) -> str:
    return _xml_escape(value).replace('"', "&quot;").replace("'", "&#39;")


__all__ = ["ExceedanceCountCurveWidget"]
