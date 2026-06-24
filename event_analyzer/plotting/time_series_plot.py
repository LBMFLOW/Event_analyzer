from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt6.QtCore import QPoint, QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QCursor, QPen
from PyQt6.QtWidgets import QFileDialog, QLabel, QMenu, QSlider, QVBoxLayout, QWidget

from event_analyzer.data.downsample import min_max_downsample
from event_analyzer.plotting.colors import color_for_index
from event_analyzer.plotting.time_axis_item import TimeAxisItem


@dataclass(slots=True)
class CurveHandle:
    name: str
    item: pg.PlotDataItem
    color: str
    axis_id: str
    role: str
    dashed: bool
    visible: bool = True


@dataclass(slots=True)
class DividerHandle:
    divider_id: str
    line: pg.InfiniteLine
    label: pg.TextItem


class TimeSeriesPlotWidget(QWidget):
    """Interactive PyQtGraph time-series plot for targets and auxiliaries."""

    slider_moved = pyqtSignal(float)
    plot_clicked = pyqtSignal(float, float)
    divider_moved = pyqtSignal(str, float)
    threshold_moved = pyqtSignal(float)
    request_add_divider = pyqtSignal(float)
    request_add_threshold = pyqtSignal(float)

    # Compatibility signals for the earlier skeleton controllers.
    context_requested = pyqtSignal(float, float, QPoint)
    region_clicked = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None, *, max_plot_points: int = 12_000) -> None:
        super().__init__(parent)
        self.time_axis_item = TimeAxisItem()
        self.plot_widget = pg.PlotWidget(axisItems={"bottom": self.time_axis_item})
        self.plotItem = self.plot_widget.plotItem
        self.legend = self.plotItem.addLegend(offset=(12, 12))
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.readout_label = QLabel("Time: - | Case: - | Value: -")

        self._max_plot_points = max_plot_points
        self._time: np.ndarray = np.asarray([], dtype=float)
        self._source_valid_time_mask: np.ndarray | None = None
        self._source_order: np.ndarray | None = None
        self._active_case: str | None = None
        self._title = "Time-series plot"
        self._time_label = "Time"
        self._curves: dict[str, CurveHandle] = {}
        self._target_values: dict[str, np.ndarray] = {}
        self._auxiliary_values: dict[str, np.ndarray] = {}
        self._display_threshold: float | None = None
        self._display_region: tuple[float | None, float | None] | None = None
        self._legend_visible = True
        self._extra_axes: dict[str, pg.AxisItem] = {}
        self._extra_views: dict[str, pg.ViewBox] = {}
        self._dividers: dict[str, DividerHandle] = {}
        self._threshold_line: pg.InfiniteLine | None = None
        self._updating_slider = False

        self._tracer_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen("#27272a", width=1.4, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen("#111827", width=2),
        )
        self._selected_region = pg.LinearRegionItem(
            values=(0.0, 1.0),
            movable=False,
            brush=(37, 99, 235, 40),
            pen=pg.mkPen("#2563eb", width=1),
        )
        self._selected_region.setZValue(-10)
        self._trace_marker = pg.ScatterPlotItem(
            size=9,
            brush=pg.mkBrush("#facc15"),
            pen=pg.mkPen("#92400e", width=1.2),
        )
        self._trace_marker.setZValue(40)
        self._trace_label = pg.TextItem(
            text="",
            color="#111827",
            anchor=(0, 1),
            fill=pg.mkBrush("#fef08a"),
            border=pg.mkPen("#a16207", width=1),
        )
        self._trace_label.setZValue(41)
        self._trace_x_axis_label = pg.TextItem(
            text="",
            color="#111827",
            anchor=(0.5, 0),
            fill=pg.mkBrush("#fef08a"),
            border=pg.mkPen("#a16207", width=1),
        )
        self._trace_x_axis_label.setZValue(42)
        self._trace_y_axis_label = pg.TextItem(
            text="",
            color="#111827",
            anchor=(0, 0.5),
            fill=pg.mkBrush("#fef08a"),
            border=pg.mkPen("#a16207", width=1),
        )
        self._trace_y_axis_label.setZValue(42)
        self._trace_annotation_position: tuple[float, float] | None = None

        self._configure_plot()
        self._build_layout()
        self._connect_signals()

    def set_data(
        self,
        time: Sequence[float],
        values_by_case: Mapping[str, Sequence[float]],
        *,
        time_label: str = "Time",
        colors: Mapping[str, str] | None = None,
        clear_auxiliary: bool = True,
    ) -> None:
        """Set target/case data and redraw target curves on y1."""
        self.clear_curves(role="target")
        if clear_auxiliary:
            self.clear_curves(role="auxiliary")
            self._auxiliary_values.clear()

        self._set_time(time, time_label=time_label)
        self._target_values.clear()
        for index, (name, values) in enumerate(values_by_case.items()):
            aligned = self._align_values(values, name=name)
            self._target_values[name] = aligned
            self._add_curve(
                name=name,
                values=aligned,
                color=(colors or {}).get(name, color_for_index(index)),
                axis_id="y1",
                role="target",
                dashed=False,
            )

        if self._active_case not in self._target_values:
            self._active_case = next(iter(self._target_values), None)
        self.reset_view()
        if self._time.size:
            self.set_slider_time(float(self._time[0]))

    def set_auxiliary_data(
        self,
        auxiliary_values: Mapping[str, Sequence[float]],
        axis_assignments: Mapping[str, str] | None = None,
        *,
        colors: Mapping[str, str] | None = None,
    ) -> None:
        """Set auxiliary parameter curves assigned to y1, y2, y3, etc."""
        self.clear_curves(role="auxiliary")
        self._auxiliary_values.clear()

        assignments = axis_assignments or {}
        color_offset = len(self._target_values)
        for index, (name, values) in enumerate(auxiliary_values.items()):
            axis_id = assignments.get(name, f"y{index + 2}")
            aligned = self._align_values(values, name=name)
            self._auxiliary_values[name] = aligned
            self._add_curve(
                name=name,
                values=aligned,
                color=(colors or {}).get(name, color_for_index(color_offset + index)),
                axis_id=axis_id,
                role="auxiliary",
                dashed=True,
            )

    def set_active_case(self, case_name: str | None) -> None:
        """Set the target used for slider/tracer readout."""
        if case_name in self._target_values:
            self._active_case = case_name
        elif self._target_values:
            self._active_case = next(iter(self._target_values), None)
        else:
            self._active_case = None
        self._update_readout(self._current_slider_time())

    def set_slider_time(self, time_value: float) -> None:
        """Move the horizontal slider and vertical tracer to a time value."""
        if not self._time.size:
            return
        clamped = min(float(self._time[-1]), max(float(self._time[0]), float(time_value)))
        self._set_tracer_time(clamped)
        self._set_slider_from_time(clamped)
        self._update_readout(clamped)

    def add_divider(self, time_value: float, divider_id: str | None = None, label: str | None = None) -> str:
        """Add a draggable vertical divider and return its id."""
        actual_id = divider_id or f"D-{uuid4().hex[:8]}"
        line = pg.InfiniteLine(
            pos=float(time_value),
            angle=90,
            movable=True,
            pen=pg.mkPen("#52525b", width=1.4, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen("#111827", width=2),
        )
        text = pg.TextItem(label or actual_id, color="#111827", anchor=(0, 1))
        self.plotItem.addItem(line, ignoreBounds=True)
        self.plotItem.addItem(text, ignoreBounds=True)
        handle = DividerHandle(divider_id=actual_id, line=line, label=text)
        self._dividers[actual_id] = handle
        line.sigPositionChangeFinished.connect(lambda moved_line, did=actual_id: self._divider_line_moved(did, moved_line))
        self._update_divider_label(handle)
        return actual_id

    def update_divider(self, divider_id: str, time_value: float, label: str | None = None) -> None:
        """Move an existing divider to a new time."""
        handle = self._dividers[divider_id]
        handle.line.setValue(float(time_value))
        if label is not None:
            handle.label.setText(label)
        self._update_divider_label(handle)

    def remove_divider(self, divider_id: str) -> None:
        """Remove a divider by id."""
        handle = self._dividers.pop(divider_id)
        self.plotItem.removeItem(handle.line)
        self.plotItem.removeItem(handle.label)

    def set_selected_region(self, region_start: float | None, region_end: float | None) -> None:
        """Highlight the selected analysis region."""
        self._display_region = (region_start, region_end)
        if region_start is None or region_end is None:
            self._selected_region.hide()
            self._refresh_display_curves(role="target")
            return
        start, end = sorted((float(region_start), float(region_end)))
        self._selected_region.setRegion((start, end))
        self._selected_region.show()
        self._refresh_display_curves(role="target")

    def set_threshold(self, value: float | None) -> None:
        """Set, move, or remove the horizontal threshold line."""
        self._display_threshold = None if value is None else float(value)
        if value is None:
            if self._threshold_line is not None:
                self.plotItem.removeItem(self._threshold_line)
                self._threshold_line = None
            self._refresh_display_curves(role="target")
            return

        if self._threshold_line is None:
            self._threshold_line = pg.InfiniteLine(
                pos=float(value),
                angle=0,
                movable=True,
                pen=pg.mkPen("#b91c1c", width=1.7),
                hoverPen=pg.mkPen("#7f1d1d", width=2.2),
            )
            self._threshold_line.sigPositionChangeFinished.connect(self._threshold_line_moved)
            self.plotItem.addItem(self._threshold_line, ignoreBounds=True)
        else:
            self._threshold_line.setValue(float(value))
        self._refresh_display_curves(role="target")

    def highlight_exceeding_cases(self, case_names: Sequence[str]) -> None:
        """Emphasize target curves whose values exceed the active threshold."""
        exceeding = set(case_names)
        for name, handle in self._curves.items():
            if handle.role != "target":
                continue
            handle.item.setPen(self._make_pen(handle.color, emphasized=name in exceeding, dashed=handle.dashed))

    def export_svg(self, path: str | Path) -> None:
        """Export the current PyQtGraph plot state as SVG."""
        exporter = pg.exporters.SVGExporter(self.plotItem)
        exporter.export(str(path))

    def set_legend_visible(self, visible: bool) -> None:
        """Show or hide the plot legend without changing plotted curves."""
        self._legend_visible = bool(visible)
        self.legend.setVisible(self._legend_visible)

    def set_title(self, title: str) -> None:
        """Set the plot title shown in the widget and SVG export."""
        self._title = title
        self.plotItem.setTitle(title)

    def reset_view(self) -> None:
        """Auto-range all visible axes."""
        self.plotItem.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plotItem.autoRange()
        for view in self._extra_views.values():
            view.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            view.autoRange()
        self._update_divider_labels()

    def set_curve_visible(self, name: str, visible: bool) -> None:
        """Toggle a target or auxiliary curve."""
        handle = self._curves.get(name)
        if handle is None:
            return
        handle.visible = visible
        handle.item.setVisible(visible)

    def set_curve_color(self, name: str, color: str) -> None:
        """Update a curve color without rebuilding the whole plot."""
        handle = self._curves.get(name)
        if handle is None:
            return
        handle.color = color
        handle.item.setPen(self._make_pen(color, emphasized=False, dashed=handle.dashed))

    def set_curve_emphasized(self, name: str, emphasized: bool) -> None:
        """Compatibility helper used by older plot controllers."""
        handle = self._curves.get(name)
        if handle is not None:
            handle.item.setPen(self._make_pen(handle.color, emphasized=emphasized, dashed=handle.dashed))

    def set_tracer_time(self, value: float | None) -> None:
        """Compatibility helper for earlier code."""
        if value is None:
            self._tracer_line.hide()
            self._hide_trace_annotation()
        else:
            self.set_slider_time(value)

    def set_time_axis(self, axis: Any) -> None:
        """Compatibility helper for earlier code that supplies a TimeAxis."""
        self.time_axis_item.set_time_axis(axis)
        self._time_label = axis.name
        self.plotItem.setLabel("bottom", axis.name)

    def clear_curves(self, role: str | None = None) -> None:
        """Remove target, auxiliary, or all curves."""
        for name, handle in list(self._curves.items()):
            if role is not None and handle.role != role:
                continue
            if handle.axis_id == "y1":
                self.plotItem.removeItem(handle.item)
            else:
                view = self._extra_views.get(handle.axis_id)
                if view is not None:
                    view.removeItem(handle.item)
            try:
                self.legend.removeItem(handle.item)
            except Exception:
                pass
            self._curves.pop(name, None)

    def add_curve(
        self,
        name: str,
        x: Sequence[float],
        y: Sequence[float],
        *,
        color: str,
        axis_id: str = "y1",
        dashed: bool = False,
        visible: bool = True,
    ) -> None:
        """Compatibility helper for older PlotController code."""
        if not self._time.size:
            self._set_time(x, time_label=self._time_label)
        aligned = self._align_values(y, name=name)
        role = "target" if not dashed else "auxiliary"
        if role == "target":
            self._target_values[name] = aligned
        else:
            self._auxiliary_values[name] = aligned
        self._add_curve(name=name, values=aligned, color=color, axis_id=axis_id, role=role, dashed=dashed)
        self.set_curve_visible(name, visible)

    def view_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        x_range, y_range = self.plotItem.vb.viewRange()
        return (float(x_range[0]), float(x_range[1])), (float(y_range[0]), float(y_range[1]))

    def _configure_plot(self) -> None:
        self.plot_widget.setBackground("w")
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plotItem.setTitle(self._title)
        self.plotItem.setLabel("bottom", self._time_label)
        self.plotItem.setLabel("left", "y1")
        self.plotItem.setMenuEnabled(False)
        self.plotItem.addItem(self._tracer_line, ignoreBounds=True)
        self.plotItem.addItem(self._trace_marker, ignoreBounds=True)
        self.plotItem.addItem(self._trace_label, ignoreBounds=True)
        self.plotItem.addItem(self._trace_x_axis_label, ignoreBounds=True)
        self.plotItem.addItem(self._trace_y_axis_label, ignoreBounds=True)
        self.plotItem.addItem(self._selected_region, ignoreBounds=True)
        self._tracer_line.hide()
        self._trace_marker.hide()
        self._trace_label.hide()
        self._trace_x_axis_label.hide()
        self._trace_y_axis_label.hide()
        self._selected_region.hide()
        self.time_slider.setEnabled(False)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.time_slider)
        layout.addWidget(self.readout_label)

    def _connect_signals(self) -> None:
        self.time_slider.valueChanged.connect(self._slider_value_changed)
        self._tracer_line.sigPositionChanged.connect(self._tracer_line_moved)
        self.scene().sigMouseClicked.connect(self._scene_mouse_clicked)
        self.plotItem.vb.sigResized.connect(self._update_extra_views)
        self.plotItem.vb.sigRangeChanged.connect(self._plot_range_changed)

    def _set_time(self, time: Sequence[float], *, time_label: str) -> None:
        raw = np.asarray(time, dtype=float)
        if raw.ndim != 1:
            raise ValueError("time must be a 1D array-like object.")
        valid = np.isfinite(raw)
        if not valid.any():
            raise ValueError("time contains no finite values.")
        order = np.argsort(raw[valid], kind="mergesort")
        self._source_valid_time_mask = valid
        self._source_order = order
        self._time = raw[valid][order]
        self._time_label = time_label
        self.time_axis_item.set_time_axis(None)
        self.plotItem.setLabel("bottom", time_label)
        self._configure_slider()

    def _align_values(self, values: Sequence[float], *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"Series '{name}' must be a 1D array-like object.")
        if array.size == self._time.size:
            return array
        if self._source_valid_time_mask is not None and self._source_order is not None and array.size == self._source_valid_time_mask.size:
            return array[self._source_valid_time_mask][self._source_order]
        raise ValueError(f"Series '{name}' has {array.size} values for {self._time.size} time samples.")

    def _add_curve(self, *, name: str, values: np.ndarray, color: str, axis_id: str, role: str, dashed: bool) -> None:
        x_plot, y_plot = self._display_xy_for_curve(values, role=role)
        item = pg.PlotDataItem(
            x_plot,
            y_plot,
            pen=self._make_pen(color, emphasized=False, dashed=dashed),
            name=name,
            skipFiniteCheck=True,
        )
        if axis_id == "y1":
            self.plotItem.addItem(item)
        else:
            self._ensure_extra_axis(axis_id).addItem(item)
        self.legend.addItem(item, name)
        self._curves[name] = CurveHandle(name=name, item=item, color=color, axis_id=axis_id, role=role, dashed=dashed)

    def _display_xy_for_curve(self, values: np.ndarray, *, role: str) -> tuple[np.ndarray, np.ndarray]:
        threshold = self._display_threshold if role == "target" else None
        region = self._display_region if role == "target" else None
        return min_max_downsample(
            self._time,
            values,
            self._max_plot_points,
            threshold=threshold,
            region=region,
        )

    def _refresh_display_curves(self, role: str | None = None) -> None:
        """Recompute lightweight display buffers from full-resolution arrays."""
        if not self._time.size:
            return
        for name, handle in self._curves.items():
            if role is not None and handle.role != role:
                continue
            source = self._target_values.get(name) if handle.role == "target" else self._auxiliary_values.get(name)
            if source is None:
                continue
            x_plot, y_plot = self._display_xy_for_curve(source, role=handle.role)
            handle.item.setData(x_plot, y_plot)

    def _ensure_extra_axis(self, axis_id: str) -> pg.ViewBox:
        if axis_id in self._extra_views:
            return self._extra_views[axis_id]

        axis_number = _axis_number(axis_id)
        orientation = "right" if axis_number % 2 == 0 else "left"
        axis = pg.AxisItem(orientation)
        axis.setLabel(axis_id)
        view = pg.ViewBox()
        self.plotItem.scene().addItem(view)
        axis.linkToView(view)
        view.setXLink(self.plotItem)

        # PyQtGraph's PlotItem layout does not support inserting new columns
        # before the built-in left axis after construction. Extra axes are
        # therefore placed in additional columns to the right; their tick
        # orientation alternates left/right so y2, y3, y4 remain visually distinct.
        column = 3 + len(self._extra_axes)
        self.plotItem.layout.addItem(axis, 2, column)
        self._extra_axes[axis_id] = axis
        self._extra_views[axis_id] = view
        self._update_extra_views()
        return view

    def _update_extra_views(self) -> None:
        rect = self.plotItem.vb.sceneBoundingRect()
        for view in self._extra_views.values():
            view.setGeometry(rect)
            view.linkedViewChanged(self.plotItem.vb, view.XAxis)

    def _configure_slider(self) -> None:
        self.time_slider.setEnabled(self._time.size > 0)
        self.time_slider.setRange(0, 10_000)
        with QSignalBlocker(self.time_slider):
            self.time_slider.setValue(0)

    def _slider_value_changed(self, value: int) -> None:
        if self._updating_slider or not self._time.size:
            return
        time_value = self._time_from_slider_value(value)
        self._set_tracer_time(time_value)
        self._update_readout(time_value)
        self.slider_moved.emit(time_value)

    def _tracer_line_moved(self, line: pg.InfiniteLine) -> None:
        if self._updating_slider or not self._time.size:
            return
        time_value = min(float(self._time[-1]), max(float(self._time[0]), float(line.value())))
        self._set_slider_from_time(time_value)
        self._update_readout(time_value)
        self.slider_moved.emit(time_value)

    def _set_tracer_time(self, time_value: float) -> None:
        self._updating_slider = True
        try:
            self._tracer_line.setValue(float(time_value))
            self._tracer_line.show()
        finally:
            self._updating_slider = False

    def _set_slider_from_time(self, time_value: float) -> None:
        if not self._time.size:
            return
        start = float(self._time[0])
        end = float(self._time[-1])
        span = end - start
        slider_value = 0 if span <= 0 else round(((time_value - start) / span) * self.time_slider.maximum())
        with QSignalBlocker(self.time_slider):
            self.time_slider.setValue(int(min(self.time_slider.maximum(), max(0, slider_value))))

    def _time_from_slider_value(self, slider_value: int) -> float:
        start = float(self._time[0])
        end = float(self._time[-1])
        fraction = slider_value / max(1, self.time_slider.maximum())
        return start + fraction * (end - start)

    def _current_slider_time(self) -> float:
        if not self._time.size:
            return float("nan")
        return self._time_from_slider_value(self.time_slider.value())

    def _update_readout(self, time_value: float) -> None:
        case_name = self._active_case or "-"
        value = np.nan
        if self._active_case in self._target_values:
            value = _interpolate_at(self._time, self._target_values[self._active_case], time_value)
        self.readout_label.setText(
            f"Time: {_format_float(time_value)} | Case: {case_name} | Value: {_format_float(value)}"
        )
        self._update_trace_annotation(time_value, value, case_name)

    def _update_trace_annotation(self, time_value: float, value: float, case_name: str) -> None:
        if not np.isfinite(time_value) or not np.isfinite(value) or case_name == "-":
            self._hide_trace_annotation()
            return
        self._trace_annotation_position = (float(time_value), float(value))
        self._trace_marker.setData([time_value], [value])
        self._trace_label.setText(
            f"{case_name}\nx: {_format_float(time_value)}\ny: {_format_float(value)}"
        )
        self._trace_label.setPos(float(time_value), float(value))
        self._trace_x_axis_label.setText(f"x: {_format_float(time_value)}")
        self._trace_y_axis_label.setText(f"y: {_format_float(value)}")
        self._position_trace_axis_labels()
        self._trace_marker.show()
        self._trace_label.show()
        self._trace_x_axis_label.show()
        self._trace_y_axis_label.show()

    def _hide_trace_annotation(self) -> None:
        self._trace_annotation_position = None
        self._trace_marker.hide()
        self._trace_label.hide()
        self._trace_x_axis_label.hide()
        self._trace_y_axis_label.hide()

    def _scene_mouse_clicked(self, event) -> None:
        if not self.plotItem.vb.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.plotItem.vb.mapSceneToView(event.scenePos())
        x = float(point.x())
        y = float(point.y())
        if event.button() == Qt.MouseButton.RightButton:
            screen_pos = QCursor.pos()
            if hasattr(event, "screenPos"):
                screen_pos = event.screenPos().toPoint()
            self.context_requested.emit(x, y, screen_pos)
            self._show_context_menu(x, y, screen_pos)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            self.plot_clicked.emit(x, y)
            self.region_clicked.emit(x)

    def _show_context_menu(self, x: float, y: float, screen_pos: QPoint) -> None:
        menu = QMenu(self)
        add_divider_action = QAction("Add divider here", menu)
        add_threshold_action = QAction("Add threshold here", menu)
        legend_action = QAction("Show legend", menu)
        legend_action.setCheckable(True)
        legend_action.setChecked(self._legend_visible)
        reset_view_action = QAction("Reset view", menu)
        export_svg_action = QAction("Export SVG", menu)
        menu.addAction(add_divider_action)
        menu.addAction(add_threshold_action)
        menu.addAction(legend_action)
        menu.addSeparator()
        menu.addAction(reset_view_action)
        menu.addAction(export_svg_action)

        action = menu.exec(screen_pos)
        if action == add_divider_action:
            self.request_add_divider.emit(x)
        elif action == add_threshold_action:
            self.request_add_threshold.emit(y)
        elif action == legend_action:
            self.set_legend_visible(legend_action.isChecked())
        elif action == reset_view_action:
            self.reset_view()
        elif action == export_svg_action:
            self._export_svg_dialog()

    def _export_svg_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export plot SVG", "time_series.svg", "SVG files (*.svg)")
        if path:
            self.export_svg(path)

    def _divider_line_moved(self, divider_id: str, line: pg.InfiniteLine) -> None:
        handle = self._dividers.get(divider_id)
        if handle is None:
            return
        self._update_divider_label(handle)
        self.divider_moved.emit(divider_id, float(line.value()))

    def _threshold_line_moved(self, line: pg.InfiniteLine) -> None:
        value = float(line.value())
        self.threshold_moved.emit(value)

    def _plot_range_changed(self, *_args) -> None:
        self._update_divider_labels()
        self._position_trace_axis_labels()

    def _position_trace_axis_labels(self) -> None:
        if self._trace_annotation_position is None:
            return
        time_value, value = self._trace_annotation_position
        x_range, y_range = self.view_range()
        x_span = max(abs(x_range[1] - x_range[0]), 1.0)
        y_span = max(abs(y_range[1] - y_range[0]), 1.0)
        self._trace_x_axis_label.setPos(time_value, y_range[0] + y_span * 0.025)
        self._trace_y_axis_label.setPos(x_range[0] + x_span * 0.01, value)

    def _update_divider_labels(self) -> None:
        for handle in self._dividers.values():
            self._update_divider_label(handle)

    def _update_divider_label(self, handle: DividerHandle) -> None:
        _, y_range = self.view_range()
        handle.label.setPos(float(handle.line.value()), y_range[1])

    def scene(self):
        return self.plot_widget.scene()

    def _make_pen(self, color: str, *, emphasized: bool, dashed: bool) -> QPen:
        style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        return pg.mkPen(color, width=3.2 if emphasized else 1.7, style=style)


def _axis_number(axis_id: str) -> int:
    try:
        return int(axis_id.lower().removeprefix("y"))
    except ValueError:
        return 2


def _interpolate_at(time: np.ndarray, values: np.ndarray, x: float) -> float:
    finite = np.isfinite(time) & np.isfinite(values)
    if finite.sum() == 0:
        return float("nan")
    time_f = time[finite]
    values_f = values[finite]
    if x <= time_f[0]:
        return float(values_f[0])
    if x >= time_f[-1]:
        return float(values_f[-1])
    return float(np.interp(x, time_f, values_f))


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    if abs(value) >= 1000 or abs(value) < 0.01:
        return f"{value:.6g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


# Backward-compatible name used by earlier controllers in this project.
TimeSeriesPlot = TimeSeriesPlotWidget

__all__ = ["TimeSeriesPlotWidget", "TimeSeriesPlot"]
