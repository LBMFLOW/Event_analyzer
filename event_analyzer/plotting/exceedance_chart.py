from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from event_analyzer.analysis.exceedance import ExceedanceEvent
from event_analyzer.plotting.colors import color_for_index

try:  # Matplotlib is part of the normal app requirements, but keep imports graceful.
    from matplotlib import colormaps
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only in partial installations.
    colormaps = None
    FigureCanvasQTAgg = None
    Figure = None
    MATPLOTLIB_AVAILABLE = False


class ExceedanceBarChartWidget(QWidget):
    """Grouped bar chart for contiguous threshold exceedance durations.

    Matplotlib is used when available because grouped categorical bars, labels,
    click picking, and SVG export are more reliable there. A PyQtGraph fallback
    renders the same grouped bars in partial installations so the UI never shows
    only a placeholder for valid exceedance events.
    """

    event_selected = pyqtSignal(str, int, float, float, float, float)
    peak_time_selected = pyqtSignal(float)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        time_unit: str = "time units",
        show_value_labels: bool = True,
        value_label_limit: int = 60,
    ) -> None:
        super().__init__(parent)
        self.time_unit = time_unit
        self.show_value_labels = show_value_labels
        self.value_label_limit = value_label_limit
        self.figure = Figure(figsize=(9, 3.8), tight_layout=True) if MATPLOTLIB_AVAILABLE else None
        self.canvas = FigureCanvasQTAgg(self.figure) if MATPLOTLIB_AVAILABLE else pg.PlotWidget()
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setWidget(self.canvas)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self._events: list[ExceedanceEvent] = []
        self._patch_events: dict[object, ExceedanceEvent] = {}
        self._fallback_bar_bounds: list[tuple[float, float, float, ExceedanceEvent]] = []
        self._plot_adapter = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)
        layout.addWidget(self.status_label)

        if MATPLOTLIB_AVAILABLE:
            self.canvas.mpl_connect("pick_event", self._bar_picked)
        elif isinstance(self.canvas, pg.PlotWidget):
            self._configure_fallback_plot()
            self.canvas.scene().sigMouseClicked.connect(self._fallback_mouse_clicked)
        self.set_events([])

    @property
    def events(self) -> list[ExceedanceEvent]:
        return list(self._events)

    def set_time_unit(self, time_unit: str) -> None:
        """Update the duration unit displayed on the y-axis."""
        self.time_unit = time_unit
        self.set_events(self._events)

    def set_plot_adapter(self, plot_adapter) -> None:
        """Attach an optional plot object with ``set_slider_time(float)``."""
        self._plot_adapter = plot_adapter

    def set_events(self, events: list[ExceedanceEvent]) -> None:
        """Render grouped duration bars from exceedance events."""
        self._events = sorted(events, key=lambda event: (event.case_name, event.event_index, event.start_time))
        self._patch_events.clear()
        self._fallback_bar_bounds.clear()
        if not MATPLOTLIB_AVAILABLE or self.figure is None:
            self._render_pyqtgraph_fallback()
            return
        self.figure.clear()
        axis = self.figure.add_subplot(111)

        if not self._events:
            self._set_canvas_width(case_count=1)
            axis.set_title("Exceedance durations")
            axis.set_xlabel("Case")
            axis.set_ylabel(f"Duration above threshold ({self.time_unit})")
            axis.text(0.5, 0.5, "No exceedance events", transform=axis.transAxes, ha="center", va="center")
            axis.grid(axis="y", alpha=0.25)
            self.status_label.setText("No exceedance events")
            self.canvas.draw_idle()
            return

        grouped = _group_events_by_case(self._events)
        cases = list(grouped)
        max_events_for_case = max(len(case_events) for case_events in grouped.values())
        x_positions = np.arange(len(cases), dtype=float)
        width = min(0.82 / max_events_for_case, 0.24)
        colors = colormaps["tab20"].colors
        total_bars = len(self._events)
        show_labels = self.show_value_labels and total_bars <= self.value_label_limit

        self._set_canvas_width(case_count=len(cases))

        for event_slot in range(max_events_for_case):
            slot_events: list[ExceedanceEvent | None] = []
            heights: list[float] = []
            for case in cases:
                case_events = grouped[case]
                event = case_events[event_slot] if event_slot < len(case_events) else None
                slot_events.append(event)
                heights.append(0.0 if event is None else event.duration)

            offset = (event_slot - (max_events_for_case - 1) / 2) * width
            bars = axis.bar(
                x_positions + offset,
                heights,
                width=width,
                label=f"Event {event_slot + 1}",
                color=colors[event_slot % len(colors)],
                picker=True,
            )

            for bar, event in zip(bars, slot_events):
                if event is None:
                    bar.set_alpha(0.0)
                    bar.set_picker(False)
                    continue
                self._patch_events[bar] = event
                if show_labels:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        _format_number(event.duration),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        axis.set_title("Exceedance durations by case")
        axis.set_xlabel("Case")
        axis.set_ylabel(f"Duration above threshold ({self.time_unit})")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [_short_case_label(case, max_length=_axis_label_length(len(cases))) for case in cases],
            rotation=_label_rotation(len(cases)),
            ha="right",
            rotation_mode="anchor",
            fontsize=_axis_label_font_size(len(cases)),
        )
        axis.grid(axis="y", alpha=0.25)
        axis.margins(x=0.02)
        self.figure.set_tight_layout(False)
        self.figure.subplots_adjust(
            left=0.075,
            right=0.99,
            top=0.88,
            bottom=_bottom_margin(len(cases)),
        )

        if max_events_for_case <= 8:
            axis.legend(loc="best", fontsize=8, ncols=2 if max_events_for_case > 4 else 1)
        else:
            axis.text(
                0.99,
                0.98,
                "Bars are ordered by event index within each case",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )

        self.status_label.setText(
            f"{len(self._events)} events across {len(cases)} cases. "
            "Click a bar to select its event and move the plot tracer to peak time."
        )
        self.canvas.draw_idle()

    def export_svg(self, path: str | Path) -> None:
        """Save the current bar chart as SVG."""
        if not MATPLOTLIB_AVAILABLE or self.figure is None:
            _export_fallback_svg(self._events, path)
            return
        self.figure.savefig(path, format="svg")

    save_svg = export_svg

    def export_csv(self, path: str | Path) -> None:
        """Export the current event table to CSV using the stdlib csv module."""
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "case_name",
                    "event_index",
                    "start_time",
                    "end_time",
                    "duration",
                    "peak_value",
                    "peak_time",
                    "threshold",
                    "region_start",
                    "region_end",
                ]
            )
            for event in self._events:
                writer.writerow(
                    [
                        event.case_name,
                        event.event_index,
                        event.start_time,
                        event.end_time,
                        event.duration,
                        event.peak_value,
                        event.peak_time,
                        event.threshold,
                        event.region_start,
                        event.region_end,
                    ]
                )

    def _bar_picked(self, pick_event) -> None:
        event = self._patch_events.get(pick_event.artist)
        if event is None:
            return
        self._select_event(event)

    def _select_event(self, event: ExceedanceEvent) -> None:
        self.status_label.setText(
            f"Selected {event.case_name} event {event.event_index}: "
            f"{_format_number(event.duration)} {self.time_unit}, peak {_format_number(event.peak_value)}"
        )
        self.event_selected.emit(
            event.case_name,
            event.event_index,
            event.start_time,
            event.end_time,
            event.duration,
            event.peak_value,
        )
        self.peak_time_selected.emit(event.peak_time)
        if self._plot_adapter is not None and hasattr(self._plot_adapter, "set_slider_time"):
            self._plot_adapter.set_slider_time(event.peak_time)

    def _configure_fallback_plot(self) -> None:
        if not isinstance(self.canvas, pg.PlotWidget):
            return
        self.canvas.setBackground("w")
        self.canvas.showGrid(x=True, y=True, alpha=0.25)
        self.canvas.plotItem.setMenuEnabled(False)

    def _render_pyqtgraph_fallback(self) -> None:
        if not isinstance(self.canvas, pg.PlotWidget):
            return
        plot_item = self.canvas.plotItem
        plot_item.clear()
        self._configure_fallback_plot()
        plot_item.setTitle("Exceedance durations by case")
        plot_item.setLabel("bottom", "Case")
        plot_item.setLabel("left", f"Duration above threshold ({self.time_unit})")

        if not self._events:
            self._set_fallback_canvas_width(case_count=1)
            plot_item.getAxis("bottom").setTicks([[]])
            plot_item.setXRange(-0.5, 0.5, padding=0)
            plot_item.setYRange(0, 1, padding=0)
            label = pg.TextItem("No exceedance events", color="#52525b", anchor=(0.5, 0.5))
            plot_item.addItem(label)
            label.setPos(0.0, 0.5)
            self.status_label.setText("No exceedance events")
            return

        grouped = _group_events_by_case(self._events)
        cases = list(grouped)
        max_events_for_case = max(len(case_events) for case_events in grouped.values())
        x_positions = np.arange(len(cases), dtype=float)
        width = min(0.82 / max_events_for_case, 0.24)
        total_bars = len(self._events)
        show_labels = self.show_value_labels and total_bars <= self.value_label_limit

        self._set_fallback_canvas_width(case_count=len(cases))

        for event_slot in range(max_events_for_case):
            x_values: list[float] = []
            heights: list[float] = []
            slot_events: list[ExceedanceEvent] = []
            offset = (event_slot - (max_events_for_case - 1) / 2) * width
            for case_index, case in enumerate(cases):
                case_events = grouped[case]
                if event_slot >= len(case_events):
                    continue
                event = case_events[event_slot]
                x_value = float(x_positions[case_index] + offset)
                x_values.append(x_value)
                heights.append(float(event.duration))
                slot_events.append(event)
                self._fallback_bar_bounds.append((x_value, width, float(event.duration), event))
                self._patch_events[(event.case_name, event.event_index, event.start_time)] = event

            if not slot_events:
                continue
            bar_item = pg.BarGraphItem(
                x=x_values,
                height=heights,
                width=width * 0.9,
                brush=pg.mkBrush(color_for_index(event_slot)),
                pen=pg.mkPen("#3f3f46", width=0.7),
            )
            plot_item.addItem(bar_item)

            if show_labels:
                for x_value, height in zip(x_values, heights):
                    label = pg.TextItem(_format_number(height), color="#111827", anchor=(0.5, 1))
                    plot_item.addItem(label)
                    label.setPos(x_value, height)

        max_duration = max((event.duration for event in self._events), default=1.0)
        y_max = max(1.0, max_duration * 1.15)
        label_space = y_max * _fallback_label_space_fraction(len(cases))
        bottom_axis = plot_item.getAxis("bottom")
        bottom_axis.setTicks([[(float(position), "") for position in x_positions]])
        bottom_axis.setHeight(_fallback_axis_height(len(cases)))
        self._add_fallback_case_labels(plot_item, cases, x_positions, label_y=-label_space * 0.1)
        plot_item.setXRange(-0.75, len(cases) - 0.25, padding=0)
        plot_item.setYRange(-label_space, y_max, padding=0)
        self.status_label.setText(
            f"{len(self._events)} events across {len(cases)} cases. "
            "Click a bar to select its event and move the plot tracer to peak time."
        )

    def _add_fallback_case_labels(self, plot_item, cases: list[str], x_positions: np.ndarray, *, label_y: float) -> None:
        angle = -_label_rotation(len(cases))
        label_length = _axis_label_length(len(cases))
        for position, case_name in zip(x_positions, cases):
            label = pg.TextItem(
                _short_case_label(case_name, max_length=label_length),
                color="#71717a",
                anchor=(1, 0.5),
            )
            try:
                label.setAngle(angle)
            except AttributeError:
                pass
            plot_item.addItem(label, ignoreBounds=True)
            label.setPos(float(position), label_y)

    def _fallback_mouse_clicked(self, mouse_event) -> None:
        if not isinstance(self.canvas, pg.PlotWidget):
            return
        if mouse_event.button() != Qt.MouseButton.LeftButton:
            return
        view_box = self.canvas.plotItem.vb
        if not view_box.sceneBoundingRect().contains(mouse_event.scenePos()):
            return
        point = view_box.mapSceneToView(mouse_event.scenePos())
        x_value = float(point.x())
        y_value = float(point.y())
        candidates = [
            (abs(x_value - x_center), event)
            for x_center, width, height, event in self._fallback_bar_bounds
            if x_center - width / 2 <= x_value <= x_center + width / 2 and 0 <= y_value <= height
        ]
        if not candidates:
            return
        _distance, event = min(candidates, key=lambda item: item[0])
        self._select_event(event)
        mouse_event.accept()

    def _set_canvas_width(self, *, case_count: int) -> None:
        if not MATPLOTLIB_AVAILABLE or self.figure is None:
            return
        width_inches = max(9.0, min(80.0, 0.58 * max(1, case_count) + 2.5))
        height_inches = _canvas_height_inches(case_count)
        self.figure.set_size_inches(width_inches, height_inches, forward=True)
        dpi = self.figure.dpi
        self.canvas.setMinimumSize(int(width_inches * dpi), int(height_inches * dpi))

    def _set_fallback_canvas_width(self, *, case_count: int) -> None:
        if not isinstance(self.canvas, pg.PlotWidget):
            return
        width = max(900, min(12000, int(58 * max(1, case_count) + 260)))
        self.canvas.setMinimumSize(width, _fallback_canvas_height(case_count))


def _group_events_by_case(events: list[ExceedanceEvent]) -> dict[str, list[ExceedanceEvent]]:
    grouped: dict[str, list[ExceedanceEvent]] = defaultdict(list)
    for event in events:
        grouped[event.case_name].append(event)
    for case_events in grouped.values():
        case_events.sort(key=lambda event: (event.event_index, event.start_time))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _label_rotation(case_count: int) -> int:
    if case_count <= 6:
        return 35
    if case_count <= 20:
        return 55
    return 70


def _axis_label_length(case_count: int) -> int:
    if case_count <= 8:
        return 28
    if case_count <= 25:
        return 20
    return 14


def _axis_label_font_size(case_count: int) -> int:
    if case_count <= 12:
        return 8
    if case_count <= 30:
        return 7
    return 6


def _bottom_margin(case_count: int) -> float:
    if case_count <= 8:
        return 0.24
    if case_count <= 25:
        return 0.31
    return 0.38


def _canvas_height_inches(case_count: int) -> float:
    if case_count <= 8:
        return 4.2
    if case_count <= 25:
        return 4.8
    return 5.3


def _fallback_axis_height(case_count: int) -> int:
    if case_count <= 8:
        return 88
    if case_count <= 25:
        return 118
    return 138


def _fallback_canvas_height(case_count: int) -> int:
    if case_count <= 8:
        return 430
    if case_count <= 25:
        return 480
    return 540


def _fallback_label_space_fraction(case_count: int) -> float:
    if case_count <= 8:
        return 0.24
    if case_count <= 25:
        return 0.34
    return 0.45


def _short_case_label(case_name: str, *, max_length: int = 24) -> str:
    if len(case_name) <= max_length:
        return case_name
    return f"{case_name[: max_length - 1]}..."


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.4g}"


def _export_fallback_svg(events: list[ExceedanceEvent], path: str | Path) -> None:
    grouped = _group_events_by_case(events)
    cases = list(grouped)
    max_events_for_case = max((len(case_events) for case_events in grouped.values()), default=1)
    width = max(900, min(12000, int(62 * max(1, len(cases)) + 260)))
    height = 420
    plot_x = 78
    plot_y = 54
    plot_width = width - 130
    plot_height = 250
    max_duration = max((event.duration for event in events), default=1.0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<text x="24" y="32" font-family="Arial" font-size="20">Exceedance durations by case</text>',
        f'<text x="24" y="52" font-family="Arial" font-size="12">Events: {len(events)}</text>',
        f'<line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height}" stroke="#111827"/>',
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_height}" stroke="#111827"/>',
        f'<text x="{plot_x + plot_width / 2:.1f}" y="{height - 18}" font-family="Arial" font-size="13" text-anchor="middle">Case</text>',
        f'<text x="18" y="{plot_y + plot_height / 2:.1f}" font-family="Arial" font-size="13" transform="rotate(-90 18 {plot_y + plot_height / 2:.1f})" text-anchor="middle">Duration above threshold</text>',
    ]
    if not events:
        lines.append(
            f'<text x="{plot_x + plot_width / 2:.1f}" y="{plot_y + plot_height / 2:.1f}" '
            'font-family="Arial" font-size="14" text-anchor="middle">No exceedance events</text>'
        )
    else:
        case_width = plot_width / max(1, len(cases))
        bar_width = min(26.0, case_width * 0.78 / max_events_for_case)
        for event_slot in range(max_events_for_case):
            color = color_for_index(event_slot)
            for case_index, case in enumerate(cases):
                case_events = grouped[case]
                if event_slot >= len(case_events):
                    continue
                event = case_events[event_slot]
                center = plot_x + case_width * (case_index + 0.5)
                offset = (event_slot - (max_events_for_case - 1) / 2) * bar_width
                bar_height = (event.duration / max_duration) * plot_height if max_duration > 0 else 0
                x = center + offset - bar_width * 0.45
                y = plot_y + plot_height - bar_height
                lines.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.9:.1f}" height="{bar_height:.1f}" '
                    f'fill="{color}" stroke="#3f3f46" stroke-width="0.5"/>'
                )
        label_step = max(1, len(cases) // 40)
        for case_index, case in enumerate(cases):
            if case_index % label_step != 0:
                continue
            x = plot_x + case_width * (case_index + 0.5)
            label = _xml_escape(_short_case_label(case, max_length=18))
            lines.append(
                f'<text x="{x:.1f}" y="{plot_y + plot_height + 18}" font-family="Arial" font-size="10" '
                f'text-anchor="end" transform="rotate(-45 {x:.1f} {plot_y + plot_height + 18})">{label}</text>'
            )
        for event_slot in range(min(max_events_for_case, 8)):
            x = plot_x + 12 + event_slot * 86
            y = plot_y - 24
            lines.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" fill="{color_for_index(event_slot)}"/>')
            lines.append(
                f'<text x="{x + 16}" y="{y}" font-family="Arial" font-size="11">Event {event_slot + 1}</text>'
            )
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Backward-compatible name used by earlier code in this repository.
ExceedanceChart = ExceedanceBarChartWidget

__all__ = ["ExceedanceBarChartWidget", "ExceedanceChart"]
