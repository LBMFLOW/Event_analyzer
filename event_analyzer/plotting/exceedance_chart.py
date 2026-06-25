from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import textwrap

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
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
        self._fallback_case_labels: list[pg.TextItem] = []
        self._case_display_labels: dict[str, str] = {}
        self._region_name = ""
        self._x_axis_title = ""
        self._y_axis_title = ""
        self._y_range: tuple[float, float] | None = None
        self._axis_title_font_size = 14
        self._tick_label_font_size = 12
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

    def set_case_display_labels(self, labels: dict[str, str] | None) -> None:
        """Set optional display-only labels keyed by internal case name."""
        self._case_display_labels = {str(key): str(value) for key, value in (labels or {}).items() if str(value)}
        self.set_events(self._events)

    def set_region_name(self, name: str) -> None:
        """Set the selected-region name shown in the chart and SVG export."""
        self._region_name = str(name or "").strip()
        self.set_events(self._events)

    def set_axis_titles(self, *, x_axis_title: str = "", y_axis_title: str = "") -> None:
        """Set optional custom axis titles for the chart and SVG export."""
        self._x_axis_title = str(x_axis_title or "").strip()
        self._y_axis_title = str(y_axis_title or "").strip()
        self.set_events(self._events)

    def set_y_range(self, value_range: tuple[float, float] | None) -> None:
        """Set an optional manual y-axis range for the duration chart."""
        self._y_range = _validate_optional_range(value_range, "chart y range")
        self.set_events(self._events)

    def set_font_sizes(self, *, axis_title_font_size: int, tick_label_font_size: int) -> None:
        """Set axis title and tick-label font sizes for the chart and SVG export."""
        self._axis_title_font_size = int(max(6, min(72, axis_title_font_size)))
        self._tick_label_font_size = int(max(6, min(72, tick_label_font_size)))
        self.set_events(self._events)

    def set_events(self, events: list[ExceedanceEvent]) -> None:
        """Render grouped duration bars from exceedance events."""
        self._events = sorted(events, key=lambda event: (event.case_name, event.event_index, event.start_time))
        self._patch_events.clear()
        self._fallback_bar_bounds.clear()
        self._fallback_case_labels.clear()
        if not MATPLOTLIB_AVAILABLE or self.figure is None:
            self._render_pyqtgraph_fallback()
            return
        self.figure.clear()
        axis = self.figure.add_subplot(111)

        if not self._events:
            self._set_canvas_width(case_count=1)
            self.figure.suptitle("Exceedance durations", fontsize=18, y=0.96)
            if self._region_name:
                self.figure.text(0.085, 0.90, f"Region: {self._region_name}", fontsize=11, color="#111827")
            axis.set_xlabel(self._resolved_x_axis_title(), fontsize=self._axis_title_font_size, labelpad=1)
            axis.set_ylabel(self._resolved_y_axis_title(), fontsize=self._axis_title_font_size)
            axis.set_ylim(*_resolve_y_range(1.0, self._y_range))
            axis.text(
                0.5,
                0.5,
                "No exceedance events",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=13,
            )
            axis.tick_params(axis="both", labelsize=self._tick_label_font_size)
            axis.grid(axis="y", alpha=0.25)
            self.figure.set_tight_layout(False)
            self.figure.subplots_adjust(left=0.085, right=0.985, top=0.78 if self._region_name else 0.82, bottom=0.18)
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
        show_labels = self.show_value_labels and total_bars <= max(self.value_label_limit, 120)

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
                        fontsize=11,
                    )

        axis.set_ylim(*_resolve_y_range(max((event.duration for event in self._events), default=1.0) * 1.16, self._y_range))
        self.figure.suptitle("Exceedance durations by case", fontsize=18, y=0.975)
        self.figure.text(0.075, 0.925, f"Events: {len(self._events)}", fontsize=11, color="#111827")
        if self._region_name:
            self.figure.text(0.075, 0.895, f"Region: {self._region_name}", fontsize=11, color="#111827")
        axis.set_xlabel(self._resolved_x_axis_title(), fontsize=self._axis_title_font_size, labelpad=1)
        axis.set_ylabel(self._resolved_y_axis_title(), fontsize=self._axis_title_font_size)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [
                _wrap_label_for_matplotlib(self._display_case_label(case), max_chars=_matplotlib_label_line_length(len(cases)))
                for case in cases
            ],
            rotation=52,
            ha="right",
            rotation_mode="anchor",
            fontsize=self._tick_label_font_size,
        )
        axis.tick_params(axis="y", labelsize=self._tick_label_font_size)
        axis.grid(axis="y", alpha=0.25)
        axis.margins(x=0.02)
        self.figure.set_tight_layout(False)
        self.figure.subplots_adjust(
            left=0.085,
            right=0.985,
            top=0.72 if self._region_name else 0.76,
            bottom=_bottom_margin(len(cases)),
        )

        if max_events_for_case <= 8:
            axis.legend(
                loc="upper left",
                bbox_to_anchor=(0.0, 1.13 if self._region_name else 1.16),
                fontsize=11,
                ncols=min(max_events_for_case, 4),
                frameon=False,
            )
        else:
            axis.text(
                0.99,
                0.98,
                "Bars are ordered by event index within each case",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=11,
            )

        self.status_label.setText(
            f"{len(self._events)} events across {len(cases)} cases. "
            "Click a bar to select its event and move the plot tracer to peak time."
        )
        self.canvas.draw_idle()

    def export_svg(self, path: str | Path) -> None:
        """Save the current bar chart as SVG."""
        if not MATPLOTLIB_AVAILABLE or self.figure is None:
            _export_fallback_svg(
                self._events,
                path,
                case_display_labels=self._case_display_labels,
                region_name=self._region_name,
                x_axis_title=self._resolved_x_axis_title(),
                y_axis_title=self._resolved_y_axis_title(),
                y_range=self._y_range,
                axis_title_font_size=self._axis_title_font_size,
                tick_label_font_size=self._tick_label_font_size,
            )
            return
        self.figure.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.12)

    save_svg = export_svg

    def export_csv(self, path: str | Path) -> None:
        """Export the current event table to CSV using the stdlib csv module."""
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            headers = [
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
            writer.writerow(["region_name", *headers] if self._region_name else headers)
            for event in self._events:
                row = [
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
                writer.writerow([self._region_name, *row] if self._region_name else row)

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
        title = "Exceedance durations by case"
        if self._region_name:
            title = f"{title}<br><span style='font-size:10pt'>Region: {_html_escape(self._region_name)}</span>"
        plot_item.setTitle(title)
        label_style = {"font-size": f"{self._axis_title_font_size}pt"}
        tick_font = QFont()
        tick_font.setPointSize(self._tick_label_font_size)
        plot_item.setLabel("bottom", self._resolved_x_axis_title(), **label_style)
        plot_item.setLabel("left", self._resolved_y_axis_title(), **label_style)
        plot_item.getAxis("bottom").setStyle(tickFont=tick_font)
        plot_item.getAxis("left").setStyle(tickFont=tick_font)

        if not self._events:
            self._set_fallback_canvas_width(case_count=1)
            plot_item.getAxis("bottom").setTicks([[]])
            plot_item.setXRange(-0.5, 0.5, padding=0)
            plot_item.setYRange(*_resolve_y_range(1.0, self._y_range), padding=0)
            label = pg.TextItem("No exceedance events", color="#52525b", anchor=(0.5, 0.5))
            plot_item.addItem(label)
            y_min, y_max = _resolve_y_range(1.0, self._y_range)
            label.setPos(0.0, y_min + (y_max - y_min) * 0.5)
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
        y_min, y_max = _resolve_y_range(max_duration * 1.15, self._y_range)
        y_span = max(abs(y_max - y_min), 1.0)
        label_space = y_span * _fallback_label_space_fraction(len(cases))
        bottom_axis = plot_item.getAxis("bottom")
        bottom_axis.setTicks([[(float(position), "") for position in x_positions]])
        bottom_axis.setHeight(_fallback_axis_height(len(cases)))
        self._add_fallback_case_labels(plot_item, cases, x_positions, label_y=y_min - label_space * 0.08)
        plot_item.setXRange(-0.75, len(cases) - 0.25, padding=0)
        plot_item.setYRange(y_min - label_space, y_max, padding=0)
        self.status_label.setText(
            f"{len(self._events)} events across {len(cases)} cases. "
            "Click a bar to select its event and move the plot tracer to peak time."
        )

    def _add_fallback_case_labels(self, plot_item, cases: list[str], x_positions: np.ndarray, *, label_y: float) -> None:
        angle = _label_rotation(len(cases))
        label_length = _axis_label_length(len(cases))
        for position, case_name in zip(x_positions, cases):
            label = pg.TextItem(
                _short_case_label(self._display_case_label(case_name), max_length=label_length),
                color="#71717a",
                anchor=(1, 0),
            )
            font = QFont()
            font.setPointSize(self._tick_label_font_size)
            label.setFont(font)
            try:
                label.setAngle(angle)
            except AttributeError:
                pass
            label.setZValue(20)
            plot_item.addItem(label, ignoreBounds=True)
            label.setPos(float(position), label_y)
            self._fallback_case_labels.append(label)

    def _display_case_label(self, case_name: str) -> str:
        return self._case_display_labels.get(case_name, case_name)

    def _resolved_x_axis_title(self) -> str:
        return self._x_axis_title or "Case"

    def _resolved_y_axis_title(self) -> str:
        return self._y_axis_title or f"Duration above threshold ({self.time_unit})"

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
        # Keep the figure close to a slide-friendly aspect ratio. Very wide
        # SVGs are hard to paste into PowerPoint because they are scaled down
        # aggressively; wrapped labels carry the full case names instead.
        width_inches = max(10.5, min(14.2, 9.8 + 0.105 * max(1, case_count)))
        height_inches = _canvas_height_inches(case_count)
        self.figure.set_size_inches(width_inches, height_inches, forward=True)
        dpi = self.figure.dpi
        self.canvas.setMinimumSize(int(width_inches * dpi), int(height_inches * dpi))

    def _set_fallback_canvas_width(self, *, case_count: int) -> None:
        if not isinstance(self.canvas, pg.PlotWidget):
            return
        width = max(900, min(2400, int(38 * max(1, case_count) + 320)))
        self.canvas.setMinimumSize(width, _fallback_canvas_height(case_count))


def _group_events_by_case(events: list[ExceedanceEvent]) -> dict[str, list[ExceedanceEvent]]:
    grouped: dict[str, list[ExceedanceEvent]] = defaultdict(list)
    for event in events:
        grouped[event.case_name].append(event)
    for case_events in grouped.values():
        case_events.sort(key=lambda event: (event.event_index, event.start_time))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _validate_optional_range(value_range: tuple[float, float] | None, name: str) -> tuple[float, float] | None:
    if value_range is None:
        return None
    start, end = float(value_range[0]), float(value_range[1])
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError(f"{name} must contain finite values with minimum less than maximum.")
    return start, end


def _resolve_y_range(auto_max: float, manual_range: tuple[float, float] | None) -> tuple[float, float]:
    if manual_range is not None:
        return manual_range
    return 0.0, max(1.0, float(auto_max))


def _label_rotation(case_count: int) -> int:
    if case_count <= 6:
        return 35
    if case_count <= 20:
        return 48
    return 56


def _axis_label_length(case_count: int) -> int:
    if case_count <= 8:
        return 28
    if case_count <= 25:
        return 20
    return 14


def _matplotlib_label_line_length(case_count: int) -> int:
    if case_count <= 8:
        return 26
    if case_count <= 25:
        return 20
    if case_count <= 60:
        return 16
    return 12


def _wrap_label_for_matplotlib(label: str, *, max_chars: int) -> str:
    return "\n".join(_wrap_label_lines(label, max_chars=max_chars))


def _bottom_margin(case_count: int) -> float:
    if case_count <= 8:
        return 0.30
    if case_count <= 25:
        return 0.38
    if case_count <= 60:
        return 0.46
    return 0.52


def _canvas_height_inches(case_count: int) -> float:
    if case_count <= 8:
        return 6.6
    if case_count <= 25:
        return 7.3
    if case_count <= 60:
        return 7.9
    return 8.6


def _fallback_axis_height(case_count: int) -> int:
    if case_count <= 8:
        return 110
    if case_count <= 25:
        return 150
    return 178


def _fallback_canvas_height(case_count: int) -> int:
    if case_count <= 8:
        return 520
    if case_count <= 25:
        return 610
    return 700


def _fallback_label_space_fraction(case_count: int) -> float:
    if case_count <= 8:
        return 0.30
    if case_count <= 25:
        return 0.40
    return 0.52


def _short_case_label(case_name: str, *, max_length: int = 24) -> str:
    if len(case_name) <= max_length:
        return case_name
    return f"{case_name[: max_length - 1]}..."


def _svg_label_line_length(case_count: int) -> int:
    if case_count <= 8:
        return 28
    if case_count <= 25:
        return 20
    if case_count <= 60:
        return 15
    return 12


def _wrap_label_lines(label: str, *, max_chars: int) -> list[str]:
    normalized = " ".join(str(label).split())
    if not normalized:
        return [""]
    lines = textwrap.wrap(
        normalized,
        width=max(4, max_chars),
        break_long_words=True,
        break_on_hyphens=False,
    )
    return lines or [normalized]


def _svg_multiline_case_label(
    label: str,
    *,
    x: float,
    y: float,
    lines: list[str],
    font_size: int,
    angle: int,
) -> str:
    escaped_label = _xml_escape(label)
    output = [
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial" font-size="{font_size}" '
        f'text-anchor="end" transform="rotate({angle} {x:.1f} {y:.1f})">',
        f"<title>{escaped_label}</title>",
    ]
    for index, line in enumerate(lines):
        dy = "0" if index == 0 else f"{font_size * 1.12:.1f}"
        output.append(f'<tspan x="{x:.1f}" dy="{dy}">{_xml_escape(line)}</tspan>')
    output.append("</text>")
    return "\n".join(output)


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.4g}"


def _svg_y_for_value(value: float, *, y_min: float, y_span: float, plot_y: float, plot_height: float) -> float:
    return plot_y + plot_height - ((value - y_min) / y_span) * plot_height


def _export_fallback_svg(
    events: list[ExceedanceEvent],
    path: str | Path,
    *,
    case_display_labels: dict[str, str] | None = None,
    region_name: str = "",
    x_axis_title: str = "Case",
    y_axis_title: str = "Duration above threshold",
    y_range: tuple[float, float] | None = None,
    axis_title_font_size: int = 18,
    tick_label_font_size: int = 14,
) -> None:
    grouped = _group_events_by_case(events)
    cases = list(grouped)
    display_labels = {str(key): str(value) for key, value in (case_display_labels or {}).items() if str(value)}
    max_events_for_case = max((len(case_events) for case_events in grouped.values()), default=1)
    case_count = max(1, len(cases))
    label_line_length = _svg_label_line_length(case_count)
    wrapped_labels = {
        case: _wrap_label_lines(display_labels.get(case, case), max_chars=label_line_length)
        for case in cases
    }
    max_label_lines = max((len(lines) for lines in wrapped_labels.values()), default=1)
    longest_label_line = max((len(line) for lines in wrapped_labels.values() for line in lines), default=8)
    legend_slots = min(max_events_for_case, 8)
    legend_columns = max(1, min(legend_slots, 4))
    legend_rows = max(1, (legend_slots + legend_columns - 1) // legend_columns)

    width = max(1200, min(2200, int(40 * case_count + 460)))
    left_margin = 128
    right_margin = 54
    region_text = str(region_name or "").strip()
    legend_y = 126 if region_text else 96
    top_margin = legend_y + legend_rows * 30 + 28
    bottom_margin = max(150, min(400, int(88 + max_label_lines * 34 + longest_label_line * 4.2)))
    plot_height = 440
    plot_width = width - left_margin - right_margin
    height = top_margin + plot_height + bottom_margin
    plot_x = left_margin
    plot_y = top_margin
    max_duration = max((event.duration for event in events), default=1.0)
    y_min, y_max = _resolve_y_range(max_duration * 1.2, _validate_optional_range(y_range, "chart y range"))
    y_span = max(abs(y_max - y_min), 1.0)
    legend_x = plot_x
    value_font_size = 14 if case_count <= 45 else 13
    x_axis_label_y = plot_y + plot_height + bottom_margin - 22
    x_axis_title = str(x_axis_title or "Case")
    y_axis_title = str(y_axis_title or "Duration above threshold")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<text x="28" y="42" font-family="Arial" font-size="28">Exceedance durations by case</text>',
        f'<text x="28" y="70" font-family="Arial" font-size="16">Events: {len(events)}</text>',
    ]
    if region_text:
        lines.append(f'<text x="28" y="96" font-family="Arial" font-size="16">Region: {_xml_escape(region_text)}</text>')
    lines.extend(
        [
            f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#d4d4d8" stroke-width="0.7"/>',
            f'<line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height}" stroke="#111827"/>',
            f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_height}" stroke="#111827"/>',
            f'<text x="{plot_x + plot_width / 2:.1f}" y="{x_axis_label_y:.1f}" font-family="Arial" font-size="{axis_title_font_size}" text-anchor="middle">{_xml_escape(x_axis_title)}</text>',
            f'<text x="28" y="{plot_y + plot_height / 2:.1f}" font-family="Arial" font-size="{axis_title_font_size}" transform="rotate(-90 28 {plot_y + plot_height / 2:.1f})" text-anchor="middle">{_xml_escape(y_axis_title)}</text>',
        ]
    )
    for tick_index in range(5):
        value = y_min + y_span * tick_index / 4
        y = _svg_y_for_value(value, y_min=y_min, y_span=y_span, plot_y=plot_y, plot_height=plot_height)
        lines.append(
            f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_width}" y2="{y:.1f}" '
            'stroke="#e4e4e7" stroke-width="0.6"/>'
        )
        lines.append(
            f'<text x="{plot_x - 10}" y="{y + 5:.1f}" font-family="Arial" font-size="{tick_label_font_size}" '
            f'text-anchor="end">{_format_number(value)}</text>'
        )
    if not events:
        lines.append(
            f'<text x="{plot_x + plot_width / 2:.1f}" y="{plot_y + plot_height / 2:.1f}" '
            'font-family="Arial" font-size="18" text-anchor="middle">No exceedance events</text>'
        )
    else:
        case_width = plot_width / case_count
        bar_width = max(3.0, min(30.0, case_width * 0.78 / max_events_for_case))
        for event_slot in range(max_events_for_case):
            color = color_for_index(event_slot)
            for case_index, case in enumerate(cases):
                case_events = grouped[case]
                if event_slot >= len(case_events):
                    continue
                event = case_events[event_slot]
                center = plot_x + case_width * (case_index + 0.5)
                offset = (event_slot - (max_events_for_case - 1) / 2) * bar_width
                visible_low = max(y_min, min(0.0, event.duration))
                visible_high = min(y_max, max(0.0, event.duration))
                if visible_high <= visible_low:
                    continue
                bar_height = ((visible_high - visible_low) / y_span) * plot_height
                x = center + offset - bar_width * 0.45
                y = _svg_y_for_value(visible_high, y_min=y_min, y_span=y_span, plot_y=plot_y, plot_height=plot_height)
                duration_label = _xml_escape(_format_number(event.duration))
                lines.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.9:.1f}" height="{bar_height:.1f}" '
                    f'fill="{color}" stroke="#3f3f46" stroke-width="0.5"/>'
                )
                lines.append(
                    f'<text x="{x + bar_width * 0.45:.1f}" y="{max(plot_y + 18, y - 7):.1f}" '
                    f'font-family="Arial" font-size="{value_font_size}" text-anchor="middle">{duration_label}</text>'
                )
        for case_index, case in enumerate(cases):
            x = plot_x + case_width * (case_index + 0.5)
            y = plot_y + plot_height + 28
            display_label = display_labels.get(case, case)
            lines.append(
                _svg_multiline_case_label(
                    display_label,
                    x=x,
                    y=y,
                    lines=wrapped_labels[case],
                    font_size=tick_label_font_size,
                    angle=-56,
                )
            )
        for event_slot in range(legend_slots):
            column = event_slot % legend_columns
            row = event_slot // legend_columns
            x = legend_x + column * 150
            y = legend_y + row * 30
            lines.append(f'<rect x="{x}" y="{y - 14}" width="16" height="16" fill="{color_for_index(event_slot)}"/>')
            lines.append(
                f'<text x="{x + 22}" y="{y}" font-family="Arial" font-size="16">Event {event_slot + 1}</text>'
            )
        if max_events_for_case > legend_slots:
            lines.append(
                f'<text x="{legend_x + legend_columns * 150 + 8}" y="{legend_y}" font-family="Arial" font-size="16">'
                f'Events 1-{max_events_for_case}</text>'
            )
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html_escape(value: str) -> str:
    return _xml_escape(value).replace('"', "&quot;")


# Backward-compatible name used by earlier code in this repository.
ExceedanceChart = ExceedanceBarChartWidget

__all__ = ["ExceedanceBarChartWidget", "ExceedanceChart"]
