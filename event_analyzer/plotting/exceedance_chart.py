from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from event_analyzer.analysis.exceedance import ExceedanceEvent

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

    The widget is Matplotlib-backed rather than PyQtGraph-backed because grouped
    categorical bars, labels, click picking, and SVG export are more reliable in
    Matplotlib. For many cases the canvas grows horizontally inside a scroll
    area, keeping labels readable instead of compressing all cases into one view.
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
        self.canvas = FigureCanvasQTAgg(self.figure) if MATPLOTLIB_AVAILABLE else QLabel("Matplotlib is not installed.")
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setWidget(self.canvas)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self._events: list[ExceedanceEvent] = []
        self._patch_events: dict[object, ExceedanceEvent] = {}
        self._plot_adapter = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)
        layout.addWidget(self.status_label)

        if MATPLOTLIB_AVAILABLE:
            self.canvas.mpl_connect("pick_event", self._bar_picked)
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
        if not MATPLOTLIB_AVAILABLE or self.figure is None:
            self.status_label.setText(
                f"{len(self._events)} exceedance event(s). Install matplotlib for the grouped bar chart view."
            )
            if isinstance(self.canvas, QLabel):
                self.canvas.setText("Grouped exceedance chart requires matplotlib.")
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
        axis.set_xticklabels(cases, rotation=_label_rotation(len(cases)), ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.margins(x=0.02)

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

    def _set_canvas_width(self, *, case_count: int) -> None:
        if not MATPLOTLIB_AVAILABLE or self.figure is None:
            return
        width_inches = max(9.0, min(80.0, 0.58 * max(1, case_count) + 2.5))
        height_inches = 3.8
        self.figure.set_size_inches(width_inches, height_inches, forward=True)
        dpi = self.figure.dpi
        self.canvas.setMinimumSize(int(width_inches * dpi), int(height_inches * dpi))


def _group_events_by_case(events: list[ExceedanceEvent]) -> dict[str, list[ExceedanceEvent]]:
    grouped: dict[str, list[ExceedanceEvent]] = defaultdict(list)
    for event in events:
        grouped[event.case_name].append(event)
    for case_events in grouped.values():
        case_events.sort(key=lambda event: (event.event_index, event.start_time))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _label_rotation(case_count: int) -> int:
    if case_count <= 6:
        return 20
    if case_count <= 20:
        return 35
    return 60


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.4g}"


def _export_fallback_svg(events: list[ExceedanceEvent], path: str | Path) -> None:
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300" viewBox="0 0 900 300">',
        '<rect width="900" height="300" fill="white"/>',
        '<text x="24" y="36" font-family="Arial" font-size="20">Exceedance durations</text>',
        '<text x="24" y="66" font-family="Arial" font-size="13">Install matplotlib for the grouped bar chart rendering.</text>',
        f'<text x="24" y="96" font-family="Arial" font-size="13">Events: {len(events)}</text>',
    ]
    for index, event in enumerate(events[:8], start=1):
        y = 100 + index * 22
        lines.append(
            '<text x="24" y="{y}" font-family="Arial" font-size="12">'
            "{case} event {event_index}: duration {duration}</text>".format(
                y=y,
                case=_xml_escape(event.case_name),
                event_index=event.event_index,
                duration=_format_number(event.duration),
            )
        )
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Backward-compatible name used by earlier code in this repository.
ExceedanceChart = ExceedanceBarChartWidget

__all__ = ["ExceedanceBarChartWidget", "ExceedanceChart"]
