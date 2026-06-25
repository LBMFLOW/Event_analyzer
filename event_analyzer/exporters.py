from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from event_analyzer.analysis.exceedance import ExceedanceEvent


class ExportError(RuntimeError):
    """Raised when an export cannot be completed."""


EVENT_HEADERS = [
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


def export_main_plot_svg(plot: Any, path: str | Path) -> None:
    """Export a PyQtGraph time-series plot to SVG."""
    try:
        if hasattr(plot, "export_svg"):
            plot.export_svg(path)
        else:
            import pyqtgraph.exporters

            exporter = pyqtgraph.exporters.SVGExporter(plot.plotItem)
            exporter.export(str(path))
    except Exception as exc:
        raise ExportError(f"Could not export main plot SVG: {exc}") from exc


def export_bar_chart_svg(chart: Any, path: str | Path) -> None:
    """Export an exceedance bar chart to SVG."""
    try:
        if hasattr(chart, "export_svg"):
            chart.export_svg(path)
        elif hasattr(chart, "save_svg"):
            chart.save_svg(path)
        else:
            raise ExportError("The chart widget does not support SVG export.")
    except ExportError:
        raise
    except Exception as exc:
        raise ExportError(f"Could not export bar chart SVG: {exc}") from exc


def export_count_curve_svg(chart: Any, path: str | Path) -> None:
    """Export an exceedance count curve to SVG."""
    try:
        if hasattr(chart, "export_svg"):
            chart.export_svg(path)
        elif hasattr(chart, "save_svg"):
            chart.save_svg(path)
        else:
            raise ExportError("The count-curve widget does not support SVG export.")
    except ExportError:
        raise
    except Exception as exc:
        raise ExportError(f"Could not export count curve SVG: {exc}") from exc


def export_events_csv(
    events: Sequence[ExceedanceEvent],
    path: str | Path,
    *,
    region_name: str = "",
) -> None:
    """Export exceedance events as CSV."""
    try:
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            headers = ["region_name", *EVENT_HEADERS] if region_name else EVENT_HEADERS
            writer.writerow(headers)
            for event in events:
                row = [getattr(event, header) for header in EVENT_HEADERS]
                writer.writerow([region_name, *row] if region_name else row)
    except Exception as exc:
        raise ExportError(f"Could not export exceedance events CSV: {exc}") from exc


def export_selected_region_csv(
    data: Any,
    region: tuple[float | None, float | None] | None,
    path: str | Path,
    *,
    region_name: str = "",
) -> None:
    """Export selected-region data from a supported loaded-data object.

    Supported shapes:
    - ``LoadedData`` from ``DataManager`` with ``time_values``, ``targets`` and
      ``auxiliaries``.
    - ``TimeSeriesDataset`` with ``time_axis``, ``targets`` and ``auxiliaries``.
    - Mapping with ``time`` plus case/auxiliary arrays.
    """
    if data is None:
        raise ExportError("No loaded data is available to export.")

    table = _tabular_region_data(data, region)
    try:
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            headers = list(table["headers"])
            rows = list(table["rows"])
            if region_name:
                headers = ["region_name", *headers]
                rows = [[region_name, *row] for row in rows]
            writer.writerow(headers)
            writer.writerows(rows)
    except Exception as exc:
        raise ExportError(f"Could not export selected-region CSV: {exc}") from exc


def export_analysis_summary_json(summary: Any, events: Sequence[ExceedanceEvent], path: str | Path) -> None:
    """Export a compact analysis summary as JSON."""
    payload = analysis_summary_payload(summary, events)
    try:
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        raise ExportError(f"Could not export analysis summary JSON: {exc}") from exc


def export_analysis_summary_csv(summary: Any, events: Sequence[ExceedanceEvent], path: str | Path) -> None:
    """Export a compact analysis summary as a two-column CSV."""
    payload = analysis_summary_payload(summary, events)
    try:
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for key, value in payload.items():
                writer.writerow([key, _csv_value(value)])
    except Exception as exc:
        raise ExportError(f"Could not export analysis summary CSV: {exc}") from exc


def analysis_summary_payload(summary: Any, events: Sequence[ExceedanceEvent]) -> dict[str, Any]:
    """Create a serializable summary payload that is safe with missing threshold data."""
    if is_dataclass(summary):
        payload = asdict(summary)
    elif isinstance(summary, Mapping):
        payload = dict(summary)
    else:
        payload = {}

    payload.setdefault("threshold_enabled", False)
    payload.setdefault("threshold", None)
    payload.setdefault("exceeding_cases", sorted({event.case_name for event in events}))
    payload.setdefault("event_count", len(events))
    payload.setdefault("max_exceedance_value", max((event.peak_value for event in events), default=None))
    payload.setdefault("longest_exceedance_duration", max((event.duration for event in events), default=None))
    payload.setdefault("region_start", None)
    payload.setdefault("region_end", None)
    return payload


def _tabular_region_data(data: Any, region: tuple[float | None, float | None] | None) -> dict[str, Any]:
    if hasattr(data, "time_values"):
        time = np.asarray(data.time_values, dtype=float)
        columns: dict[str, np.ndarray] = {"time": time}
        columns.update({name: np.asarray(values) for name, values in getattr(data, "targets", {}).items()})
        columns.update({name: np.asarray(values) for name, values in getattr(data, "auxiliaries", {}).items()})
        return _table_from_columns(columns, time, region)

    if hasattr(data, "time_axis"):
        time_axis = data.time_axis
        time = np.asarray(time_axis.values, dtype=float)
        columns = {getattr(time_axis, "name", "time"): time}
        for series in getattr(data, "targets", []):
            columns[series.name] = np.asarray(series.values)
        for series in getattr(data, "auxiliaries", []):
            columns[series.name] = np.asarray(series.values)
        return _table_from_columns(columns, time, region)

    if isinstance(data, Mapping):
        if "time" not in data:
            raise ExportError("Mapping data must contain a 'time' column.")
        time = np.asarray(data["time"], dtype=float)
        columns = {str(name): np.asarray(values) for name, values in data.items()}
        return _table_from_columns(columns, time, region)

    raise ExportError("Unsupported data object for selected-region export.")


def _table_from_columns(
    columns: Mapping[str, np.ndarray],
    time: np.ndarray,
    region: tuple[float | None, float | None] | None,
) -> dict[str, Any]:
    if time.ndim != 1:
        raise ExportError("Time data must be one-dimensional.")

    mask = np.ones(time.size, dtype=bool)
    if region is not None:
        start, end = region
        if start is not None:
            mask &= time >= float(start)
        if end is not None:
            mask &= time <= float(end)

    headers = list(columns)
    rows = []
    for index in np.flatnonzero(mask):
        row = []
        for header in headers:
            values = columns[header]
            if values.size != time.size:
                raise ExportError(f"Column '{header}' has {values.size} rows for {time.size} time values.")
            row.append(values[index].item() if hasattr(values[index], "item") else values[index])
        rows.append(row)
    return {"headers": headers, "rows": rows}


def _csv_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return "" if value is None else str(value)


__all__ = [
    "ExportError",
    "analysis_summary_payload",
    "export_analysis_summary_csv",
    "export_analysis_summary_json",
    "export_bar_chart_svg",
    "export_count_curve_svg",
    "export_events_csv",
    "export_main_plot_svg",
    "export_selected_region_csv",
]
