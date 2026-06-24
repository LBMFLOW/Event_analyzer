from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
import numpy as np

from event_analyzer.analysis.exceedance import ExceedanceEvent

if TYPE_CHECKING:
    from event_analyzer.data.models import TimeSeriesDataset


def region_statistics(
    dataset: TimeSeriesDataset,
    region: tuple[float | None, float | None] | None,
    *,
    threshold: float | None = None,
    events: Sequence[ExceedanceEvent] | None = None,
) -> list[dict[str, float | str]]:
    time = dataset.time_axis.values
    start = dataset.time_axis.start if region is None or region[0] is None else region[0]
    end = dataset.time_axis.end if region is None or region[1] is None else region[1]
    mask = (time >= start) & (time <= end)
    event_counts = Counter(event.case_name for event in events or [])
    above_threshold = _duration_above_by_case(events) if events is not None else {}

    rows: list[dict[str, float | str]] = []
    for series in dataset.targets:
        values = series.values[mask]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            rows.append(
                {
                    "case": series.name,
                    "unit": series.unit,
                    "min": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "samples": 0,
                    "time_above_threshold": above_threshold.get(series.name, 0.0),
                    "event_count": event_counts.get(series.name, 0),
                }
            )
            continue
        rows.append(
            {
                "case": series.name,
                "unit": series.unit,
                "min": float(np.nanmin(finite)),
                "max": float(np.nanmax(finite)),
                "mean": float(np.nanmean(finite)),
                "median": float(np.nanmedian(finite)),
                "std": float(np.nanstd(finite)),
                "samples": int(finite.size),
                "time_above_threshold": _time_above_threshold(time[mask], values, threshold)
                if events is None
                else above_threshold.get(series.name, 0.0),
                "event_count": event_counts.get(series.name, 0),
            }
        )
    return rows


def region_statistics_from_arrays(
    time: Sequence[float],
    values_by_case: Mapping[str, Sequence[float]],
    region: tuple[float | None, float | None] | None,
    *,
    threshold: float | None = None,
    events: Sequence[ExceedanceEvent] | None = None,
    units: Mapping[str, str] | None = None,
) -> list[dict[str, float | str]]:
    """Compute selected-region statistics without requiring a dataset object."""
    time_array = np.asarray(time, dtype=float)
    if time_array.size == 0:
        return []
    start = float(np.nanmin(time_array)) if region is None or region[0] is None else float(region[0])
    end = float(np.nanmax(time_array)) if region is None or region[1] is None else float(region[1])
    if start > end:
        start, end = end, start
    mask = (time_array >= start) & (time_array <= end)
    event_counts = Counter(event.case_name for event in events or [])
    above_threshold = _duration_above_by_case(events) if events is not None else {}
    rows: list[dict[str, float | str]] = []
    for case_name, raw_values in values_by_case.items():
        values = np.asarray(raw_values, dtype=float)
        selected = values[mask]
        finite = selected[np.isfinite(selected)]
        row: dict[str, float | str] = {
            "case": str(case_name),
            "unit": (units or {}).get(case_name, ""),
            "samples": int(finite.size),
            "time_above_threshold": above_threshold.get(str(case_name), 0.0),
            "event_count": event_counts.get(str(case_name), 0),
        }
        if finite.size == 0:
            row.update({"min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "std": np.nan})
        else:
            row.update(
                {
                    "min": float(np.nanmin(finite)),
                    "max": float(np.nanmax(finite)),
                    "mean": float(np.nanmean(finite)),
                    "median": float(np.nanmedian(finite)),
                    "std": float(np.nanstd(finite)),
                    "time_above_threshold": _time_above_threshold(time_array[mask], selected, threshold)
                    if events is None
                    else above_threshold.get(str(case_name), 0.0),
                }
            )
        rows.append(row)
    return rows


def _duration_above_by_case(events: Sequence[ExceedanceEvent]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for event in events:
        durations[event.case_name] = durations.get(event.case_name, 0.0) + float(event.duration)
    return durations


def _time_above_threshold(time: np.ndarray, values: np.ndarray, threshold: float | None) -> float:
    if threshold is None or time.size < 2:
        return 0.0
    total = 0.0
    for index in range(time.size - 1):
        t0 = float(time[index])
        t1 = float(time[index + 1])
        y0 = float(values[index])
        y1 = float(values[index + 1])
        if not all(np.isfinite(value) for value in (t0, t1, y0, y1)) or t1 <= t0:
            continue
        above0 = y0 > threshold
        above1 = y1 > threshold
        if above0 and above1:
            total += t1 - t0
        elif above0 != above1 and y1 != y0:
            crossing = t0 + ((threshold - y0) / (y1 - y0)) * (t1 - t0)
            crossing = min(t1, max(t0, crossing))
            total += (crossing - t0) if above0 else (t1 - crossing)
    return float(total)


__all__ = ["region_statistics", "region_statistics_from_arrays"]
