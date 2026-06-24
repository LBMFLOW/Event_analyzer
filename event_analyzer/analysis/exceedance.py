from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ExceedanceEvent:
    """A contiguous interval where one case is above the selected threshold."""

    case_name: str
    event_index: int
    start_time: float
    end_time: float
    duration: float
    peak_value: float
    peak_time: float
    threshold: float
    region_start: float
    region_end: float


@dataclass(slots=True)
class _OpenEvent:
    start_time: float
    end_time: float
    peak_value: float
    peak_time: float


def detect_exceedance_events(
    time: Sequence[float],
    values_by_case: Mapping[str, Sequence[float]],
    threshold: float,
    region_start: float | None,
    region_end: float | None,
    *,
    cancel_token: object | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[ExceedanceEvent]:
    """Detect threshold exceedance events in a selected time region.

    Args:
        time: 1D numeric time coordinates. Datetime columns should be converted
            to numeric values, such as Unix seconds, before calling this function.
        values_by_case: Mapping of case name to one numeric array per case.
        threshold: Numeric y-value. Only values strictly greater than this value
            are considered exceedances.
        region_start: Selected region start. ``None`` uses the first finite time.
        region_end: Selected region end. ``None`` uses the last finite time.
        cancel_token: Optional object with an ``is_cancelled`` attribute or
            method. Cancellation is checked between cases.
        progress_callback: Optional callback receiving
            ``(completed_cases, total_cases, message)``.

    Returns:
        Events sorted by case name and start time. ``event_index`` is 1-based
        within each case after sorting.

    Notes:
        The detector treats the data as piecewise linear between adjacent
        samples. Missing values and duplicate or decreasing sample times break
        events. If a threshold crossing cannot be interpolated safely, the
        segment endpoint time is used as a conservative fallback.
    """
    time_array = _as_1d_float_array(time, name="time")
    if time_array.size < 2:
        return []

    start, end = _normalise_region(time_array, region_start, region_end)
    if start >= end:
        return []

    order = np.argsort(time_array, kind="mergesort")
    sorted_time = time_array[order]
    finite_time = np.isfinite(sorted_time)
    sorted_time = sorted_time[finite_time]
    value_order = order[finite_time]
    if sorted_time.size < 2:
        return []

    threshold_value = float(threshold)
    all_events: list[ExceedanceEvent] = []
    total_cases = len(values_by_case)
    if progress_callback is not None:
        progress_callback(0, total_cases, "Starting threshold analysis")

    for completed_cases, (case_name, raw_values) in enumerate(values_by_case.items(), start=1):
        if _is_cancelled(cancel_token):
            raise RuntimeError("Threshold analysis was cancelled.")
        value_array = _as_1d_float_array(raw_values, name=case_name)
        if value_array.shape != time_array.shape:
            raise ValueError(
                f"Case '{case_name}' has {value_array.size} values for "
                f"{time_array.size} time samples."
            )

        case_events = _detect_case_events(
            case_name=case_name,
            time=sorted_time,
            values=value_array[value_order],
            threshold=threshold_value,
            region_start=start,
            region_end=end,
        )
        all_events.extend(case_events)
        if progress_callback is not None:
            progress_callback(completed_cases, total_cases, f"Analyzed {completed_cases:,}/{total_cases:,} cases")

    all_events.sort(key=lambda event: (event.case_name, event.start_time, event.end_time))
    return _renumber_events(all_events)


def analyze_exceedances(
    time: Sequence[float],
    cases: Mapping[str, Sequence[float]],
    threshold: float,
    region_start: float | None = None,
    region_end: float | None = None,
    region: tuple[float | None, float | None] | None = None,
) -> list[ExceedanceEvent]:
    """Backward-compatible wrapper around :func:`detect_exceedance_events`.

    New code should pass ``region_start`` and ``region_end`` explicitly. Existing
    callers may still pass ``region=(start, end)``.
    """
    if region is not None:
        region_start, region_end = region
    return detect_exceedance_events(time, cases, threshold, region_start, region_end)


def _detect_case_events(
    *,
    case_name: str,
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
    region_start: float,
    region_end: float,
) -> list[ExceedanceEvent]:
    if time.size < 2:
        return []

    open_event: _OpenEvent | None = None
    completed: list[_OpenEvent] = []

    for index in range(time.size - 1):
        t0 = float(time[index])
        t1 = float(time[index + 1])
        y0 = float(values[index])
        y1 = float(values[index + 1])

        if t1 <= t0 or t1 < region_start or t0 > region_end:
            open_event = _close_event(open_event, completed)
            continue

        if not (np.isfinite(y0) and np.isfinite(y1)):
            open_event = _close_event(open_event, completed)
            continue

        segment_start = max(t0, region_start)
        segment_end = min(t1, region_end)
        if segment_start >= segment_end:
            continue

        start_value = _interpolate_or_endpoint(t0, y0, t1, y1, segment_start, prefer="left")
        end_value = _interpolate_or_endpoint(t0, y0, t1, y1, segment_end, prefer="right")
        interval = _segment_exceedance_interval(
            segment_start,
            start_value,
            segment_end,
            end_value,
            threshold,
        )

        if interval is None:
            open_event = _close_event(open_event, completed)
            continue

        interval_start, interval_end = interval
        peak_value, peak_time = _segment_peak(
            interval_start,
            interval_end,
            segment_start,
            start_value,
            segment_end,
            end_value,
        )

        if open_event is None:
            open_event = _OpenEvent(
                start_time=interval_start,
                end_time=interval_end,
                peak_value=peak_value,
                peak_time=peak_time,
            )
        elif interval_start <= open_event.end_time:
            open_event.end_time = max(open_event.end_time, interval_end)
            if peak_value > open_event.peak_value:
                open_event.peak_value = peak_value
                open_event.peak_time = peak_time
        else:
            completed.append(open_event)
            open_event = _OpenEvent(
                start_time=interval_start,
                end_time=interval_end,
                peak_value=peak_value,
                peak_time=peak_time,
            )

    open_event = _close_event(open_event, completed)
    return [
        ExceedanceEvent(
            case_name=case_name,
            event_index=index,
            start_time=event.start_time,
            end_time=event.end_time,
            duration=event.end_time - event.start_time,
            peak_value=event.peak_value,
            peak_time=event.peak_time,
            threshold=threshold,
            region_start=region_start,
            region_end=region_end,
        )
        for index, event in enumerate(completed, start=1)
        if event.end_time > event.start_time
    ]


def _segment_exceedance_interval(
    t0: float,
    y0: float,
    t1: float,
    y1: float,
    threshold: float,
) -> tuple[float, float] | None:
    above0 = y0 > threshold
    above1 = y1 > threshold

    if above0 and above1:
        return t0, t1
    if above0 and not above1:
        return t0, _crossing_time(t0, y0, t1, y1, threshold, fallback=t1)
    if not above0 and above1:
        return _crossing_time(t0, y0, t1, y1, threshold, fallback=t0), t1
    return None


def _segment_peak(
    interval_start: float,
    interval_end: float,
    segment_start: float,
    start_value: float,
    segment_end: float,
    end_value: float,
) -> tuple[float, float]:
    candidates: list[tuple[float, float]] = []
    if interval_start == segment_start:
        candidates.append((start_value, interval_start))
    if interval_end == segment_end:
        candidates.append((end_value, interval_end))
    if not candidates:
        candidates.append((max(start_value, end_value), interval_start))
    return max(candidates, key=lambda item: item[0])


def _crossing_time(
    t0: float,
    y0: float,
    t1: float,
    y1: float,
    threshold: float,
    *,
    fallback: float,
) -> float:
    if t1 == t0 or not all(np.isfinite(value) for value in (y0, y1, threshold)):
        return fallback
    delta = y1 - y0
    if delta == 0:
        return fallback
    fraction = (threshold - y0) / delta
    if not np.isfinite(fraction):
        return fallback
    fraction = min(1.0, max(0.0, fraction))
    return t0 + fraction * (t1 - t0)


def _interpolate_or_endpoint(
    t0: float,
    y0: float,
    t1: float,
    y1: float,
    x: float,
    *,
    prefer: str,
) -> float:
    if x <= t0:
        return y0
    if x >= t1:
        return y1
    if t1 == t0 or not (np.isfinite(y0) and np.isfinite(y1)):
        return y0 if prefer == "left" else y1
    return y0 + ((x - t0) / (t1 - t0)) * (y1 - y0)


def _normalise_region(
    time: np.ndarray,
    region_start: float | None,
    region_end: float | None,
) -> tuple[float, float]:
    finite_time = time[np.isfinite(time)]
    if finite_time.size == 0:
        return 0.0, 0.0

    data_start = float(np.nanmin(finite_time))
    data_end = float(np.nanmax(finite_time))
    requested_start = data_start if region_start is None else float(region_start)
    requested_end = data_end if region_end is None else float(region_end)
    if requested_start > requested_end:
        requested_start, requested_end = requested_end, requested_start
    if requested_end <= data_start or requested_start >= data_end:
        return data_start, data_start
    start = max(data_start, requested_start)
    end = min(data_end, requested_end)
    return start, end


def _close_event(open_event: _OpenEvent | None, completed: list[_OpenEvent]) -> None:
    if open_event is not None and open_event.end_time > open_event.start_time:
        completed.append(open_event)
    return None


def _renumber_events(events: list[ExceedanceEvent]) -> list[ExceedanceEvent]:
    counts_by_case: dict[str, int] = {}
    renumbered: list[ExceedanceEvent] = []
    for event in events:
        next_index = counts_by_case.get(event.case_name, 0) + 1
        counts_by_case[event.case_name] = next_index
        renumbered.append(
            ExceedanceEvent(
                case_name=event.case_name,
                event_index=next_index,
                start_time=event.start_time,
                end_time=event.end_time,
                duration=event.duration,
                peak_value=event.peak_value,
                peak_time=event.peak_time,
                threshold=event.threshold,
                region_start=event.region_start,
                region_end=event.region_end,
            )
        )
    return renumbered


def _as_1d_float_array(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like object.")
    return array


def _is_cancelled(cancel_token: object | None) -> bool:
    if cancel_token is None:
        return False
    cancelled = getattr(cancel_token, "is_cancelled", False)
    if callable(cancelled):
        return bool(cancelled())
    return bool(cancelled)


__all__ = ["ExceedanceEvent", "analyze_exceedances", "detect_exceedance_events"]
