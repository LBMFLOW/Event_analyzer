from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import polars as pl

from event_analyzer.data.models import TimeAxis


def coerce_time_axis(name: str, series: pl.Series) -> tuple[TimeAxis, list[str]]:
    """Convert a CSV time column into monotonic numeric coordinates for plotting.

    Numeric time is used as-is. Datetime values are stored as Unix seconds because
    PyQtGraph expects numeric coordinates and its DateAxisItem follows the same
    convention. Warnings are returned instead of raising for partially bad input.
    """
    warnings: list[str] = []

    numeric = _series_to_float(series)
    finite_ratio = _finite_ratio(numeric)
    if finite_ratio > 0.9:
        if finite_ratio < 1.0:
            warnings.append(f"Time column '{name}' contains missing or invalid numeric values; those rows were ignored.")
        return TimeAxis(name=name, values=numeric, is_datetime=False), warnings

    datetime_values = _series_to_datetime64(series)
    if datetime_values is not None:
        invalid = np.isnat(datetime_values)
        seconds = datetime_values.astype("datetime64[ns]").astype("int64").astype(float) / 1_000_000_000.0
        seconds[invalid] = np.nan
        bad = ~np.isfinite(seconds)
        if bad.any():
            warnings.append(f"Time column '{name}' contains unparseable datetime values; those rows were ignored.")
        return TimeAxis(name=name, values=seconds, is_datetime=True), warnings

    raise ValueError(
        f"Column '{name}' could not be interpreted as numeric time or datetime. "
        "Choose a numeric timestamp column or an ISO-like datetime column."
    )


def parse_manual_time(value: str, axis: TimeAxis) -> float:
    text = value.strip()
    if not text:
        raise ValueError("Enter a time value.")

    if not axis.is_datetime:
        return float(text)

    try:
        return float(text)
    except ValueError:
        pass

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("Enter a datetime such as 2026-01-01 12:00:00 or a Unix timestamp.") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _series_to_float(series: pl.Series) -> np.ndarray:
    try:
        cast = series.cast(pl.Float64, strict=False)
        return cast.to_numpy().astype(float, copy=False)
    except Exception:
        values = series.to_list()
        output = np.full(len(values), np.nan, dtype=float)
        for index, value in enumerate(values):
            try:
                output[index] = float(value)
            except (TypeError, ValueError):
                output[index] = np.nan
        return output


def _series_to_datetime64(series: pl.Series) -> np.ndarray | None:
    if "Datetime" in str(series.dtype) or "Date" in str(series.dtype):
        values = series.to_numpy()
        if np.issubdtype(values.dtype, np.datetime64):
            return values.astype("datetime64[ns]")

    try:
        parsed = series.cast(pl.Utf8).str.to_datetime(strict=False, exact=False)
        values = parsed.to_numpy()
        if np.issubdtype(values.dtype, np.datetime64) and _finite_datetime_ratio(values) > 0.9:
            return values.astype("datetime64[ns]")
    except Exception:
        pass

    values = series.to_list()
    output: list[np.datetime64] = []
    parsed_count = 0
    for value in values:
        dt = _parse_datetime_value(value)
        if dt is None:
            output.append(np.datetime64("NaT"))
        else:
            output.append(np.datetime64(dt.replace(tzinfo=None), "ns"))
            parsed_count += 1
    if values and parsed_count / len(values) > 0.9:
        return np.asarray(output, dtype="datetime64[ns]")
    return None


def _parse_datetime_value(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _finite_ratio(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.isfinite(values).sum() / values.size)


def _finite_datetime_ratio(values: Sequence[np.datetime64]) -> float:
    if len(values) == 0:
        return 0.0
    array = np.asarray(values, dtype="datetime64[ns]")
    valid = ~np.isnat(array)
    return float(valid.sum() / array.size)
