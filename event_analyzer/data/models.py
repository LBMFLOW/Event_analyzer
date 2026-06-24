from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import polars as pl
except Exception:  # pragma: no cover - depends on optional runtime installation.
    pl = None


@dataclass(slots=True)
class ColumnInfo:
    name: str
    dtype: str
    is_numeric: bool
    numeric_ratio: float
    is_likely_time: bool = False
    null_count: int | None = None
    note: str = ""


@dataclass(slots=True)
class CsvInspection:
    path: Path
    columns: list[ColumnInfo]
    preview_headers: list[str]
    preview_rows: list[list[object]]
    row_count_estimate: int | None = None

    @property
    def likely_time_columns(self) -> list[str]:
        return [column.name for column in self.columns if column.is_likely_time]


@dataclass(slots=True)
class TimeAxis:
    name: str
    values: np.ndarray
    is_datetime: bool
    display_unit: str = ""

    @property
    def start(self) -> float:
        return float(np.nanmin(self.values))

    @property
    def end(self) -> float:
        return float(np.nanmax(self.values))

    def format_value(self, value: float) -> str:
        if not np.isfinite(value):
            return ""
        if self.is_datetime:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None)
            return dt.isoformat(sep=" ", timespec="seconds")
        if abs(value) >= 1000 or abs(value) < 0.01:
            return f"{value:.6g}"
        return f"{value:.3f}".rstrip("0").rstrip(".")


@dataclass(slots=True)
class SeriesData:
    name: str
    values: np.ndarray
    color: str = ""
    unit: str = ""
    visible: bool = True
    axis_id: str = "y1"

    def finite_values(self) -> np.ndarray:
        return self.values[np.isfinite(self.values)]


@dataclass(slots=True)
class TimeSeriesDataset:
    file_path: Path
    time_axis: TimeAxis
    targets: list[SeriesData]
    auxiliaries: list[SeriesData] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return int(self.time_axis.values.size)

    @property
    def time_range(self) -> tuple[float, float]:
        return self.time_axis.start, self.time_axis.end

    def all_series(self) -> list[SeriesData]:
        return [*self.targets, *self.auxiliaries]

    def target_by_name(self, name: str) -> SeriesData | None:
        return next((series for series in self.targets if series.name == name), None)

    def series_by_name(self, name: str) -> SeriesData | None:
        return next((series for series in self.all_series() if series.name == name), None)

    def target_names(self) -> list[str]:
        return [series.name for series in self.targets]

    def to_polars_frame(
        self,
        *,
        include_auxiliary: bool = True,
        region: tuple[float | None, float | None] | None = None,
    ) -> object:
        if pl is None:
            raise RuntimeError("Polars is required to convert a dataset to a Polars DataFrame.")
        data: dict[str, Iterable[float]] = {self.time_axis.name: self.time_axis.values}
        for series in self.targets:
            data[series.name] = series.values
        if include_auxiliary:
            for series in self.auxiliaries:
                data[series.name] = series.values

        frame = pl.DataFrame(data)
        if region is not None:
            start, end = region
            if start is not None:
                frame = frame.filter(pl.col(self.time_axis.name) >= start)
            if end is not None:
                frame = frame.filter(pl.col(self.time_axis.name) <= end)
        return frame
