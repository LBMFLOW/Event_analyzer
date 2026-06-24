from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from event_analyzer.data.models import ColumnInfo, CsvInspection, SeriesData, TimeSeriesDataset
from event_analyzer.data.time_utils import coerce_time_axis


NUMERIC_DTYPE_MARKERS = ("Int", "UInt", "Float", "Decimal")
TIME_NAME_MARKERS = ("time", "date", "timestamp", "datetime", "seconds", "sec")


class CSVLoader:
    """CSV access layer built around Polars lazy scans.

    The UI first inspects only schema and a small preview. When the user chooses
    columns, only those columns are collected. That keeps memory bounded by the
    selected analysis surface instead of the full CSV width.
    """

    def __init__(self, preview_rows: int = 100, infer_schema_length: int = 10_000) -> None:
        self.preview_rows = preview_rows
        self.infer_schema_length = infer_schema_length

    def inspect(self, path: str | Path) -> CsvInspection:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

        preview = pl.read_csv(
            csv_path,
            n_rows=self.preview_rows,
            infer_schema_length=min(self.preview_rows, self.infer_schema_length),
            ignore_errors=True,
        )
        schema = self._collect_schema(csv_path, preview)
        columns = [
            self._inspect_column(name, dtype, preview[name] if name in preview.columns else None)
            for name, dtype in schema.items()
        ]

        likely_names = self._detect_time_columns(columns)
        for column in columns:
            column.is_likely_time = column.name in likely_names

        return CsvInspection(
            path=csv_path,
            columns=columns,
            preview_headers=preview.columns,
            preview_rows=preview.rows(),
            row_count_estimate=None,
        )

    def load_dataset(
        self,
        path: str | Path,
        time_column: str,
        target_columns: list[str],
        auxiliary_columns: list[str] | None = None,
    ) -> TimeSeriesDataset:
        csv_path = Path(path)
        auxiliaries = auxiliary_columns or []
        selected = _unique_preserving_order([time_column, *target_columns, *auxiliaries])
        if not target_columns:
            raise ValueError("Select at least one target/case column.")

        lazy_frame = pl.scan_csv(
            csv_path,
            infer_schema_length=self.infer_schema_length,
            ignore_errors=True,
        ).select(selected)
        frame = self._collect(lazy_frame)

        time_axis, warnings = coerce_time_axis(time_column, frame[time_column])
        valid_time = np.isfinite(time_axis.values)
        if not valid_time.any():
            raise ValueError("No valid time values were found in the selected time column.")

        order = np.argsort(time_axis.values[valid_time], kind="mergesort")
        sorted_time = time_axis.values[valid_time][order]
        time_axis.values = sorted_time

        targets = [
            self._series_from_column(frame, name, valid_time, order, warnings, axis_id="y1")
            for name in target_columns
        ]
        aux_series = [
            self._series_from_column(frame, name, valid_time, order, warnings, axis_id=f"y{index + 2}")
            for index, name in enumerate(auxiliaries)
        ]

        return TimeSeriesDataset(
            file_path=csv_path,
            time_axis=time_axis,
            targets=targets,
            auxiliaries=aux_series,
            warnings=warnings,
        )

    def _collect_schema(self, csv_path: Path, preview: pl.DataFrame) -> dict[str, pl.DataType]:
        try:
            schema = pl.scan_csv(
                csv_path,
                infer_schema_length=self.infer_schema_length,
                ignore_errors=True,
            ).collect_schema()
            return dict(schema)
        except Exception:
            return dict(zip(preview.columns, preview.dtypes))

    def _inspect_column(self, name: str, dtype: pl.DataType, sample: pl.Series | None) -> ColumnInfo:
        dtype_text = str(dtype)
        is_numeric = _is_numeric_dtype(dtype_text)
        numeric_ratio = 1.0 if is_numeric else 0.0
        null_count = None
        note = ""

        if sample is not None:
            null_count = int(sample.null_count())
            try:
                cast = sample.cast(pl.Float64, strict=False)
                non_null = max(1, sample.len() - null_count)
                numeric_count = int(cast.is_not_null().sum())
                numeric_ratio = numeric_count / non_null
            except Exception:
                numeric_ratio = 1.0 if is_numeric else 0.0
            if not is_numeric and numeric_ratio > 0.95:
                note = "Looks numeric in preview."
            elif numeric_ratio == 0 and not _looks_like_time_name(name):
                note = "Not numeric in preview."

        return ColumnInfo(
            name=name,
            dtype=dtype_text,
            is_numeric=is_numeric or numeric_ratio > 0.95,
            numeric_ratio=numeric_ratio,
            null_count=null_count,
            note=note,
        )

    def _detect_time_columns(self, columns: list[ColumnInfo]) -> list[str]:
        named = [column.name for column in columns if _looks_like_time_name(column.name)]
        if named:
            return named
        numeric = [column.name for column in columns if column.is_numeric]
        return numeric[:1]

    def _series_from_column(
        self,
        frame: pl.DataFrame,
        name: str,
        valid_time: np.ndarray,
        order: np.ndarray,
        warnings: list[str],
        *,
        axis_id: str,
    ) -> SeriesData:
        cast = frame[name].cast(pl.Float64, strict=False)
        values = cast.to_numpy().astype(float, copy=False)
        source_non_null = int(frame[name].is_not_null().sum())
        numeric_non_null = int(np.isfinite(values).sum())
        if numeric_non_null < source_non_null:
            dropped = source_non_null - numeric_non_null
            warnings.append(
                f"Column '{name}' has {dropped} non-numeric or missing values; they are plotted as gaps."
            )
        return SeriesData(name=name, values=values[valid_time][order], axis_id=axis_id)

    def _collect(self, lazy_frame: pl.LazyFrame) -> pl.DataFrame:
        try:
            return lazy_frame.collect(engine="streaming")
        except TypeError:
            return lazy_frame.collect(streaming=True)

def _is_numeric_dtype(dtype_text: str) -> bool:
    return any(marker in dtype_text for marker in NUMERIC_DTYPE_MARKERS)


def _looks_like_time_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in TIME_NAME_MARKERS)


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output
