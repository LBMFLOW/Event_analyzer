from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    import polars as pl
except Exception:  # pragma: no cover - depends on optional runtime installation.
    pl = None


NUMERIC_DTYPE_MARKERS = ("Int", "UInt", "Float", "Decimal")
TIME_NAME_MARKERS = ("time", "date", "timestamp", "datetime", "seconds", "secs", "sec")


class DataManagerError(Exception):
    """Base class for data-loading errors that can be shown directly to users."""


class MissingFileError(FileNotFoundError, DataManagerError):
    """Raised when the requested CSV path does not exist."""


class UnreadableCSVError(DataManagerError):
    """Raised when a CSV cannot be parsed or inspected."""


class NonNumericColumnError(DataManagerError):
    """Raised when a selected target or auxiliary column is not numeric."""


class InvalidTimeColumnError(DataManagerError):
    """Raised when the selected time column is neither numeric nor datetime-like."""


class EmptyDataError(DataManagerError):
    """Raised when no usable rows remain after loading and validation."""


class TooManyInvalidValuesError(DataManagerError):
    """Raised when a selected numeric column has too many missing/invalid values."""


class MemoryLimitError(DataManagerError):
    """Raised when a requested CSV load exceeds the configured cell budget."""


@dataclass(slots=True)
class ColumnProfile:
    """Summary of one CSV column based on schema and preview values."""

    name: str
    dtype: str
    is_numeric: bool
    numeric_valid_ratio: float
    is_likely_time: bool = False
    datetime_valid_ratio: float = 0.0
    null_count_in_preview: int = 0


@dataclass(slots=True)
class CSVPreview:
    """Small row preview for display or tests."""

    headers: list[str]
    rows: list[dict[str, object]]


@dataclass(slots=True)
class CSVLayout:
    """1-based row positions used to interpret a CSV file."""

    header_row: int = 1
    units_row: int | None = None
    data_start_row: int = 2


@dataclass(slots=True)
class CSVMetadata:
    """Metadata available after opening a CSV file."""

    path: Path
    columns: list[ColumnProfile]
    preview: CSVPreview
    layout: CSVLayout = field(default_factory=CSVLayout)
    units: dict[str, str] = field(default_factory=dict)

    @property
    def column_names(self) -> list[str]:
        """Return CSV columns in file order."""
        return [column.name for column in self.columns]

    @property
    def numeric_columns(self) -> list[str]:
        """Return columns that appear numeric enough for plotting."""
        return [column.name for column in self.columns if column.is_numeric]

    @property
    def likely_time_columns(self) -> list[str]:
        """Return columns that look like numeric or datetime time axes."""
        return [column.name for column in self.columns if column.is_likely_time]


@dataclass(slots=True)
class LoadedSeries:
    """One selected target or auxiliary series prepared for plotting."""

    name: str
    values: np.ndarray
    invalid_ratio: float


@dataclass(slots=True)
class LoadedData:
    """Selected CSV data converted to sorted NumPy arrays."""

    path: Path
    time_column: str
    time_values: np.ndarray
    time_is_datetime: bool
    targets: dict[str, np.ndarray]
    auxiliaries: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        """Number of rows with valid time values."""
        return int(self.time_values.size)

    @property
    def time_range(self) -> tuple[float, float]:
        """Minimum and maximum plotting time values."""
        if self.time_values.size == 0:
            raise EmptyDataError("No loaded time values are available.")
        return float(self.time_values[0]), float(self.time_values[-1])


class DataManager:
    """Load and validate large CSV time-series files for plotting and analysis.

    Polars is used because it can inspect schemas and lazily scan CSV files
    without loading unused columns. Once the user chooses the time, target, and
    auxiliary columns, only those selected columns are collected and converted to
    NumPy arrays. Missing numeric values are represented as ``np.nan`` so plots
    can show gaps and analysis code can use finite-value masks.
    """

    def __init__(
        self,
        *,
        preview_rows: int = 100,
        infer_schema_length: int = 10_000,
        min_numeric_valid_ratio: float = 0.9,
        min_time_valid_ratio: float = 0.9,
        max_loaded_cells: int | None = None,
    ) -> None:
        self.preview_rows = preview_rows
        self.infer_schema_length = infer_schema_length
        self.min_numeric_valid_ratio = min_numeric_valid_ratio
        self.min_time_valid_ratio = min_time_valid_ratio
        self.max_loaded_cells = max_loaded_cells
        self._metadata: CSVMetadata | None = None

    @property
    def metadata(self) -> CSVMetadata:
        """Metadata for the currently opened CSV file."""
        if self._metadata is None:
            raise DataManagerError("No CSV file is open. Open a CSV before reading metadata.")
        return self._metadata

    @property
    def column_names(self) -> list[str]:
        """Column names for the currently opened CSV."""
        return self.metadata.column_names

    @property
    def numeric_columns(self) -> list[str]:
        """Columns inferred as numeric for the currently opened CSV."""
        return self.metadata.numeric_columns

    @property
    def likely_time_columns(self) -> list[str]:
        """Columns inferred as likely time axes for the currently opened CSV."""
        return self.metadata.likely_time_columns

    def open_csv(
        self,
        path: str | Path,
        *,
        header_row: int | None = None,
        units_row: int | None = None,
        data_start_row: int | None = None,
        cancel_token: object | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> CSVMetadata:
        """Open a CSV path, inspect columns, and cache metadata.

        Args:
            path: CSV file path.

        Raises:
            MissingFileError: The file does not exist.
            UnreadableCSVError: The file cannot be read as CSV.
            EmptyDataError: The file has no columns.
        """
        _report(progress_callback, 0, 4, "Checking CSV path")
        _raise_if_cancelled(cancel_token)
        csv_path = Path(path)
        if not csv_path.exists():
            raise MissingFileError(f"CSV file does not exist: {csv_path}")
        if not csv_path.is_file():
            raise MissingFileError(f"CSV path is not a file: {csv_path}")

        layout = _resolve_csv_layout(
            csv_path,
            header_row=header_row,
            units_row=units_row,
            data_start_row=data_start_row,
        )
        unique_headers = _read_unique_csv_headers(csv_path, header_row=layout.header_row)
        units = _read_units(csv_path, unique_headers, units_row=layout.units_row)
        if pl is None:
            return self._open_csv_stdlib(
                csv_path,
                layout=layout,
                unique_headers=unique_headers,
                units=units,
                cancel_token=cancel_token,
                progress_callback=progress_callback,
            )

        try:
            _report(progress_callback, 1, 4, "Reading preview rows")
            preview_frame = pl.read_csv(
                csv_path,
                n_rows=self.preview_rows,
                has_header=False,
                skip_rows=layout.data_start_row - 1,
                infer_schema_length=min(self.preview_rows, self.infer_schema_length),
                new_columns=unique_headers,
                ignore_errors=False,
            )
        except Exception as exc:
            raise UnreadableCSVError(f"Could not read CSV file '{csv_path}': {exc}") from exc

        if not preview_frame.columns:
            raise EmptyDataError(f"CSV file '{csv_path}' does not contain any columns.")

        _raise_if_cancelled(cancel_token)
        _report(progress_callback, 2, 4, "Inspecting CSV schema")
        schema = self._read_schema(csv_path, preview_frame, unique_headers=unique_headers, layout=layout)
        _raise_if_cancelled(cancel_token)
        _report(progress_callback, 3, 4, "Profiling columns")
        profiles = [
            self._profile_column(name, dtype, preview_frame[name] if name in preview_frame.columns else None)
            for name, dtype in schema.items()
        ]
        likely_time = set(self._infer_likely_time_column_names(profiles))
        for profile in profiles:
            profile.is_likely_time = profile.name in likely_time

        metadata = CSVMetadata(
            path=csv_path,
            columns=profiles,
            preview=self._frame_to_preview(preview_frame),
            layout=layout,
            units=units,
        )
        self._metadata = metadata
        _report(progress_callback, 4, 4, "CSV preview ready")
        return metadata

    def preview(self, rows: int | None = None) -> CSVPreview:
        """Return a preview of the first ``rows`` rows from the opened CSV."""
        metadata = self.metadata
        if rows is None or rows <= self.preview_rows:
            return CSVPreview(headers=metadata.preview.headers, rows=metadata.preview.rows[: rows or self.preview_rows])

        try:
            if pl is None:
                headers, rows = _read_csv_preview(
                    metadata.path,
                    rows=rows,
                    unique_headers=metadata.column_names,
                    data_start_row=metadata.layout.data_start_row,
                )
                return CSVPreview(headers=headers, rows=rows)
            frame = pl.read_csv(
                metadata.path,
                n_rows=rows,
                has_header=False,
                skip_rows=metadata.layout.data_start_row - 1,
                infer_schema_length=min(rows, self.infer_schema_length),
                new_columns=metadata.column_names,
                ignore_errors=False,
            )
        except Exception as exc:
            raise UnreadableCSVError(f"Could not read preview rows from '{metadata.path}': {exc}") from exc
        return self._frame_to_preview(frame)

    def infer_numeric_columns(self) -> list[str]:
        """Return columns inferred as numeric from schema and preview values."""
        return self.numeric_columns

    def infer_likely_time_columns(self) -> list[str]:
        """Return columns that look suitable for the time axis."""
        return self.likely_time_columns

    def select_columns(
        self,
        *,
        time_column: str,
        target_columns: Sequence[str],
        auxiliary_columns: Sequence[str] | None = None,
        ignore_invalid_targets: bool = False,
        cancel_token: object | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> LoadedData:
        """Load selected columns and convert them into sorted plotting arrays.

        Args:
            time_column: Numeric or datetime-like column to use as x-axis.
            target_columns: One or more numeric case/target columns.
            auxiliary_columns: Optional numeric auxiliary columns.

        Raises:
            DataManagerError: No CSV is open or requested columns are missing.
            NonNumericColumnError: A selected target/auxiliary column is not numeric.
            InvalidTimeColumnError: The time column is not valid numeric/datetime data.
            EmptyDataError: No usable rows remain after invalid time rows are removed.
            TooManyInvalidValuesError: A selected numeric column exceeds the invalid limit.
        """
        metadata = self.metadata
        if pl is None:
            return self._select_columns_stdlib(
                time_column=time_column,
                target_columns=target_columns,
                auxiliary_columns=auxiliary_columns,
                ignore_invalid_targets=ignore_invalid_targets,
                cancel_token=cancel_token,
                progress_callback=progress_callback,
            )

        targets = list(target_columns)
        auxiliaries = list(auxiliary_columns or [])
        total_series = len(targets) + len(auxiliaries)
        total_steps = 4 + total_series
        _report(progress_callback, 0, total_steps, "Validating selected columns")
        _raise_if_cancelled(cancel_token)
        if not targets:
            raise DataManagerError("Select at least one target column.")

        selected_columns = _unique_preserving_order([time_column, *targets, *auxiliaries])
        self._validate_columns_exist(selected_columns)
        self._validate_selected_numeric(targets, role="target")
        self._validate_selected_numeric(auxiliaries, role="auxiliary")

        try:
            _report(progress_callback, 1, total_steps, "Loading selected CSV columns")
            frame = self._collect_selected(metadata.path, selected_columns)
        except DataManagerError:
            raise
        except Exception as exc:
            raise UnreadableCSVError(f"Could not load selected CSV columns: {exc}") from exc

        self._validate_loaded_cell_budget(frame.height, len(selected_columns))
        if frame.height == 0:
            raise EmptyDataError("CSV file contains no data rows.")

        _raise_if_cancelled(cancel_token)
        _report(progress_callback, 2, total_steps, "Converting time column")
        time_values, time_is_datetime, warnings = self._coerce_time_column(time_column, frame[time_column])
        valid_time_mask = np.isfinite(time_values)
        if not valid_time_mask.any():
            raise EmptyDataError(f"No usable rows remain after validating time column '{time_column}'.")

        order = np.argsort(time_values[valid_time_mask], kind="mergesort")
        sorted_time = time_values[valid_time_mask][order]

        loaded_targets: dict[str, np.ndarray] = {}
        progress_step = 3
        for offset, name in enumerate(targets, start=1):
            _raise_if_cancelled(cancel_token)
            _report(
                progress_callback,
                progress_step,
                total_steps,
                f"Converting target {offset:,}/{total_series:,}: {name}",
            )
            try:
                loaded_targets[name] = self._coerce_numeric_series(
                    frame[name],
                    name,
                    valid_time_mask,
                    order,
                    warnings,
                    role="target",
                )
            except NonNumericColumnError:
                if not ignore_invalid_targets:
                    raise
                warnings.append(f"Skipped target column '{name}' because it contains no usable numeric values.")
            progress_step += 1

        loaded_auxiliaries: dict[str, np.ndarray] = {}
        for offset, name in enumerate(auxiliaries, start=len(targets) + 1):
            _raise_if_cancelled(cancel_token)
            _report(
                progress_callback,
                progress_step,
                total_steps,
                f"Converting auxiliary {offset:,}/{total_series:,}: {name}",
            )
            loaded_auxiliaries[name] = self._coerce_numeric_series(
                frame[name],
                name,
                valid_time_mask,
                order,
                warnings,
                role="auxiliary",
            )
            progress_step += 1

        _report(progress_callback, total_steps, total_steps, "Selected data ready")
        if not loaded_targets:
            raise DataManagerError("No selected target columns contain usable numeric values.")
        return LoadedData(
            path=metadata.path,
            time_column=time_column,
            time_values=sorted_time,
            time_is_datetime=time_is_datetime,
            targets=loaded_targets,
            auxiliaries=loaded_auxiliaries,
            warnings=warnings,
            units=dict(metadata.units),
        )

    load_selected = select_columns

    def _open_csv_stdlib(
        self,
        csv_path: Path,
        *,
        layout: CSVLayout,
        unique_headers: Sequence[str],
        units: Mapping[str, str],
        cancel_token: object | None,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> CSVMetadata:
        _report(progress_callback, 1, 4, "Reading preview rows")
        headers, preview_rows = _read_csv_preview(
            csv_path,
            rows=self.preview_rows,
            unique_headers=unique_headers,
            data_start_row=layout.data_start_row,
        )
        if not headers:
            raise EmptyDataError(f"CSV file '{csv_path}' does not contain any columns.")

        _raise_if_cancelled(cancel_token)
        _report(progress_callback, 3, 4, "Profiling columns")
        column_values = {header: [row.get(header) for row in preview_rows] for header in headers}
        profiles = [self._profile_column_values(name, column_values[name]) for name in headers]
        likely_time = set(self._infer_likely_time_column_names(profiles))
        for profile in profiles:
            profile.is_likely_time = profile.name in likely_time

        metadata = CSVMetadata(
            path=csv_path,
            columns=profiles,
            preview=CSVPreview(headers=headers, rows=preview_rows),
            layout=layout,
            units=dict(units),
        )
        self._metadata = metadata
        _report(progress_callback, 4, 4, "CSV preview ready")
        return metadata

    def _select_columns_stdlib(
        self,
        *,
        time_column: str,
        target_columns: Sequence[str],
        auxiliary_columns: Sequence[str] | None,
        ignore_invalid_targets: bool,
        cancel_token: object | None,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> LoadedData:
        metadata = self.metadata
        targets = list(target_columns)
        auxiliaries = list(auxiliary_columns or [])
        total_series = len(targets) + len(auxiliaries)
        total_steps = 4 + total_series
        _report(progress_callback, 0, total_steps, "Validating selected columns")
        _raise_if_cancelled(cancel_token)
        if not targets:
            raise DataManagerError("Select at least one target column.")

        selected_columns = _unique_preserving_order([time_column, *targets, *auxiliaries])
        self._validate_columns_exist(selected_columns)
        self._validate_selected_numeric(targets, role="target")
        self._validate_selected_numeric(auxiliaries, role="auxiliary")

        _report(progress_callback, 1, total_steps, "Loading selected CSV columns")
        rows = _read_selected_csv_rows(
            metadata.path,
            selected_columns,
            headers=metadata.column_names,
            data_start_row=metadata.layout.data_start_row,
        )
        self._validate_loaded_cell_budget(len(rows), len(selected_columns))
        if not rows:
            raise EmptyDataError("CSV file contains no data rows.")

        columns = {name: [row.get(name) for row in rows] for name in selected_columns}
        _raise_if_cancelled(cancel_token)
        _report(progress_callback, 2, total_steps, "Converting time column")
        time_values, time_is_datetime, warnings = self._coerce_time_values(time_column, columns[time_column])
        valid_time_mask = np.isfinite(time_values)
        if not valid_time_mask.any():
            raise EmptyDataError(f"No usable rows remain after validating time column '{time_column}'.")

        order = np.argsort(time_values[valid_time_mask], kind="mergesort")
        sorted_time = time_values[valid_time_mask][order]
        loaded_targets: dict[str, np.ndarray] = {}
        progress_step = 3
        for offset, name in enumerate(targets, start=1):
            _raise_if_cancelled(cancel_token)
            _report(progress_callback, progress_step, total_steps, f"Converting target {offset:,}/{total_series:,}: {name}")
            try:
                loaded_targets[name] = self._coerce_numeric_values(
                    columns[name],
                    name,
                    valid_time_mask,
                    order,
                    warnings,
                    role="target",
                )
            except NonNumericColumnError:
                if not ignore_invalid_targets:
                    raise
                warnings.append(f"Skipped target column '{name}' because it contains no usable numeric values.")
            progress_step += 1

        loaded_auxiliaries: dict[str, np.ndarray] = {}
        for offset, name in enumerate(auxiliaries, start=len(targets) + 1):
            _raise_if_cancelled(cancel_token)
            _report(
                progress_callback,
                progress_step,
                total_steps,
                f"Converting auxiliary {offset:,}/{total_series:,}: {name}",
            )
            loaded_auxiliaries[name] = self._coerce_numeric_values(
                columns[name],
                name,
                valid_time_mask,
                order,
                warnings,
                role="auxiliary",
            )
            progress_step += 1

        _report(progress_callback, total_steps, total_steps, "Selected data ready")
        if not loaded_targets:
            raise DataManagerError("No selected target columns contain usable numeric values.")
        return LoadedData(
            path=metadata.path,
            time_column=time_column,
            time_values=sorted_time,
            time_is_datetime=time_is_datetime,
            targets=loaded_targets,
            auxiliaries=loaded_auxiliaries,
            warnings=warnings,
            units=dict(metadata.units),
        )

    def _read_schema(
        self,
        csv_path: Path,
        preview_frame: pl.DataFrame,
        *,
        unique_headers: Sequence[str],
        layout: CSVLayout,
    ) -> Mapping[str, pl.DataType]:
        try:
            return dict(
                pl.scan_csv(
                    csv_path,
                    has_header=False,
                    skip_rows=layout.data_start_row - 1,
                    infer_schema_length=self.infer_schema_length,
                    new_columns=list(unique_headers),
                    ignore_errors=False,
                ).collect_schema()
            )
        except Exception:
            return dict(zip(preview_frame.columns, preview_frame.dtypes))

    def _profile_column(self, name: str, dtype: pl.DataType, sample: pl.Series | None) -> ColumnProfile:
        dtype_text = str(dtype)
        dtype_is_numeric = _is_numeric_dtype(dtype_text)
        numeric_ratio = 1.0 if dtype_is_numeric else 0.0
        datetime_ratio = 0.0
        null_count = 0

        if sample is not None and sample.len() > 0:
            null_count = int(sample.null_count())
            numeric_ratio = self._numeric_ratio(sample)
            datetime_ratio = self._datetime_ratio(sample)

        is_numeric = dtype_is_numeric or numeric_ratio >= self.min_numeric_valid_ratio
        return ColumnProfile(
            name=name,
            dtype=dtype_text,
            is_numeric=is_numeric,
            numeric_valid_ratio=numeric_ratio,
            datetime_valid_ratio=datetime_ratio,
            null_count_in_preview=null_count,
        )

    def _profile_column_values(self, name: str, values: Sequence[object]) -> ColumnProfile:
        numeric = _values_to_float(values)
        numeric_ratio = _finite_ratio(numeric)
        datetime_ratio = _datetime_finite_ratio(_values_to_datetime64(values))
        is_numeric = numeric_ratio >= self.min_numeric_valid_ratio
        dtype = "Float64" if is_numeric else "Utf8"
        return ColumnProfile(
            name=name,
            dtype=dtype,
            is_numeric=is_numeric,
            numeric_valid_ratio=numeric_ratio,
            datetime_valid_ratio=datetime_ratio,
            null_count_in_preview=sum(1 for value in values if value in (None, "")),
        )

    def _infer_likely_time_column_names(self, profiles: Sequence[ColumnProfile]) -> list[str]:
        named_time = [
            profile.name
            for profile in profiles
            if _looks_like_time_name(profile.name)
            and (profile.is_numeric or profile.datetime_valid_ratio >= self.min_time_valid_ratio)
        ]
        if named_time:
            return named_time

        datetime_like = [
            profile.name
            for profile in profiles
            if profile.datetime_valid_ratio >= self.min_time_valid_ratio
        ]
        if datetime_like:
            return datetime_like

        numeric = [profile.name for profile in profiles if profile.is_numeric]
        return numeric[:1]

    def _validate_columns_exist(self, columns: Sequence[str]) -> None:
        available = set(self.column_names)
        missing = [column for column in columns if column not in available]
        if missing:
            raise DataManagerError(f"CSV file does not contain column(s): {', '.join(missing)}")

    def _validate_selected_numeric(self, columns: Sequence[str], *, role: str) -> None:
        profiles = {profile.name: profile for profile in self.metadata.columns}
        for column in columns:
            profile = profiles[column]
            if role == "target":
                # Target/case columns are allowed to be ragged: a case may end
                # before the global time axis ends, leaving blanks that become
                # NaN plot gaps. Full numeric validation happens after loading,
                # where we only require at least one usable numeric value.
                continue
            if not profile.is_numeric:
                raise NonNumericColumnError(
                    f"Selected {role} column '{column}' is not numeric enough for plotting "
                    f"({profile.numeric_valid_ratio:.1%} numeric values in preview; "
                    f"required at least {self.min_numeric_valid_ratio:.0%})."
                )

    def _collect_selected(self, csv_path: Path, selected_columns: Sequence[str]) -> pl.DataFrame:
        self._validate_requested_cell_budget(csv_path, selected_columns)
        metadata = self.metadata
        lazy_frame = pl.scan_csv(
            csv_path,
            has_header=False,
            skip_rows=metadata.layout.data_start_row - 1,
            infer_schema_length=self.infer_schema_length,
            new_columns=metadata.column_names,
            ignore_errors=True,
        ).select(list(selected_columns))
        try:
            return lazy_frame.collect(engine="streaming")
        except TypeError:
            return lazy_frame.collect(streaming=True)

    def _validate_requested_cell_budget(self, csv_path: Path, selected_columns: Sequence[str]) -> None:
        if self.max_loaded_cells is None:
            return
        try:
            metadata = self.metadata
            row_count_frame = (
                pl.scan_csv(
                    csv_path,
                    has_header=False,
                    skip_rows=metadata.layout.data_start_row - 1,
                    infer_schema_length=self.infer_schema_length,
                    new_columns=metadata.column_names,
                    ignore_errors=True,
                )
                .select(pl.len())
                .collect()
            )
            row_count = int(row_count_frame.item())
        except Exception:
            return
        requested_cells = row_count * len(selected_columns)
        if requested_cells > self.max_loaded_cells:
            raise MemoryLimitError(
                f"Selected data would load about {requested_cells:,} cells "
                f"({row_count:,} rows x {len(selected_columns):,} columns), "
                f"which exceeds the configured limit of {self.max_loaded_cells:,} cells. "
                "Select fewer columns or raise the memory limit."
            )

    def _validate_loaded_cell_budget(self, row_count: int, column_count: int) -> None:
        if self.max_loaded_cells is None:
            return
        loaded_cells = row_count * column_count
        if loaded_cells > self.max_loaded_cells:
            raise MemoryLimitError(
                f"Loaded data has {loaded_cells:,} cells, exceeding the configured limit of "
                f"{self.max_loaded_cells:,} cells. Select fewer columns or raise the memory limit."
            )

    def _coerce_time_column(self, name: str, series: pl.Series) -> tuple[np.ndarray, bool, list[str]]:
        warnings: list[str] = []

        numeric = _series_to_float(series)
        numeric_ratio = _finite_ratio(numeric)
        if numeric_ratio >= self.min_time_valid_ratio:
            if numeric_ratio < 1.0:
                warnings.append(
                    f"Time column '{name}' has {(1.0 - numeric_ratio):.1%} missing or invalid values; "
                    "those rows were ignored."
                )
            return numeric, False, warnings

        datetime_values = _series_to_datetime64(series)
        datetime_ratio = _datetime_finite_ratio(datetime_values)
        if datetime_ratio >= self.min_time_valid_ratio:
            invalid = np.isnat(datetime_values)
            seconds = datetime_values.astype("datetime64[ns]").astype("int64").astype(float) / 1_000_000_000.0
            seconds[invalid] = np.nan
            if datetime_ratio < 1.0:
                warnings.append(
                    f"Time column '{name}' has {(1.0 - datetime_ratio):.1%} unparseable datetime values; "
                    "those rows were ignored."
                )
            return seconds, True, warnings

        raise InvalidTimeColumnError(
            f"Selected time column '{name}' is invalid. It must be numeric or datetime-like, "
            f"with at least {self.min_time_valid_ratio:.0%} valid values."
        )

    def _coerce_time_values(self, name: str, values: Sequence[object]) -> tuple[np.ndarray, bool, list[str]]:
        warnings: list[str] = []
        numeric = _values_to_float(values)
        numeric_ratio = _finite_ratio(numeric)
        if numeric_ratio >= self.min_time_valid_ratio:
            if numeric_ratio < 1.0:
                warnings.append(
                    f"Time column '{name}' has {(1.0 - numeric_ratio):.1%} missing or invalid values; "
                    "those rows were ignored."
                )
            return numeric, False, warnings

        datetime_values = _values_to_datetime64(values)
        datetime_ratio = _datetime_finite_ratio(datetime_values)
        if datetime_ratio >= self.min_time_valid_ratio:
            invalid = np.isnat(datetime_values)
            seconds = datetime_values.astype("datetime64[ns]").astype("int64").astype(float) / 1_000_000_000.0
            seconds[invalid] = np.nan
            if datetime_ratio < 1.0:
                warnings.append(
                    f"Time column '{name}' has {(1.0 - datetime_ratio):.1%} unparseable datetime values; "
                    "those rows were ignored."
                )
            return seconds, True, warnings

        raise InvalidTimeColumnError(
            f"Selected time column '{name}' is invalid. It must be numeric or datetime-like, "
            f"with at least {self.min_time_valid_ratio:.0%} valid values."
        )

    def _coerce_numeric_series(
        self,
        series: pl.Series,
        name: str,
        valid_time_mask: np.ndarray,
        order: np.ndarray,
        warnings: list[str],
        *,
        role: str,
    ) -> np.ndarray:
        values = _series_to_float(series)
        values_with_valid_time = values[valid_time_mask]
        valid_ratio = _finite_ratio(values_with_valid_time)
        invalid_ratio = 1.0 - valid_ratio

        if valid_ratio == 0.0:
            raise NonNumericColumnError(
                f"Selected {role} column '{name}' contains no usable numeric values after time filtering."
            )

        if role != "target" and valid_ratio < self.min_numeric_valid_ratio:
            raise TooManyInvalidValuesError(
                f"Selected {role} column '{name}' has {invalid_ratio:.1%} missing or invalid values "
                f"after time filtering; allowed maximum is {(1.0 - self.min_numeric_valid_ratio):.0%}."
            )

        if invalid_ratio > 0:
            if role == "target":
                warnings.append(
                    f"Target column '{name}' has {invalid_ratio:.1%} missing or invalid values; "
                    "those samples are shown as plot gaps and ignored by threshold analysis."
                )
            else:
                warnings.append(
                    f"Column '{name}' has {invalid_ratio:.1%} missing or invalid values; "
                    "they are represented as NaN gaps."
                )

        return values_with_valid_time[order]

    def _coerce_numeric_values(
        self,
        values: Sequence[object],
        name: str,
        valid_time_mask: np.ndarray,
        order: np.ndarray,
        warnings: list[str],
        *,
        role: str,
    ) -> np.ndarray:
        numeric = _values_to_float(values)
        values_with_valid_time = numeric[valid_time_mask]
        valid_ratio = _finite_ratio(values_with_valid_time)
        invalid_ratio = 1.0 - valid_ratio

        if valid_ratio == 0.0:
            raise NonNumericColumnError(
                f"Selected {role} column '{name}' contains no usable numeric values after time filtering."
            )

        if role != "target" and valid_ratio < self.min_numeric_valid_ratio:
            raise TooManyInvalidValuesError(
                f"Selected {role} column '{name}' has {invalid_ratio:.1%} missing or invalid values "
                f"after time filtering; allowed maximum is {(1.0 - self.min_numeric_valid_ratio):.0%}."
            )

        if invalid_ratio > 0:
            if role == "target":
                warnings.append(
                    f"Target column '{name}' has {invalid_ratio:.1%} missing or invalid values; "
                    "those samples are shown as plot gaps and ignored by threshold analysis."
                )
            else:
                warnings.append(
                    f"Column '{name}' has {invalid_ratio:.1%} missing or invalid values; "
                    "they are represented as NaN gaps."
                )
        return values_with_valid_time[order]

    def _numeric_ratio(self, series: pl.Series) -> float:
        values = _series_to_float(series)
        return _finite_ratio(values)

    def _datetime_ratio(self, series: pl.Series) -> float:
        return _datetime_finite_ratio(_series_to_datetime64(series))

    def _frame_to_preview(self, frame: pl.DataFrame) -> CSVPreview:
        rows: list[dict[str, object]] = []
        for row in frame.iter_rows(named=True):
            rows.append(dict(row))
        return CSVPreview(headers=list(frame.columns), rows=rows)


def _series_to_float(series: pl.Series) -> np.ndarray:
    try:
        return series.cast(pl.Float64, strict=False).to_numpy().astype(float, copy=False)
    except Exception:
        values = series.to_list()
        output = np.full(len(values), np.nan, dtype=float)
        for index, value in enumerate(values):
            try:
                output[index] = float(value)
            except (TypeError, ValueError):
                output[index] = np.nan
        return output


def _series_to_datetime64(series: pl.Series) -> np.ndarray:
    if "Datetime" in str(series.dtype) or "Date" in str(series.dtype):
        values = series.to_numpy()
        if np.issubdtype(values.dtype, np.datetime64):
            return values.astype("datetime64[ns]")

    try:
        parsed = series.cast(pl.Utf8).str.to_datetime(strict=False, exact=False)
        values = parsed.to_numpy()
        if np.issubdtype(values.dtype, np.datetime64):
            return values.astype("datetime64[ns]")
    except Exception:
        pass

    parsed_values: list[np.datetime64] = []
    for value in series.to_list():
        parsed = _parse_datetime_value(value)
        if parsed is None:
            parsed_values.append(np.datetime64("NaT", "ns"))
        else:
            parsed_values.append(np.datetime64(parsed.replace(tzinfo=None), "ns"))
    return np.asarray(parsed_values, dtype="datetime64[ns]")


def _values_to_float(values: Sequence[object]) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    for index, value in enumerate(values):
        if value is None or value == "":
            continue
        try:
            output[index] = float(value)
        except (TypeError, ValueError):
            output[index] = np.nan
    return output


def _values_to_datetime64(values: Sequence[object]) -> np.ndarray:
    parsed_values: list[np.datetime64] = []
    for value in values:
        parsed = _parse_datetime_value(value)
        if parsed is None:
            parsed_values.append(np.datetime64("NaT", "ns"))
        else:
            parsed_values.append(np.datetime64(parsed.replace(tzinfo=None), "ns"))
    return np.asarray(parsed_values, dtype="datetime64[ns]")


def _read_csv_preview(
    path: Path,
    *,
    rows: int,
    unique_headers: Sequence[str] | None = None,
    data_start_row: int = 2,
) -> tuple[list[str], list[dict[str, object]]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            raw_headers = _row_at(reader, 1) or []
            headers = list(unique_headers or _unique_column_names(raw_headers))
            preview: list[dict[str, object]] = []
            for row_number, values in enumerate(reader, start=2):
                if row_number < data_start_row:
                    continue
                if len(preview) >= rows:
                    break
                preview.append(_row_dict(headers, values))
            return headers, preview
    except Exception as exc:
        raise UnreadableCSVError(f"Could not read CSV file '{path}': {exc}") from exc


def _read_selected_csv_rows(
    path: Path,
    selected_columns: Sequence[str],
    *,
    headers: Sequence[str],
    data_start_row: int,
) -> list[dict[str, object]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            rows: list[dict[str, object]] = []
            for row_number, values in enumerate(reader, start=1):
                if row_number < data_start_row:
                    continue
                row = _row_dict(headers, values)
                rows.append({column: row.get(column) for column in selected_columns})
            return rows
    except Exception as exc:
        raise UnreadableCSVError(f"Could not load selected CSV columns: {exc}") from exc


def _read_unique_csv_headers(path: Path, *, header_row: int = 1) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            raw_headers = _row_at(reader, header_row) or []
            return _unique_column_names(raw_headers)
    except Exception as exc:
        raise UnreadableCSVError(f"Could not read CSV header from '{path}': {exc}") from exc


def _read_units(path: Path, headers: Sequence[str], *, units_row: int | None) -> dict[str, str]:
    if units_row is None or units_row <= 0:
        return {header: "" for header in headers}
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            raw_units = _row_at(reader, units_row) or []
    except Exception as exc:
        raise UnreadableCSVError(f"Could not read CSV units row from '{path}': {exc}") from exc
    result: dict[str, str] = {}
    for header, value in zip_longest(headers, raw_units, fillvalue=""):
        if not header:
            continue
        result[str(header)] = str(value).strip()
    return result


def _resolve_csv_layout(
    path: Path,
    *,
    header_row: int | None,
    units_row: int | None,
    data_start_row: int | None,
) -> CSVLayout:
    detected = _detect_csv_layout(path)
    resolved_header = _positive_row(header_row, detected.header_row)
    if units_row is None:
        resolved_units = detected.units_row
    elif units_row <= 0:
        resolved_units = None
    else:
        resolved_units = int(units_row)
    resolved_data_start = _positive_row(data_start_row, detected.data_start_row)
    if resolved_data_start <= resolved_header:
        resolved_data_start = resolved_header + 1
    if resolved_units is not None and resolved_units >= resolved_data_start:
        resolved_units = None
    return CSVLayout(
        header_row=resolved_header,
        units_row=resolved_units,
        data_start_row=resolved_data_start,
    )


def _detect_csv_layout(path: Path) -> CSVLayout:
    rows = _read_raw_csv_rows(path, limit=50)
    if not rows:
        return CSVLayout()

    header_index = 0
    best_score = float("-inf")
    search_limit = min(len(rows), 20)
    for index in range(search_limit):
        row = rows[index]
        non_empty_ratio = _row_non_empty_ratio(row)
        if non_empty_ratio == 0:
            continue
        data_ratio = _row_data_ratio(row)
        text_ratio = _row_text_ratio(row)
        next_data_ratio = max((_row_data_ratio(candidate) for candidate in rows[index + 1 : index + 5]), default=0.0)
        time_bonus = 0.45 if any(_looks_like_time_name(str(value)) for value in row if str(value).strip()) else 0.0
        score = non_empty_ratio + text_ratio + next_data_ratio + time_bonus - data_ratio
        if score > best_score:
            best_score = score
            header_index = index

    data_index = None
    for index in range(header_index + 1, len(rows)):
        if _row_data_ratio(rows[index]) >= 0.4:
            data_index = index
            break
    if data_index is None:
        data_index = min(header_index + 1, len(rows))

    units_index = None
    for index in range(header_index + 1, data_index):
        if _row_non_empty_ratio(rows[index]) > 0 and _row_data_ratio(rows[index]) < 0.35:
            units_index = index
            break

    return CSVLayout(
        header_row=header_index + 1,
        units_row=None if units_index is None else units_index + 1,
        data_start_row=data_index + 1,
    )


def _read_raw_csv_rows(path: Path, *, limit: int) -> list[list[str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            rows: list[list[str]] = []
            for index, row in enumerate(reader):
                if index >= limit:
                    break
                rows.append([str(value).strip() for value in row])
            return rows
    except Exception as exc:
        raise UnreadableCSVError(f"Could not inspect CSV rows from '{path}': {exc}") from exc


def _row_at(reader: object, row_number: int) -> list[str] | None:
    for current, row in enumerate(reader, start=1):
        if current == row_number:
            return [str(value).strip() for value in row]
    return None


def _positive_row(value: int | None, default: int) -> int:
    if value is None:
        return default
    return max(1, int(value))


def _row_non_empty_ratio(row: Sequence[object]) -> float:
    if not row:
        return 0.0
    return sum(1 for value in row if str(value).strip()) / len(row)


def _row_data_ratio(row: Sequence[object]) -> float:
    non_empty = [value for value in row if str(value).strip()]
    if not non_empty:
        return 0.0
    usable = 0
    for value in non_empty:
        text = str(value).strip()
        if _is_float_like(text) or _parse_datetime_value(text) is not None:
            usable += 1
    return usable / len(non_empty)


def _row_text_ratio(row: Sequence[object]) -> float:
    non_empty = [value for value in row if str(value).strip()]
    if not non_empty:
        return 0.0
    return 1.0 - _row_data_ratio(non_empty)


def _is_float_like(value: object) -> bool:
    try:
        float(str(value).strip())
    except (TypeError, ValueError):
        return False
    return True


def _unique_column_names(raw_headers: Sequence[object]) -> list[str]:
    counts: dict[str, int] = {}
    names: list[str] = []
    for index, raw_header in enumerate(raw_headers, start=1):
        base = str(raw_header).strip() if raw_header is not None else ""
        if not base:
            base = f"Column {index}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        names.append(base if count == 1 else f"{base} {count}")
    return names


def _row_dict(headers: Sequence[str], values: Sequence[object]) -> dict[str, object]:
    return {
        header: "" if value is None else value
        for header, value in zip_longest(headers, values, fillvalue="")
        if header
    }


def _parse_datetime_value(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _finite_ratio(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.isfinite(values).sum() / values.size)


def _datetime_finite_ratio(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float((~np.isnat(values)).sum() / values.size)


def _is_numeric_dtype(dtype_text: str) -> bool:
    return any(marker in dtype_text for marker in NUMERIC_DTYPE_MARKERS)


def _looks_like_time_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in TIME_NAME_MARKERS)


def _unique_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _report(
    progress_callback: Callable[[int, int, str], None] | None,
    current: int,
    total: int,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(current, total, message)


def _raise_if_cancelled(cancel_token: object | None) -> None:
    if cancel_token is None:
        return
    cancelled = getattr(cancel_token, "is_cancelled", False)
    if callable(cancelled):
        cancelled = cancelled()
    if bool(cancelled):
        raise DataManagerError("CSV loading was cancelled.")
