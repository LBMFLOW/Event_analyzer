from __future__ import annotations

import csv
from pathlib import Path

import pytest

from event_analyzer.data.data_manager import (
    DataManager,
    DataManagerError,
    InvalidTimeColumnError,
    NonNumericColumnError,
    TooManyInvalidValuesError,
)
from scripts.generate_sample_csv import generate


def test_csv_loading_with_numeric_time(workspace_tmp: Path) -> None:
    path = workspace_tmp / "numeric_time.csv"
    generate(path, rows=40, seed=11)

    manager = DataManager(preview_rows=10)
    metadata = manager.open_csv(path)
    loaded = manager.select_columns(
        time_column="time_s",
        target_columns=["case_alpha", "case_beta"],
        auxiliary_columns=["aux_temperature"],
    )

    assert "time_s" in metadata.likely_time_columns
    assert loaded.time_is_datetime is False
    assert loaded.row_count == 40
    assert list(loaded.targets) == ["case_alpha", "case_beta"]
    assert list(loaded.auxiliaries) == ["aux_temperature"]
    assert loaded.time_values[0] <= loaded.time_values[-1]


def test_csv_loading_with_datetime_time(workspace_tmp: Path) -> None:
    path = workspace_tmp / "datetime_time.csv"
    generate(path, rows=25, seed=12)

    manager = DataManager(preview_rows=10)
    manager.open_csv(path)
    loaded = manager.select_columns(time_column="timestamp", target_columns=["case_alpha"])

    assert loaded.time_is_datetime is True
    assert loaded.row_count == 25
    assert loaded.time_values[0] > 0


def test_csv_loading_handles_missing_values_as_nan_gaps(workspace_tmp: Path) -> None:
    path = workspace_tmp / "missing_values.csv"
    _write_rows(
        path,
        ["time_s", "case_a", "aux_temperature"],
        [
            [0, 1.0, 20.0],
            [1, "", 21.0],
            [2, 3.0, ""],
            [3, 4.0, 23.0],
        ],
    )

    manager = DataManager(preview_rows=4, min_numeric_valid_ratio=0.5)
    manager.open_csv(path)
    loaded = manager.select_columns(
        time_column="time_s",
        target_columns=["case_a"],
        auxiliary_columns=["aux_temperature"],
    )

    assert loaded.row_count == 4
    assert loaded.targets["case_a"][1] != loaded.targets["case_a"][1]
    assert loaded.auxiliaries["aux_temperature"][2] != loaded.auxiliaries["aux_temperature"][2]
    assert loaded.warnings


def test_csv_loading_rejects_invalid_columns(workspace_tmp: Path) -> None:
    path = workspace_tmp / "invalid_columns.csv"
    _write_rows(
        path,
        ["time_s", "case_a", "operator_note"],
        [[0, 1.0, "ok"], [1, 2.0, "bad"]],
    )

    manager = DataManager(preview_rows=2)
    manager.open_csv(path)

    with pytest.raises(DataManagerError):
        manager.select_columns(time_column="time_s", target_columns=["missing_case"])
    with pytest.raises(NonNumericColumnError):
        manager.select_columns(time_column="time_s", target_columns=["operator_note"])
    with pytest.raises(InvalidTimeColumnError):
        manager.select_columns(time_column="operator_note", target_columns=["case_a"])


def test_csv_loading_accepts_ragged_target_columns(workspace_tmp: Path) -> None:
    path = workspace_tmp / "ragged_target.csv"
    _write_rows(
        path,
        ["time_s", "ends_early"],
        [[0, 1.0], [1, ""], [2, ""], [3, ""]],
    )

    manager = DataManager(preview_rows=1, min_numeric_valid_ratio=0.75)
    manager.open_csv(path)
    loaded = manager.select_columns(time_column="time_s", target_columns=["ends_early"])

    assert loaded.row_count == 4
    assert loaded.targets["ends_early"][0] == 1.0
    assert loaded.targets["ends_early"][1] != loaded.targets["ends_early"][1]
    assert loaded.warnings


def test_csv_loading_rejects_auxiliary_columns_with_too_many_missing_values(workspace_tmp: Path) -> None:
    path = workspace_tmp / "too_many_missing_aux.csv"
    _write_rows(
        path,
        ["time_s", "case_a", "mostly_missing_aux"],
        [[0, 1.0, 1.0], [1, 2.0, ""], [2, 3.0, ""], [3, 4.0, ""]],
    )

    manager = DataManager(preview_rows=1, min_numeric_valid_ratio=0.75)
    manager.open_csv(path)

    with pytest.raises(TooManyInvalidValuesError):
        manager.select_columns(
            time_column="time_s",
            target_columns=["case_a"],
            auxiliary_columns=["mostly_missing_aux"],
        )


def test_target_and_auxiliary_column_selection_are_loaded_separately(workspace_tmp: Path) -> None:
    path = workspace_tmp / "selection.csv"
    generate(path, rows=15, seed=13)

    manager = DataManager(preview_rows=10)
    manager.open_csv(path)
    loaded = manager.select_columns(
        time_column="time_s",
        target_columns=["case_alpha", "case_gamma"],
        auxiliary_columns=["aux_pressure"],
    )

    assert set(loaded.targets) == {"case_alpha", "case_gamma"}
    assert set(loaded.auxiliaries) == {"aux_pressure"}
    assert "case_beta" not in loaded.targets
    assert "aux_temperature" not in loaded.auxiliaries


def test_auto_target_loading_can_skip_text_only_leftover_columns(workspace_tmp: Path) -> None:
    path = workspace_tmp / "auto_targets_with_note.csv"
    _write_rows(
        path,
        ["time_s", "case_a", "operator_note"],
        [[0, 1.0, "ok"], [1, 2.0, "maintenance"]],
    )

    manager = DataManager(preview_rows=2)
    manager.open_csv(path)
    loaded = manager.select_columns(
        time_column="time_s",
        target_columns=["case_a", "operator_note"],
        ignore_invalid_targets=True,
    )

    assert list(loaded.targets) == ["case_a"]
    assert any("Skipped target column 'operator_note'" in warning for warning in loaded.warnings)


def test_duplicate_target_headers_are_numbered_and_loaded_separately(workspace_tmp: Path) -> None:
    path = workspace_tmp / "duplicate_headers.csv"
    _write_rows(
        path,
        ["time_s", "Cell voltage", "Cell voltage", "Cell voltage", "aux_current"],
        [
            [0, 3.1, 3.2, 3.3, 10],
            [1, 3.4, 3.5, 3.6, 11],
            [2, 3.7, "", 3.9, 12],
        ],
    )

    manager = DataManager(preview_rows=3)
    metadata = manager.open_csv(path)

    assert metadata.column_names == [
        "time_s",
        "Cell voltage",
        "Cell voltage 2",
        "Cell voltage 3",
        "aux_current",
    ]

    loaded = manager.select_columns(
        time_column="time_s",
        target_columns=["Cell voltage", "Cell voltage 2", "Cell voltage 3"],
        auxiliary_columns=["aux_current"],
    )

    assert list(loaded.targets) == ["Cell voltage", "Cell voltage 2", "Cell voltage 3"]
    assert loaded.targets["Cell voltage"][0] == 3.1
    assert loaded.targets["Cell voltage 2"][0] == 3.2
    assert loaded.targets["Cell voltage 3"][0] == 3.3
    assert loaded.targets["Cell voltage 2"][2] != loaded.targets["Cell voltage 2"][2]
    assert list(loaded.auxiliaries) == ["aux_current"]


def test_csv_layout_autodetects_header_units_and_data_start_rows(workspace_tmp: Path) -> None:
    path = workspace_tmp / "header_units_rows.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Simulation export", "", ""])
        writer.writerow(["time_s", "case_a", "aux_pressure"])
        writer.writerow(["s", "V", "Pa"])
        writer.writerows([[0, 1.0, 101.0], [1, 2.0, 102.0], [2, 3.0, 103.0]])

    manager = DataManager(preview_rows=3)
    metadata = manager.open_csv(path)
    loaded = manager.select_columns(
        time_column="time_s",
        target_columns=["case_a"],
        auxiliary_columns=["aux_pressure"],
    )

    assert metadata.layout.header_row == 2
    assert metadata.layout.units_row == 3
    assert metadata.layout.data_start_row == 4
    assert metadata.units == {"time_s": "s", "case_a": "V", "aux_pressure": "Pa"}
    assert loaded.units["time_s"] == "s"
    assert loaded.units["case_a"] == "V"
    assert loaded.row_count == 3


def test_sample_data_generator_writes_expected_columns(workspace_tmp: Path) -> None:
    path = workspace_tmp / "sample.csv"

    generate(path, rows=3, seed=14)

    with path.open(newline="", encoding="utf-8") as handle:
        headers = next(csv.reader(handle))
    assert headers == [
        "timestamp",
        "time_s",
        "case_alpha",
        "case_beta",
        "case_gamma",
        "aux_temperature",
        "aux_pressure",
        "operator_note",
    ]


def _write_rows(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
