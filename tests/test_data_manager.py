from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

POLARS_AVAILABLE = importlib.util.find_spec("polars") is not None

if POLARS_AVAILABLE:
    from event_analyzer.data.data_manager import (
        DataManager,
        EmptyDataError,
        InvalidTimeColumnError,
        MissingFileError,
        NonNumericColumnError,
    )
    from scripts.generate_sample_csv import generate


@unittest.skipUnless(POLARS_AVAILABLE, "polars is required for DataManager tests")
class DataManagerTests(unittest.TestCase):
    def test_open_csv_infers_columns_and_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            generate(path, rows=50, seed=1)

            manager = DataManager(preview_rows=10)
            metadata = manager.open_csv(path)

            self.assertIn("time_s", metadata.column_names)
            self.assertIn("case_alpha", metadata.numeric_columns)
            self.assertIn("timestamp", metadata.likely_time_columns)
            self.assertEqual(len(manager.preview(5).rows), 5)

    def test_select_columns_with_numeric_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            generate(path, rows=25, seed=2)

            manager = DataManager(preview_rows=10)
            manager.open_csv(path)
            loaded = manager.select_columns(
                time_column="time_s",
                target_columns=["case_alpha", "case_beta"],
                auxiliary_columns=["aux_temperature"],
            )

            self.assertFalse(loaded.time_is_datetime)
            self.assertEqual(loaded.row_count, 25)
            self.assertEqual(set(loaded.targets), {"case_alpha", "case_beta"})
            self.assertEqual(set(loaded.auxiliaries), {"aux_temperature"})

    def test_select_columns_with_datetime_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            generate(path, rows=20, seed=3)

            manager = DataManager(preview_rows=10)
            manager.open_csv(path)
            loaded = manager.select_columns(time_column="timestamp", target_columns=["case_alpha"])

            self.assertTrue(loaded.time_is_datetime)
            self.assertEqual(loaded.row_count, 20)

    def test_missing_file_has_clear_error(self) -> None:
        manager = DataManager()
        with self.assertRaises(MissingFileError):
            manager.open_csv("does-not-exist.csv")

    def test_non_numeric_target_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            generate(path, rows=20, seed=4)

            manager = DataManager(preview_rows=20)
            manager.open_csv(path)

            with self.assertRaises(NonNumericColumnError):
                manager.select_columns(time_column="time_s", target_columns=["operator_note"])

    def test_invalid_time_column_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            generate(path, rows=20, seed=5)

            manager = DataManager(preview_rows=20)
            manager.open_csv(path)

            with self.assertRaises(InvalidTimeColumnError):
                manager.select_columns(time_column="operator_note", target_columns=["case_alpha"])

    def test_empty_data_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.csv"
            path.write_text("time_s,case_alpha\n", encoding="utf-8")

            manager = DataManager()
            manager.open_csv(path)

            with self.assertRaises(EmptyDataError):
                manager.select_columns(time_column="time_s", target_columns=["case_alpha"])

    def test_ragged_target_columns_are_loaded_as_nan_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ragged_target.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time_s", "ends_early"])
                for index in range(10):
                    writer.writerow([index, index if index < 2 else ""])

            manager = DataManager(preview_rows=10, min_numeric_valid_ratio=0.8)
            manager.open_csv(path)
            loaded = manager.select_columns(time_column="time_s", target_columns=["ends_early"])

            self.assertEqual(loaded.row_count, 10)
            self.assertTrue(loaded.targets["ends_early"][2] != loaded.targets["ends_early"][2])
            self.assertTrue(loaded.warnings)

    def test_ragged_auxiliary_columns_are_loaded_as_nan_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ragged_auxiliary.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time_s", "case_a", "ends_early_aux"])
                for index in range(10):
                    writer.writerow([index, index, index if index < 2 else ""])

            manager = DataManager(preview_rows=1, min_numeric_valid_ratio=0.8)
            manager.open_csv(path)

            loaded = manager.select_columns(
                time_column="time_s",
                target_columns=["case_a"],
                auxiliary_columns=["ends_early_aux"],
            )

            self.assertEqual(loaded.row_count, 10)
            self.assertEqual(loaded.auxiliaries["ends_early_aux"][1], 1.0)
            self.assertTrue(loaded.auxiliaries["ends_early_aux"][2] != loaded.auxiliaries["ends_early_aux"][2])
            self.assertTrue(loaded.warnings)

    def test_duplicate_headers_preserve_source_column_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicate_headers.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time_s", "Cell voltage", "Cell voltage", "Cell voltage"])
                writer.writerows([[0, 3.1, 3.2, 3.3], [1, 3.4, 3.5, 3.6]])

            manager = DataManager(preview_rows=2)
            metadata = manager.open_csv(path)
            loaded = manager.select_columns(
                time_column="time_s",
                target_columns=["Cell voltage", "Cell voltage 2", "Cell voltage 3"],
            )

            self.assertEqual(metadata.source_columns["Cell voltage 2"], "Cell voltage")
            self.assertEqual(metadata.source_columns["Cell voltage 3"], "Cell voltage")
            self.assertEqual(loaded.source_columns["Cell voltage 2"], "Cell voltage")


if __name__ == "__main__":
    unittest.main()
