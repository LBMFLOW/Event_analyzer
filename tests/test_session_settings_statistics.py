from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest

import numpy as np

from event_analyzer.analysis.exceedance import ExceedanceEvent
from event_analyzer.analysis.statistics import region_statistics_from_arrays
from event_analyzer.controllers.app_settings import AppSettings
from event_analyzer.controllers.session import SessionState
from event_analyzer.data.units import split_column_unit, unit_for_column


class SessionSettingsStatisticsTests(unittest.TestCase):
    def tearDown(self) -> None:
        for path in Path.cwd().glob("test_session_settings_*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    def test_session_round_trip_includes_advanced_state(self) -> None:
        path = Path.cwd() / "test_session_settings_project.json"
        state = SessionState(
            csv_path="sample.csv",
            time_column="time",
            target_columns=["case_a"],
            auxiliary_columns=["temperature"],
            auxiliary_axes={"temperature": "y2"},
            header_row=2,
            units_row=3,
            data_start_row=4,
            plot_title="My plot",
            x_axis_title="Time (s)",
            y_axis_title="Voltage (V)",
            main_time_range=(10.0, 20.0),
            main_target_range=(0.2, 0.8),
            chart_y_range=(0.0, 100.0),
            chart_x_axis_title="Selected cases",
            chart_y_axis_title="Seconds above limit",
            chart_axis_title_font_size=18,
            chart_tick_label_font_size=16,
            chart_value_label_font_size=9,
            chart_value_label_angle=90,
            count_curve_settings={
                "threshold_range": (0.0, 5.0),
                "levels": 500,
                "title": "Count curve",
                "x_axis_title": "Threshold",
                "y_axis_title": "Cases",
            },
            dividers=[{"id": "d1", "time": 1.0}],
            threshold=2.5,
            region=(1.0, 3.0),
            region_name="Discharge 1",
            colors={"case_a": "#ff0000"},
            visibility={"case_a": False},
            trace_boxes_visible=False,
            theme="dark",
        )

        state.save(path)
        restored = SessionState.load(path)

        self.assertEqual(restored.csv_path, "sample.csv")
        self.assertEqual(restored.auxiliary_axes, {"temperature": "y2"})
        self.assertEqual(restored.header_row, 2)
        self.assertEqual(restored.units_row, 3)
        self.assertEqual(restored.data_start_row, 4)
        self.assertEqual(restored.plot_title, "My plot")
        self.assertEqual(restored.x_axis_title, "Time (s)")
        self.assertEqual(restored.y_axis_title, "Voltage (V)")
        self.assertEqual(restored.main_time_range, (10.0, 20.0))
        self.assertEqual(restored.main_target_range, (0.2, 0.8))
        self.assertEqual(restored.chart_y_range, (0.0, 100.0))
        self.assertEqual(restored.chart_x_axis_title, "Selected cases")
        self.assertEqual(restored.chart_y_axis_title, "Seconds above limit")
        self.assertEqual(restored.chart_axis_title_font_size, 18)
        self.assertEqual(restored.chart_tick_label_font_size, 16)
        self.assertEqual(restored.chart_value_label_font_size, 9)
        self.assertEqual(restored.chart_value_label_angle, 90)
        self.assertEqual(restored.count_curve_settings["threshold_range"], [0.0, 5.0])
        self.assertEqual(restored.count_curve_settings["x_axis_title"], "Threshold")
        self.assertEqual(restored.dividers[0]["time"], 1.0)
        self.assertEqual(restored.region, (1.0, 3.0))
        self.assertEqual(restored.region_name, "Discharge 1")
        self.assertEqual(restored.colors["case_a"], "#ff0000")
        self.assertFalse(restored.visibility["case_a"])
        self.assertFalse(restored.trace_boxes_visible)
        self.assertEqual(restored.theme, "dark")

    def test_session_load_accepts_legacy_divider_times(self) -> None:
        path = Path.cwd() / "test_session_settings_legacy.json"
        path.write_text(json.dumps({"dividers": [1.0, 2.0]}), encoding="utf-8")

        restored = SessionState.load(path)

        self.assertEqual(restored.dividers, [{"time": 1.0}, {"time": 2.0}])

    def test_app_settings_keeps_recent_files_unique_and_capped(self) -> None:
        settings = AppSettings()
        for index in range(12):
            settings.add_recent_file(f"file_{index}.csv")
        settings.add_recent_file("file_5.csv")

        self.assertEqual(settings.recent_files[0], "file_5.csv")
        self.assertEqual(len(settings.recent_files), 10)
        self.assertEqual(len(set(settings.recent_files)), 10)

    def test_app_settings_remembers_last_csv_directory(self) -> None:
        directory = Path.cwd() / "test_session_settings_csv_dir"
        directory.mkdir()
        csv_path = directory / "sample.csv"
        csv_path.write_text("time,value\n0,1\n", encoding="utf-8")
        settings_path = directory / "settings.json"
        settings = AppSettings()

        settings.add_recent_file(csv_path)
        settings.save(settings_path)
        restored = AppSettings.load(settings_path)

        self.assertEqual(restored.last_csv_directory, str(directory))
        self.assertEqual(restored.open_csv_directory(), str(directory))

    def test_app_settings_loads_legacy_settings_without_last_csv_directory(self) -> None:
        path = Path.cwd() / "test_session_settings_app_legacy.json"
        path.write_text(json.dumps({"recent_files": ["sample.csv"], "theme": "dark"}), encoding="utf-8")

        restored = AppSettings.load(path)

        self.assertEqual(restored.theme, "dark")
        self.assertEqual(restored.last_csv_directory, "")
        self.assertEqual(restored.last_save_directory, "")

    def test_app_settings_remembers_last_save_directory(self) -> None:
        directory = Path.cwd() / "test_session_settings_save_dir"
        directory.mkdir()
        export_path = directory / "chart.svg"
        settings_path = directory / "settings.json"
        settings = AppSettings()

        settings.remember_save_path(export_path)
        settings.save(settings_path)
        restored = AppSettings.load(settings_path)

        self.assertEqual(restored.last_save_directory, str(directory))
        self.assertEqual(restored.save_directory(), str(directory))

    def test_unit_parsing(self) -> None:
        self.assertEqual(split_column_unit("pressure [Pa]"), ("pressure", "Pa"))
        self.assertEqual(unit_for_column("temperature (C)"), "C")
        self.assertEqual(unit_for_column("case_alpha"), "")

    def test_region_statistics_include_requested_metrics(self) -> None:
        events = [ExceedanceEvent("case_a", 1, 1.0, 2.0, 1.0, 3.0, 1.5, 1.0, 0.0, 3.0)]

        rows = region_statistics_from_arrays(
            [0.0, 1.0, 2.0, 3.0],
            {"case_a": [0.0, 2.0, 3.0, 0.0]},
            (0.0, 3.0),
            threshold=1.0,
            events=events,
            units={"case_a": "V"},
        )

        row = rows[0]
        self.assertEqual(row["unit"], "V")
        self.assertAlmostEqual(row["min"], 0.0)
        self.assertAlmostEqual(row["max"], 3.0)
        self.assertAlmostEqual(row["mean"], 1.25)
        self.assertAlmostEqual(row["median"], 1.0)
        self.assertAlmostEqual(row["std"], float(np.std([0.0, 2.0, 3.0, 0.0])))
        self.assertAlmostEqual(row["time_above_threshold"], 1.0)
        self.assertEqual(row["event_count"], 1)


if __name__ == "__main__":
    unittest.main()
