from __future__ import annotations

import json
from pathlib import Path
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

    def test_session_round_trip_includes_advanced_state(self) -> None:
        path = Path.cwd() / "test_session_settings_project.json"
        state = SessionState(
            csv_path="sample.csv",
            time_column="time",
            target_columns=["case_a"],
            auxiliary_columns=["temperature"],
            auxiliary_axes={"temperature": "y2"},
            dividers=[{"id": "d1", "time": 1.0}],
            threshold=2.5,
            region=(1.0, 3.0),
            colors={"case_a": "#ff0000"},
            visibility={"case_a": False},
            theme="dark",
        )

        state.save(path)
        restored = SessionState.load(path)

        self.assertEqual(restored.csv_path, "sample.csv")
        self.assertEqual(restored.auxiliary_axes, {"temperature": "y2"})
        self.assertEqual(restored.dividers[0]["time"], 1.0)
        self.assertEqual(restored.region, (1.0, 3.0))
        self.assertEqual(restored.colors["case_a"], "#ff0000")
        self.assertFalse(restored.visibility["case_a"])
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
