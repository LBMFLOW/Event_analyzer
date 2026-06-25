from __future__ import annotations

import json
from pathlib import Path
import unittest

from event_analyzer.analysis.exceedance import ExceedanceEvent
from event_analyzer.controllers.threshold_manager import ThresholdSummary
from event_analyzer.exporters import (
    export_analysis_summary_json,
    export_events_csv,
    export_selected_region_csv,
)


class ExporterTests(unittest.TestCase):
    def tearDown(self) -> None:
        for path in Path.cwd().glob("test_exporters_*"):
            if path.is_file():
                path.unlink()

    def test_export_events_csv_writes_all_event_fields(self) -> None:
        event = ExceedanceEvent("case_a", 1, 0.5, 2.5, 2.0, 3.0, 1.0, 1.0, 0.0, 4.0)

        path = Path.cwd() / "test_exporters_events.csv"
        export_events_csv([event], path)
        text = path.read_text(encoding="utf-8")

        self.assertIn("case_name,event_index,start_time", text)
        self.assertIn("case_a,1,0.5,2.5,2.0,3.0,1.0,1.0,0.0,4.0", text)

    def test_export_events_csv_can_include_region_name(self) -> None:
        event = ExceedanceEvent("case_a", 1, 0.5, 2.5, 2.0, 3.0, 1.0, 1.0, 0.0, 4.0)

        path = Path.cwd() / "test_exporters_events_region_name.csv"
        export_events_csv([event], path, region_name="Discharge 1")
        rows = path.read_text(encoding="utf-8").strip().splitlines()

        self.assertTrue(rows[0].startswith("region_name,case_name,event_index"))
        self.assertTrue(rows[1].startswith("Discharge 1,case_a,1"))

    def test_export_selected_region_csv_from_mapping(self) -> None:
        data = {
            "time": [0.0, 1.0, 2.0, 3.0],
            "case_a": [10.0, 11.0, 12.0, 13.0],
            "aux": [100.0, 101.0, 102.0, 103.0],
        }

        path = Path.cwd() / "test_exporters_region.csv"
        export_selected_region_csv(data, (1.0, 2.0), path)
        rows = path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(rows[0], "time,case_a,aux")
        self.assertEqual(rows[1], "1.0,11.0,101.0")
        self.assertEqual(rows[2], "2.0,12.0,102.0")
        self.assertEqual(len(rows), 3)

    def test_export_selected_region_csv_can_include_region_name(self) -> None:
        data = {
            "time": [0.0, 1.0, 2.0],
            "case_a": [10.0, 11.0, 12.0],
        }

        path = Path.cwd() / "test_exporters_region_name.csv"
        export_selected_region_csv(data, (1.0, 2.0), path, region_name="Charge step")
        rows = path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(rows[0], "region_name,time,case_a")
        self.assertEqual(rows[1], "Charge step,1.0,11.0")

    def test_export_analysis_summary_json_handles_disabled_threshold(self) -> None:
        summary = ThresholdSummary(
            threshold_enabled=False,
            threshold=None,
            exceeding_cases=(),
            event_count=0,
            max_exceedance_value=None,
            longest_exceedance_duration=None,
            region_start=None,
            region_end=None,
        )

        path = Path.cwd() / "test_exporters_summary.json"
        export_analysis_summary_json(summary, [], path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertFalse(payload["threshold_enabled"])
        self.assertIsNone(payload["threshold"])
        self.assertEqual(payload["event_count"], 0)


if __name__ == "__main__":
    unittest.main()
