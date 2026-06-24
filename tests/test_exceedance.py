from __future__ import annotations

from datetime import datetime, timezone
import math
import unittest

from event_analyzer.analysis.exceedance import analyze_exceedances, detect_exceedance_events


class ExceedanceEventDetectionTests(unittest.TestCase):
    def test_no_exceedance(self) -> None:
        events = detect_exceedance_events(
            time=[0, 1, 2],
            values_by_case={"case_a": [0, 1, 1]},
            threshold=1,
            region_start=0,
            region_end=2,
        )

        self.assertEqual(events, [])

    def test_one_exceedance(self) -> None:
        events = detect_exceedance_events(
            time=[0, 1, 2, 3],
            values_by_case={"case_a": [0, 2, 2, 0]},
            threshold=1,
            region_start=0,
            region_end=3,
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.case_name, "case_a")
        self.assertEqual(event.event_index, 1)
        self.assertAlmostEqual(event.start_time, 0.5)
        self.assertAlmostEqual(event.end_time, 2.5)
        self.assertAlmostEqual(event.duration, 2.0)
        self.assertAlmostEqual(event.peak_value, 2.0)
        self.assertAlmostEqual(event.peak_time, 1.0)
        self.assertEqual(event.threshold, 1)
        self.assertEqual(event.region_start, 0)
        self.assertEqual(event.region_end, 3)

    def test_multiple_exceedances_are_indexed_per_case(self) -> None:
        events = detect_exceedance_events(
            time=[0, 1, 2, 3, 4, 5],
            values_by_case={
                "case_b": [0, 2, 0, 0, 4, 0],
                "case_a": [0, 0, 2, 0, 0, 0],
            },
            threshold=1,
            region_start=0,
            region_end=5,
        )

        self.assertEqual([event.case_name for event in events], ["case_a", "case_b", "case_b"])
        self.assertEqual([event.event_index for event in events], [1, 1, 2])
        self.assertAlmostEqual(events[0].start_time, 1.5)
        self.assertAlmostEqual(events[0].end_time, 2.5)
        self.assertAlmostEqual(events[1].start_time, 0.5)
        self.assertAlmostEqual(events[1].end_time, 1.5)
        self.assertAlmostEqual(events[2].start_time, 3.25)
        self.assertAlmostEqual(events[2].end_time, 4.75)

    def test_exceedance_touching_region_boundary_is_clipped(self) -> None:
        events = detect_exceedance_events(
            time=[0, 10],
            values_by_case={"case_a": [10, 10]},
            threshold=5,
            region_start=2,
            region_end=8,
        )

        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].start_time, 2)
        self.assertAlmostEqual(events[0].end_time, 8)
        self.assertAlmostEqual(events[0].duration, 6)
        self.assertAlmostEqual(events[0].peak_value, 10)

    def test_nonuniform_time_spacing_uses_interpolated_crossings(self) -> None:
        events = detect_exceedance_events(
            time=[0, 1, 3, 4, 7, 10],
            values_by_case={"case_a": [0, 2, 2, 0, 3, 0]},
            threshold=1,
            region_start=0,
            region_end=10,
        )

        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0].start_time, 0.5)
        self.assertAlmostEqual(events[0].end_time, 3.5)
        self.assertAlmostEqual(events[0].duration, 3.0)
        self.assertAlmostEqual(events[1].start_time, 5.0)
        self.assertAlmostEqual(events[1].end_time, 9.0)
        self.assertAlmostEqual(events[1].duration, 4.0)
        self.assertAlmostEqual(events[1].peak_value, 3.0)
        self.assertAlmostEqual(events[1].peak_time, 7.0)

    def test_nan_values_break_events_safely(self) -> None:
        events = detect_exceedance_events(
            time=[0, 1, 2, 3, 4],
            values_by_case={"case_a": [0, 2, math.nan, 2, 0]},
            threshold=1,
            region_start=0,
            region_end=4,
        )

        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0].start_time, 0.5)
        self.assertAlmostEqual(events[0].end_time, 1.0)
        self.assertAlmostEqual(events[1].start_time, 3.0)
        self.assertAlmostEqual(events[1].end_time, 3.5)

    def test_values_equal_to_threshold_are_not_exceedances(self) -> None:
        events = detect_exceedance_events(
            time=[0, 1, 2, 3],
            values_by_case={
                "equal_only": [1, 1, 1, 1],
                "touch_then_above": [0, 1, 2, 1],
            },
            threshold=1,
            region_start=0,
            region_end=3,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].case_name, "touch_then_above")
        self.assertAlmostEqual(events[0].start_time, 1.0)
        self.assertAlmostEqual(events[0].end_time, 3.0)

    def test_datetime_like_numeric_input(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        time = [base, base + 60, base + 120]

        events = detect_exceedance_events(
            time=time,
            values_by_case={"case_a": [0, 2, 0]},
            threshold=1,
            region_start=base,
            region_end=base + 120,
        )

        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].start_time, base + 30)
        self.assertAlmostEqual(events[0].end_time, base + 90)
        self.assertAlmostEqual(events[0].duration, 60)

    def test_legacy_region_argument_still_works(self) -> None:
        events = analyze_exceedances(
            [0, 10],
            {"case_a": [0, 10]},
            threshold=5,
            region=(2, 8),
        )

        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].start_time, 5.0)
        self.assertAlmostEqual(events[0].end_time, 8.0)


if __name__ == "__main__":
    unittest.main()
