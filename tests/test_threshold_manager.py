from __future__ import annotations

import unittest

from event_analyzer.controllers.threshold_manager import ThresholdError, ThresholdManager
from event_analyzer.analysis.exceedance import detect_exceedance_events


class FakePlot:
    def __init__(self) -> None:
        self.threshold_values: list[float | None] = []
        self.highlighted_cases: list[tuple[str, ...]] = []

    def set_threshold(self, value: float | None) -> None:
        self.threshold_values.append(value)

    def highlight_exceeding_cases(self, case_names) -> None:
        self.highlighted_cases.append(tuple(case_names))


class FakeEventsView:
    def __init__(self) -> None:
        self.events = []

    def set_events(self, events) -> None:
        self.events = list(events)


class FakeSummaryView:
    def __init__(self) -> None:
        self.summary = None

    def set_threshold_summary(self, summary) -> None:
        self.summary = summary


class ThresholdManagerTests(unittest.TestCase):
    def test_threshold_detects_and_highlights_target_cases_only(self) -> None:
        plot = FakePlot()
        chart = FakeEventsView()
        table = FakeEventsView()
        summary_view = FakeSummaryView()
        manager = ThresholdManager(
            plot_adapter=plot,
            chart_adapter=chart,
            table_adapter=table,
            summary_adapter=summary_view,
        )
        manager.set_target_data(
            [0, 1, 2, 3],
            {
                "case_a": [0, 2, 2, 0],
                "case_b": [0, 0, 0, 0],
            },
        )

        manager.set_threshold(1)

        self.assertEqual(plot.threshold_values[-1], 1)
        self.assertEqual(plot.highlighted_cases[-1], ("case_a",))
        self.assertEqual(len(manager.events), 1)
        self.assertEqual(chart.events, manager.events)
        self.assertEqual(table.events, manager.events)
        self.assertEqual(summary_view.summary.exceeding_cases, ("case_a",))
        self.assertEqual(summary_view.summary.exceeding_case_count, 1)
        self.assertEqual(summary_view.summary.event_count, 1)
        self.assertAlmostEqual(summary_view.summary.max_exceedance_value, 2)
        self.assertAlmostEqual(summary_view.summary.longest_exceedance_duration, 2)

    def test_selected_region_restricts_analysis(self) -> None:
        plot = FakePlot()
        summary_view = FakeSummaryView()
        manager = ThresholdManager(plot_adapter=plot, summary_adapter=summary_view)
        manager.set_target_data([0, 1, 2, 3], {"case_a": [0, 2, 2, 0]})
        manager.set_region(1.25, 1.75)

        manager.set_threshold(1)

        self.assertEqual(len(manager.events), 1)
        self.assertAlmostEqual(manager.events[0].start_time, 1.25)
        self.assertAlmostEqual(manager.events[0].end_time, 1.75)
        self.assertEqual(summary_view.summary.region_start, 1.25)
        self.assertEqual(summary_view.summary.region_end, 1.75)

    def test_create_edit_drag_and_disable_threshold(self) -> None:
        plot = FakePlot()
        chart = FakeEventsView()
        summary_view = FakeSummaryView()
        manager = ThresholdManager(plot_adapter=plot, chart_adapter=chart, summary_adapter=summary_view)
        manager.set_target_data([0, 1, 2], {"case_a": [0, 3, 0]})

        manager.create_from_plot(1)
        self.assertTrue(manager.state.enabled)
        self.assertEqual(manager.value, 1)

        manager.edit_threshold(2)
        self.assertEqual(manager.value, 2)

        manager.drag_threshold(0.5)
        self.assertEqual(manager.value, 0.5)

        manager.disable_threshold()
        self.assertFalse(manager.state.enabled)
        self.assertIsNone(manager.value)
        self.assertEqual(manager.events, [])
        self.assertEqual(chart.events, [])
        self.assertEqual(plot.threshold_values[-1], None)
        self.assertEqual(plot.highlighted_cases[-1], ())
        self.assertEqual(summary_view.summary.event_count, 0)

    def test_invalid_threshold_is_rejected(self) -> None:
        manager = ThresholdManager()

        with self.assertRaises(ThresholdError):
            manager.set_threshold(float("nan"))

    def test_detector_reports_progress_and_checks_cancellation_between_cases(self) -> None:
        progress: list[tuple[int, int, str]] = []

        class Token:
            def __init__(self) -> None:
                self.is_cancelled = False

        token = Token()

        def on_progress(current: int, total: int, message: str) -> None:
            progress.append((current, total, message))
            if current == 1:
                token.is_cancelled = True

        with self.assertRaises(RuntimeError):
            detect_exceedance_events(
                [0, 1, 2],
                {"case_a": [0, 2, 0], "case_b": [0, 2, 0]},
                1,
                None,
                None,
                cancel_token=token,
                progress_callback=on_progress,
            )

        self.assertEqual(progress[0][0], 0)
        self.assertEqual(progress[1][0], 1)


if __name__ == "__main__":
    unittest.main()
