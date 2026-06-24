from __future__ import annotations

import unittest

from event_analyzer.controllers.divider_manager import (
    DividerManager,
    DividerOutOfRangeError,
    DividerStyle,
    DuplicateDividerTimeError,
)
from event_analyzer.controllers.region_selector import RegionSelector


class FakePlotAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[float | None, float | None]] = []

    def set_selected_region(self, region_start: float | None, region_end: float | None) -> None:
        self.calls.append((region_start, region_end))


class DividerManagerTests(unittest.TestCase):
    def test_add_dividers_sorts_and_relabels(self) -> None:
        manager = DividerManager(time_range=(0, 10))
        second = manager.add_divider(8)
        first = manager.add_divider(2)

        self.assertEqual(manager.times(), [2.0, 8.0])
        self.assertEqual(manager.labels_and_times(), [("D1", 2.0), ("D2", 8.0)])
        self.assertEqual(manager.get_divider(first.id).label, "D1")
        self.assertEqual(manager.get_divider(second.id).label, "D2")

    def test_edit_and_drag_divider_time_resorts(self) -> None:
        manager = DividerManager(time_range=(0, 10))
        left = manager.add_divider(2)
        right = manager.add_divider(8)

        updated = manager.edit_divider_time(right.id, 1)
        self.assertEqual(updated.label, "D1")
        self.assertEqual(manager.labels_and_times(), [("D1", 1.0), ("D2", 2.0)])

        dragged = manager.drag_divider(left.id, 9)
        self.assertEqual(dragged.label, "D2")
        self.assertEqual(manager.labels_and_times(), [("D1", 1.0), ("D2", 9.0)])

    def test_duplicate_times_are_rejected(self) -> None:
        manager = DividerManager(time_range=(0, 10), duplicate_tolerance=1e-6)
        first = manager.add_divider(5)

        with self.assertRaises(DuplicateDividerTimeError):
            manager.add_divider(5 + 1e-7)

        second = manager.add_divider(7)
        with self.assertRaises(DuplicateDividerTimeError):
            manager.edit_divider_time(second.id, first.time)

    def test_divider_outside_data_range_is_rejected(self) -> None:
        manager = DividerManager(time_range=(0, 10))

        with self.assertRaises(DividerOutOfRangeError):
            manager.add_divider(-1)

        with self.assertRaises(DividerOutOfRangeError):
            manager.add_divider(11)

    def test_delete_clear_and_serialize(self) -> None:
        manager = DividerManager(time_range=(0, 10))
        manager.add_divider(8, divider_id="right", style=DividerStyle(color="#ff0000", line_style="solid", width=2))
        manager.add_divider(2, divider_id="left")

        removed = manager.delete_divider("left")
        self.assertEqual(removed.id, "left")
        self.assertEqual(manager.labels_and_times(), [("D1", 8.0)])

        records = manager.serialize()
        restored = DividerManager.from_serialized(records, time_range=(0, 10))
        self.assertEqual(restored.labels_and_times(), [("D1", 8.0)])
        self.assertEqual(restored.get_divider("right").style.color, "#ff0000")

        restored.clear_all_dividers()
        self.assertEqual(restored.dividers, [])


class RegionSelectorTests(unittest.TestCase):
    def test_no_dividers_creates_whole_range_region(self) -> None:
        selector = RegionSelector(time_range=(0, 10))

        regions = selector.regions()

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].index, 0)
        self.assertEqual(regions[0].label, "All data")
        self.assertEqual((regions[0].start_time, regions[0].end_time), (0.0, 10.0))

    def test_regions_before_between_and_after_dividers(self) -> None:
        manager = DividerManager(time_range=(0, 10))
        manager.add_divider(2)
        manager.add_divider(7)
        selector = RegionSelector(time_range=(0, 10))

        regions = selector.regions(manager)

        self.assertEqual([region.label for region in regions], ["Before D1", "D1 to D2", "After D2"])
        self.assertEqual([(r.start_time, r.end_time) for r in regions], [(0.0, 2.0), (2.0, 7.0), (7.0, 10.0)])

    def test_click_inside_region_selects_region(self) -> None:
        manager = DividerManager(time_range=(0, 10))
        manager.add_divider(2)
        manager.add_divider(7)
        selector = RegionSelector(time_range=(0, 10))

        region = selector.select_at(4, manager)

        self.assertIsNotNone(region)
        self.assertEqual(region.index, 1)
        self.assertEqual(region.label, "D1 to D2")
        self.assertEqual(selector.current_region_tuple(), (2.0, 7.0))

    def test_click_exactly_on_divider_selects_right_region_by_default(self) -> None:
        manager = DividerManager(time_range=(0, 10))
        manager.add_divider(2)
        manager.add_divider(7)
        selector = RegionSelector(time_range=(0, 10))

        region = selector.select_at(2, manager)

        self.assertIsNotNone(region)
        self.assertEqual(region.index, 1)
        self.assertEqual(region.label, "D1 to D2")

    def test_click_exactly_on_divider_can_select_left_region(self) -> None:
        manager = DividerManager(time_range=(0, 10))
        manager.add_divider(2)
        selector = RegionSelector(time_range=(0, 10), boundary_policy="left")

        region = selector.select_at(2, manager)

        self.assertIsNotNone(region)
        self.assertEqual(region.index, 0)
        self.assertEqual(region.label, "Before D1")

    def test_outside_divider_times_are_ignored_for_regions(self) -> None:
        selector = RegionSelector(time_range=(0, 10))

        regions = selector.regions([-5, 2, 20])

        self.assertEqual([(r.start_time, r.end_time) for r in regions], [(0.0, 2.0), (2.0, 10.0)])

    def test_duplicate_divider_times_are_collapsed_for_region_calculation(self) -> None:
        selector = RegionSelector(time_range=(0, 10))

        regions = selector.regions([2, 2, 5])

        self.assertEqual([(r.start_time, r.end_time) for r in regions], [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0)])

    def test_selected_region_highlights_plot_adapter(self) -> None:
        plot = FakePlotAdapter()
        selector = RegionSelector(time_range=(0, 10), plot_adapter=plot)

        selector.select_at(4, [2, 7])
        selector.clear_selection()

        self.assertEqual(plot.calls, [(2.0, 7.0), (None, None)])

    def test_deleting_divider_reconciles_selected_region(self) -> None:
        manager = DividerManager(time_range=(0, 10))
        left = manager.add_divider(2)
        manager.add_divider(7)
        selector = RegionSelector(time_range=(0, 10))
        selector.select_at(4, manager)

        manager.delete_divider(left.id)
        region = selector.reconcile_after_divider_change(manager)

        self.assertIsNotNone(region)
        self.assertEqual(region.index, 0)
        self.assertEqual(region.label, "Before D1")
        self.assertEqual(selector.current_region_tuple(), (0.0, 7.0))


if __name__ == "__main__":
    unittest.main()

