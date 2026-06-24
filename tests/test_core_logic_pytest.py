from __future__ import annotations

from pathlib import Path

import pytest

from event_analyzer.analysis.exceedance import detect_exceedance_events
from event_analyzer.controllers.divider_manager import DividerManager
from event_analyzer.controllers.region_selector import RegionSelector
from event_analyzer.exporters import export_main_plot_svg


def test_divider_sorting_and_relabeling() -> None:
    manager = DividerManager(time_range=(0, 10))

    manager.add_divider(8)
    manager.add_divider(2)
    manager.add_divider(5)

    assert manager.times() == [2.0, 5.0, 8.0]
    assert manager.labels_and_times() == [("D1", 2.0), ("D2", 5.0), ("D3", 8.0)]


def test_divider_editing_resorts_and_keeps_ids() -> None:
    manager = DividerManager(time_range=(0, 10))
    left = manager.add_divider(2)
    right = manager.add_divider(8)

    edited = manager.edit_divider_time(right.id, 1)

    assert edited.id == right.id
    assert edited.label == "D1"
    assert manager.get_divider(left.id).label == "D2"
    assert manager.times() == [1.0, 2.0]


def test_region_selection_between_dividers() -> None:
    manager = DividerManager(time_range=(0, 10))
    manager.add_divider(2)
    manager.add_divider(7)
    selector = RegionSelector(time_range=(0, 10))

    region = selector.select_at(4, manager)

    assert region is not None
    assert region.label == "D1 to D2"
    assert region.start_time == 2.0
    assert region.end_time == 7.0
    assert selector.current_region_tuple() == (2.0, 7.0)


def test_threshold_event_detection_uses_interpolated_crossing_times() -> None:
    events = detect_exceedance_events(
        time=[0, 1, 2, 3],
        values_by_case={"case_a": [0, 2, 2, 0]},
        threshold=1,
        region_start=0,
        region_end=3,
    )

    assert len(events) == 1
    assert events[0].case_name == "case_a"
    assert events[0].start_time == pytest.approx(0.5)
    assert events[0].end_time == pytest.approx(2.5)
    assert events[0].duration == pytest.approx(2.0)


def test_multiple_exceedance_events_per_case() -> None:
    events = detect_exceedance_events(
        time=[0, 1, 2, 3, 4, 5],
        values_by_case={"case_a": [0, 2, 0, 0, 3, 0]},
        threshold=1,
        region_start=0,
        region_end=5,
    )

    assert [event.event_index for event in events] == [1, 2]
    assert events[0].start_time == pytest.approx(0.5)
    assert events[0].end_time == pytest.approx(1.5)
    assert events[1].start_time == pytest.approx(3 + 1 / 3)
    assert events[1].end_time == pytest.approx(4 + 2 / 3)


def test_nonuniform_time_spacing_exceedance_duration() -> None:
    events = detect_exceedance_events(
        time=[0, 0.5, 3.0, 10.0],
        values_by_case={"case_a": [0, 2, 2, 0]},
        threshold=1,
        region_start=0,
        region_end=10,
    )

    assert len(events) == 1
    assert events[0].start_time == pytest.approx(0.25)
    assert events[0].end_time == pytest.approx(6.5)
    assert events[0].duration == pytest.approx(6.25)


def test_svg_export_wrapper_does_not_crash(workspace_tmp: Path) -> None:
    path = workspace_tmp / "plot.svg"
    fake_plot = _FakeSvgPlot()

    export_main_plot_svg(fake_plot, path)

    assert fake_plot.exported_to == path
    assert path.read_text(encoding="utf-8").startswith("<svg")


class _FakeSvgPlot:
    def __init__(self) -> None:
        self.exported_to: Path | None = None

    def export_svg(self, path: str | Path) -> None:
        self.exported_to = Path(path)
        self.exported_to.write_text("<svg></svg>\n", encoding="utf-8")
