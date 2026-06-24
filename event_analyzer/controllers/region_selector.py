from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from event_analyzer.controllers.divider_manager import Divider, DividerManager


class RegionError(ValueError):
    """Raised for invalid region selection state."""


class RegionPlotAdapter(Protocol):
    """Minimal plot API needed to visualize region selection."""

    def set_selected_region(self, region_start: float | None, region_end: float | None) -> None:
        ...


@dataclass(frozen=True, slots=True)
class SelectedRegion:
    """A selected interval derived from the current divider boundaries."""

    index: int
    label: str
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class RegionSelector:
    """Compute and select analysis regions from sorted vertical dividers.

    The class is intentionally UI-independent. A PyQtGraph plot can be attached
    through :meth:`set_plot_adapter`; the selector only calls
    ``set_selected_region(start, end)`` on that adapter.
    """

    def __init__(
        self,
        *,
        time_range: tuple[float, float],
        divider_tolerance: float = 1e-9,
        boundary_policy: str = "right",
        plot_adapter: RegionPlotAdapter | None = None,
    ) -> None:
        if boundary_policy not in {"left", "right"}:
            raise RegionError("boundary_policy must be 'left' or 'right'.")
        self._time_range = _normalise_range(time_range)
        self.divider_tolerance = divider_tolerance
        self.boundary_policy = boundary_policy
        self._selected_region: SelectedRegion | None = None
        self._plot_adapter = plot_adapter

    @property
    def time_range(self) -> tuple[float, float]:
        return self._time_range

    @property
    def selected_region(self) -> SelectedRegion | None:
        return self._selected_region

    def set_time_range(self, start: float, end: float) -> None:
        """Update data time range and clear invalid region selections."""
        self._time_range = _normalise_range((start, end))
        if self._selected_region is not None:
            midpoint = (self._selected_region.start_time + self._selected_region.end_time) / 2
            if not self._contains_time(midpoint):
                self.clear_selection()

    def set_plot_adapter(self, plot_adapter: RegionPlotAdapter | None) -> None:
        """Attach or detach a plot object that can highlight selected regions."""
        self._plot_adapter = plot_adapter
        self._sync_plot()

    def regions(self, dividers: DividerManager | Iterable[Divider] | Sequence[float] | None = None) -> list[SelectedRegion]:
        """Return all selectable regions for the current divider state."""
        start, end = self._time_range
        divider_times = self._in_range_unique_divider_times(dividers)
        if not divider_times:
            return [SelectedRegion(index=0, label="All data", start_time=start, end_time=end)]

        boundaries = [start, *divider_times, end]
        regions: list[SelectedRegion] = []
        for index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if index == 0:
                label = f"Before D1"
            elif index == len(boundaries) - 2:
                label = f"After D{index}"
            else:
                label = f"D{index} to D{index + 1}"
            regions.append(SelectedRegion(index=index, label=label, start_time=left, end_time=right))
        return regions

    def select_at(
        self,
        time_value: float,
        dividers: DividerManager | Iterable[Divider] | Sequence[float] | None = None,
    ) -> SelectedRegion | None:
        """Select the region containing a clicked time.

        Clicking exactly on a divider follows ``boundary_policy``. The default
        selects the region to the right of the divider.
        """
        time = float(time_value)
        if not self._contains_time(time):
            return None

        regions = self.regions(dividers)
        divider_times = self._in_range_unique_divider_times(dividers)
        boundary_index = self._matching_divider_index(time, divider_times)
        if boundary_index is not None:
            if self.boundary_policy == "right":
                selected_index = min(boundary_index + 1, len(regions) - 1)
            else:
                selected_index = max(boundary_index, 0)
            return self.select_region_by_index(selected_index, dividers)

        for region in regions:
            if _time_in_region(time, region, tolerance=self.divider_tolerance):
                self._selected_region = region
                self._sync_plot()
                return region
        return None

    def select_region_by_index(
        self,
        index: int,
        dividers: DividerManager | Iterable[Divider] | Sequence[float] | None = None,
    ) -> SelectedRegion:
        """Select a region by its 0-based index."""
        regions = self.regions(dividers)
        if index < 0 or index >= len(regions):
            raise RegionError(f"Region index {index} is outside the available range 0 to {len(regions) - 1}.")
        self._selected_region = regions[index]
        self._sync_plot()
        return self._selected_region

    def reconcile_after_divider_change(
        self,
        dividers: DividerManager | Iterable[Divider] | Sequence[float] | None = None,
    ) -> SelectedRegion | None:
        """Keep selection valid after divider add/edit/delete operations.

        The previous selected-region midpoint is reselected against the new
        boundaries. This handles deleting a divider by expanding or shifting the
        selected region to the interval that now contains the same midpoint.
        """
        if self._selected_region is None:
            return None
        midpoint = (self._selected_region.start_time + self._selected_region.end_time) / 2
        if not self._contains_time(midpoint):
            self.clear_selection()
            return None
        return self.select_at(midpoint, dividers)

    def clear_selection(self) -> None:
        self._selected_region = None
        self._sync_plot()

    def current_region_tuple(self) -> tuple[float, float] | None:
        if self._selected_region is None:
            return None
        return self._selected_region.start_time, self._selected_region.end_time

    # Compatibility helpers for earlier code in this repository.
    def current_region(self) -> tuple[float, float] | None:
        return self.current_region_tuple()

    def set_region(self, start: float, end: float, *, emit: bool = True) -> None:
        del emit
        region = SelectedRegion(index=-1, label="Custom", start_time=float(start), end_time=float(end))
        if region.start_time > region.end_time:
            region = SelectedRegion(
                index=-1,
                label="Custom",
                start_time=region.end_time,
                end_time=region.start_time,
            )
        self._selected_region = region
        self._sync_plot()

    def _sync_plot(self) -> None:
        if self._plot_adapter is None:
            return
        if self._selected_region is None:
            self._plot_adapter.set_selected_region(None, None)
        else:
            self._plot_adapter.set_selected_region(
                self._selected_region.start_time,
                self._selected_region.end_time,
            )

    def _contains_time(self, time: float) -> bool:
        start, end = self._time_range
        return start - self.divider_tolerance <= time <= end + self.divider_tolerance

    def _matching_divider_index(self, time: float, divider_times: Sequence[float]) -> int | None:
        for index, divider_time in enumerate(divider_times):
            if abs(time - divider_time) <= self.divider_tolerance:
                return index
        return None

    def _in_range_unique_divider_times(
        self,
        dividers: DividerManager | Iterable[Divider] | Sequence[float] | None,
    ) -> list[float]:
        start, end = self._time_range
        times = _extract_times(dividers)
        in_range = sorted(time for time in times if start < time < end)
        unique: list[float] = []
        for time in in_range:
            if unique and abs(time - unique[-1]) <= self.divider_tolerance:
                continue
            unique.append(time)
        return unique


def _extract_times(dividers: DividerManager | Iterable[Divider] | Sequence[float] | None) -> list[float]:
    if dividers is None:
        return []
    if isinstance(dividers, DividerManager):
        return dividers.times()

    times: list[float] = []
    for divider in dividers:
        if isinstance(divider, Divider):
            times.append(float(divider.time))
        else:
            times.append(float(divider))
    return times


def _time_in_region(time: float, region: SelectedRegion, *, tolerance: float) -> bool:
    return region.start_time - tolerance <= time <= region.end_time + tolerance


def _normalise_range(time_range: tuple[float, float]) -> tuple[float, float]:
    start, end = float(time_range[0]), float(time_range[1])
    if start > end:
        start, end = end, start
    if start == end:
        raise RegionError("Time range must have non-zero duration.")
    return start, end


__all__ = ["RegionError", "RegionPlotAdapter", "RegionSelector", "SelectedRegion"]

