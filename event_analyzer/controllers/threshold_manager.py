from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Mapping, Protocol, Sequence

import numpy as np

from event_analyzer.analysis.exceedance import ExceedanceEvent, detect_exceedance_events


class ThresholdError(ValueError):
    """Raised for invalid threshold operations."""


class ThresholdAnalysisCancelled(RuntimeError):
    """Raised when cooperative threshold analysis cancellation is requested."""


class CancellationToken(Protocol):
    """Small protocol shared with background workers without importing PyQt."""

    @property
    def is_cancelled(self) -> bool:
        ...


class ThresholdPlotAdapter(Protocol):
    """Minimal plot API needed by ThresholdManager."""

    def set_threshold(self, value: float | None) -> None:
        ...

    def highlight_exceeding_cases(self, case_names: Sequence[str]) -> None:
        ...


class ThresholdEventsAdapter(Protocol):
    """Receives the current exceedance events."""

    def set_events(self, events: list[ExceedanceEvent]) -> None:
        ...


class ThresholdSummaryAdapter(Protocol):
    """Receives the current threshold summary."""

    def set_threshold_summary(self, summary: "ThresholdSummary") -> None:
        ...


@dataclass(frozen=True, slots=True)
class ThresholdState:
    """Current threshold value and enabled/disabled state."""

    enabled: bool = False
    value: float | None = None


@dataclass(frozen=True, slots=True)
class ThresholdSummary:
    """Compact summary shown in the control panel."""

    threshold_enabled: bool
    threshold: float | None
    exceeding_cases: tuple[str, ...]
    event_count: int
    max_exceedance_value: float | None
    longest_exceedance_duration: float | None
    region_start: float | None
    region_end: float | None

    @property
    def exceeding_case_count(self) -> int:
        return len(self.exceeding_cases)


class ThresholdManager:
    """Manage threshold state, analysis refresh, and highlighting.

    The manager is UI-framework independent. It owns the threshold value, target
    arrays, selected region, latest exceedance events, and adapter calls needed
    to update the plot, chart, summary table, and control-panel summary.
    Threshold analysis intentionally uses target/case data only; auxiliary
    series are never passed to the detector.
    """

    def __init__(
        self,
        *,
        plot_adapter: ThresholdPlotAdapter | None = None,
        chart_adapter: ThresholdEventsAdapter | None = None,
        table_adapter: ThresholdEventsAdapter | None = None,
        summary_adapter: ThresholdSummaryAdapter | None = None,
    ) -> None:
        self._state = ThresholdState()
        self._time: np.ndarray | None = None
        self._target_values: dict[str, np.ndarray] = {}
        self._region: tuple[float | None, float | None] = (None, None)
        self._plot_adapter = plot_adapter
        self._chart_adapter = chart_adapter
        self._table_adapter = table_adapter
        self._summary_adapter = summary_adapter
        self._events: list[ExceedanceEvent] = []
        self._summary = ThresholdSummary(False, None, (), 0, None, None, None, None)

    @property
    def state(self) -> ThresholdState:
        return self._state

    @property
    def value(self) -> float | None:
        return self._state.value if self._state.enabled else None

    @property
    def events(self) -> list[ExceedanceEvent]:
        return list(self._events)

    @property
    def summary(self) -> ThresholdSummary:
        return self._summary

    def set_adapters(
        self,
        *,
        plot_adapter: ThresholdPlotAdapter | None = None,
        chart_adapter: ThresholdEventsAdapter | None = None,
        table_adapter: ThresholdEventsAdapter | None = None,
        summary_adapter: ThresholdSummaryAdapter | None = None,
    ) -> None:
        """Attach or replace UI adapters."""
        if plot_adapter is not None:
            self._plot_adapter = plot_adapter
        if chart_adapter is not None:
            self._chart_adapter = chart_adapter
        if table_adapter is not None:
            self._table_adapter = table_adapter
        if summary_adapter is not None:
            self._summary_adapter = summary_adapter
        self._push_updates()

    def set_target_data(self, time: Sequence[float], values_by_case: Mapping[str, Sequence[float]]) -> None:
        """Set target/case data. Auxiliary columns should not be passed here."""
        time_array = np.asarray(time, dtype=float)
        if time_array.ndim != 1:
            raise ThresholdError("time must be a 1D array-like object.")
        self._time = time_array
        self._target_values = {}
        for case_name, values in values_by_case.items():
            array = np.asarray(values, dtype=float)
            if array.ndim != 1:
                raise ThresholdError(f"Target case '{case_name}' must be a 1D array-like object.")
            if array.shape != time_array.shape:
                raise ThresholdError(
                    f"Target case '{case_name}' has {array.size} values for {time_array.size} time samples."
                )
            self._target_values[str(case_name)] = array
        self.refresh()

    def set_region(self, region_start: float | None, region_end: float | None) -> None:
        """Set selected region used for threshold analysis."""
        self._region = (
            None if region_start is None else float(region_start),
            None if region_end is None else float(region_end),
        )
        self.refresh()

    def create_from_plot(self, value: float) -> None:
        """Create or move threshold from a plot context-menu y value."""
        self.set_threshold(value)

    def set_threshold(self, value: float) -> None:
        """Enable threshold using a manually entered or programmatic value."""
        self._state = ThresholdState(enabled=True, value=_finite_threshold(value))
        self.refresh()

    def edit_threshold(self, value: float) -> None:
        """Edit threshold from a numeric input box."""
        self.set_threshold(value)

    def drag_threshold(self, value: float) -> None:
        """Update threshold after the horizontal plot line is dragged."""
        self.set_threshold(value)

    def disable_threshold(self) -> None:
        """Disable threshold analysis while keeping data and region state."""
        self._state = ThresholdState(enabled=False, value=None)
        self._events = []
        self._summary = self._build_summary([])
        self._push_updates()

    delete_threshold = disable_threshold

    def refresh(
        self,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        """Rerun exceedance detection and push all UI updates."""
        if not self._state.enabled or self._state.value is None:
            self.disable_threshold()
            return
        if self._time is None or not self._target_values:
            self._events = []
            self._summary = self._build_summary([])
            self._push_updates()
            return

        self._events = self.compute_events(cancel_token=cancel_token, progress_callback=progress_callback)
        self._summary = self._build_summary(self._events)
        self._push_updates()

    def compute_events(
        self,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[ExceedanceEvent]:
        """Compute events from full-resolution target arrays without UI updates."""
        if not self._state.enabled or self._state.value is None or self._time is None:
            return []

        try:
            return detect_exceedance_events(
                self._time,
                self._target_values,
                self._state.value,
                self._region[0],
                self._region[1],
                cancel_token=cancel_token,
                progress_callback=progress_callback,
            )
        except RuntimeError as exc:
            if "cancelled" in str(exc).lower():
                raise ThresholdAnalysisCancelled(str(exc)) from exc
            raise

    def apply_events(self, events: Sequence[ExceedanceEvent]) -> None:
        """Apply precomputed events on the UI thread and notify adapters."""
        self._events = list(events)
        self._summary = self._build_summary(self._events)
        self._push_updates()

    def _build_summary(self, events: list[ExceedanceEvent]) -> ThresholdSummary:
        exceeding_cases = tuple(sorted({event.case_name for event in events}))
        max_value = max((event.peak_value for event in events), default=None)
        longest = max((event.duration for event in events), default=None)
        return ThresholdSummary(
            threshold_enabled=self._state.enabled,
            threshold=self._state.value if self._state.enabled else None,
            exceeding_cases=exceeding_cases,
            event_count=len(events),
            max_exceedance_value=max_value,
            longest_exceedance_duration=longest,
            region_start=self._region[0],
            region_end=self._region[1],
        )

    def _push_updates(self) -> None:
        if self._plot_adapter is not None:
            self._plot_adapter.set_threshold(self._state.value if self._state.enabled else None)
            self._plot_adapter.highlight_exceeding_cases(self._summary.exceeding_cases)
        if self._chart_adapter is not None:
            self._chart_adapter.set_events(self.events)
        if self._table_adapter is not None:
            self._table_adapter.set_events(self.events)
        if self._summary_adapter is not None:
            self._summary_adapter.set_threshold_summary(self._summary)


def _finite_threshold(value: float) -> float:
    threshold = float(value)
    if not np.isfinite(threshold):
        raise ThresholdError("Threshold must be a finite numeric value.")
    return threshold


__all__ = [
    "CancellationToken",
    "ThresholdAnalysisCancelled",
    "ThresholdError",
    "ThresholdEventsAdapter",
    "ThresholdManager",
    "ThresholdPlotAdapter",
    "ThresholdState",
    "ThresholdSummary",
    "ThresholdSummaryAdapter",
]
