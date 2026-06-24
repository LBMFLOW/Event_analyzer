from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
from uuid import uuid4


class DividerError(ValueError):
    """Base error for invalid divider operations."""


class DividerNotFoundError(DividerError):
    """Raised when a divider id does not exist."""


class DuplicateDividerTimeError(DividerError):
    """Raised when a divider time duplicates another divider boundary."""


class DividerOutOfRangeError(DividerError):
    """Raised when a divider is outside the configured data range."""


@dataclass(frozen=True, slots=True)
class DividerStyle:
    """Optional display styling for a vertical divider."""

    color: str = "#52525b"
    line_style: str = "dash"
    width: float = 1.5


@dataclass(frozen=True, slots=True)
class Divider:
    """A vertical time divider used to split the x-axis into regions."""

    id: str
    label: str
    time: float
    style: DividerStyle = field(default_factory=DividerStyle)


class DividerManager:
    """Manage sorted vertical time dividers independent of the PyQt UI.

    A plotting/controller layer should call :meth:`drag_divider` when a plot
    divider is moved, and should mirror returned divider state into the plot.
    Duplicate times are rejected because they create zero-width, ambiguous
    regions.
    """

    def __init__(
        self,
        *,
        time_range: tuple[float, float] | None = None,
        duplicate_tolerance: float = 1e-9,
        allow_outside_range: bool = False,
    ) -> None:
        self.duplicate_tolerance = duplicate_tolerance
        self.allow_outside_range = allow_outside_range
        self._time_range = _normalise_range(time_range) if time_range is not None else None
        self._dividers: list[Divider] = []

    @property
    def time_range(self) -> tuple[float, float] | None:
        return self._time_range

    @property
    def dividers(self) -> list[Divider]:
        """Return sorted divider snapshots."""
        return list(self._dividers)

    def set_time_range(self, start: float, end: float) -> None:
        """Set the data time range used for divider validation."""
        self._time_range = _normalise_range((start, end))
        if self.allow_outside_range:
            return
        for divider in self._dividers:
            self._validate_in_range(divider.time)

    def add_divider(
        self,
        time: float,
        *,
        divider_id: str | None = None,
        style: DividerStyle | None = None,
    ) -> Divider:
        """Add a divider, sort all dividers, and relabel them D1, D2, ..."""
        divider_time = _finite_time(time)
        actual_id = divider_id or f"divider-{uuid4().hex[:8]}"
        if any(divider.id == actual_id for divider in self._dividers):
            raise DividerError(f"Divider id already exists: {actual_id}")
        self._validate_time_available(divider_time)
        divider = Divider(
            id=actual_id,
            label="",
            time=divider_time,
            style=style or DividerStyle(),
        )
        self._dividers.append(divider)
        self._resort_and_relabel()
        return self.get_divider(actual_id)

    def edit_divider_time(self, divider_id: str, time: float) -> Divider:
        """Edit a divider time from manual user input."""
        return self._replace_time(divider_id, time)

    def drag_divider(self, divider_id: str, time: float) -> Divider:
        """Update a divider after it was dragged on the plot."""
        return self._replace_time(divider_id, time)

    update_divider_time = edit_divider_time

    def delete_divider(self, divider_id: str) -> Divider:
        """Delete a divider by id and relabel remaining dividers."""
        index = self._index_for_id(divider_id)
        removed = self._dividers.pop(index)
        self._resort_and_relabel()
        return removed

    def clear_all_dividers(self) -> None:
        """Remove all dividers."""
        self._dividers.clear()

    clear = clear_all_dividers

    def get_divider(self, divider_id: str) -> Divider:
        return self._dividers[self._index_for_id(divider_id)]

    def times(self) -> list[float]:
        return [divider.time for divider in self._dividers]

    def labels_and_times(self) -> list[tuple[str, float]]:
        return [(divider.label, divider.time) for divider in self._dividers]

    def serialize(self) -> list[dict[str, Any]]:
        """Serialize dividers for session persistence."""
        return [
            {
                "id": divider.id,
                "label": divider.label,
                "time": divider.time,
                "style": asdict(divider.style),
            }
            for divider in self._dividers
        ]

    def deserialize(self, records: Iterable[dict[str, Any]], *, replace: bool = True) -> list[Divider]:
        """Load dividers from serialized records.

        Labels are recalculated after sorting so restored sessions remain D1,
        D2, D3 in time order even if the saved list was not sorted.
        """
        existing = [] if replace else list(self._dividers)
        loaded: list[Divider] = []
        seen_ids = {divider.id for divider in existing}
        seen_times = [divider.time for divider in existing]

        for record in records:
            divider_id = str(record.get("id") or f"divider-{uuid4().hex[:8]}")
            if divider_id in seen_ids:
                raise DividerError(f"Divider id already exists: {divider_id}")
            time = _finite_time(record["time"])
            self._validate_in_range(time)
            if self._has_duplicate_time(time, seen_times):
                raise DuplicateDividerTimeError(f"Divider time already exists: {time:g}")

            style_data = record.get("style") or {}
            default_style = DividerStyle()
            style = DividerStyle(
                color=str(style_data.get("color", default_style.color)),
                line_style=str(style_data.get("line_style", default_style.line_style)),
                width=float(style_data.get("width", default_style.width)),
            )
            loaded.append(Divider(id=divider_id, label="", time=time, style=style))
            seen_ids.add(divider_id)
            seen_times.append(time)

        self._dividers = [*existing, *loaded]
        self._resort_and_relabel()
        return self.dividers

    @classmethod
    def from_serialized(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        time_range: tuple[float, float] | None = None,
        duplicate_tolerance: float = 1e-9,
        allow_outside_range: bool = False,
    ) -> "DividerManager":
        manager = cls(
            time_range=time_range,
            duplicate_tolerance=duplicate_tolerance,
            allow_outside_range=allow_outside_range,
        )
        manager.deserialize(records)
        return manager

    def _replace_time(self, divider_id: str, time: float) -> Divider:
        divider_time = _finite_time(time)
        index = self._index_for_id(divider_id)
        self._validate_time_available(divider_time, ignore_id=divider_id)
        current = self._dividers[index]
        self._dividers[index] = Divider(
            id=current.id,
            label=current.label,
            time=divider_time,
            style=current.style,
        )
        self._resort_and_relabel()
        return self.get_divider(divider_id)

    def _validate_time_available(self, time: float, *, ignore_id: str | None = None) -> None:
        self._validate_in_range(time)
        other_times = [divider.time for divider in self._dividers if divider.id != ignore_id]
        if self._has_duplicate_time(time, other_times):
            raise DuplicateDividerTimeError(f"Divider time already exists: {time:g}")

    def _validate_in_range(self, time: float) -> None:
        if self.allow_outside_range or self._time_range is None:
            return
        start, end = self._time_range
        if time < start or time > end:
            raise DividerOutOfRangeError(
                f"Divider time {time:g} is outside the data range {start:g} to {end:g}."
            )

    def _has_duplicate_time(self, time: float, existing_times: Iterable[float]) -> bool:
        return any(abs(time - existing) <= self.duplicate_tolerance for existing in existing_times)

    def _index_for_id(self, divider_id: str) -> int:
        for index, divider in enumerate(self._dividers):
            if divider.id == divider_id:
                return index
        raise DividerNotFoundError(f"Divider does not exist: {divider_id}")

    def _resort_and_relabel(self) -> None:
        ordered = sorted(self._dividers, key=lambda divider: (divider.time, divider.id))
        self._dividers = [
            Divider(id=divider.id, label=f"D{index}", time=divider.time, style=divider.style)
            for index, divider in enumerate(ordered, start=1)
        ]


def _finite_time(value: float) -> float:
    time = float(value)
    if time != time or time in (float("inf"), float("-inf")):
        raise DividerError("Divider time must be a finite number.")
    return time


def _normalise_range(time_range: tuple[float, float] | None) -> tuple[float, float] | None:
    if time_range is None:
        return None
    start, end = float(time_range[0]), float(time_range[1])
    if start > end:
        start, end = end, start
    if start == end:
        raise DividerError("Time range must have non-zero duration.")
    return start, end


__all__ = [
    "Divider",
    "DividerError",
    "DividerManager",
    "DividerNotFoundError",
    "DividerOutOfRangeError",
    "DividerStyle",
    "DuplicateDividerTimeError",
]
