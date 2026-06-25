from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ExceedanceCountCurve:
    """Count of cases exceeding candidate threshold levels."""

    thresholds: np.ndarray
    counts: np.ndarray
    case_maxima: dict[str, float]
    threshold_min: float
    threshold_max: float
    region_start: float | None
    region_end: float | None


def compute_exceedance_count_curve(
    time: Sequence[float],
    values_by_case: Mapping[str, Sequence[float]],
    *,
    threshold_min: float,
    threshold_max: float,
    levels: int = 500,
    region_start: float | None = None,
    region_end: float | None = None,
) -> ExceedanceCountCurve:
    """Compute how many cases exceed each candidate threshold.

    A case is counted at a threshold if any finite target value in the selected
    region is strictly greater than that threshold. Internally this reduces to
    comparing the candidate levels against each case maximum, which keeps the
    computation cheap for many levels.
    """
    start = float(threshold_min)
    end = float(threshold_max)
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError("Target parameter range must contain finite values with minimum less than maximum.")
    level_count = int(levels)
    if level_count < 2:
        raise ValueError("Level count must be at least 2.")

    time_array = np.asarray(time, dtype=float)
    if time_array.ndim != 1:
        raise ValueError("time must be one-dimensional.")
    if time_array.size == 0:
        raise ValueError("time contains no samples.")

    base_mask = np.isfinite(time_array)
    if region_start is not None:
        base_mask &= time_array >= float(region_start)
    if region_end is not None:
        base_mask &= time_array <= float(region_end)

    thresholds = np.linspace(start, end, level_count, dtype=float)
    case_maxima: dict[str, float] = {}
    finite_maxima: list[float] = []

    for case_name, values in values_by_case.items():
        value_array = np.asarray(values, dtype=float)
        if value_array.ndim != 1:
            raise ValueError(f"Case '{case_name}' values must be one-dimensional.")
        sample_count = min(time_array.size, value_array.size)
        if sample_count == 0:
            case_maxima[case_name] = float("nan")
            continue
        mask = base_mask[:sample_count] & np.isfinite(value_array[:sample_count])
        if not mask.any():
            case_maxima[case_name] = float("nan")
            continue
        maximum = float(np.max(value_array[:sample_count][mask]))
        case_maxima[case_name] = maximum
        finite_maxima.append(maximum)

    if finite_maxima:
        sorted_maxima = np.sort(np.asarray(finite_maxima, dtype=float))
        counts = sorted_maxima.size - np.searchsorted(sorted_maxima, thresholds, side="right")
        counts = counts.astype(int)
    else:
        counts = np.zeros(thresholds.size, dtype=int)

    return ExceedanceCountCurve(
        thresholds=thresholds,
        counts=counts,
        case_maxima=case_maxima,
        threshold_min=start,
        threshold_max=end,
        region_start=None if region_start is None else float(region_start),
        region_end=None if region_end is None else float(region_end),
    )


__all__ = ["ExceedanceCountCurve", "compute_exceedance_count_curve"]
