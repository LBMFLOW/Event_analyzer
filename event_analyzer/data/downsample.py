from __future__ import annotations

import numpy as np


def downsample_indices(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
    *,
    threshold: float | None = None,
    region: tuple[float | None, float | None] | None = None,
) -> np.ndarray:
    """Return source indices for plotting a reduced but representative series.

    The reducer is designed for interactive display only. It preserves the
    first/last finite samples, per-bucket minima and maxima, optional selected
    region boundaries, and a capped set of samples adjacent to threshold
    crossings. The returned indices reference the original arrays so callers can
    keep full-resolution analysis arrays separate from display arrays.
    """
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have the same shape.")

    if max_points <= 0:
        return np.asarray([], dtype=int)
    if x_array.size <= max_points:
        return np.arange(x_array.size, dtype=int)

    finite = np.isfinite(x_array) & np.isfinite(y_array)
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size <= max_points:
        return finite_indices.astype(int, copy=False)

    mandatory = _mandatory_indices(
        x_array,
        y_array,
        finite,
        finite_indices,
        max_points=max_points,
        threshold=threshold,
        region=region,
    )
    if mandatory.size >= max_points:
        return _thin_indices(mandatory, max_points)

    remaining = max_points - mandatory.size
    selected = set(int(index) for index in mandatory)

    # Each bucket can contribute two points. This keeps narrow spikes visible
    # without drawing every sample.
    bucket_count = max(1, remaining // 2)
    edges = np.linspace(0, finite_indices.size, bucket_count + 1, dtype=int)
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        source_indices = finite_indices[start:end]
        segment = y_array[source_indices]
        local_min = int(source_indices[int(np.nanargmin(segment))])
        local_max = int(source_indices[int(np.nanargmax(segment))])
        if local_min <= local_max:
            selected.update((local_min, local_max))
        else:
            selected.update((local_max, local_min))

    selected_array = np.fromiter(selected, dtype=int)
    selected_array.sort()
    if selected_array.size <= max_points:
        return selected_array

    mandatory_set = set(int(index) for index in mandatory)
    optional = np.asarray([index for index in selected_array if int(index) not in mandatory_set], dtype=int)
    optional_budget = max_points - mandatory.size
    optional = _thin_indices(optional, optional_budget)
    combined = np.concatenate((mandatory, optional))
    combined = np.unique(combined.astype(int, copy=False))
    combined.sort()
    if combined.size > max_points:
        return _thin_indices(combined, max_points)
    return combined


def min_max_downsample(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
    *,
    threshold: float | None = None,
    region: tuple[float | None, float | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bucketed min/max downsampling that preserves spikes and crossings."""
    if max_points <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if np.asarray(x).size <= max_points:
        return x, y

    indices = downsample_indices(x, y, max_points, threshold=threshold, region=region)
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    return x_array[indices], y_array[indices]


def _mandatory_indices(
    x: np.ndarray,
    y: np.ndarray,
    finite: np.ndarray,
    finite_indices: np.ndarray,
    *,
    max_points: int,
    threshold: float | None,
    region: tuple[float | None, float | None] | None,
) -> np.ndarray:
    selected: set[int] = {int(finite_indices[0]), int(finite_indices[-1])}

    normalised_region = _normalise_region(region)
    if normalised_region is not None:
        for boundary in normalised_region:
            nearest = _nearest_finite_index(x, finite_indices, boundary)
            if nearest is not None:
                selected.add(nearest)

    if threshold is not None and np.isfinite(threshold):
        crossing_indices = _threshold_crossing_indices(x, y, finite, float(threshold), normalised_region)
        if crossing_indices.size:
            # Crossings can be very dense for noisy signals. Reserve roughly
            # half the budget for them so the plot remains responsive.
            crossing_budget = max(2, max_points // 2)
            selected.update(int(index) for index in _thin_indices(crossing_indices, crossing_budget))

    mandatory = np.fromiter(selected, dtype=int)
    mandatory.sort()
    return mandatory


def _threshold_crossing_indices(
    x: np.ndarray,
    y: np.ndarray,
    finite: np.ndarray,
    threshold: float,
    region: tuple[float, float] | None,
) -> np.ndarray:
    if x.size < 2:
        return np.asarray([], dtype=int)

    pair_mask = finite[:-1] & finite[1:]
    pair_mask &= x[1:] > x[:-1]
    if region is not None:
        start, end = region
        pair_mask &= (x[:-1] <= end) & (x[1:] >= start)

    above = y > threshold
    crossing_positions = np.flatnonzero(pair_mask & (above[:-1] != above[1:]))
    if crossing_positions.size == 0:
        return np.asarray([], dtype=int)
    indices = np.concatenate((crossing_positions, crossing_positions + 1))
    indices = np.unique(indices.astype(int, copy=False))
    indices.sort()
    return indices


def _nearest_finite_index(x: np.ndarray, finite_indices: np.ndarray, value: float) -> int | None:
    if finite_indices.size == 0 or not np.isfinite(value):
        return None
    x_finite = x[finite_indices]
    position = int(np.searchsorted(x_finite, value, side="left"))
    candidates: list[int] = []
    if position < finite_indices.size:
        candidates.append(int(finite_indices[position]))
    if position > 0:
        candidates.append(int(finite_indices[position - 1]))
    if not candidates:
        return None
    return min(candidates, key=lambda index: abs(float(x[index]) - value))


def _normalise_region(region: tuple[float | None, float | None] | None) -> tuple[float, float] | None:
    if region is None:
        return None
    start, end = region
    if start is None or end is None:
        return None
    start_value = float(start)
    end_value = float(end)
    if not (np.isfinite(start_value) and np.isfinite(end_value)):
        return None
    if start_value > end_value:
        start_value, end_value = end_value, start_value
    return start_value, end_value


def _thin_indices(indices: np.ndarray, budget: int) -> np.ndarray:
    if budget <= 0 or indices.size == 0:
        return np.asarray([], dtype=int)
    indices = np.unique(indices.astype(int, copy=False))
    indices.sort()
    if indices.size <= budget:
        return indices
    positions = np.linspace(0, indices.size - 1, budget)
    positions = np.rint(positions).astype(int)
    thinned = np.unique(indices[positions])
    thinned.sort()
    return thinned


__all__ = ["downsample_indices", "min_max_downsample"]
