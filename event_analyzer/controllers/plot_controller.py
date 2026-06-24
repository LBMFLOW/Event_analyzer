from __future__ import annotations

import numpy as np

from event_analyzer.data.models import SeriesData, TimeSeriesDataset
from event_analyzer.plotting.colors import color_for_index
from event_analyzer.plotting.time_series_plot import TimeSeriesPlot


class PlotController:
    def __init__(self, plot: TimeSeriesPlot) -> None:
        self.plot = plot
        self.dataset: TimeSeriesDataset | None = None

    def set_dataset(self, dataset: TimeSeriesDataset) -> None:
        self.dataset = dataset
        self.plot.clear_curves()
        self.plot.set_time_axis(dataset.time_axis)

        for index, series in enumerate(dataset.targets):
            if not series.color:
                series.color = color_for_index(index)
            self.plot.add_curve(
                series.name,
                dataset.time_axis.values,
                series.values,
                color=series.color,
                axis_id="y1",
                dashed=False,
                visible=series.visible,
            )

        for index, series in enumerate(dataset.auxiliaries):
            if not series.color:
                series.color = color_for_index(index + len(dataset.targets))
            self.plot.add_curve(
                series.name,
                dataset.time_axis.values,
                series.values,
                color=series.color,
                axis_id=series.axis_id,
                dashed=True,
                visible=series.visible,
            )

        start, end = dataset.time_range
        self.plot.plotItem.setXRange(start, end, padding=0.02)

    def target_case_values(self) -> dict[str, np.ndarray]:
        if self.dataset is None:
            return {}
        return {series.name: series.values for series in self.dataset.targets if series.visible}

    def interpolate_values(self, time_value: float, case_names: list[str] | None = None) -> dict[str, float]:
        if self.dataset is None:
            return {}
        selected = self.dataset.targets
        if case_names is not None:
            wanted = set(case_names)
            selected = [series for series in selected if series.name in wanted]
        return {
            series.name: _interpolate_at(self.dataset.time_axis.values, series.values, time_value)
            for series in selected
            if series.visible
        }

    def update_threshold_highlighting(
        self,
        threshold: float | None,
        region: tuple[float | None, float | None] | None,
    ) -> list[str]:
        if self.dataset is None:
            return []
        exceeded: list[str] = []
        time = self.dataset.time_axis.values
        start = self.dataset.time_axis.start if region is None or region[0] is None else region[0]
        end = self.dataset.time_axis.end if region is None or region[1] is None else region[1]
        mask = (time >= start) & (time <= end)

        for series in self.dataset.targets:
            has_exceedance = False
            if threshold is not None and mask.any():
                values = series.values[mask]
                has_exceedance = bool(np.nanmax(values) > threshold) if np.isfinite(values).any() else False
            self.plot.set_curve_emphasized(series.name, has_exceedance)
            if has_exceedance:
                exceeded.append(series.name)
        return exceeded

    def set_series_visible(self, name: str, visible: bool) -> None:
        if self.dataset is None:
            return
        series = self.dataset.series_by_name(name)
        if series is not None:
            series.visible = visible
            self.plot.set_curve_visible(name, visible)

    def set_series_color(self, name: str, color: str) -> None:
        if self.dataset is None:
            return
        series = self.dataset.series_by_name(name)
        if series is not None:
            series.color = color
            if hasattr(self.plot, "set_curve_color"):
                self.plot.set_curve_color(name, color)


def _interpolate_at(time: np.ndarray, values: np.ndarray, x: float) -> float:
    finite = np.isfinite(time) & np.isfinite(values)
    if finite.sum() == 0:
        return float("nan")
    time_f = time[finite]
    values_f = values[finite]
    if x <= time_f[0]:
        return float(values_f[0])
    if x >= time_f[-1]:
        return float(values_f[-1])
    return float(np.interp(x, time_f, values_f))
