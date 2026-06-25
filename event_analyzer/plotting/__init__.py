"""Plotting package."""

__all__ = [
    "ExceedanceBarChartWidget",
    "ExceedanceChart",
    "ExceedanceCountCurveWidget",
    "TimeSeriesPlot",
    "TimeSeriesPlotWidget",
]


def __getattr__(name: str):
    if name in __all__:
        from event_analyzer.plotting.exceedance_chart import ExceedanceBarChartWidget, ExceedanceChart
        from event_analyzer.plotting.exceedance_count_curve import ExceedanceCountCurveWidget
        from event_analyzer.plotting.time_series_plot import TimeSeriesPlot, TimeSeriesPlotWidget

        return {
            "ExceedanceBarChartWidget": ExceedanceBarChartWidget,
            "ExceedanceChart": ExceedanceChart,
            "ExceedanceCountCurveWidget": ExceedanceCountCurveWidget,
            "TimeSeriesPlot": TimeSeriesPlot,
            "TimeSeriesPlotWidget": TimeSeriesPlotWidget,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
