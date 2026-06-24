from __future__ import annotations

from typing import Any

import pyqtgraph as pg


class TimeAxisItem(pg.AxisItem):
    def __init__(self) -> None:
        super().__init__(orientation="bottom")
        self._time_axis: Any | None = None

    def set_time_axis(self, axis: Any | None) -> None:
        self._time_axis = axis
        self.setLabel(axis.name if axis else "Time")

    def tickStrings(self, values, scale, spacing):  # noqa: N802 - PyQtGraph API
        if self._time_axis is None:
            return super().tickStrings(values, scale, spacing)
        return [self._time_axis.format_value(float(value)) for value in values]
