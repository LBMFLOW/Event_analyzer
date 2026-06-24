from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtWidgets import QSlider


class TimeSlider(QSlider):
    context_requested = pyqtSignal(float, QPoint)
    time_changed = pyqtSignal(float)

    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.setRange(0, 10_000)
        self._start = 0.0
        self._end = 1.0
        self.valueChanged.connect(self._emit_time)

    def set_time_range(self, start: float, end: float) -> None:
        self._start = float(start)
        self._end = float(end) if end != start else float(start) + 1.0
        self.setValue(0)

    def set_time_value(self, time_value: float) -> None:
        fraction = (float(time_value) - self._start) / (self._end - self._start)
        fraction = min(1.0, max(0.0, fraction))
        self.setValue(int(round(fraction * self.maximum())))

    def current_time(self) -> float:
        fraction = self.value() / max(1, self.maximum())
        return self._start + fraction * (self._end - self._start)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            fraction = event.position().x() / max(1, self.width())
            time_value = self._start + min(1.0, max(0.0, fraction)) * (self._end - self._start)
            self.context_requested.emit(float(time_value), event.globalPosition().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def _emit_time(self, _value: int) -> None:
        self.time_changed.emit(self.current_time())

