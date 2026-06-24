from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class SearchableColumnSelector(QWidget):
    """Searchable check-list used for target and auxiliary column selection."""

    selection_changed = pyqtSignal()

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search columns")
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(title))
        layout.addWidget(self.search_edit)
        layout.addWidget(self.list_widget)

        self.search_edit.textChanged.connect(self._filter_items)
        self.list_widget.itemChanged.connect(lambda _item: self.selection_changed.emit())

    def set_columns(self, columns: list[str], disabled_columns: set[str] | None = None) -> None:
        disabled = disabled_columns or set()
        with QSignalBlocker(self.list_widget):
            self.list_widget.clear()
            for name in columns:
                item = QListWidgetItem(name)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                if name in disabled:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                    item.setToolTip("This column was not inferred as numeric.")
                self.list_widget.addItem(item)
        self._filter_items(self.search_edit.text())

    def selected_columns(self) -> list[str]:
        return [
            self.list_widget.item(index).text()
            for index in range(self.list_widget.count())
            if self.list_widget.item(index).checkState() == Qt.CheckState.Checked
        ]

    def set_selected_columns(self, columns: list[str]) -> None:
        selected = set(columns)
        with QSignalBlocker(self.list_widget):
            for index in range(self.list_widget.count()):
                item = self.list_widget.item(index)
                item.setCheckState(Qt.CheckState.Checked if item.text() in selected else Qt.CheckState.Unchecked)
        self.selection_changed.emit()

    def _filter_items(self, text: str) -> None:
        needle = text.casefold().strip()
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            item.setHidden(bool(needle and needle not in item.text().casefold()))

