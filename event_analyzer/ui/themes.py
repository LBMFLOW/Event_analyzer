from __future__ import annotations

from PyQt6.QtWidgets import QApplication


LIGHT_THEME = """
QWidget {
    font-size: 10pt;
}
QGroupBox {
    font-weight: 600;
}
"""


DARK_THEME = """
QWidget {
    background: #18181b;
    color: #e4e4e7;
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #3f3f46;
    border-radius: 6px;
    margin-top: 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 3px;
}
QLineEdit, QComboBox, QDoubleSpinBox, QListWidget, QTableWidget {
    background: #27272a;
    color: #f4f4f5;
    border: 1px solid #52525b;
    selection-background-color: #2563eb;
}
QPushButton {
    background: #3f3f46;
    border: 1px solid #71717a;
    border-radius: 4px;
    padding: 4px 8px;
}
QPushButton:hover {
    background: #52525b;
}
QMenuBar, QMenu {
    background: #27272a;
    color: #f4f4f5;
}
QHeaderView::section {
    background: #3f3f46;
    color: #f4f4f5;
    border: 1px solid #52525b;
}
"""


def apply_theme(app: QApplication, theme: str) -> str:
    """Apply and return the normalized theme name."""
    normalized = "dark" if theme == "dark" else "light"
    app.setStyleSheet(DARK_THEME if normalized == "dark" else LIGHT_THEME)
    return normalized


__all__ = ["apply_theme"]
