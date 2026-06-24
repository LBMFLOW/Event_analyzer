from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from event_analyzer.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Event Analyzer")
    app.setOrganizationName("EventAnalyzer")

    window = MainWindow()
    window.resize(1500, 920)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

