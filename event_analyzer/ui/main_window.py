from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMainWindow, QMenu, QSplitter

from event_analyzer.ui.control_panel import ControlPanel
from event_analyzer.ui.main_window_controller import MainWindowController
from event_analyzer.ui.plot_workspace import PlotWorkspace


@dataclass(slots=True)
class MainWindowActions:
    """Menu and toolbar actions exposed to the controller."""

    open_csv: QAction
    save_session: QAction
    load_session: QAction
    save_main_plot_svg: QAction
    save_bar_chart_svg: QAction
    save_count_curve_svg: QAction
    export_events_csv: QAction
    export_region_csv: QAction
    export_analysis_summary_json: QAction
    export_analysis_summary_csv: QAction
    add_divider: QAction
    add_threshold: QAction
    reset_view: QAction
    toggle_theme: QAction
    exit_app: QAction


class MainWindow(QMainWindow):
    """Top-level UI shell for Event Analyzer.

    This class deliberately only composes widgets and menus. The controller owns
    signal wiring and will later call the data, plotting, divider, threshold,
    region, tracer, and export services.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Event Analyzer")

        self.control_panel = ControlPanel()
        self.workspace = PlotWorkspace()
        self.actions = self._create_actions()
        self.recent_files_menu: QMenu | None = None

        self._build_menu()
        self._build_layout()
        self.controller = MainWindowController(self)
        self.statusBar().showMessage("Open a CSV file to begin.")

    def _create_actions(self) -> MainWindowActions:
        open_csv = QAction("Open CSV...", self)
        open_csv.setShortcut(QKeySequence.StandardKey.Open)

        save_session = QAction("Save Project...", self)
        save_session.setShortcut(QKeySequence.StandardKey.Save)

        load_session = QAction("Load Project...", self)
        load_session.setShortcut(QKeySequence("Ctrl+Shift+O"))

        save_main_plot_svg = QAction("Save Main Plot as SVG...", self)
        save_main_plot_svg.setShortcut(QKeySequence("Ctrl+Shift+P"))

        save_bar_chart_svg = QAction("Save Bar Chart as SVG...", self)
        save_bar_chart_svg.setShortcut(QKeySequence("Ctrl+Shift+B"))

        save_count_curve_svg = QAction("Save Exceedance Count Curve as SVG...", self)
        save_count_curve_svg.setShortcut(QKeySequence("Ctrl+Shift+C"))

        export_events_csv = QAction("Export Exceedance Events CSV...", self)
        export_events_csv.setShortcut(QKeySequence("Ctrl+Shift+E"))

        export_region_csv = QAction("Export Selected Region CSV...", self)
        export_region_csv.setShortcut(QKeySequence("Ctrl+Shift+R"))

        export_analysis_summary_json = QAction("Export Analysis Summary JSON...", self)
        export_analysis_summary_json.setShortcut(QKeySequence("Ctrl+Shift+J"))

        export_analysis_summary_csv = QAction("Export Analysis Summary CSV...", self)
        export_analysis_summary_csv.setShortcut(QKeySequence("Ctrl+Shift+M"))

        add_divider = QAction("Add Divider...", self)
        add_divider.setShortcut(QKeySequence("Ctrl+D"))

        add_threshold = QAction("Add Threshold...", self)
        add_threshold.setShortcut(QKeySequence("Ctrl+T"))

        reset_view = QAction("Reset View", self)
        reset_view.setShortcut(QKeySequence("Ctrl+0"))

        toggle_theme = QAction("Dark Theme", self)
        toggle_theme.setCheckable(True)

        exit_app = QAction("Exit", self)
        exit_app.setShortcut(QKeySequence.StandardKey.Quit)
        exit_app.triggered.connect(self.close)

        return MainWindowActions(
            open_csv=open_csv,
            save_session=save_session,
            load_session=load_session,
            save_main_plot_svg=save_main_plot_svg,
            save_bar_chart_svg=save_bar_chart_svg,
            save_count_curve_svg=save_count_curve_svg,
            export_events_csv=export_events_csv,
            export_region_csv=export_region_csv,
            export_analysis_summary_json=export_analysis_summary_json,
            export_analysis_summary_csv=export_analysis_summary_csv,
            add_divider=add_divider,
            add_threshold=add_threshold,
            reset_view=reset_view,
            toggle_theme=toggle_theme,
            exit_app=exit_app,
        )

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.actions.open_csv)
        self.recent_files_menu = file_menu.addMenu("Recent Files")
        file_menu.addSeparator()
        file_menu.addAction(self.actions.load_session)
        file_menu.addAction(self.actions.save_session)
        file_menu.addSeparator()
        file_menu.addAction(self.actions.save_main_plot_svg)
        file_menu.addAction(self.actions.save_bar_chart_svg)
        file_menu.addAction(self.actions.save_count_curve_svg)
        file_menu.addAction(self.actions.export_events_csv)
        file_menu.addAction(self.actions.export_region_csv)
        file_menu.addAction(self.actions.export_analysis_summary_json)
        file_menu.addAction(self.actions.export_analysis_summary_csv)
        file_menu.addSeparator()
        file_menu.addAction(self.actions.exit_app)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.actions.add_divider)
        edit_menu.addAction(self.actions.add_threshold)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.actions.reset_view)
        view_menu.addAction(self.actions.toggle_theme)

    def _build_layout(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.workspace)
        splitter.setSizes([380, 1120])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
