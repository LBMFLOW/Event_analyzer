from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox

from event_analyzer.analysis.statistics import region_statistics_from_arrays
from event_analyzer.controllers.app_settings import AppSettings
from event_analyzer.controllers.divider_manager import DividerError, DividerManager
from event_analyzer.controllers.region_selector import RegionSelector, SelectedRegion
from event_analyzer.controllers.session import SessionState
from event_analyzer.controllers.threshold_manager import ThresholdManager
from event_analyzer.data.models import TimeAxis
from event_analyzer.data.units import unit_for_column
from event_analyzer.exporters import (
    ExportError,
    export_analysis_summary_csv,
    export_analysis_summary_json,
    export_bar_chart_svg,
    export_events_csv,
    export_main_plot_svg,
    export_selected_region_csv,
)
from event_analyzer.plotting.colors import color_for_index
from event_analyzer.ui.themes import apply_theme
from event_analyzer.ui.workers import BackgroundTaskRunner


class MainWindowController:
    """Coordinate UI widgets with data, plotting, analysis, and persistence services."""

    def __init__(self, window: Any) -> None:
        self.window = window
        self.tasks = BackgroundTaskRunner(window)
        self.settings = AppSettings.load()
        self.current_csv_path = ""
        self.loaded_data: Any | None = None
        self.loaded_data_for_export: Any | None = None
        self.selected_region_for_export: tuple[float | None, float | None] | None = None
        self.pending_session: SessionState | None = None
        self.case_colors: dict[str, str] = {}
        self.case_visibility: dict[str, bool] = {}
        self.case_units: dict[str, str] = {}
        self.current_slider_time: float | None = None
        self.divider_manager = DividerManager(allow_outside_range=True)
        self.region_selector: RegionSelector | None = None

        self.threshold_manager = ThresholdManager(
            plot_adapter=self.window.workspace.main_plot.plot_widget,
            chart_adapter=self.window.workspace.bar_chart,
            table_adapter=self.window.workspace.summary_table,
            summary_adapter=self.window.control_panel,
        )

        self.tasks.busy_changed.connect(self.window.control_panel.set_busy)
        self.tasks.progress_changed.connect(self.window.control_panel.set_progress)
        self.tasks.progress_changed.connect(self.task_progress)

        self._connect_signals()
        self._apply_theme(self.settings.theme)
        self._refresh_recent_files_menu()

    def _connect_signals(self) -> None:
        actions = self.window.actions
        panel = self.window.control_panel
        plot = self.window.workspace.main_plot.plot_widget

        actions.open_csv.triggered.connect(self.open_csv)
        actions.save_session.triggered.connect(self.save_session)
        actions.load_session.triggered.connect(self.load_session)
        actions.save_main_plot_svg.triggered.connect(self.save_main_plot_svg)
        actions.save_bar_chart_svg.triggered.connect(self.save_bar_chart_svg)
        actions.export_events_csv.triggered.connect(self.export_exceedance_events_csv)
        actions.export_region_csv.triggered.connect(self.export_selected_region_csv)
        actions.export_analysis_summary_json.triggered.connect(self.export_analysis_summary_json)
        actions.export_analysis_summary_csv.triggered.connect(self.export_analysis_summary_csv)
        actions.add_divider.triggered.connect(self.add_divider)
        actions.add_threshold.triggered.connect(self.add_threshold)
        actions.reset_view.triggered.connect(plot.reset_view)
        actions.toggle_theme.triggered.connect(self.toggle_theme)

        panel.open_csv_requested.connect(self.open_csv)
        panel.update_plot_requested.connect(self.update_plot)
        panel.active_case_changed.connect(self.active_trace_case_changed)
        panel.threshold_changed.connect(self.threshold_changed)
        panel.threshold_disabled_requested.connect(self.disable_threshold)
        panel.divider_add_requested.connect(self.add_divider)
        panel.divider_edit_requested.connect(self.edit_divider)
        panel.divider_delete_requested.connect(self.delete_divider)
        panel.column_selection_changed.connect(self.column_selection_changed)
        panel.cancel_task_requested.connect(self.cancel_task)
        panel.case_visibility_changed.connect(self.case_visibility_changed)
        panel.case_color_changed.connect(self.case_color_changed)
        panel.legend_visibility_changed.connect(plot.set_legend_visible)

        plot.request_add_divider.connect(self.add_divider_at)
        plot.request_add_threshold.connect(self.create_threshold_from_plot)
        plot.threshold_moved.connect(self.threshold_dragged)
        plot.divider_moved.connect(self.divider_dragged)
        plot.plot_clicked.connect(self.plot_clicked)
        plot.slider_moved.connect(self.slider_moved)

    def open_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Open CSV",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            self.open_csv_path(path)

    def open_csv_path(self, path: str) -> None:
        if not Path(path).exists():
            self._warn(f"CSV file does not exist:\n{path}")
            return
        started = self.tasks.start(
            self._inspect_csv,
            path,
            on_finished=self.csv_loaded,
            on_failed=self.task_failed,
            pass_task_context=True,
        )
        if started:
            self.window.statusBar().showMessage("Inspecting CSV...")
        else:
            self.window.statusBar().showMessage("A background task is already running.")

    def csv_loaded(self, result: object) -> None:
        metadata = result if isinstance(result, dict) else {}
        path = str(metadata.get("path", ""))
        columns = list(metadata.get("columns", []))
        numeric_columns = list(metadata.get("numeric_columns", []))
        likely_time_columns = list(metadata.get("likely_time_columns", []))
        preview = metadata.get("preview", {})

        self.current_csv_path = path
        self.settings.add_recent_file(path)
        self.settings.save()
        self._refresh_recent_files_menu()

        panel = self.window.control_panel
        panel.set_file_info(path, metadata.get("row_count"))
        panel.set_columns(columns, numeric_columns=numeric_columns, likely_time_columns=likely_time_columns)
        self.window.workspace.preview_table.set_preview(
            list(preview.get("headers", [])),
            list(preview.get("rows", [])),
        )

        if self.pending_session is not None:
            self._apply_session_to_controls(self.pending_session)
            self.update_plot()

        self.window.statusBar().showMessage(f"Loaded CSV metadata: {Path(path).name}")

    def update_plot(self) -> None:
        if not self.current_csv_path:
            self._info("Open a CSV file before plotting.")
            return

        panel = self.window.control_panel
        time_column = panel.selected_time_column()
        target_columns = panel.selected_target_columns()
        auxiliary_columns = panel.selected_auxiliary_columns()
        auxiliary_axes = panel.auxiliary_axis_assignments()
        if not time_column:
            self._info("Select a time column before plotting.")
            return
        if not target_columns:
            self._info("Select at least one target/case column before plotting.")
            return

        started = self.tasks.start(
            self._load_selected_data,
            self.current_csv_path,
            time_column,
            target_columns,
            auxiliary_columns,
            on_finished=self.plot_data_loaded,
            on_failed=self.task_failed,
            pass_task_context=True,
        )
        if not started:
            self.window.statusBar().showMessage("A background task is already running.")
            return
        self._pending_auxiliary_axes = auxiliary_axes
        self.window.statusBar().showMessage("Loading selected columns...")

    def plot_data_loaded(self, result: object) -> None:
        data = result
        self.loaded_data = data
        self.set_loaded_data_for_export(data, self._selected_region_tuple())

        target_names = list(data.targets)
        auxiliary_names = list(data.auxiliaries)
        self.case_units = {name: unit_for_column(name) for name in target_names}
        self._ensure_case_defaults(target_names, auxiliary_names)

        plot = self.window.workspace.main_plot.plot_widget
        target_colors = {name: self.case_colors[name] for name in target_names}
        auxiliary_colors = {name: self.case_colors[name] for name in auxiliary_names}
        auxiliary_axes = getattr(self, "_pending_auxiliary_axes", {})

        plot.set_data(
            data.time_values,
            data.targets,
            time_label=data.time_column,
            colors=target_colors,
        )
        plot.set_time_axis(TimeAxis(data.time_column, data.time_values, bool(data.time_is_datetime)))
        plot.set_auxiliary_data(data.auxiliaries, auxiliary_axes, colors=auxiliary_colors)
        for name, visible in self.case_visibility.items():
            plot.set_curve_visible(name, visible)

        self.window.control_panel.set_case_styles(
            target_names,
            colors={name: self.case_colors[name] for name in target_names},
            visibility={name: self.case_visibility.get(name, True) for name in target_names},
            units=self.case_units,
        )
        self.window.control_panel.set_active_cases(target_names)
        plot.set_active_case(self.window.control_panel.active_case_combo.currentText())

        time_range = data.time_range
        self.divider_manager = DividerManager(time_range=time_range)
        self.region_selector = RegionSelector(time_range=time_range, plot_adapter=plot)

        if self.pending_session is not None:
            self._restore_session_after_data_load(self.pending_session)
            self.pending_session = None

        self.threshold_manager.set_target_data(data.time_values, data.targets)
        self.threshold_manager.set_region(*self._selected_region_tuple_or_none_args())
        self._refresh_statistics()
        self.window.statusBar().showMessage(f"Plotted {len(target_names)} target case(s).")

    def set_loaded_data_for_export(
        self,
        data: Any,
        selected_region: tuple[float | None, float | None] | None = None,
    ) -> None:
        self.loaded_data_for_export = data
        self.selected_region_for_export = selected_region

    def save_session(self) -> None:
        path = self._choose_save_path("Save project", "event_analyzer_project.json", "Project files (*.json)")
        if not path:
            return
        state = self._session_state()
        try:
            state.save(path)
        except Exception as exc:
            self._warn(f"Could not save project:\n{exc}", title="Save project failed")
            return
        self.window.statusBar().showMessage(f"Saved project: {path}")

    def load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Load project",
            "",
            "Project files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            session = SessionState.load(path)
        except Exception as exc:
            self._warn(f"Could not load project:\n{exc}", title="Load project failed")
            return
        if not session.csv_path:
            self._warn("Project does not contain a CSV path.", title="Load project failed")
            return
        self.pending_session = session
        self._apply_theme(session.theme)
        self.open_csv_path(session.csv_path)

    def save_main_plot_svg(self) -> None:
        path = self._choose_save_path("Save main plot SVG", "time_series.svg", "SVG files (*.svg)")
        if path:
            self._run_export(
                lambda: export_main_plot_svg(self.window.workspace.main_plot.plot_widget, path),
                f"Saved main plot SVG: {path}",
            )

    def save_bar_chart_svg(self) -> None:
        path = self._choose_save_path("Save bar chart SVG", "exceedance_durations.svg", "SVG files (*.svg)")
        if path:
            self._run_export(lambda: export_bar_chart_svg(self.window.workspace.bar_chart, path), f"Saved bar chart SVG: {path}")

    def export_exceedance_events_csv(self) -> None:
        path = self._choose_save_path("Export exceedance events CSV", "exceedance_events.csv", "CSV files (*.csv)")
        if path:
            self._run_export(lambda: export_events_csv(self.threshold_manager.events, path), f"Exported exceedance events CSV: {path}")

    def export_selected_region_csv(self) -> None:
        if self.loaded_data_for_export is None:
            self._info("No loaded data is available to export.")
            return
        path = self._choose_save_path("Export selected region CSV", "selected_region.csv", "CSV files (*.csv)")
        if path:
            self._run_export(
                lambda: export_selected_region_csv(
                    self.loaded_data_for_export,
                    self.selected_region_for_export,
                    path,
                ),
                f"Exported selected region CSV: {path}",
            )

    def export_analysis_summary_json(self) -> None:
        path = self._choose_save_path("Export analysis summary JSON", "analysis_summary.json", "JSON files (*.json)")
        if path:
            self._run_export(
                lambda: export_analysis_summary_json(self.threshold_manager.summary, self.threshold_manager.events, path),
                f"Exported analysis summary JSON: {path}",
            )

    def export_analysis_summary_csv(self) -> None:
        path = self._choose_save_path("Export analysis summary CSV", "analysis_summary.csv", "CSV files (*.csv)")
        if path:
            self._run_export(
                lambda: export_analysis_summary_csv(self.threshold_manager.summary, self.threshold_manager.events, path),
                f"Exported analysis summary CSV: {path}",
            )

    def active_trace_case_changed(self, case_name: str) -> None:
        self.window.workspace.main_plot.plot_widget.set_active_case(case_name)
        self.window.statusBar().showMessage(f"Active trace case: {case_name}")

    def threshold_changed(self, value: float) -> None:
        self.threshold_manager.edit_threshold(value)
        self._refresh_statistics()
        self.window.statusBar().showMessage(f"Threshold changed: {value:.6g}")

    def add_threshold(self) -> None:
        value, accepted = QInputDialog.getDouble(
            self.window,
            "Add threshold",
            "Threshold value",
            self.threshold_manager.value or 0.0,
            -1e18,
            1e18,
            8,
        )
        if accepted:
            self.window.control_panel.set_threshold_value(value)
            self.threshold_manager.set_threshold(value)
            self._refresh_statistics()

    def create_threshold_from_plot(self, value: float) -> None:
        self.window.control_panel.set_threshold_value(value)
        self.threshold_manager.create_from_plot(value)
        self._refresh_statistics()
        self.window.statusBar().showMessage(f"Threshold created: {value:.6g}")

    def threshold_dragged(self, value: float) -> None:
        self.window.control_panel.set_threshold_value(value)
        self.threshold_manager.drag_threshold(value)
        self._refresh_statistics()
        self.window.statusBar().showMessage(f"Threshold moved: {value:.6g}")

    def disable_threshold(self) -> None:
        self.threshold_manager.disable_threshold()
        self._refresh_statistics()
        self.window.statusBar().showMessage("Threshold disabled.")

    def add_divider(self) -> None:
        if self.loaded_data is None:
            self._info("Load and plot data before adding dividers.")
            return
        default = self.current_slider_time
        if default is None:
            start, end = self.loaded_data.time_range
            default = (start + end) / 2
        value, accepted = QInputDialog.getDouble(
            self.window,
            "Add divider",
            "Divider time",
            float(default),
            -1e18,
            1e18,
            8,
        )
        if accepted:
            self.add_divider_at(value)

    def add_divider_at(self, time_value: float) -> None:
        try:
            self.divider_manager.add_divider(time_value)
        except DividerError as exc:
            self._warn(str(exc), title="Divider error")
            return
        self._sync_dividers()
        self._reconcile_region()

    def edit_divider(self) -> None:
        row = self.window.control_panel.selected_divider_row()
        dividers = self.divider_manager.dividers
        if row < 0 or row >= len(dividers):
            self._info("Select a divider to edit.")
            return
        divider = dividers[row]
        value, accepted = QInputDialog.getDouble(
            self.window,
            "Edit divider",
            f"{divider.label} time",
            divider.time,
            -1e18,
            1e18,
            8,
        )
        if not accepted:
            return
        try:
            self.divider_manager.edit_divider_time(divider.id, value)
        except DividerError as exc:
            self._warn(str(exc), title="Divider error")
            return
        self._sync_dividers()
        self._reconcile_region()

    def delete_divider(self) -> None:
        row = self.window.control_panel.selected_divider_row()
        dividers = self.divider_manager.dividers
        if row < 0 or row >= len(dividers):
            self._info("Select a divider to delete.")
            return
        self.divider_manager.delete_divider(dividers[row].id)
        self._sync_dividers()
        self._reconcile_region()

    def divider_dragged(self, divider_id: str, time_value: float) -> None:
        try:
            self.divider_manager.drag_divider(divider_id, time_value)
        except DividerError as exc:
            self._warn(str(exc), title="Divider error")
            self._sync_dividers()
            return
        self._sync_dividers()
        self._reconcile_region()

    def plot_clicked(self, time_value: float, _y_value: float) -> None:
        if self.region_selector is None:
            return
        region = self.region_selector.select_at(time_value, self.divider_manager)
        if region is not None:
            self._set_active_region(region)

    def slider_moved(self, time_value: float) -> None:
        self.current_slider_time = time_value

    def case_visibility_changed(self, case_name: str, visible: bool) -> None:
        self.case_visibility[case_name] = visible
        self.window.workspace.main_plot.plot_widget.set_curve_visible(case_name, visible)

    def case_color_changed(self, case_name: str, color: str) -> None:
        self.case_colors[case_name] = color
        self.window.workspace.main_plot.plot_widget.set_curve_color(case_name, color)

    def column_selection_changed(self) -> None:
        cases = self.window.control_panel.selected_target_columns()
        self.window.control_panel.set_active_cases(cases)
        self.window.control_panel.set_case_styles(
            cases,
            colors={name: self.case_colors.get(name, "") for name in cases},
            visibility={name: self.case_visibility.get(name, True) for name in cases},
            units={name: unit_for_column(name) for name in cases},
        )

    def task_failed(self, message: str) -> None:
        self.window.statusBar().showMessage("Task failed.")
        self._warn(message)

    def task_progress(self, current: int, total: int, message: str) -> None:
        del current, total
        if message:
            self.window.statusBar().showMessage(message)

    def cancel_task(self) -> None:
        self.tasks.cancel()
        self.window.statusBar().showMessage("Cancelling background task...")

    def toggle_theme(self) -> None:
        requested = "dark" if self.window.actions.toggle_theme.isChecked() else "light"
        self._apply_theme(requested)
        self.settings.save()

    def _apply_theme(self, theme: str) -> None:
        app = QApplication.instance()
        if app is not None:
            self.settings.theme = apply_theme(app, theme)
        self.window.actions.toggle_theme.setChecked(self.settings.theme == "dark")

    def _refresh_recent_files_menu(self) -> None:
        menu = self.window.recent_files_menu
        if menu is None:
            return
        menu.clear()
        if not self.settings.recent_files:
            empty = QAction("No recent files", menu)
            empty.setEnabled(False)
            menu.addAction(empty)
            return
        for path in self.settings.recent_files:
            action = QAction(Path(path).name, menu)
            action.setToolTip(path)
            action.triggered.connect(lambda _checked=False, p=path: self.open_csv_path(p))
            menu.addAction(action)

    def _sync_dividers(self) -> None:
        plot = self.window.workspace.main_plot.plot_widget
        current_plot_ids = set(getattr(plot, "_dividers", {}).keys())
        manager_ids = {divider.id for divider in self.divider_manager.dividers}
        for divider_id in current_plot_ids - manager_ids:
            plot.remove_divider(divider_id)
        for divider in self.divider_manager.dividers:
            if divider.id in current_plot_ids:
                plot.update_divider(divider.id, divider.time, divider.label)
            else:
                plot.add_divider(divider.time, divider_id=divider.id, label=divider.label)
        self.window.control_panel.set_dividers(
            [(divider.label, f"{divider.time:.6g}") for divider in self.divider_manager.dividers]
        )

    def _reconcile_region(self) -> None:
        if self.region_selector is None:
            return
        region = self.region_selector.reconcile_after_divider_change(self.divider_manager)
        if region is not None:
            self._set_active_region(region)
        else:
            self._update_region_text()
            self.threshold_manager.set_region(*self._selected_region_tuple_or_none_args())
            self._refresh_statistics()

    def _set_active_region(self, region: SelectedRegion) -> None:
        self.window.control_panel.set_region_text(
            f"{region.label}: {region.start_time:.6g} to {region.end_time:.6g}"
        )
        self.set_loaded_data_for_export(self.loaded_data, (region.start_time, region.end_time))
        self.threshold_manager.set_region(region.start_time, region.end_time)
        self._refresh_statistics()

    def _update_region_text(self) -> None:
        region = self.region_selector.selected_region if self.region_selector is not None else None
        if region is None:
            self.window.control_panel.set_region_text("Full time range")
        else:
            self.window.control_panel.set_region_text(
                f"{region.label}: {region.start_time:.6g} to {region.end_time:.6g}"
            )

    def _refresh_statistics(self) -> None:
        if self.loaded_data is None:
            self.window.workspace.statistics_table.set_statistics([])
            return
        rows = region_statistics_from_arrays(
            self.loaded_data.time_values,
            self.loaded_data.targets,
            self._selected_region_tuple(),
            threshold=self.threshold_manager.value,
            events=self.threshold_manager.events,
            units=self.case_units,
        )
        self.window.workspace.statistics_table.set_statistics(rows)

    def _selected_region_tuple(self) -> tuple[float | None, float | None] | None:
        if self.region_selector is None or self.region_selector.selected_region is None:
            return None
        region = self.region_selector.selected_region
        return region.start_time, region.end_time

    def _selected_region_tuple_or_none_args(self) -> tuple[float | None, float | None]:
        region = self._selected_region_tuple()
        if region is None:
            return None, None
        return region

    def _session_state(self) -> SessionState:
        return SessionState(
            csv_path=self.current_csv_path,
            time_column=self.window.control_panel.selected_time_column(),
            target_columns=self.window.control_panel.selected_target_columns(),
            auxiliary_columns=self.window.control_panel.selected_auxiliary_columns(),
            auxiliary_axes=self.window.control_panel.auxiliary_axis_assignments(),
            dividers=self.divider_manager.serialize(),
            threshold=self.threshold_manager.value,
            region=self._selected_region_tuple(),
            colors=self.case_colors,
            visibility={**self.case_visibility, **self.window.control_panel.case_visibility()},
            theme=self.settings.theme,
        )

    def _apply_session_to_controls(self, session: SessionState) -> None:
        panel = self.window.control_panel
        panel.set_time_column(session.time_column)
        self.case_colors.update(session.colors)
        self.case_visibility.update(session.visibility)
        panel.set_selected_columns(
            target_columns=session.target_columns,
            auxiliary_columns=session.auxiliary_columns,
            auxiliary_axes=session.auxiliary_axes,
        )
        if session.threshold is not None:
            panel.set_threshold_value(session.threshold)

    def _restore_session_after_data_load(self, session: SessionState) -> None:
        self.case_colors.update(session.colors)
        self.case_visibility.update(session.visibility)
        try:
            self.divider_manager.deserialize(session.dividers)
        except DividerError as exc:
            self._warn(f"Some saved dividers could not be restored:\n{exc}", title="Project restore warning")
        self._sync_dividers()

        if session.region is not None and self.region_selector is not None:
            self.region_selector.set_region(session.region[0], session.region[1])
            self._update_region_text()
            self.set_loaded_data_for_export(self.loaded_data, session.region)

        if session.threshold is not None:
            self.window.control_panel.set_threshold_value(session.threshold)
            self.threshold_manager.set_threshold(session.threshold)

        for name, visible in self.case_visibility.items():
            self.window.workspace.main_plot.plot_widget.set_curve_visible(name, visible)
        for name, color in self.case_colors.items():
            self.window.workspace.main_plot.plot_widget.set_curve_color(name, color)

    def _ensure_case_defaults(self, target_names: list[str], auxiliary_names: list[str]) -> None:
        for index, name in enumerate([*target_names, *auxiliary_names]):
            self.case_colors.setdefault(name, color_for_index(index))
            self.case_visibility.setdefault(name, True)

    def _choose_save_path(self, title: str, default_name: str, file_filter: str) -> str:
        path, _ = QFileDialog.getSaveFileName(self.window, title, default_name, file_filter)
        if not path:
            return ""
        if Path(path).exists():
            answer = QMessageBox.question(
                self.window,
                "Confirm overwrite",
                f"'{Path(path).name}' already exists. Replace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return ""
        return path

    def _run_export(self, export_fn, success_message: str) -> None:
        try:
            export_fn()
        except ExportError as exc:
            self._warn(str(exc), title="Export failed")
        except Exception as exc:
            self._warn(f"Export failed:\n{exc}", title="Export failed")
        else:
            self.window.statusBar().showMessage(success_message)

    def _warn(self, message: str, *, title: str = "Event Analyzer") -> None:
        self.window.statusBar().showMessage(title)
        QMessageBox.warning(self.window, title, message)

    def _info(self, message: str) -> None:
        self.window.statusBar().showMessage(message)
        QMessageBox.information(self.window, "Event Analyzer", message)

    @staticmethod
    def _inspect_csv(
        path: str,
        *,
        cancel_token: object | None = None,
        progress_callback=None,
    ) -> dict[str, object]:
        from event_analyzer.data.data_manager import DataManager

        manager = DataManager(preview_rows=100)
        metadata = manager.open_csv(path, cancel_token=cancel_token, progress_callback=progress_callback)
        return {
            "path": str(metadata.path),
            "columns": metadata.column_names,
            "numeric_columns": metadata.numeric_columns,
            "likely_time_columns": metadata.likely_time_columns,
            "row_count": None,
            "preview": {
                "headers": metadata.preview.headers,
                "rows": metadata.preview.rows,
            },
        }

    @staticmethod
    def _load_selected_data(
        path: str,
        time_column: str,
        target_columns: list[str],
        auxiliary_columns: list[str],
        *,
        cancel_token: object | None = None,
        progress_callback=None,
    ) -> object:
        from event_analyzer.data.data_manager import DataManager

        manager = DataManager(preview_rows=100)
        manager.open_csv(path, cancel_token=cancel_token, progress_callback=progress_callback)
        return manager.select_columns(
            time_column=time_column,
            target_columns=target_columns,
            auxiliary_columns=auxiliary_columns,
            ignore_invalid_targets=True,
            cancel_token=cancel_token,
            progress_callback=progress_callback,
        )
