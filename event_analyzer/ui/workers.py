from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


class CancellationToken:
    """Thread-safe enough cooperative cancellation flag for background tasks."""

    def __init__(self) -> None:
        self._cancelled = False

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True


class Worker(QObject):
    """Run one blocking callable on a QThread and report success or failure."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        pass_task_context: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._pass_task_context = pass_task_context
        self.cancel_token = CancellationToken()

    @pyqtSlot()
    def run(self) -> None:
        try:
            kwargs = dict(self._kwargs)
            if self._pass_task_context:
                kwargs.setdefault("cancel_token", self.cancel_token)
                kwargs.setdefault("progress_callback", self._report_progress)
            self.finished.emit(self._fn(*self._args, **kwargs))
        except Exception as exc:
            self.failed.emit(str(exc))

    def _report_progress(self, current: int, total: int, message: str = "") -> None:
        self.progress.emit(int(current), int(total), str(message))


class BackgroundTaskRunner(QObject):
    """Small single-task QThread runner for CSV loading and future analysis jobs."""

    busy_changed = pyqtSignal(bool)
    progress_changed = pyqtSignal(int, int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: Worker | None = None

    @property
    def is_busy(self) -> bool:
        return self._thread is not None

    def start(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_finished: Callable[[object], None],
        on_failed: Callable[[str], None],
        on_progress: Callable[[int, int, str], None] | None = None,
        pass_task_context: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Start a background task.

        Returns ``False`` if another task is still running.
        """
        if self._thread is not None:
            return False

        thread = QThread(self)
        worker = Worker(fn, *args, pass_task_context=pass_task_context, **kwargs)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.progress.connect(self._relay_progress)
        if on_progress is not None:
            worker.progress.connect(on_progress)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup)

        self._thread = thread
        self._worker = worker
        self.busy_changed.emit(True)
        thread.start()
        return True

    def cancel(self) -> None:
        """Request cancellation for the active task.

        Cancellation is cooperative: tasks stop at explicit checks, while a
        single blocking library call may need to return before the flag is seen.
        """
        if self._worker is not None:
            self._worker.cancel_token.cancel()

    def _relay_progress(self, current: int, total: int, message: str) -> None:
        self.progress_changed.emit(current, total, message)

    def _cleanup(self) -> None:
        self._thread = None
        self._worker = None
        self.busy_changed.emit(False)


__all__ = ["BackgroundTaskRunner", "CancellationToken", "Worker"]
