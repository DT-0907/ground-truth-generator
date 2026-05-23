"""
Background processing workers using QThread / QRunnable + signals.
Replaces Flask's threading + polling pattern with instant signal-based updates.

PRD E5-4: ProcessingWorker has been converted to a ``QRunnable`` so the batch
scheduler can run many concurrently in a ``QThreadPool``. A ``QThread``-based
``ProcessingWorker`` shim is kept for any single-shot legacy callers
(processing one video on-demand from the Videos tab).
"""
import traceback
from PySide6.QtCore import QObject, QRunnable, QThread, Signal


# ---------------------------------------------------------------------------
# QRunnable-based worker (used by the batch scheduler)
# ---------------------------------------------------------------------------

class ProcessingSignals(QObject):
    """Signal carrier for ``ProcessingRunnable``.

    QRunnable can't subclass QObject so signals live on a side object that
    the runnable creates and exposes via ``.signals``.
    """
    progress         = Signal(str, int)              # session_id, percent (0-100)
    progress_detail  = Signal(str, int, float, float)  # sid, percent, fps, eta_seconds
    finished         = Signal(str)                   # session_id
    error            = Signal(str, str)              # session_id, error_message
    cancelled        = Signal(str)                   # session_id (cooperative cancel)


class ProcessingRunnable(QRunnable):
    """Process one video on a worker thread inside a ``QThreadPool``.

    The scheduler keeps a reference to this object so it can flip
    ``cancel_requested = True`` to abort the in-flight ``process_video``
    call (the cancel check runs once per frame inside ``processor.py``).
    """

    def __init__(
        self,
        video_path: str,
        tracks_dir: str,
        model: str = "yolov8m.pt",
        conf: float = 0.25,
        session_id: str = "",
        models_dir: str = "",
        processing_roi: dict = None,
        sample_rate: int = 1,
    ):
        super().__init__()
        self.video_path = video_path
        self.tracks_dir = tracks_dir
        self.model = model
        self.conf = conf
        self.session_id = session_id
        self.models_dir = models_dir
        self.processing_roi = processing_roi
        self.sample_rate = max(1, int(sample_rate))
        self.signals = ProcessingSignals()
        # Flipped from the scheduler thread; read on the worker thread.
        self.cancel_requested = False
        # Lets pool.clear() drop us before we even start.
        self.setAutoDelete(True)

    def cancel(self):
        """Request cooperative cancellation. Safe to call from any thread."""
        self.cancel_requested = True

    def run(self):
        from cctv_yolo.processor import process_video, BatchCancelled
        try:
            # If cancellation was requested before we even got scheduled,
            # skip the heavy work and emit cancelled.
            if self.cancel_requested:
                self.signals.cancelled.emit(self.session_id)
                return

            self.signals.progress.emit(self.session_id, 0)

            def _on_progress(pct):
                self.signals.progress.emit(self.session_id, min(pct, 99))

            def _on_progress_detail(pct, fps, eta):
                self.signals.progress_detail.emit(
                    self.session_id, min(int(pct), 99), float(fps), float(eta)
                )

            def _should_cancel():
                return self.cancel_requested

            process_video(
                self.video_path,
                self.tracks_dir,
                self.model,
                self.conf,
                session_id=self.session_id,
                progress_callback=_on_progress,
                progress_detail_callback=_on_progress_detail,
                models_dir=self.models_dir if self.models_dir else None,
                processing_roi=self.processing_roi,
                should_cancel=_should_cancel,
                sample_rate=self.sample_rate,
            )
            self.signals.progress.emit(self.session_id, 100)
            self.signals.finished.emit(self.session_id)
        except BatchCancelled:
            # Clean cancel — scheduler is responsible for cleanup.
            self.signals.cancelled.emit(self.session_id)
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"{e}\n\n{tb}"
            print(f"[ProcessingRunnable] Error processing {self.session_id}:\n{tb}")
            self.signals.error.emit(self.session_id, error_msg)


# ---------------------------------------------------------------------------
# Legacy QThread shim (kept so non-batch callers still work)
# ---------------------------------------------------------------------------

class ProcessingWorker(QThread):
    """Single-shot QThread wrapper for callers that don't use the batch pool.

    Public Qt signals are unchanged from the original API.
    """

    progress = Signal(str, int)              # session_id, percent (0-100)
    progress_detail = Signal(str, int, float, float)  # sid, pct, fps, eta_seconds
    finished = Signal(str)                   # session_id
    error = Signal(str, str)                 # session_id, error_message
    cancelled = Signal(str)                  # session_id (new — emitted on cooperative cancel)

    def __init__(
        self,
        video_path: str,
        tracks_dir: str,
        model: str = "yolov8m.pt",
        conf: float = 0.25,
        session_id: str = "",
        models_dir: str = "",
        processing_roi: dict = None,
        sample_rate: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.tracks_dir = tracks_dir
        self.model = model
        self.conf = conf
        self.session_id = session_id
        self.models_dir = models_dir
        self.processing_roi = processing_roi
        self.sample_rate = max(1, int(sample_rate))
        self._cancel_requested = False

    def cancel(self):
        """Request cooperative cancellation."""
        self._cancel_requested = True

    def run(self):
        from cctv_yolo.processor import process_video, BatchCancelled
        try:
            self.progress.emit(self.session_id, 0)

            def _on_progress(pct):
                self.progress.emit(self.session_id, min(pct, 99))

            def _on_progress_detail(pct, fps, eta):
                self.progress_detail.emit(
                    self.session_id, min(int(pct), 99), float(fps), float(eta)
                )

            def _should_cancel():
                return self._cancel_requested

            process_video(
                self.video_path,
                self.tracks_dir,
                self.model,
                self.conf,
                session_id=self.session_id,
                progress_callback=_on_progress,
                progress_detail_callback=_on_progress_detail,
                models_dir=self.models_dir if self.models_dir else None,
                processing_roi=self.processing_roi,
                should_cancel=_should_cancel,
                sample_rate=self.sample_rate,
            )
            self.progress.emit(self.session_id, 100)
            self.finished.emit(self.session_id)
        except BatchCancelled:
            self.cancelled.emit(self.session_id)
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"{e}\n\n{tb}"
            print(f"[ProcessingWorker] Error processing {self.session_id}:\n{tb}")
            self.error.emit(self.session_id, error_msg)


class ExportWorker(QThread):
    """Exports labeled images in a background thread.

    Signals
    -------
    progress(session_id, percent)
        Emitted periodically with 0-100 progress.
    finished(session_id, count)
        Emitted when export completes. *count* is the number of
        images written.
    error(session_id, error_message)
        Emitted when the export fails.
    """

    progress = Signal(str, int)   # session_id, percent
    finished = Signal(str, int)   # session_id, count
    error = Signal(str, str)      # session_id, error_message

    def __init__(self, data_manager, session_id: str, sample_rate: int = 1, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.session_id = session_id
        self.sample_rate = sample_rate

    def run(self):
        try:
            self.data_manager.set_export_status(
                self.session_id, "exporting", progress=0
            )
            count = self.data_manager.export_labeled_images(
                self.session_id, self.sample_rate
            )
            self.data_manager.set_export_status(
                self.session_id, "exported", progress=100
            )
            self.finished.emit(self.session_id, count)
        except Exception as e:
            self.data_manager.set_export_status(
                self.session_id, "error", error=str(e)
            )
            self.error.emit(self.session_id, str(e))


class CocoExportWorker(QThread):
    """Exports COCO annotations in a background thread.

    Signals
    -------
    finished(session_id, output_path)
        Emitted when the COCO export completes. *output_path* is the
        absolute path to the generated JSON file.
    error(session_id, error_message)
        Emitted when the export fails.
    """

    finished = Signal(str, str)   # session_id, output_path
    error = Signal(str, str)      # session_id, error_message

    def __init__(self, data_manager, session_id: str, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.session_id = session_id

    def run(self):
        try:
            output_path = self.data_manager.export_coco(self.session_id)
            self.finished.emit(self.session_id, str(output_path))
        except Exception as e:
            self.error.emit(self.session_id, str(e))
