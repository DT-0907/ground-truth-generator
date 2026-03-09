"""
Background processing workers using QThread + signals.
Replaces Flask's threading + polling pattern with instant signal-based updates.
"""
import traceback
from PySide6.QtCore import QThread, Signal


class ProcessingWorker(QThread):
    """Runs YOLO detection + tracking on a video in a background thread.

    Signals
    -------
    progress(session_id, percent)
        Emitted periodically with 0-100 progress.
    finished(session_id)
        Emitted when processing completes successfully.
    error(session_id, error_message)
        Emitted when processing fails.
    """

    progress = Signal(str, int)   # session_id, percent (0-100)
    finished = Signal(str)        # session_id
    error = Signal(str, str)      # session_id, error_message

    def __init__(
        self,
        video_path: str,
        tracks_dir: str,
        model: str = "yolov8m.pt",
        conf: float = 0.25,
        session_id: str = "",
        models_dir: str = "",
        processing_roi: dict = None,
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

    def run(self):
        try:
            self.progress.emit(self.session_id, 0)
            from cctv_yolo.processor import process_video

            def _on_progress(pct):
                self.progress.emit(self.session_id, min(pct, 99))

            process_video(
                self.video_path,
                self.tracks_dir,
                self.model,
                self.conf,
                session_id=self.session_id,
                progress_callback=_on_progress,
                models_dir=self.models_dir if self.models_dir else None,
                processing_roi=self.processing_roi,
            )
            self.progress.emit(self.session_id, 100)
            self.finished.emit(self.session_id)
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
