"""
Background processing workers using QThread + signals.
Replaces Flask's threading + polling pattern with instant signal-based updates.
"""
import time
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
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.tracks_dir = tracks_dir
        self.model = model
        self.conf = conf
        self.session_id = session_id

    def run(self):
        try:
            self.progress.emit(self.session_id, 0)
            from cctv_yolo.processor import process_video

            process_video(
                self.video_path,
                self.tracks_dir,
                self.model,
                self.conf,
                session_id=self.session_id,
            )
            self.progress.emit(self.session_id, 100)
            self.finished.emit(self.session_id)
        except Exception as e:
            self.error.emit(self.session_id, str(e))


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
