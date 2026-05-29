"""First-run GPU acceleration dialog.

Offers to download the matching CUDA build of torch (see cctv_yolo.gpu_runtime)
with a progress bar, then asks the user to restart. Used both at first launch
(when an NVIDIA GPU is detected and nothing is installed yet) and from
Settings -> "Set up / repair GPU acceleration".
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from cctv_yolo import gpu_runtime
from cctv_yolo.theme import BORDER, INDIGO, OFFWHITE, PINK, PURPLE, RADIUS, TEXT_MUTED

logger = logging.getLogger(__name__)


class _InstallWorker(QThread):
    """Runs gpu_runtime.install() off the GUI thread."""

    progress = Signal(int, int)     # done_bytes, total_bytes (current file)
    status = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, variant: str, parent=None):
        super().__init__(parent)
        self.variant = variant
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            manifest = gpu_runtime.install(
                self.variant,
                progress=lambda d, t: self.progress.emit(d, t),
                status=lambda m: self.status.emit(m),
                cancel=lambda: self._cancel,
            )
        except gpu_runtime.GpuSetupCancelled:
            self.cancelled.emit()
            return
        except Exception as e:  # noqa: BLE001 — surface any failure to the UI
            logger.exception("GPU torch install failed")
            self.failed.emit(str(e))
            return
        self.finished_ok.emit(manifest)


class GpuSetupDialog(QDialog):
    """Modal dialog: offer -> download (progress) -> restart prompt."""

    def __init__(self, variant: str, parent=None):
        super().__init__(parent)
        self._variant = variant
        self._worker: _InstallWorker | None = None
        self._done = False

        self.setWindowTitle("GPU Acceleration")
        self.setMinimumWidth(460)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {INDIGO}; }}
            QLabel {{ color: {OFFWHITE}; }}
            QLabel#muted {{ color: {TEXT_MUTED}; font-size: 12px; }}
            QProgressBar {{
                background-color: {BORDER}; border: none; border-radius: 3px;
                text-align: center; color: {OFFWHITE}; min-height: 8px;
            }}
            QProgressBar::chunk {{ background-color: {PURPLE}; border-radius: 3px; }}
            QPushButton {{
                background-color: {PURPLE}; color: {OFFWHITE};
                border: 1px solid {PURPLE}; border-radius: {RADIUS}px;
                padding: 6px 14px; font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {PINK}; border-color: {PINK}; color: {INDIGO}; }}
            QPushButton#secondary {{ background-color: transparent; color: {OFFWHITE}; border: 1px solid {BORDER}; }}
            QPushButton#secondary:hover {{ border-color: {PINK}; color: {PINK}; }}
        """)

        name = gpu_runtime.gpu_name() or "an NVIDIA GPU"
        size_mb = gpu_runtime.estimated_download_mb(variant)

        self._title = QLabel(f"GPU detected: {name}")
        f = self._title.font()
        f.setPointSize(f.pointSize() + 2)
        f.setBold(True)
        self._title.setFont(f)
        self._title.setWordWrap(True)

        self._body = QLabel(
            f"CCTV-YOLO is running in CPU mode. I can install GPU acceleration "
            f"(PyTorch {variant}) for a big speedup — a one-time download of "
            f"about {size_mb / 1000:.1f} GB. The app keeps working on CPU if you skip this."
        )
        self._body.setObjectName("muted")
        self._body.setWordWrap(True)

        self._status = QLabel("")
        self._status.setObjectName("muted")
        self._status.setWordWrap(True)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setVisible(False)

        self._btn_install = QPushButton("Install GPU support")
        self._btn_install.clicked.connect(self._start)
        self._btn_later = QPushButton("Not now")
        self._btn_later.setObjectName("secondary")
        self._btn_later.clicked.connect(self.reject)
        self._btn_never = QPushButton("Don't ask again")
        self._btn_never.setObjectName("secondary")
        self._btn_never.clicked.connect(self._decline)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setObjectName("secondary")
        self._btn_cancel.clicked.connect(self._cancel)
        self._btn_cancel.setVisible(False)
        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.accept)
        self._btn_close.setVisible(False)

        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self._btn_never)
        btns.addWidget(self._btn_later)
        btns.addWidget(self._btn_install)
        btns.addWidget(self._btn_cancel)
        btns.addWidget(self._btn_close)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.setSpacing(12)
        lay.addWidget(self._title)
        lay.addWidget(self._body)
        lay.addWidget(self._bar)
        lay.addWidget(self._status)
        lay.addLayout(btns)

    # -- actions ----------------------------------------------------------
    def _start(self):
        self._btn_install.setVisible(False)
        self._btn_later.setVisible(False)
        self._btn_never.setVisible(False)
        self._btn_cancel.setVisible(True)
        self._bar.setVisible(True)
        self._status.setText("Starting…")

        self._worker = _InstallWorker(self._variant, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._status.setText)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _on_progress(self, done: int, total: int):
        if total > 0:
            self._bar.setRange(0, 100)
            self._bar.setValue(int(done / total * 100))
        else:
            self._bar.setRange(0, 0)  # indeterminate

    def _on_done(self, manifest: dict):
        self._done = True
        self._bar.setValue(100)
        self._btn_cancel.setVisible(False)
        self._btn_close.setVisible(True)
        self._title.setText("GPU acceleration installed")
        self._body.setText(
            f"Installed PyTorch {manifest.get('torch', '')} ({manifest.get('variant', '')}). "
            "Restart CCTV-YOLO to start using your GPU."
        )
        self._status.setText("")

    def _on_failed(self, msg: str):
        self._bar.setVisible(False)
        self._btn_cancel.setVisible(False)
        self._btn_close.setVisible(True)
        self._title.setText("GPU setup didn't finish")
        self._body.setText(
            "The app will keep running on CPU. You can try again later from "
            "Settings -> GPU acceleration."
        )
        self._status.setText(msg)

    def _on_cancelled(self):
        self._bar.setVisible(False)
        self._btn_cancel.setVisible(False)
        self._btn_close.setVisible(True)
        self._title.setText("GPU setup cancelled")
        self._body.setText("No problem — the app keeps running on CPU.")
        self._status.setText("")

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._status.setText("Cancelling…")
            self._worker.cancel()

    def _decline(self):
        gpu_runtime.mark_declined()
        self.reject()

    def closeEvent(self, event):
        # Never let the dialog (the worker's parent) be destroyed while the
        # QThread is still running — that aborts the process. Cancel, wait a
        # bounded time; if it hasn't stopped yet, keep the dialog open and let
        # it close once the worker honors the cancel.
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            if not self._worker.wait(5000):
                self._status.setText("Cancelling — finishing the current step…")
                event.ignore()
                return
        super().closeEvent(event)


def maybe_run_gpu_setup(variant: str, parent=None) -> None:
    """Show the GPU setup dialog (modal). Safe to call from main() at startup.

    Never raises — GPU acceleration is best-effort; the app always works on the
    baked CPU torch.
    """
    try:
        dlg = GpuSetupDialog(variant, parent=parent)
        dlg.exec()
    except Exception:
        logger.exception("GPU setup dialog failed")
