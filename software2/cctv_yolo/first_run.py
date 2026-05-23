"""First-run wizard (PRD C7).

Triggered on first launch (no `~/Documents/CCTV-YOLO/.first_run_complete` marker).
Walks the user through: welcome → data folder → model download → tour → finish.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWizard, QWizardPage, QLabel, QVBoxLayout, QRadioButton, QLineEdit,
    QProgressDialog, QMessageBox, QPushButton, QHBoxLayout, QWidget,
)

from cctv_yolo.__version__ import __version__, __app_name__
from cctv_yolo.theme import (
    PURPLE, PINK, OFFWHITE, PANEL, INDIGO, BORDER, TEXT_MUTED, RADIUS,
)


FIRST_RUN_MARKER = Path.home() / "Documents" / "CCTV-YOLO" / ".first_run_complete"


def is_first_run() -> bool:
    return not FIRST_RUN_MARKER.exists()


def mark_first_run_complete() -> None:
    FIRST_RUN_MARKER.parent.mkdir(parents=True, exist_ok=True)
    FIRST_RUN_MARKER.touch()


_TITLE_STYLE = f"color: {PINK}; font-size: 20px; font-weight: bold;"
_BODY_STYLE  = f"color: {OFFWHITE}; font-size: 13px;"
_HINT_STYLE  = f"color: {TEXT_MUTED}; font-size: 11px;"
_BTN_STYLE   = f"""
    QPushButton {{
        background-color: {PURPLE};
        color: {OFFWHITE};
        border: none;
        border-radius: {RADIUS}px;
        padding: 6px 14px;
        font-weight: bold;
    }}
    QPushButton:hover {{ background-color: {PINK}; color: {INDIGO}; }}
"""


def _make_section(title: str, body_lines: list[str]) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(20, 16, 20, 16)
    t = QLabel(title)
    t.setStyleSheet(_TITLE_STYLE)
    lay.addWidget(t)
    for line in body_lines:
        lbl = QLabel(line)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(_BODY_STYLE)
        lay.addWidget(lbl)
    lay.addStretch(1)
    return w


class _WelcomePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome")
        section = _make_section(
            f"Welcome to {__app_name__} v{__version__}",
            [
                "Detect, track, and correct vehicles in CCTV footage.",
                "",
                "This short setup will create your data folder, optionally "
                "download a starter detection model, and point you at the "
                "Preprocessing tab to begin.",
                "",
                "Click Next to continue.",
            ],
        )
        lay = QVBoxLayout(self)
        lay.addWidget(section)


class _DataFolderPage(QWizardPage):
    def __init__(self, data_manager):
        super().__init__()
        self.dm = data_manager
        self.setTitle("Data Folder")
        section = QWidget()
        s = QVBoxLayout(section)
        s.setContentsMargins(20, 16, 20, 16)
        t = QLabel("Data Folder")
        t.setStyleSheet(_TITLE_STYLE)
        s.addWidget(t)
        body = QLabel(
            "All videos, tracks, corrections, and exports will be stored here."
        )
        body.setWordWrap(True)
        body.setStyleSheet(_BODY_STYLE)
        s.addWidget(body)
        path_edit = QLineEdit(str(self.dm.data_root))
        path_edit.setReadOnly(True)
        path_edit.setStyleSheet(
            f"background:{PANEL}; color:{OFFWHITE}; "
            f"border:1px solid {BORDER}; padding:6px; border-radius:{RADIUS}px;"
        )
        s.addWidget(path_edit)
        hint = QLabel(
            "Data location is fixed in this version. Each subfolder is "
            "available via the 📁 buttons in every tab's header."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(_HINT_STYLE)
        s.addWidget(hint)
        s.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addWidget(section)


class _ModelDownloadPage(QWizardPage):
    def __init__(self, data_manager):
        super().__init__()
        self.dm = data_manager
        self.setTitle("Detection Model")
        self._existing_models = bool(self.dm.list_models())
        section = QWidget()
        s = QVBoxLayout(section)
        s.setContentsMargins(20, 16, 20, 16)
        t = QLabel("Detection Model")
        t.setStyleSheet(_TITLE_STYLE)
        s.addWidget(t)
        body = QLabel(
            "yolov8n.pt is the smallest YOLOv8 model — ~6 MB, fast on CPU. "
            "You can switch to a larger model later from the Models tab."
        )
        body.setWordWrap(True)
        body.setStyleSheet(_BODY_STYLE)
        s.addWidget(body)
        self.rb_download = QRadioButton("Download yolov8n.pt now (recommended)")
        self.rb_skip = QRadioButton("Skip — I'll do this later")
        for rb in (self.rb_download, self.rb_skip):
            rb.setStyleSheet(f"color: {OFFWHITE}; padding: 4px;")
        if self._existing_models:
            self.rb_skip.setChecked(True)
            self.rb_download.setText("Download anyway")
            note = QLabel(
                f"Existing models detected: {', '.join(self.dm.list_models())}"
            )
            note.setStyleSheet(_HINT_STYLE)
            s.addWidget(note)
        else:
            self.rb_download.setChecked(True)
        s.addWidget(self.rb_download)
        s.addWidget(self.rb_skip)
        s.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addWidget(section)

    def validatePage(self) -> bool:  # noqa: N802
        if not self.rb_download.isChecked():
            return True
        # Trigger the download synchronously through a progress dialog so the
        # user sees what's happening before moving to the next page.
        try:
            from cctv_yolo.model_downloader import ModelDownloadWorker
        except Exception:
            QMessageBox.warning(
                self, "Download unavailable",
                "Couldn't import the model downloader. Skipping; you can "
                "download yolov8n.pt later from the Models tab.",
            )
            return True

        dlg = QProgressDialog(
            "Downloading yolov8n.pt…", "Cancel", 0, 0, self
        )
        dlg.setWindowTitle("Downloading model")
        dlg.setWindowModality(Qt.ApplicationModal)

        worker = ModelDownloadWorker("yolov8n.pt", self.dm.models_dir)
        finished_ok = {"ok": False, "err": None}

        def _done():
            finished_ok["ok"] = True
            dlg.close()

        def _err(msg: str):
            finished_ok["err"] = msg
            dlg.close()

        worker.finished_ok.connect(_done)
        worker.error.connect(_err)
        worker.start()
        dlg.exec()

        if not finished_ok["ok"]:
            QMessageBox.warning(
                self, "Download failed",
                (finished_ok["err"] or "Cancelled.")
                + "\n\nYou can download the model later from the Models tab.",
            )
            # Continue anyway — user can download later.
        return True


class _TourPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Tour")
        section = _make_section(
            "Quick Tour",
            [
                "1. Preprocessing — load videos, pick a model, click Process.",
                "2. Batch — drop a folder of videos to process at scale.",
                "3. Correction — review and fix detections, draw ROIs.",
                "4. Performance / Analytics / Insights — measure how the "
                "active model performs and find anomalies.",
                "5. Training — build datasets from your corrections and "
                "retrain. Use 'Build from unused corrections' for incremental "
                "training.",
                "6. Models / Live — manage models and run live cameras.",
                "",
                "Every tab has 📁 buttons in its top-right corner that open "
                "the underlying data folders in Finder/Explorer.",
            ],
        )
        lay = QVBoxLayout(self)
        lay.addWidget(section)


class _FinishPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("You're set!")
        section = _make_section(
            "You're set",
            [
                "Click Finish to start using CCTV-YOLO.",
                "",
                "You can revisit this guide anytime via Help → About → Tour.",
            ],
        )
        lay = QVBoxLayout(self)
        lay.addWidget(section)


class FirstRunWizard(QWizard):
    """Wizard shown only on the very first launch."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.setWindowTitle(f"Welcome to {__app_name__}")
        self.setMinimumSize(620, 480)
        self.setOption(QWizard.NoCancelButton, False)
        self.setOption(QWizard.HaveHelpButton, False)
        self.setStyleSheet(
            f"QWizard, QWizardPage {{ background: {INDIGO}; color: {OFFWHITE}; }}"
            f"QLabel {{ color: {OFFWHITE}; }}"
            f"QPushButton {{ {_BTN_STYLE} }}"
        )

        self.addPage(_WelcomePage())
        self.addPage(_DataFolderPage(self.dm))
        self.addPage(_ModelDownloadPage(self.dm))
        self.addPage(_TourPage())
        self.addPage(_FinishPage())

        self.finished.connect(self._on_finished)

    def _on_finished(self, result: int) -> None:
        if result == QWizard.Accepted:
            mark_first_run_complete()
