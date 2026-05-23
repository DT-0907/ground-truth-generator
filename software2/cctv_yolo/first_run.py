"""First-run wizard (PRD C7).

Triggered on first launch when the `.first_run_complete` marker file is
absent from the data root. Walks the user through: welcome → data folder
→ model download → tour → finish.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (
    QWizard, QWizardPage, QLabel, QVBoxLayout, QHBoxLayout, QRadioButton,
    QLineEdit, QProgressDialog, QMessageBox, QPushButton, QWidget,
    QButtonGroup,
)

from cctv_yolo.__version__ import __version__, __app_name__
from cctv_yolo.paths import get_first_run_marker
from cctv_yolo.theme import (
    PURPLE, PINK, OFFWHITE, PANEL, INDIGO, BORDER, TEXT_MUTED, RADIUS,
)


# ---------- Styles ----------------------------------------------------------

_TITLE_STYLE = f"color: {PINK}; font-size: 20px; font-weight: bold;"
_BODY_STYLE  = f"color: {OFFWHITE}; font-size: 13px;"
_HINT_STYLE  = f"color: {TEXT_MUTED}; font-size: 11px;"
_RADIO_STYLE = f"color: {OFFWHITE}; padding: 4px; font-size: 13px;"
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


def _apply_dark_palette(widget) -> None:
    """Force the dark palette on a wizard so QWizard's built-in style
    doesn't paint the body with the OS default (white on Windows), which
    would render OFFWHITE text invisible.
    """
    pal = widget.palette()
    pal.setColor(QPalette.Window,           QColor(INDIGO))
    pal.setColor(QPalette.Base,             QColor(INDIGO))
    pal.setColor(QPalette.AlternateBase,    QColor(PANEL))
    pal.setColor(QPalette.WindowText,       QColor(OFFWHITE))
    pal.setColor(QPalette.Text,             QColor(OFFWHITE))
    pal.setColor(QPalette.ButtonText,       QColor(OFFWHITE))
    pal.setColor(QPalette.PlaceholderText,  QColor(TEXT_MUTED))
    pal.setColor(QPalette.Highlight,        QColor(PURPLE))
    pal.setColor(QPalette.HighlightedText,  QColor(OFFWHITE))
    widget.setPalette(pal)


# ---------- Marker file -----------------------------------------------------

def is_first_run() -> bool:
    return not get_first_run_marker().exists()


def mark_first_run_complete() -> None:
    marker = get_first_run_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


# ---------- Pages -----------------------------------------------------------

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
        self.setTitle("")
        section = _make_section(
            f"Welcome to {__app_name__} v{__version__}",
            [
                "Detect, track, and correct vehicles in CCTV footage.",
                "",
                "This short setup will create your data folder, optionally "
                "download one or more YOLOv8 detection models, and point "
                "you at the Preprocessing tab.",
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
        self.setTitle("")
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
            "You can move this folder later — the app finds it again via "
            "search + a remembered-location marker. To force a different "
            "location, set the CCTV_YOLO_DATA_DIR environment variable."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(_HINT_STYLE)
        s.addWidget(hint)
        s.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addWidget(section)


# ---------- Model picker ----------------------------------------------------

# Variants surfaced in the wizard, smallest to largest. (yolov8s and yolov8x
# are skipped here to keep the picker simple — they're still available via
# the Models tab download menu.)
_WIZARD_MODELS = [
    ("yolov8n.pt", 6,   "fastest, lower accuracy"),
    ("yolov8m.pt", 52,  "balanced (recommended)"),
    ("yolov8l.pt", 87,  "most accurate, slower"),
]
_ALL_BYTES_MB = sum(mb for _, mb, _ in _WIZARD_MODELS)


class _ModelDownloadPage(QWizardPage):
    """Pick which YOLOv8 weights to download (or skip).

    Five mutually-exclusive choices. Defaults to "yolov8m.pt" (the recommended
    balance for our pipeline).
    """

    def __init__(self, data_manager):
        super().__init__()
        self.dm = data_manager
        self.setTitle("")
        self._existing_models = bool(self.dm.list_models())

        outer = QVBoxLayout(self)
        section = QWidget()
        s = QVBoxLayout(section)
        s.setContentsMargins(20, 16, 20, 16)

        t = QLabel("Detection Model")
        t.setStyleSheet(_TITLE_STYLE)
        s.addWidget(t)

        body = QLabel(
            "Pick which YOLOv8 weight file(s) to download. You can change "
            "or add more later from the Models tab."
        )
        body.setWordWrap(True)
        body.setStyleSheet(_BODY_STYLE)
        s.addWidget(body)

        self.group = QButtonGroup(self)
        self.rb_by_name: dict[str, QRadioButton] = {}

        for name, mb, blurb in _WIZARD_MODELS:
            rb = QRadioButton(f"  {name}  —  {blurb}  ({mb} MB)")
            rb.setStyleSheet(_RADIO_STYLE)
            self.group.addButton(rb)
            self.rb_by_name[name] = rb
            s.addWidget(rb)

        self.rb_all = QRadioButton(
            f"  Download all three (n + m + l, ~{_ALL_BYTES_MB} MB)"
        )
        self.rb_all.setStyleSheet(_RADIO_STYLE)
        self.group.addButton(self.rb_all)
        s.addWidget(self.rb_all)

        self.rb_skip = QRadioButton(
            "  Skip — I'll download or import my own model later"
        )
        self.rb_skip.setStyleSheet(_RADIO_STYLE)
        self.group.addButton(self.rb_skip)
        s.addWidget(self.rb_skip)

        # Default selection: recommended model, OR skip if models already exist.
        if self._existing_models:
            self.rb_skip.setChecked(True)
            note = QLabel(
                f"Existing models detected: {', '.join(self.dm.list_models())}"
            )
            note.setStyleSheet(_HINT_STYLE)
            s.addWidget(note)
        else:
            self.rb_by_name["yolov8m.pt"].setChecked(True)

        timing_hint = QLabel(
            "Download time depends on your connection. Typical broadband: "
            "yolov8n ~5s, yolov8m ~30s, yolov8l ~1min, all three ~2min. "
            "Slow connections (<1 MB/s) may take 5-20 minutes for all three."
        )
        timing_hint.setWordWrap(True)
        timing_hint.setStyleSheet(_HINT_STYLE)
        s.addWidget(timing_hint)

        s.addStretch(1)
        outer.addWidget(section)

    def _selected(self) -> list[str]:
        """List of model filenames the user picked (empty = skip)."""
        if self.rb_skip.isChecked():
            return []
        if self.rb_all.isChecked():
            return [name for name, _, _ in _WIZARD_MODELS]
        for name, rb in self.rb_by_name.items():
            if rb.isChecked():
                return [name]
        return []

    def validatePage(self) -> bool:  # noqa: N802
        wanted = self._selected()
        if not wanted:
            return True

        try:
            from cctv_yolo.model_downloader import ModelDownloadWorker
        except Exception:
            QMessageBox.warning(
                self, "Download unavailable",
                "Couldn't import the model downloader. You can download "
                "models later from the Models tab.",
            )
            return True

        # Download each picked model sequentially through a progress dialog.
        for i, model_name in enumerate(wanted, start=1):
            ok = self._download_one(
                ModelDownloadWorker, model_name,
                index=i, total=len(wanted),
            )
            if not ok:
                # Ask whether to keep going with the remaining downloads.
                if i < len(wanted):
                    resp = QMessageBox.question(
                        self, "Continue?",
                        f"{model_name} download failed. Try the next one?",
                        QMessageBox.Yes | QMessageBox.No,
                    )
                    if resp != QMessageBox.Yes:
                        break
        return True

    def _download_one(
        self, ModelDownloadWorker, model_name: str,
        index: int, total: int,
    ) -> bool:
        """Download a single model. Returns True on success."""
        label_suffix = f"  ({index}/{total})" if total > 1 else ""
        dlg = QProgressDialog(
            f"Downloading {model_name}{label_suffix}…",
            "Cancel", 0, 0, self,
        )
        dlg.setWindowTitle("Downloading model")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)

        worker = ModelDownloadWorker(model_name, self.dm.models_dir)
        state = {"ok": False, "err": None}

        def _done(_path: str):
            state["ok"] = True
            dlg.close()

        def _failed(msg: str):
            state["err"] = msg
            dlg.close()

        # The worker emits `done(str)` and `failed(str)` — these are the
        # correct names. (An earlier wizard listened for `finished_ok` /
        # `error`, which is why downloads silently appeared to hang.)
        worker.done.connect(_done)
        worker.failed.connect(_failed)
        worker.start()

        # Modal Qt event loop until either the worker fires or the user
        # cancels.
        dlg.exec()  # noqa: PLW0108
        if not state["ok"] and not state["err"]:
            # User clicked Cancel — try to ask the worker to stop.
            worker.requestInterruption()
            worker.wait(2000)
            QMessageBox.information(
                self, "Cancelled", f"{model_name} download was cancelled."
            )
            return False

        if not state["ok"]:
            # Offer Retry — the new downloader resumes from the partial
            # .pt.tmp file so retries are cheap.
            resp = QMessageBox.question(
                self, f"{model_name} download failed",
                (state["err"] or "Unknown error.")
                + "\n\nRetry now? (Partial file will be resumed.)",
                QMessageBox.Yes | QMessageBox.No,
            )
            if resp == QMessageBox.Yes:
                return self._download_one(
                    ModelDownloadWorker, model_name, index, total,
                )
            return False
        return True


class _TourPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("")
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
                "the underlying data folders in Finder / Explorer.",
            ],
        )
        lay = QVBoxLayout(self)
        lay.addWidget(section)


class _FinishPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("")
        section = _make_section(
            "You're set",
            [
                "Click Finish to start using CCTV-YOLO.",
                "",
                "You can revisit this guide anytime via Help → About.",
            ],
        )
        lay = QVBoxLayout(self)
        lay.addWidget(section)


# ---------- Wizard ----------------------------------------------------------

class FirstRunWizard(QWizard):
    """Wizard shown only on the very first launch."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.setWindowTitle(f"Welcome to {__app_name__}")
        self.setMinimumSize(640, 520)
        self.setOption(QWizard.NoCancelButton, False)
        self.setOption(QWizard.HaveHelpButton, False)
        self.setOption(QWizard.NoBackButtonOnStartPage, True)

        # Use ClassicStyle so the wizard body doesn't get OS-painted
        # white-background panels that wash out our OFFWHITE text on Windows.
        self.setWizardStyle(QWizard.ClassicStyle)

        # Force the dark palette across the wizard + every child page.
        _apply_dark_palette(self)

        # Plus a stylesheet for finer control of text colors, including the
        # subTitle/title areas that QWizard paints itself.
        self.setStyleSheet(f"""
            QWizard, QWizardPage, QDialog {{
                background-color: {INDIGO};
                color: {OFFWHITE};
            }}
            QWidget {{
                background-color: {INDIGO};
                color: {OFFWHITE};
            }}
            QLabel, QRadioButton, QCheckBox {{
                color: {OFFWHITE};
                background-color: transparent;
            }}
            QLineEdit {{
                background-color: {PANEL};
                color: {OFFWHITE};
                border: 1px solid {BORDER};
                border-radius: {RADIUS}px;
                padding: 6px;
            }}
            QPushButton {{
                background-color: {PURPLE};
                color: {OFFWHITE};
                border: none;
                border-radius: {RADIUS}px;
                padding: 6px 18px;
                font-weight: bold;
                min-width: 60px;
            }}
            QPushButton:hover {{
                background-color: {PINK};
                color: {INDIGO};
            }}
            QPushButton:disabled {{
                background-color: {BORDER};
                color: {TEXT_MUTED};
            }}
        """)

        # Pages: welcome → data folder → model picker → tour → finish
        self.addPage(_WelcomePage())
        self.addPage(_DataFolderPage(self.dm))
        self.addPage(_ModelDownloadPage(self.dm))
        self.addPage(_TourPage())
        self.addPage(_FinishPage())

        self.finished.connect(self._on_finished)

    def _on_finished(self, result: int) -> None:
        if result == QWizard.Accepted:
            mark_first_run_complete()
