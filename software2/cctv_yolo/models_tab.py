"""
Models tab — list installed YOLO models, set the active one, run A/B
comparison on a chosen video.
"""
import datetime as dt
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QPlainTextEdit,
    QProgressBar,
    QMessageBox,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from cctv_yolo.model_compare import ModelCompareWorker


from cctv_yolo.theme import (
    INDIGO as BG, PANEL, BORDER, PURPLE as ACCENT, OFFWHITE as TEXT,
)

ACTION_BTN = f"""
QPushButton {{
    background-color: {ACCENT};
    color: #000;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton:hover {{ background-color: #3bbb91; }}
QPushButton:disabled {{ background-color: {BORDER}; color: #666; }}
"""


class ModelsTab(QWidget):
    """Manage YOLO models + run head-to-head comparison."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self._compare_worker = None
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QLabel("Models")
        header.setStyleSheet(f"color: {ACCENT}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        # Model list
        list_box = QGroupBox("Installed models (~/Documents/CCTV-YOLO/models)")
        list_layout = QHBoxLayout(list_box)

        self.model_list = QListWidget()
        self.model_list.setStyleSheet(
            f"QListWidget {{ background-color: {PANEL}; color: {TEXT}; "
            f"border: 1px solid {BORDER}; }}"
            f"QListWidget::item:selected {{ background: {ACCENT}; color: black; }}"
        )
        list_layout.addWidget(self.model_list, stretch=1)

        action_col = QVBoxLayout()
        btn_active = QPushButton("Set as Active")
        btn_active.setStyleSheet(ACTION_BTN)
        btn_active.clicked.connect(self._set_active)
        action_col.addWidget(btn_active)

        btn_open = QPushButton("Open Models Folder")
        btn_open.clicked.connect(lambda: self.dm.open_folder("models"))
        action_col.addWidget(btn_open)

        btn_import = QPushButton("Import .pt…")
        btn_import.clicked.connect(self._import_model)
        action_col.addWidget(btn_import)

        action_col.addStretch()
        list_layout.addLayout(action_col)

        layout.addWidget(list_box)

        # Comparison
        cmp_box = QGroupBox("A/B compare on a video")
        cmp_layout = QVBoxLayout(cmp_box)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Video:"))
        self.video_combo = QComboBox()
        self.video_combo.setMinimumWidth(280)
        params_row.addWidget(self.video_combo, stretch=1)

        params_row.addWidget(QLabel("Model A:"))
        self.model_a = QComboBox()
        self.model_a.setMinimumWidth(140)
        params_row.addWidget(self.model_a)

        params_row.addWidget(QLabel("Model B:"))
        self.model_b = QComboBox()
        self.model_b.setMinimumWidth(140)
        params_row.addWidget(self.model_b)

        params_row.addWidget(QLabel("Conf:"))
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.25)
        params_row.addWidget(self.conf)

        params_row.addWidget(QLabel("Stride:"))
        self.stride = QSpinBox()
        self.stride.setRange(1, 30)
        self.stride.setValue(1)
        self.stride.setToolTip(
            "Process every Nth frame. Use 2-5 to speed up comparison runs."
        )
        params_row.addWidget(self.stride)
        cmp_layout.addLayout(params_row)

        action_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Comparison")
        self.btn_run.setStyleSheet(ACTION_BTN)
        self.btn_run.clicked.connect(self._run_compare)
        action_row.addWidget(self.btn_run)

        self.progress_a = QProgressBar()
        self.progress_a.setFormat("A %p%")
        action_row.addWidget(self.progress_a, stretch=1)
        self.progress_b = QProgressBar()
        self.progress_b.setFormat("B %p%")
        action_row.addWidget(self.progress_b, stretch=1)
        cmp_layout.addLayout(action_row)

        # Results table
        self.results = QTableWidget(0, 4)
        self.results.setHorizontalHeaderLabels(["Metric", "A", "B", "Δ (B - A)"])
        self.results.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 4):
            self.results.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.results.setStyleSheet(
            f"QTableWidget {{ background-color: {PANEL}; color: {TEXT}; "
            f"gridline-color: {BORDER}; border: 1px solid {BORDER}; }}"
            f"QHeaderView::section {{ background-color: {PANEL}; color: {ACCENT}; "
            f"border: 0; padding: 4px; }}"
        )
        self.results.setMaximumHeight(260)
        cmp_layout.addWidget(self.results)

        # Log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(800)
        self.log.setFont(QFont("Menlo", 11))
        self.log.setStyleSheet(
            f"QPlainTextEdit {{ background-color: #0e1424; color: {TEXT}; "
            f"border: 1px solid {BORDER}; }}"
        )
        cmp_layout.addWidget(self.log)

        layout.addWidget(cmp_box, stretch=1)

    def refresh(self):
        # Models
        self.model_list.clear()
        self.model_a.clear()
        self.model_b.clear()
        models = self.dm.list_models()
        active = self.dm.get_last_model()
        for m in models:
            label = f"{m}   ★" if m == active else m
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, m)
            self.model_list.addItem(it)
            self.model_a.addItem(m)
            self.model_b.addItem(m)

        if not models:
            for combo in (self.model_a, self.model_b):
                combo.addItem("yolov8m.pt")

        # Pick a sensible default for B if possible
        if len(models) > 1:
            self.model_b.setCurrentIndex(1)

        # Videos for comparison source
        self.video_combo.clear()
        for v in self.dm.get_videos():
            self.video_combo.addItem(v["display_name"], v["session_id"])

    def _set_active(self):
        it = self.model_list.currentItem()
        if not it:
            return
        m = it.data(Qt.UserRole)
        self.dm.set_last_model(m)
        self.refresh()

    def _import_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import .pt model", "", "YOLO models (*.pt)"
        )
        if not path:
            return
        src = Path(path)
        dest = self.dm.models_dir / src.name
        if dest.exists():
            reply = QMessageBox.question(
                self, "Overwrite?", f"{src.name} already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        import shutil
        shutil.copy2(src, dest)
        self.refresh()

    def _run_compare(self):
        sid = self.video_combo.currentData()
        if not sid:
            QMessageBox.warning(self, "No video", "Pick a video first.")
            return
        video_path = self.dm.get_video_path(sid)
        if not video_path or not video_path.exists():
            QMessageBox.warning(self, "No video", "Video file missing.")
            return
        a = self.model_a.currentText()
        b = self.model_b.currentText()
        if a == b:
            reply = QMessageBox.question(
                self, "Same model",
                "Model A and B are the same. Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self.log.clear()
        self.results.setRowCount(0)
        self.progress_a.setValue(0)
        self.progress_b.setValue(0)
        self.btn_run.setEnabled(False)

        worker = ModelCompareWorker(
            video_path=video_path, model_a=a, model_b=b,
            models_dir=self.dm.models_dir, conf=self.conf.value(),
            stride=self.stride.value(),
        )
        worker.log_line.connect(self.log.appendPlainText)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        self._compare_worker = worker
        worker.start()

    def _on_progress(self, which: str, pct: int):
        if which == "A":
            self.progress_a.setValue(pct)
        else:
            self.progress_b.setValue(pct)

    def _on_done(self, payload: dict):
        self.btn_run.setEnabled(True)
        self.progress_a.setValue(100)
        self.progress_b.setValue(100)
        a = payload["a"]
        b = payload["b"]
        d = payload["delta"]

        rows = [
            ("Total tracks", a["total_tracks"], b["total_tracks"], d["total_tracks"]),
            ("Total detections", a["total_detections"], b["total_detections"],
             d["total_detections"]),
            ("Mean confidence", a["mean_conf"], b["mean_conf"], d["mean_conf"]),
            ("Median track length", a["median_track_length"],
             b["median_track_length"], d["median_track_length"]),
        ]
        for c in sorted(set(a["by_class"]) | set(b["by_class"])):
            rows.append((
                f"  class:{c}",
                a["by_class"].get(c, 0),
                b["by_class"].get(c, 0),
                d["by_class"].get(c, 0),
            ))

        self.results.setRowCount(len(rows))
        for i, (m, av, bv, dv) in enumerate(rows):
            self.results.setItem(i, 0, QTableWidgetItem(str(m)))
            self.results.setItem(i, 1, QTableWidgetItem(str(av)))
            self.results.setItem(i, 2, QTableWidgetItem(str(bv)))
            d_item = QTableWidgetItem(f"{dv:+}" if isinstance(dv, int) else f"{dv:+.3f}")
            if isinstance(dv, (int, float)) and dv != 0:
                d_item.setForeground(Qt.GlobalColor.green if dv > 0 else Qt.GlobalColor.red)
            self.results.setItem(i, 3, d_item)

    def _on_failed(self, msg: str):
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Comparison failed", msg)
