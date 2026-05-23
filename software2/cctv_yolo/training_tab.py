"""
Training tab — active-learning queue + dataset build + retrain.
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
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QPlainTextEdit,
    QProgressBar,
    QMessageBox,
)

from cctv_yolo.training import (
    rank_sessions_by_uncertainty,
    DatasetBuildWorker,
    TrainingWorker,
)


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

DANGER_BTN = f"""
QPushButton {{
    background-color: #c0392b;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton:hover {{ background-color: #a82a1f; }}
"""


class TrainingTab(QWidget):
    """Active-learning queue + retrain workflow."""

    review_requested = Signal(str)

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self._build_worker = None
        self._train_worker = None
        self._dataset_yaml = None
        self._last_build_id: str | None = None
        self._setup_ui()
        # PRD F4-1 / J5 — auto-refresh AL queue + unused counter on every save
        self.dm.corrections_changed.connect(lambda _sid: self.refresh())
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QLabel("Active Learning + Training")
        header.setStyleSheet(f"color: {ACCENT}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        # Active-learning queue
        al_box = QGroupBox("Sessions ranked by review priority (most uncertain first)")
        al_layout = QVBoxLayout(al_box)
        self.al_table = QTableWidget(0, 6)
        self.al_table.setHorizontalHeaderLabels(
            ["Video", "Tracks", "Mean conf", "Low-conf", "Short", "Score"]
        )
        self.al_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 6):
            self.al_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.al_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.al_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.al_table.setStyleSheet(
            f"QTableWidget {{ background-color: {PANEL}; color: {TEXT}; "
            f"gridline-color: {BORDER}; border: 1px solid {BORDER}; }}"
            f"QHeaderView::section {{ background-color: {PANEL}; color: {ACCENT}; "
            f"border: 0; padding: 6px; font-weight: bold; }}"
        )
        al_layout.addWidget(self.al_table)
        row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh ranking")
        btn_refresh.clicked.connect(self.refresh)
        row.addWidget(btn_refresh)

        btn_review = QPushButton("Review most-uncertain")
        btn_review.setStyleSheet(ACTION_BTN)
        btn_review.clicked.connect(self._review_top)
        row.addWidget(btn_review)
        row.addStretch()
        al_layout.addLayout(row)
        layout.addWidget(al_box)

        # Dataset / training
        ds_box = QGroupBox("Build dataset + train")
        ds_layout = QVBoxLayout(ds_box)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Sample every"))
        self.sample_n = QSpinBox()
        self.sample_n.setRange(1, 100)
        self.sample_n.setValue(5)
        self.sample_n.setSuffix(" frames")
        params_row.addWidget(self.sample_n)

        params_row.addWidget(QLabel("Base model:"))
        self.base_model = QComboBox()
        self.base_model.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
        self.base_model.setCurrentText("yolov8n.pt")
        params_row.addWidget(self.base_model)

        params_row.addWidget(QLabel("Epochs:"))
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(30)
        params_row.addWidget(self.epochs)

        params_row.addWidget(QLabel("Image size:"))
        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1536)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        params_row.addWidget(self.imgsz)

        params_row.addWidget(QLabel("Batch:"))
        self.batch = QSpinBox()
        self.batch.setRange(1, 128)
        self.batch.setValue(16)
        params_row.addWidget(self.batch)
        params_row.addStretch()
        ds_layout.addLayout(params_row)

        action_row = QHBoxLayout()
        self.btn_build = QPushButton("1. Build Dataset")
        self.btn_build.setStyleSheet(ACTION_BTN)
        self.btn_build.clicked.connect(self._build_dataset)
        action_row.addWidget(self.btn_build)

        # PRD J2: iterative loop — build only from corrections that have
        # changed (or sessions that have been re-processed) since the last
        # successful training run.
        self.btn_build_unused = QPushButton("Build from unused corrections")
        self.btn_build_unused.setStyleSheet(ACTION_BTN)
        self.btn_build_unused.clicked.connect(self._build_dataset_unused)
        action_row.addWidget(self.btn_build_unused)

        self.btn_train = QPushButton("2. Train (uses last-built dataset)")
        self.btn_train.setStyleSheet(ACTION_BTN)
        self.btn_train.clicked.connect(self._start_training)
        self.btn_train.setEnabled(False)
        action_row.addWidget(self.btn_train)

        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.setStyleSheet(DANGER_BTN)
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_stop.setEnabled(False)
        action_row.addWidget(self.btn_stop)

        action_row.addStretch()

        # "X unused corrections since last build" hint label
        self.lbl_unused = QLabel("")
        self.lbl_unused.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
        action_row.addWidget(self.lbl_unused)
        ds_layout.addLayout(action_row)

        # OpenLocationBar — quick access to all Training-related folders (PRD J6)
        from cctv_yolo.widgets.open_location_bar import OpenLocationBar
        ol_row = OpenLocationBar(self)
        ol_row.add_folder("Datasets", self.dm.data_root / "training")
        ol_row.add_folder("Models", self.dm.models_dir)
        ol_row.add_file("Last Trained", self._latest_trained_model)
        ds_layout.addWidget(ol_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        ds_layout.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        f = QFont("Menlo", 11)
        self.log.setFont(f)
        self.log.setStyleSheet(
            f"QPlainTextEdit {{ background-color: #0e1424; color: {TEXT}; "
            f"border: 1px solid {BORDER}; }}"
        )
        ds_layout.addWidget(self.log, stretch=1)

        layout.addWidget(ds_box, stretch=1)

    def _latest_trained_model(self):
        """Return path to the most recent models/trained_*.pt, or None."""
        candidates = sorted(self.dm.models_dir.glob("trained_*.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else self.dm.models_dir

    # ----- Active learning -----
    def refresh(self):
        rows = rank_sessions_by_uncertainty(self.dm)
        self.al_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            cell = QTableWidgetItem(r["video_name"])
            cell.setData(Qt.UserRole, r["session_id"])
            self.al_table.setItem(i, 0, cell)
            self.al_table.setItem(i, 1, QTableWidgetItem(str(r["tracks"])))
            self.al_table.setItem(i, 2, QTableWidgetItem(f"{r['mean_conf']:.3f}"))
            self.al_table.setItem(i, 3, QTableWidgetItem(str(r["low_conf"])))
            self.al_table.setItem(i, 4, QTableWidgetItem(str(r["short"])))
            self.al_table.setItem(i, 5, QTableWidgetItem(f"{r['score']:.3f}"))
        self._refresh_unused_count()

    def _refresh_unused_count(self):
        """Update the 'X unused corrections since last build' hint label."""
        unused = self.dm.list_unused_corrections()
        last = self.dm.get_last_training_build()
        if last:
            built_at = last.get("built_at", "")[:16].replace("T", " ")
            self.lbl_unused.setText(
                f"{len(unused)} unused correction(s) since last build ({built_at})"
            )
        else:
            self.lbl_unused.setText(
                f"{len(unused)} corrected session(s) — no prior build"
            )

    def _review_top(self):
        if self.al_table.rowCount() == 0:
            return
        cell = self.al_table.item(0, 0)
        if cell:
            sid = cell.data(Qt.UserRole)
            if sid:
                self.review_requested.emit(sid)

    # ----- Dataset build -----
    def _build_dataset(self, restrict_to: list[str] | None = None):
        """Build a YOLO dataset. If restrict_to is set, only those session ids
        are included (used by 'Build from unused corrections')."""
        out = self.dm.data_root / "training" / dt.datetime.now().strftime("ds_%Y%m%d_%H%M%S")
        self._append_log(f"Building dataset at {out}")
        if restrict_to is not None:
            self._append_log(f"  Restricted to {len(restrict_to)} session(s): "
                             f"{', '.join(restrict_to[:5])}"
                             f"{'…' if len(restrict_to) > 5 else ''}")
        self.progress.setValue(0)
        self.btn_build.setEnabled(False)
        self.btn_build_unused.setEnabled(False)

        worker = DatasetBuildWorker(
            self.dm, out,
            sample_every_n=self.sample_n.value(),
        )
        # PRD J2 — pass restrict list as an attribute the worker can consume.
        worker._restrict_session_ids = restrict_to
        worker.progress.connect(self.progress.setValue)
        worker.finished_ok.connect(self._on_dataset_built)
        worker.failed.connect(self._on_failed)
        self._build_worker = worker
        worker.start()

    def _build_dataset_unused(self):
        """PRD J2 — build only from corrections that have changed since the
        last training build."""
        unused = self.dm.list_unused_corrections()
        if not unused:
            QMessageBox.information(
                self, "Nothing to build",
                "No corrections have changed since the last training build.\n"
                "Make new corrections in the Correction tab first."
            )
            return
        session_ids = [s["id"] for s in unused]
        reasons = {s["_reason"]: 0 for s in unused}
        for s in unused:
            reasons[s["_reason"]] = reasons.get(s["_reason"], 0) + 1
        self._append_log(
            f"Found {len(unused)} unused correction(s): "
            + ", ".join(f"{n} {r}" for r, n in reasons.items())
        )
        self._build_dataset(restrict_to=session_ids)

    def _on_dataset_built(self, stats: dict):
        self.btn_build.setEnabled(True)
        self.btn_build_unused.setEnabled(True)
        self.progress.setValue(100)
        self._dataset_yaml = stats["yaml_path"]
        # PRD J2 — remember the build id so we can stamp the snapshot when
        # training actually succeeds (don't snapshot on a failed train).
        self._last_build_id = Path(stats["yaml_path"]).parent.name
        self._append_log(
            f"Dataset built: {stats['images']} images, {stats['labels']} labels "
            f"from {stats['corrected_sessions']} corrected sessions"
        )
        self._append_log(f"Classes: {stats['classes']}")
        self._append_log(f"data.yaml: {self._dataset_yaml}")
        if stats["images"] > 0:
            self.btn_train.setEnabled(True)
        else:
            QMessageBox.warning(self, "Empty dataset",
                                "No corrected sessions found. Correct some sessions first.")

    # ----- Training -----
    def _start_training(self):
        if not self._dataset_yaml:
            QMessageBox.warning(self, "No dataset", "Build the dataset first.")
            return
        self._append_log("=" * 50)
        self._append_log("Starting training")
        worker = TrainingWorker(
            data_yaml=self._dataset_yaml,
            base_model=self.base_model.currentText(),
            epochs=self.epochs.value(),
            imgsz=self.imgsz.value(),
            batch=self.batch.value(),
            models_dir=str(self.dm.models_dir),
        )
        worker.log_line.connect(self._append_log)
        worker.progress.connect(self.progress.setValue)
        worker.finished_ok.connect(self._on_train_done)
        worker.failed.connect(self._on_failed)
        self._train_worker = worker
        self.btn_train.setEnabled(False)
        self.btn_build.setEnabled(False)
        self.btn_stop.setEnabled(True)
        worker.start()

    def _stop_training(self):
        if self._train_worker:
            self._train_worker.stop()
            self._append_log("Stop requested…")

    def _on_train_done(self, model_path: str):
        self.btn_build.setEnabled(True)
        self.btn_build_unused.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(f"Training complete: {model_path}")
        # PRD J2: snapshot the current corrections+tracks mtimes so the next
        # 'Build from unused corrections' diff starts from THIS build.
        if self._last_build_id:
            try:
                self.dm.record_training_build(
                    build_id=self._last_build_id,
                    trained_model=Path(model_path).name,
                )
                self._append_log(f"Training history updated (build_id={self._last_build_id}).")
            except Exception as e:
                self._append_log(f"Warning: couldn't record training history: {e}")
        QMessageBox.information(self, "Training complete",
                                f"New model saved as:\n{model_path}\n\n"
                                "It now appears in the Preprocessing model picker.")
        self.refresh()

    def _on_failed(self, msg: str):
        self.btn_build.setEnabled(True)
        self.btn_build_unused.setEnabled(True)
        self.btn_train.setEnabled(self._dataset_yaml is not None)
        self.btn_stop.setEnabled(False)
        self._append_log(f"FAILED: {msg}")

    def _append_log(self, line: str):
        self.log.appendPlainText(line)
