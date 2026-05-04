"""
Batch tab — folder-wide ingest, persistent queue, watch folder, priority/scheduling.
"""
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QProgressBar,
)

from cctv_yolo.batch_queue import BatchQueueManager

# Style constants (match the rest of the app)
BG = "#1a1a2e"
PANEL = "#16213e"
BORDER = "#2d3a5a"
ACCENT = "#4ecca3"
TEXT = "#eeeeee"

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


class BatchTab(QWidget):
    """Persistent batch-processing queue with watch folder support."""

    review_requested = Signal(str)

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.queue = BatchQueueManager(data_manager, self)
        self.queue.queue_changed.connect(self._refresh_table)
        self.queue.item_progress.connect(self._on_item_progress)

        self._setup_ui()
        self._refresh_table()
        # Auto-resume any pending work from a previous session
        self.queue.start_next_if_idle()

    # ----- UI -----
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header
        header = QLabel("Batch Processing")
        header.setStyleSheet(f"color: {ACCENT}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        # Add controls
        ctrl_box = QGroupBox("Add to queue")
        ctrl_layout = QHBoxLayout(ctrl_box)

        self.model_combo = QComboBox()
        self._reload_models()
        ctrl_layout.addWidget(QLabel("Model:"))
        ctrl_layout.addWidget(self.model_combo)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self.dm.get_last_confidence())
        self.conf_spin.setDecimals(2)
        ctrl_layout.addWidget(QLabel("Conf:"))
        ctrl_layout.addWidget(self.conf_spin)

        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(0, 100)
        self.priority_spin.setValue(0)
        ctrl_layout.addWidget(QLabel("Priority:"))
        ctrl_layout.addWidget(self.priority_spin)

        ctrl_layout.addStretch()

        btn_files = QPushButton("Add Files…")
        btn_files.setStyleSheet(ACTION_BTN)
        btn_files.clicked.connect(self._add_files)
        ctrl_layout.addWidget(btn_files)

        btn_folder = QPushButton("Add Folder (recursive)…")
        btn_folder.setStyleSheet(ACTION_BTN)
        btn_folder.clicked.connect(self._add_folder)
        ctrl_layout.addWidget(btn_folder)

        layout.addWidget(ctrl_box)

        # Watch folder
        watch_box = QGroupBox("Watch folder (auto-queue new videos)")
        watch_layout = QHBoxLayout(watch_box)
        self.watch_path = QLineEdit()
        self.watch_path.setPlaceholderText("Select a folder to watch…")
        watch_layout.addWidget(self.watch_path, stretch=1)

        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_watch_folder)
        watch_layout.addWidget(btn_browse)

        self.poll_spin = QSpinBox()
        self.poll_spin.setRange(1, 600)
        self.poll_spin.setValue(5)
        self.poll_spin.setSuffix(" s")
        watch_layout.addWidget(QLabel("Poll:"))
        watch_layout.addWidget(self.poll_spin)

        self.btn_watch = QPushButton("Start Watching")
        self.btn_watch.setStyleSheet(ACTION_BTN)
        self.btn_watch.setCheckable(True)
        self.btn_watch.toggled.connect(self._toggle_watch)
        watch_layout.addWidget(self.btn_watch)

        layout.addWidget(watch_box)

        # Queue control row
        ctrl_row = QHBoxLayout()
        self.btn_pause = QPushButton("Pause Queue")
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self._toggle_pause)
        ctrl_row.addWidget(self.btn_pause)

        btn_clear_done = QPushButton("Clear Finished")
        btn_clear_done.clicked.connect(self.queue.clear_finished)
        ctrl_row.addWidget(btn_clear_done)

        ctrl_row.addStretch()
        self.queue_status = QLabel("0 queued / 0 done")
        self.queue_status.setStyleSheet("color: #aaa;")
        ctrl_row.addWidget(self.queue_status)
        layout.addLayout(ctrl_row)

        # Queue table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Video", "Model", "Conf", "Priority", "Status", "Progress"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 6):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setStyleSheet(
            f"QTableWidget {{ background-color: {PANEL}; color: {TEXT}; "
            f"gridline-color: {BORDER}; border: 1px solid {BORDER}; }}"
            f"QHeaderView::section {{ background-color: {PANEL}; color: {ACCENT}; "
            f"border: 0; padding: 6px; font-weight: bold; }}"
        )
        layout.addWidget(self.table, stretch=1)

        # Row actions
        row_btns = QHBoxLayout()
        btn_remove = QPushButton("Remove Selected")
        btn_remove.setStyleSheet(DANGER_BTN)
        btn_remove.clicked.connect(self._remove_selected)
        row_btns.addWidget(btn_remove)

        btn_review = QPushButton("Open Review for Selected")
        btn_review.clicked.connect(self._review_selected)
        row_btns.addWidget(btn_review)

        row_btns.addStretch()
        layout.addLayout(row_btns)

    def _reload_models(self):
        self.model_combo.clear()
        models = self.dm.list_models()
        if not models:
            models = ["yolov8m.pt"]
        self.model_combo.addItems(models)
        last = self.dm.get_last_model()
        if last and last in models:
            self.model_combo.setCurrentText(last)

    # ----- Add ops -----
    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select videos", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)",
        )
        if not files:
            return
        added, skipped = 0, 0
        for f in files:
            sid = self.queue.add(
                f,
                model=self.model_combo.currentText(),
                conf=self.conf_spin.value(),
                priority=self.priority_spin.value(),
            )
            if sid:
                added += 1
            else:
                skipped += 1
        self.queue.start_next_if_idle()
        self._toast(f"Added {added}, skipped {skipped}")

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if not folder:
            return
        exts = {".mp4", ".mov", ".avi", ".mkv"}
        added, skipped = 0, 0
        for p in Path(folder).rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                sid = self.queue.add(
                    str(p),
                    model=self.model_combo.currentText(),
                    conf=self.conf_spin.value(),
                    priority=self.priority_spin.value(),
                )
                if sid:
                    added += 1
                else:
                    skipped += 1
        self.queue.start_next_if_idle()
        self._toast(f"Folder ingest: added {added}, skipped {skipped}")

    def _browse_watch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select watch folder")
        if folder:
            self.watch_path.setText(folder)

    def _toggle_watch(self, checked: bool):
        if checked:
            folder = self.watch_path.text().strip()
            if not folder or not Path(folder).exists():
                QMessageBox.warning(self, "Watch folder", "Please pick a valid folder first.")
                self.btn_watch.setChecked(False)
                return
            self.queue.start_watch_folder(
                Path(folder),
                model=self.model_combo.currentText(),
                conf=self.conf_spin.value(),
                poll_seconds=self.poll_spin.value(),
            )
            self.btn_watch.setText("Stop Watching")
        else:
            self.queue.stop_watch_folder()
            self.btn_watch.setText("Start Watching")

    def _toggle_pause(self, checked: bool):
        if checked:
            self.queue.pause_all()
            self.btn_pause.setText("Resume Queue")
        else:
            self.queue.resume_all()
            self.btn_pause.setText("Pause Queue")

    # ----- Row ops -----
    def _selected_session(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return item.data(Qt.UserRole) if item else None

    def _remove_selected(self):
        sid = self._selected_session()
        if not sid:
            return
        self.queue.remove(sid)

    def _review_selected(self):
        sid = self._selected_session()
        if sid:
            self.review_requested.emit(sid)

    # ----- Refresh -----
    def _refresh_table(self):
        items = self.queue.get_items()
        self.table.setRowCount(len(items))
        done = 0
        queued = 0
        for r, it in enumerate(items):
            name = Path(it["video_path"]).name
            cell = QTableWidgetItem(name)
            cell.setData(Qt.UserRole, it["session_id"])
            cell.setToolTip(it["video_path"])
            self.table.setItem(r, 0, cell)
            self.table.setItem(r, 1, QTableWidgetItem(it["model"]))
            self.table.setItem(r, 2, QTableWidgetItem(f"{it['conf']:.2f}"))
            self.table.setItem(r, 3, QTableWidgetItem(str(it.get("priority", 0))))

            status = it["status"]
            status_item = QTableWidgetItem(status)
            color = {
                "queued": "#888",
                "processing": ACCENT,
                "done": "#5dade2",
                "error": "#e74c3c",
                "paused": "#f39c12",
            }.get(status, TEXT)
            status_item.setForeground(Qt.GlobalColor.white if status != "processing" else Qt.black)
            status_item.setData(Qt.ToolTipRole, it.get("error") or "")
            self.table.setItem(r, 4, status_item)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(it.get("progress", 0)))
            bar.setTextVisible(True)
            self.table.setCellWidget(r, 5, bar)

            if status == "done":
                done += 1
            elif status in ("queued", "paused"):
                queued += 1

        self.queue_status.setText(f"{queued} queued / {done} done / {len(items)} total")

    def _on_item_progress(self, session_id: str, pct: int):
        for r in range(self.table.rowCount()):
            cell = self.table.item(r, 0)
            if cell and cell.data(Qt.UserRole) == session_id:
                bar = self.table.cellWidget(r, 5)
                if bar:
                    bar.setValue(pct)
                break

    def refresh(self):
        self._reload_models()
        self._refresh_table()

    def shutdown(self):
        self.queue.shutdown()

    # ----- Toast -----
    def _toast(self, msg: str):
        self.queue_status.setText(msg)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
