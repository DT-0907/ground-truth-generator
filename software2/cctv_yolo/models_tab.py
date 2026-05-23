"""
Models tab — installed YOLO models with metadata, A/B comparison, and a
download bootstrap for fresh installs.

PRD Part K (Models tab):
- K2-1 Metadata columns (filename, size, inference speed, Active badge)
- K2-2 Delete (right-click)
- K2-3 Rename (right-click) — keeps active flag in sync
- K2-4 Source provenance (.meta.json sidecar) for trained models
- K2-5 Comparison history (saved JSON in data/exports/model_compare/)
- K2-6 Download YOLOv8 from Ultralytics
- C11  Theme tokens (no raw hex)
- C12  OpenLocationBar in header
- J4c  "Compare on Dataset val split" mode (P2 bonus)
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QFont, QAction, QBrush, QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
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
    QMenu,
    QInputDialog,
    QProgressDialog,
    QButtonGroup,
    QRadioButton,
    QAbstractItemView,
)

from cctv_yolo.model_compare import (
    ModelCompareWorker,
    DatasetCompareWorker,
)
from cctv_yolo.model_downloader import (
    ModelDownloadWorker,
    variant_labels,
    variant_from_label,
)
from cctv_yolo.widgets.open_location_bar import OpenLocationBar, open_path
from cctv_yolo.theme import (
    INDIGO,
    PURPLE,
    PINK,
    OFFWHITE,
    PANEL,
    BORDER,
    TEXT_MUTED,
    TYPE_TITLE,
    TYPE_HINT,
    RADIUS,
    GAP,
    PAD,
)


# ---------------------------------------------------------------------------
# Stylesheet snippets — all sourced from theme tokens
# ---------------------------------------------------------------------------

ACTION_BTN = f"""
QPushButton {{
    background-color: {PURPLE};
    color: {OFFWHITE};
    border: none;
    border-radius: {RADIUS}px;
    padding: 6px 14px;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton:hover {{ background-color: {PINK}; color: {INDIGO}; }}
QPushButton:disabled {{ background-color: {BORDER}; color: {TEXT_MUTED}; }}
"""

GHOST_BTN = f"""
QPushButton {{
    background-color: transparent;
    color: {OFFWHITE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    padding: 6px 12px;
    font-size: 12px;
}}
QPushButton:hover {{ color: {PINK}; border-color: {PINK}; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; }}
"""

TABLE_STYLE = f"""
QTableWidget {{
    background-color: {PANEL};
    color: {OFFWHITE};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    selection-background-color: rgba(152, 37, 152, 0.30);
    selection-color: {OFFWHITE};
}}
QTableWidget::item {{ padding: 4px 6px; }}
QTableWidget::item:hover {{ background-color: rgba(228, 145, 201, 0.15); }}
QTableWidget::item:selected {{ background-color: rgba(152, 37, 152, 0.30); color: {OFFWHITE}; }}
QHeaderView::section {{
    background-color: {PANEL};
    color: {PURPLE};
    border: 0;
    padding: 6px;
    font-weight: bold;
}}
"""


# ---------------------------------------------------------------------------
# Inference-speed cache helpers (PRD K2-1)
# ---------------------------------------------------------------------------

def _speed_cache_path(models_dir: Path) -> Path:
    return Path(models_dir) / ".inference_speed.json"


def _load_speed_cache(models_dir: Path) -> dict:
    p = _speed_cache_path(models_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _meta_for(models_dir: Path, model_name: str) -> dict | None:
    """Return .meta.json sidecar for ``model_name`` or None if absent."""
    meta_path = Path(models_dir) / (Path(model_name).stem + ".meta.json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tab widget
# ---------------------------------------------------------------------------

class ModelsTab(QWidget):
    """Manage YOLO models + run head-to-head comparison."""

    # Column indices for the models table
    COL_NAME, COL_SIZE, COL_SPEED, COL_ACTIVE, COL_PROVENANCE = range(5)

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self._compare_worker = None
        self._history_files: list[Path] = []
        self._setup_ui()
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(PAD, PAD, PAD, PAD)
        layout.setSpacing(GAP)

        # ---- Header with OpenLocationBar (PRD C12)
        header_row = QHBoxLayout()
        header = QLabel("Models")
        header.setStyleSheet(
            f"color: {PURPLE}; font-size: {TYPE_TITLE}px; font-weight: bold;"
        )
        header_row.addWidget(header)
        header_row.addStretch(1)

        self.locations = OpenLocationBar(self)
        self.locations.add_folder("Models Folder", self.dm.models_dir)
        self.locations.add_file(
            "Active Model",
            lambda: self.dm.models_dir / (self.dm.get_last_model() or "yolov8m.pt"),
        )
        self.locations.add_folder(
            "Comparisons",
            lambda: self.dm.exports_dir / "model_compare",
        )
        header_row.addWidget(self.locations)
        layout.addLayout(header_row)

        # ---- Installed models table
        list_box = QGroupBox("Installed models")
        list_layout = QVBoxLayout(list_box)

        self.model_table = QTableWidget(0, 5)
        self.model_table.setHorizontalHeaderLabels(
            ["Model", "Size", "Inference", "Active", "Provenance"]
        )
        self.model_table.setStyleSheet(TABLE_STYLE)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.model_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.model_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model_table.customContextMenuRequested.connect(self._open_row_menu)
        h = self.model_table.horizontalHeader()
        h.setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        h.setSectionResizeMode(self.COL_SIZE, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(self.COL_SPEED, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(self.COL_ACTIVE, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(self.COL_PROVENANCE, QHeaderView.Stretch)
        self.model_table.setMaximumHeight(260)
        list_layout.addWidget(self.model_table)

        btn_row = QHBoxLayout()
        btn_active = QPushButton("Set as Active")
        btn_active.setStyleSheet(ACTION_BTN)
        btn_active.clicked.connect(self._set_active)
        btn_row.addWidget(btn_active)

        btn_import = QPushButton("Import .pt...")
        btn_import.setStyleSheet(GHOST_BTN)
        btn_import.clicked.connect(self._import_model)
        btn_row.addWidget(btn_import)

        btn_download = QPushButton("Download YOLOv8...")
        btn_download.setStyleSheet(ACTION_BTN)
        btn_download.clicked.connect(self._download_model)
        btn_row.addWidget(btn_download)

        hint = QLabel("Right-click a row to rename or delete.")
        hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {TYPE_HINT}px;")
        btn_row.addWidget(hint)
        btn_row.addStretch(1)
        list_layout.addLayout(btn_row)

        layout.addWidget(list_box)

        # ---- Comparison panel
        cmp_box = QGroupBox("A/B comparison")
        cmp_layout = QVBoxLayout(cmp_box)

        # Mode toggle (PRD J4c)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_video = QRadioButton("Single video")
        self.mode_video.setChecked(True)
        self.mode_dataset = QRadioButton("Dataset val split")
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.mode_video)
        self._mode_group.addButton(self.mode_dataset)
        self.mode_video.toggled.connect(self._update_mode_visibility)
        mode_row.addWidget(self.mode_video)
        mode_row.addWidget(self.mode_dataset)
        mode_row.addStretch(1)

        # History picker (PRD K2-5)
        mode_row.addWidget(QLabel("History:"))
        self.history_combo = QComboBox()
        self.history_combo.setMinimumWidth(280)
        self.history_combo.addItem("recent runs", None)
        self.history_combo.currentIndexChanged.connect(self._load_history_item)
        mode_row.addWidget(self.history_combo)
        btn_refresh_hist = QPushButton("Reload")
        btn_refresh_hist.setStyleSheet(GHOST_BTN)
        btn_refresh_hist.setToolTip("Refresh history list")
        btn_refresh_hist.clicked.connect(self._refresh_history)
        mode_row.addWidget(btn_refresh_hist)
        cmp_layout.addLayout(mode_row)

        # Source row (changes by mode)
        self._source_row = QHBoxLayout()
        self._source_video_label = QLabel("Video:")
        self._source_row.addWidget(self._source_video_label)
        self.video_combo = QComboBox()
        self.video_combo.setMinimumWidth(280)
        self._source_row.addWidget(self.video_combo, stretch=1)

        self._source_dataset_label = QLabel("Dataset:")
        self._source_row.addWidget(self._source_dataset_label)
        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumWidth(220)
        self._source_row.addWidget(self.dataset_combo, stretch=1)
        cmp_layout.addLayout(self._source_row)

        # Param row
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Model A:"))
        self.model_a = QComboBox()
        self.model_a.setMinimumWidth(160)
        params_row.addWidget(self.model_a)

        params_row.addWidget(QLabel("Model B:"))
        self.model_b = QComboBox()
        self.model_b.setMinimumWidth(160)
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
            "Process every Nth frame (video mode only). Use 2-5 to speed up."
        )
        params_row.addWidget(self.stride)
        params_row.addStretch(1)
        cmp_layout.addLayout(params_row)

        # Run row
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
        self.results.setHorizontalHeaderLabels(["Metric", "A", "B", "delta (B - A)"])
        self.results.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 4):
            self.results.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeToContents
            )
        self.results.setStyleSheet(TABLE_STYLE)
        self.results.setMaximumHeight(280)
        cmp_layout.addWidget(self.results)

        # Log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(800)
        self.log.setFont(QFont("Menlo", 11))
        self.log.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {INDIGO}; color: {OFFWHITE}; "
            f"border: 1px solid {BORDER}; border-radius: {RADIUS}px; }}"
        )
        cmp_layout.addWidget(self.log)

        layout.addWidget(cmp_box, stretch=1)

        self._update_mode_visibility()

    # ------------------------------------------------------------------ refresh
    def refresh(self):
        self._populate_models_table()
        self._populate_model_combos()
        self._populate_video_combo()
        self._populate_dataset_combo()
        self._refresh_history()

    def _populate_models_table(self):
        speed_cache = _load_speed_cache(self.dm.models_dir)
        active = self.dm.get_last_model()
        models = self.dm.list_models()

        # Clear any pre-existing spans
        self.model_table.clearSpans()
        self.model_table.setRowCount(len(models) if models else 1)

        if not models:
            empty = QTableWidgetItem(
                "No models yet. Click 'Download YOLOv8...' to fetch one."
            )
            empty.setForeground(QBrush(QColor(TEXT_MUTED)))
            self.model_table.setItem(0, 0, empty)
            self.model_table.setSpan(0, 0, 1, 5)
            return

        for row, m in enumerate(models):
            full = self.dm.models_dir / m
            size_mb = (
                f"{full.stat().st_size / (1024 * 1024):.1f} MB"
                if full.exists()
                else "-"
            )
            speed = speed_cache.get(m)
            speed_text = (
                f"{speed:.1f} ms/f"
                if isinstance(speed, (int, float))
                else "-"
            )

            name_item = QTableWidgetItem(m)
            name_item.setData(Qt.UserRole, m)
            if m == active:
                f = name_item.font()
                f.setBold(True)
                name_item.setFont(f)
            self.model_table.setItem(row, self.COL_NAME, name_item)
            self.model_table.setItem(row, self.COL_SIZE, QTableWidgetItem(size_mb))
            self.model_table.setItem(row, self.COL_SPEED, QTableWidgetItem(speed_text))

            # Active badge cell — PRD C11: PINK background, INDIGO text
            badge_item = QTableWidgetItem("Active" if m == active else "")
            if m == active:
                badge_item.setBackground(QBrush(QColor(PINK)))
                badge_item.setForeground(QBrush(QColor(INDIGO)))
                bf = badge_item.font()
                bf.setBold(True)
                badge_item.setFont(bf)
                badge_item.setTextAlignment(Qt.AlignCenter)
            self.model_table.setItem(row, self.COL_ACTIVE, badge_item)

            # Provenance (sidecar)
            meta = _meta_for(self.dm.models_dir, m)
            if meta:
                base = meta.get("base_model", "?")
                epochs = meta.get("epochs", "?")
                vmap = meta.get("best_val_map")
                vmap_text = (
                    f" - mAP {vmap:.3f}"
                    if isinstance(vmap, (int, float))
                    else ""
                )
                ds = meta.get("dataset_id") or ""
                prov_text = f"{base} - {epochs}e{vmap_text}"
                if ds:
                    prov_text += f" - {ds}"
            else:
                prov_text = "pretrained"
            prov_item = QTableWidgetItem(prov_text)
            prov_item.setForeground(QBrush(QColor(TEXT_MUTED)))
            self.model_table.setItem(row, self.COL_PROVENANCE, prov_item)

    def _populate_model_combos(self):
        models = self.dm.list_models()
        self.model_a.clear()
        self.model_b.clear()
        for m in models:
            self.model_a.addItem(m)
            self.model_b.addItem(m)
        if not models:
            for combo in (self.model_a, self.model_b):
                combo.addItem("yolov8m.pt")
        if len(models) > 1:
            self.model_b.setCurrentIndex(1)

    def _populate_video_combo(self):
        self.video_combo.clear()
        for v in self.dm.get_videos():
            self.video_combo.addItem(v["display_name"], v["session_id"])

    def _populate_dataset_combo(self):
        self.dataset_combo.clear()
        training_root = self.dm.data_dir / "training"
        if training_root.exists():
            for ds in sorted(training_root.iterdir()):
                yaml_path = ds / "data.yaml"
                val_dir = ds / "images" / "val"
                if yaml_path.exists() and val_dir.exists():
                    n_imgs = sum(1 for _ in val_dir.glob("*.jpg")) + sum(
                        1 for _ in val_dir.glob("*.png")
                    )
                    self.dataset_combo.addItem(
                        f"{ds.name}  ({n_imgs} val imgs)", str(ds)
                    )
        if self.dataset_combo.count() == 0:
            self.dataset_combo.addItem(
                "(no datasets - build one in Training tab)", None
            )

    def _update_mode_visibility(self):
        is_video = self.mode_video.isChecked()
        self._source_video_label.setVisible(is_video)
        self.video_combo.setVisible(is_video)
        self._source_dataset_label.setVisible(not is_video)
        self.dataset_combo.setVisible(not is_video)
        self.stride.setEnabled(is_video)

    # ------------------------------------------------------------------ row menu
    def _open_row_menu(self, pos: QPoint):
        item = self.model_table.itemAt(pos)
        if not item:
            return
        row = item.row()
        name_item = self.model_table.item(row, self.COL_NAME)
        if not name_item:
            return
        model_name = name_item.data(Qt.UserRole)
        if not model_name:
            return

        menu = QMenu(self)
        act_active = QAction("Set as Active", self)
        act_active.triggered.connect(lambda: self._set_active_by_name(model_name))
        menu.addAction(act_active)

        act_rename = QAction("Rename...", self)
        act_rename.triggered.connect(lambda: self._rename_model(model_name))
        menu.addAction(act_rename)

        act_reveal = QAction("Reveal in Folder", self)
        act_reveal.triggered.connect(lambda: self._reveal_model(model_name))
        menu.addAction(act_reveal)

        menu.addSeparator()
        act_delete = QAction("Delete...", self)
        act_delete.triggered.connect(lambda: self._delete_model(model_name))
        menu.addAction(act_delete)

        menu.exec(self.model_table.viewport().mapToGlobal(pos))

    def _reveal_model(self, model_name: str):
        open_path(self.dm.models_dir / model_name, select=True)

    def _set_active_by_name(self, model_name: str):
        self.dm.set_last_model(model_name)
        self.refresh()

    def _set_active(self):
        row = self.model_table.currentRow()
        if row < 0:
            return
        item = self.model_table.item(row, self.COL_NAME)
        if item:
            self._set_active_by_name(item.data(Qt.UserRole))

    def _rename_model(self, model_name: str):
        new_name, ok = QInputDialog.getText(
            self, "Rename model",
            f"Rename '{model_name}' to:",
            text=model_name,
        )
        if not ok or not new_name or new_name == model_name:
            return
        if not new_name.endswith(".pt"):
            new_name = new_name + ".pt"
        src = self.dm.models_dir / model_name
        dest = self.dm.models_dir / new_name
        if dest.exists():
            QMessageBox.warning(
                self, "Name in use",
                f"A model named '{new_name}' already exists."
            )
            return
        try:
            src.rename(dest)
            # Rename the sidecar too if present
            sidecar = self.dm.models_dir / (Path(model_name).stem + ".meta.json")
            if sidecar.exists():
                sidecar.rename(
                    self.dm.models_dir / (Path(new_name).stem + ".meta.json")
                )
            # Keep active flag in sync if needed
            if self.dm.get_last_model() == model_name:
                self.dm.set_last_model(new_name)
            self.refresh()
        except Exception as e:
            QMessageBox.critical(self, "Rename failed", str(e))

    def _delete_model(self, model_name: str):
        reply = QMessageBox.question(
            self, "Delete model",
            f"Delete {model_name}? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            target = self.dm.models_dir / model_name
            if target.exists():
                target.unlink()
            sidecar = self.dm.models_dir / (Path(model_name).stem + ".meta.json")
            if sidecar.exists():
                sidecar.unlink()
            if self.dm.get_last_model() == model_name:
                # Demote to whatever's still installed; if nothing, clear it.
                remaining = self.dm.list_models()
                self.dm.set_last_model(remaining[0] if remaining else "")
            self.refresh()
        except Exception as e:
            QMessageBox.critical(self, "Delete failed", str(e))

    # ------------------------------------------------------------------ import / download
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
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        shutil.copy2(src, dest)
        self.refresh()

    def _download_model(self):
        """PRD C1 / K2-6 — bootstrap downloader."""
        labels = variant_labels()
        choice, ok = QInputDialog.getItem(
            self, "Download YOLOv8 model",
            "Which model would you like to download?\n"
            "(Downloaded once, cached in ~/Documents/CCTV-YOLO/models/)",
            labels, 0, False,
        )
        if not ok:
            return
        model_name = variant_from_label(choice)
        dest = self.dm.models_dir / model_name
        if dest.exists():
            QMessageBox.information(
                self, "Already installed",
                f"{model_name} is already in your models folder."
            )
            return

        progress = QProgressDialog(
            f"Downloading {model_name}...\n(This can take a minute.)",
            None, 0, 0, self,
        )
        progress.setWindowTitle("Downloading model")
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.show()

        self._dl_worker = ModelDownloadWorker(model_name, self.dm.models_dir)

        def _ok(path: str):
            progress.close()
            QMessageBox.information(
                self, "Download complete",
                f"{model_name} saved to:\n{path}"
            )
            self.refresh()

        def _fail(msg: str):
            progress.close()
            QMessageBox.critical(self, "Download failed", msg)

        self._dl_worker.done.connect(_ok)
        self._dl_worker.failed.connect(_fail)
        self._dl_worker.start()

    # ------------------------------------------------------------------ run
    def _run_compare(self):
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

        if self.mode_video.isChecked():
            self._run_video_compare(a, b)
        else:
            self._run_dataset_compare(a, b)

    def _run_video_compare(self, a: str, b: str):
        sid = self.video_combo.currentData()
        if not sid:
            QMessageBox.warning(self, "No video", "Pick a video first.")
            self.btn_run.setEnabled(True)
            return
        video_path = self.dm.get_video_path(sid)
        if not video_path or not video_path.exists():
            QMessageBox.warning(self, "No video", "Video file missing.")
            self.btn_run.setEnabled(True)
            return

        worker = ModelCompareWorker(
            video_path=video_path, model_a=a, model_b=b,
            models_dir=self.dm.models_dir, conf=self.conf.value(),
            stride=self.stride.value(),
            exports_dir=self.dm.exports_dir,
            session_id=sid,
        )
        worker.log_line.connect(self.log.appendPlainText)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_video_done)
        worker.failed.connect(self._on_failed)
        self._compare_worker = worker
        worker.start()

    def _run_dataset_compare(self, a: str, b: str):
        ds_path = self.dataset_combo.currentData()
        if not ds_path:
            QMessageBox.warning(
                self, "No dataset",
                "Build a dataset in the Training tab first."
            )
            self.btn_run.setEnabled(True)
            return

        worker = DatasetCompareWorker(
            dataset_root=Path(ds_path),
            model_a=a, model_b=b,
            models_dir=self.dm.models_dir,
            conf=self.conf.value(),
            exports_dir=self.dm.exports_dir,
        )
        worker.log_line.connect(self.log.appendPlainText)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_dataset_done)
        worker.failed.connect(self._on_failed)
        self._compare_worker = worker
        worker.start()

    def _on_progress(self, which: str, pct: int):
        if which == "A":
            self.progress_a.setValue(pct)
        else:
            self.progress_b.setValue(pct)

    # ------------------------------------------------------------------ results
    def _on_video_done(self, payload: dict):
        self.btn_run.setEnabled(True)
        self.progress_a.setValue(100)
        self.progress_b.setValue(100)
        self._show_video_payload(payload)
        self._refresh_history()

    def _on_dataset_done(self, payload: dict):
        self.btn_run.setEnabled(True)
        self.progress_a.setValue(100)
        self.progress_b.setValue(100)
        self._show_dataset_payload(payload)
        self._refresh_history()

    def _on_failed(self, msg: str):
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Comparison failed", msg)

    def _show_video_payload(self, payload: dict):
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
        self._render_rows(rows)

    def _show_dataset_payload(self, payload: dict):
        cm_a = payload["a"]
        cm_b = payload["b"]
        agg_a = cm_a["aggregate"]
        agg_b = cm_b["aggregate"]

        def _delta(x, y):
            return round((y - x), 4)

        rows = [
            ("Precision (agg)", agg_a["precision"], agg_b["precision"],
             _delta(agg_a["precision"], agg_b["precision"])),
            ("Recall (agg)", agg_a["recall"], agg_b["recall"],
             _delta(agg_a["recall"], agg_b["recall"])),
            ("F1 (agg)", agg_a["f1"], agg_b["f1"],
             _delta(agg_a["f1"], agg_b["f1"])),
            ("mAP", agg_a["mAP"], agg_b["mAP"],
             _delta(agg_a["mAP"], agg_b["mAP"])),
        ]
        classes = sorted(set(cm_a["per_class"]) | set(cm_b["per_class"]))
        for c in classes:
            pa = cm_a["per_class"].get(c, {"precision": 0, "recall": 0, "f1": 0})
            pb = cm_b["per_class"].get(c, {"precision": 0, "recall": 0, "f1": 0})
            rows.append((f"  {c} P", pa["precision"], pb["precision"],
                         _delta(pa["precision"], pb["precision"])))
            rows.append((f"  {c} R", pa["recall"], pb["recall"],
                         _delta(pa["recall"], pb["recall"])))
            rows.append((f"  {c} F1", pa["f1"], pb["f1"],
                         _delta(pa["f1"], pb["f1"])))
        self._render_rows(rows)

    def _render_rows(self, rows: list[tuple]):
        self.results.setRowCount(len(rows))
        for i, (m, av, bv, dv) in enumerate(rows):
            self.results.setItem(i, 0, QTableWidgetItem(str(m)))
            self.results.setItem(i, 1, QTableWidgetItem(str(av)))
            self.results.setItem(i, 2, QTableWidgetItem(str(bv)))
            d_text = f"{dv:+}" if isinstance(dv, int) else f"{dv:+.3f}"
            d_item = QTableWidgetItem(d_text)
            if isinstance(dv, (int, float)) and dv != 0:
                d_item.setForeground(
                    Qt.GlobalColor.green if dv > 0 else Qt.GlobalColor.red
                )
            self.results.setItem(i, 3, d_item)

    # ------------------------------------------------------------------ history
    def _refresh_history(self):
        out_dir = self.dm.exports_dir / "model_compare"
        self._history_files = []
        if out_dir.exists():
            self._history_files = sorted(
                out_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        self.history_combo.blockSignals(True)
        self.history_combo.clear()
        self.history_combo.addItem("recent runs", None)
        for p in self._history_files[:50]:
            self.history_combo.addItem(p.stem, str(p))
        self.history_combo.blockSignals(False)

    def _load_history_item(self, idx: int):
        if idx <= 0:
            return
        path = self.history_combo.currentData()
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text())
        except Exception as e:
            QMessageBox.warning(self, "Couldn't load run", str(e))
            return
        self.log.appendPlainText(f"Loaded history: {Path(path).name}")
        if payload.get("mode") == "dataset":
            self._show_dataset_payload(payload)
        else:
            self._show_video_payload(payload)
