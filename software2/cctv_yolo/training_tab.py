"""
Training tab — active-learning queue + dataset build + retrain.

PRD Part J: J2 unused-corrections build, J3 filter row, J4 manual selection,
J4b combine datasets, J4c train-against-any-dataset, J5 auto-refresh,
J6 OpenLocationBar, J7 promote prompt, J8 ROI-aware build, J9 dynamic classes.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QDateTime
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
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QDateTimeEdit,
    QInputDialog,
    QDialog,
    QFormLayout,
)

from cctv_yolo.training import (
    rank_sessions_by_uncertainty,
    DatasetBuildWorker,
    TrainingWorker,
    combine_datasets,
)
from cctv_yolo.training_history import (
    list_datasets,
    compare_models_summary,
)


from cctv_yolo.theme import (
    INDIGO as BG, PANEL, BORDER, PURPLE as ACCENT, OFFWHITE as TEXT,
    PINK, ERROR, TEXT_MUTED, INDIGO,
)

ACTION_BTN = f"""
QPushButton {{
    background-color: {ACCENT};
    color: {TEXT};
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton:hover {{ background-color: {PINK}; color: {INDIGO}; }}
QPushButton:disabled {{ background-color: {BORDER}; color: {TEXT_MUTED}; }}
"""

DANGER_BTN = f"""
QPushButton {{
    background-color: {ERROR};
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton:hover {{ background-color: {PINK}; color: {INDIGO}; }}
"""


# ---------------------------------------------------------------------------
# Promote prompt dialog (PRD J7)
# ---------------------------------------------------------------------------

class _PromoteDialog(QDialog):
    """Side-by-side current-vs-new model comparison."""

    PROMOTE = 1
    PROMOTE_AFTER_COMPARE = 2
    KEEP = 3

    def __init__(self, summary: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Promote new model?")
        self.choice = self.KEEP
        layout = QVBoxLayout(self)

        header = QLabel("Training finished — promote the new model to active?")
        header.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 14px;")
        layout.addWidget(header)

        row = QHBoxLayout()
        for side_name, side in (("Current", summary["current"]), ("New", summary["new"])):
            box = QGroupBox(side_name)
            form = QFormLayout(box)
            form.addRow("Name:", QLabel(str(side.get("name") or "-")))
            form.addRow("Size (MB):", QLabel(str(side.get("size_mb") or "-")))
            form.addRow("Training images:", QLabel(str(side.get("training_images") or "-")))
            form.addRow("Epochs:", QLabel(str(side.get("epochs") or "-")))
            m50 = side.get("val_map50")
            m95 = side.get("val_map5095")
            form.addRow("val mAP@50:", QLabel(f"{m50:.3f}" if isinstance(m50, (int, float)) else "-"))
            form.addRow("val mAP@50-95:", QLabel(f"{m95:.3f}" if isinstance(m95, (int, float)) else "-"))
            row.addWidget(box)
        layout.addLayout(row)

        btns = QHBoxLayout()
        keep_btn = QPushButton("Keep current")
        keep_btn.clicked.connect(lambda: self._done(self.KEEP))
        btns.addWidget(keep_btn)
        cmp_btn = QPushButton("Promote after comparison")
        cmp_btn.clicked.connect(lambda: self._done(self.PROMOTE_AFTER_COMPARE))
        btns.addWidget(cmp_btn)
        prom_btn = QPushButton("Promote")
        prom_btn.setStyleSheet(ACTION_BTN)
        prom_btn.clicked.connect(lambda: self._done(self.PROMOTE))
        btns.addWidget(prom_btn)
        btns.addStretch()
        layout.addLayout(btns)

    def _done(self, choice: int) -> None:
        self.choice = choice
        self.accept()


# ---------------------------------------------------------------------------
# Filter persistence helpers
# ---------------------------------------------------------------------------

_CORRECTED_WINDOWS = ["Any time", "Last 24h", "Last 7d", "Last 30d", "Custom range"]


def _ui_state_path(dm) -> Path:
    return dm.config_dir / "ui_state.json"


def _load_ui_state(dm) -> dict:
    p = _ui_state_path(dm)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_ui_state(dm, state: dict) -> None:
    p = _ui_state_path(dm)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps(state, indent=2))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------

class TrainingTab(QWidget):
    """Active-learning queue + retrain workflow."""

    review_requested = Signal(str)
    # PRD J7 — Performance tab subscribes and pre-fills a comparison.
    compare_models_requested = Signal(str, str)

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self._build_worker = None
        self._train_worker = None
        self._dataset_yaml = None
        self._last_build_id: str | None = None
        self._all_sessions: list[dict] = []
        self._pending_group = None
        self._pending_model = None
        self._setup_ui()
        # PRD J5 — auto-refresh AL queue / unused counter on every save.
        self.dm.corrections_changed.connect(lambda _sid: self.refresh())
        try:
            self.dm.groups_changed.connect(self.refresh)
        except Exception:
            pass
        self._restore_filters()
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QLabel("Active Learning + Training")
        header.setStyleSheet(f"color: {ACCENT}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        # PRD J3 — filter row (search / group / corrected-window / model).
        filt = QHBoxLayout()
        filt.addWidget(QLabel("Search:"))
        self.filter_search = QLineEdit()
        self.filter_search.setPlaceholderText("session or video name…")
        self.filter_search.setMaximumWidth(220)
        self.filter_search.textChanged.connect(self._on_filter_changed)
        filt.addWidget(self.filter_search)

        filt.addWidget(QLabel("Group:"))
        self.filter_group = QComboBox()
        self.filter_group.currentIndexChanged.connect(self._on_filter_changed)
        filt.addWidget(self.filter_group)

        filt.addWidget(QLabel("Corrected:"))
        self.filter_when = QComboBox()
        for label in _CORRECTED_WINDOWS:
            self.filter_when.addItem(label)
        self.filter_when.currentIndexChanged.connect(self._on_when_changed)
        filt.addWidget(self.filter_when)

        self.filter_from = QDateTimeEdit()
        self.filter_from.setCalendarPopup(True)
        self.filter_from.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.filter_from.setDateTime(QDateTime.currentDateTime().addDays(-7))
        self.filter_from.dateTimeChanged.connect(self._on_filter_changed)
        self.filter_to = QDateTimeEdit()
        self.filter_to.setCalendarPopup(True)
        self.filter_to.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.filter_to.setDateTime(QDateTime.currentDateTime())
        self.filter_to.dateTimeChanged.connect(self._on_filter_changed)
        filt.addWidget(self.filter_from)
        filt.addWidget(QLabel("→"))
        filt.addWidget(self.filter_to)
        self.filter_from.setVisible(False)
        self.filter_to.setVisible(False)

        filt.addWidget(QLabel("Model used:"))
        self.filter_model = QComboBox()
        self.filter_model.currentIndexChanged.connect(self._on_filter_changed)
        filt.addWidget(self.filter_model)
        filt.addStretch()
        layout.addLayout(filt)

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

        # PRD J4b — datasets panel (combine / delete / open).
        ds_panel = QGroupBox("Existing datasets")
        ds_panel_layout = QVBoxLayout(ds_panel)
        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.MultiSelection)
        self.dataset_list.setStyleSheet(
            f"QListWidget {{ background-color: {PANEL}; color: {TEXT}; "
            f"border: 1px solid {BORDER}; }}"
        )
        self.dataset_list.setMaximumHeight(140)
        ds_panel_layout.addWidget(self.dataset_list)
        ds_btn_row = QHBoxLayout()
        btn_combine = QPushButton("Combine Selected")
        btn_combine.setStyleSheet(ACTION_BTN)
        btn_combine.clicked.connect(self._combine_selected)
        ds_btn_row.addWidget(btn_combine)
        btn_del_ds = QPushButton("Delete Selected")
        btn_del_ds.setStyleSheet(DANGER_BTN)
        btn_del_ds.clicked.connect(self._delete_selected_datasets)
        ds_btn_row.addWidget(btn_del_ds)
        btn_open_ds = QPushButton("Open Folder")
        btn_open_ds.clicked.connect(self._open_dataset_folder)
        ds_btn_row.addWidget(btn_open_ds)
        ds_btn_row.addStretch()
        ds_panel_layout.addLayout(ds_btn_row)
        layout.addWidget(ds_panel)

        # Dataset / training
        ds_box = QGroupBox("Build dataset + train")
        ds_layout = QVBoxLayout(ds_box)

        # PRD J4 — build-mode selector.
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Build mode:"))
        self.build_mode = QComboBox()
        self.build_mode.addItems([
            "All corrected",
            "Unused since last build",
            "Manual selection",
            "Combine existing datasets",
        ])
        self.build_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.build_mode)
        mode_row.addStretch()
        ds_layout.addLayout(mode_row)

        # Manual-selection list (only visible in manual mode).
        self.manual_list = QListWidget()
        self.manual_list.setStyleSheet(
            f"QListWidget {{ background-color: {PANEL}; color: {TEXT}; "
            f"border: 1px solid {BORDER}; }}"
        )
        self.manual_list.setMaximumHeight(180)
        self.manual_list.setVisible(False)
        ds_layout.addWidget(self.manual_list)

        # Build params
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Sample every"))
        self.sample_n = QSpinBox()
        self.sample_n.setRange(1, 100)
        self.sample_n.setValue(5)
        self.sample_n.setSuffix(" frames")
        params_row.addWidget(self.sample_n)

        # PRD J9-2 — val split spinbox.
        params_row.addWidget(QLabel("val split:"))
        self.val_split = QDoubleSpinBox()
        self.val_split.setRange(0.05, 0.30)
        self.val_split.setSingleStep(0.05)
        self.val_split.setValue(0.10)
        self.val_split.setDecimals(2)
        params_row.addWidget(self.val_split)

        # PRD J8 — ROI filter.
        params_row.addWidget(QLabel("Filter to ROI:"))
        self.roi_filter = QComboBox()
        self.roi_filter.addItem("None", userData=None)
        params_row.addWidget(self.roi_filter)

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

        # PRD J4c — train-dataset picker.
        train_row = QHBoxLayout()
        train_row.addWidget(QLabel("Train using dataset:"))
        self.train_dataset = QComboBox()
        train_row.addWidget(self.train_dataset)
        train_row.addStretch()
        ds_layout.addLayout(train_row)

        action_row = QHBoxLayout()
        self.btn_build = QPushButton("1. Build Dataset")
        self.btn_build.setStyleSheet(ACTION_BTN)
        self.btn_build.clicked.connect(self._build_dataset_clicked)
        action_row.addWidget(self.btn_build)

        self.btn_train = QPushButton("2. Train selected dataset")
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
            f"QPlainTextEdit {{ background-color: #15173D; color: {TEXT}; "
            f"border: 1px solid {BORDER}; }}"
        )
        ds_layout.addWidget(self.log, stretch=1)

        layout.addWidget(ds_box, stretch=1)

    def _latest_trained_model(self):
        """Return path to the most recent models/trained_*.pt, or None."""
        candidates = sorted(self.dm.models_dir.glob("trained_*.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else self.dm.models_dir

    # ------------------------------------------------------------------
    # Filter persistence (PRD J3)
    # ------------------------------------------------------------------

    def _restore_filters(self) -> None:
        state = _load_ui_state(self.dm).get("training_tab", {})
        if "search" in state:
            self.filter_search.setText(state["search"])
        if "when" in state:
            idx = self.filter_when.findText(state["when"])
            if idx >= 0:
                self.filter_when.setCurrentIndex(idx)
        self._pending_group = state.get("group")
        self._pending_model = state.get("model")

    def _persist_filters(self) -> None:
        state = _load_ui_state(self.dm)
        state["training_tab"] = {
            "search": self.filter_search.text(),
            "group": self.filter_group.currentData(),
            "when": self.filter_when.currentText(),
            "model": self.filter_model.currentText(),
        }
        _save_ui_state(self.dm, state)

    def _on_when_changed(self) -> None:
        is_custom = self.filter_when.currentText() == "Custom range"
        self.filter_from.setVisible(is_custom)
        self.filter_to.setVisible(is_custom)
        self._on_filter_changed()

    def _on_filter_changed(self) -> None:
        self._persist_filters()
        self._apply_filters_to_views()

    def _on_mode_changed(self) -> None:
        mode = self.build_mode.currentText()
        self.manual_list.setVisible(mode == "Manual selection")
        if mode == "Combine existing datasets":
            self.btn_build.setText("Combine selected datasets")
        else:
            self.btn_build.setText("1. Build Dataset")

    # ------------------------------------------------------------------
    # Refresh — populate filter combos + tables (PRD J3, J4c)
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        prev_group = self.filter_group.currentData()
        self.filter_group.blockSignals(True)
        self.filter_group.clear()
        self.filter_group.addItem("All", userData=None)
        try:
            for g in self.dm.list_groups():
                self.filter_group.addItem(g.get("name", g["id"]), userData=g["id"])
        except Exception:
            pass
        target_group = self._pending_group or prev_group
        if target_group:
            idx = self.filter_group.findData(target_group)
            if idx >= 0:
                self.filter_group.setCurrentIndex(idx)
        self.filter_group.blockSignals(False)
        self._pending_group = None

        # Collect sessions + model names + mtime for filtering.
        self._all_sessions = []
        model_names: set[str] = set()
        for s in self.dm.get_sessions():
            t = self.dm.load_tracks(s["id"]) or {}
            m = (t.get("model") or "").strip()
            if m:
                model_names.add(m)
            s2 = dict(s)
            s2["_model"] = m
            cf = self.dm.corrections_dir / f"{s['id']}.json"
            tf = self.dm.tracks_dir / f"{s['id']}.json"
            try:
                if cf.exists():
                    s2["_when"] = cf.stat().st_mtime
                elif tf.exists():
                    s2["_when"] = tf.stat().st_mtime
                else:
                    s2["_when"] = 0
            except OSError:
                s2["_when"] = 0
            self._all_sessions.append(s2)

        prev_model = self.filter_model.currentText()
        self.filter_model.blockSignals(True)
        self.filter_model.clear()
        self.filter_model.addItem("Any")
        for m in sorted(model_names):
            self.filter_model.addItem(m)
        target_model = self._pending_model or prev_model
        if target_model:
            idx = self.filter_model.findText(target_model)
            if idx >= 0:
                self.filter_model.setCurrentIndex(idx)
        self.filter_model.blockSignals(False)
        self._pending_model = None

        self._refresh_datasets()
        self._refresh_roi_options()
        self._apply_filters_to_views()
        self._refresh_unused_count()

    def _refresh_roi_options(self) -> None:
        prev = self.roi_filter.currentText()
        self.roi_filter.blockSignals(True)
        self.roi_filter.clear()
        self.roi_filter.addItem("None", userData=None)
        try:
            global_roi = self.dm.get_global_processing_roi()
        except Exception:
            global_roi = None
        if global_roi:
            self.roi_filter.addItem("Global processing ROI", userData=global_roi)
        idx = self.roi_filter.findText(prev)
        if idx >= 0:
            self.roi_filter.setCurrentIndex(idx)
        self.roi_filter.blockSignals(False)

    def _refresh_datasets(self) -> None:
        training_root = self.dm.data_root / "training"
        datasets = list_datasets(training_root)

        self.dataset_list.clear()
        for ds in datasets:
            classes = ", ".join(ds["classes"][:4]) + ("…" if len(ds["classes"]) > 4 else "")
            label = (f"{ds['name']}  ·  {ds['train_images']+ds['val_images']} images"
                     f"  ·  [{classes or 'no classes'}]")
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, ds["path"])
            self.dataset_list.addItem(item)

        prev = self.train_dataset.currentText()
        self.train_dataset.blockSignals(True)
        self.train_dataset.clear()
        for ds in datasets:
            self.train_dataset.addItem(ds["name"], userData=ds["path"])
        idx = self.train_dataset.findText(prev)
        if idx >= 0:
            self.train_dataset.setCurrentIndex(idx)
        self.train_dataset.blockSignals(False)
        self.btn_train.setEnabled(self.train_dataset.count() > 0)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filtered_sessions(self) -> list[dict]:
        sessions = list(self._all_sessions)
        q = self.filter_search.text().strip().lower()
        if q:
            sessions = [s for s in sessions
                        if q in s["id"].lower() or q in s.get("video_name", "").lower()]
        gid = self.filter_group.currentData()
        if gid:
            try:
                gset = {gs["id"] for gs in self.dm.get_sessions_in_group(gid)}
            except Exception:
                gset = set()
            sessions = [s for s in sessions if s["id"] in gset]
        when = self.filter_when.currentText()
        now = dt.datetime.now().timestamp()
        if when == "Last 24h":
            sessions = [s for s in sessions if now - s["_when"] <= 86400]
        elif when == "Last 7d":
            sessions = [s for s in sessions if now - s["_when"] <= 7 * 86400]
        elif when == "Last 30d":
            sessions = [s for s in sessions if now - s["_when"] <= 30 * 86400]
        elif when == "Custom range":
            f = self.filter_from.dateTime().toSecsSinceEpoch()
            t = self.filter_to.dateTime().toSecsSinceEpoch()
            sessions = [s for s in sessions if f <= s["_when"] <= t]
        mt = self.filter_model.currentText()
        if mt and mt != "Any":
            sessions = [s for s in sessions if s.get("_model") == mt]
        return sessions

    def _apply_filters_to_views(self) -> None:
        filtered_ids = {s["id"] for s in self._filtered_sessions()}

        rows = rank_sessions_by_uncertainty(self.dm)
        rows = [r for r in rows if r["session_id"] in filtered_ids]
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

        # Manual selection list, only meaningful for J4 mode.
        prev_checked = {
            self.manual_list.item(i).data(Qt.UserRole)
            for i in range(self.manual_list.count())
            if self.manual_list.item(i).checkState() == Qt.Checked
        }
        self.manual_list.clear()
        corrected_only = [s for s in self._filtered_sessions() if s.get("has_corrections")]
        for s in corrected_only:
            item = QListWidgetItem(f"{s.get('video_name', s['id'])}  ({s['id']})")
            item.setData(Qt.UserRole, s["id"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if s["id"] in prev_checked else Qt.Unchecked)
            self.manual_list.addItem(item)

    def _refresh_unused_count(self) -> None:
        try:
            unused = self.dm.list_unused_corrections()
            last = self.dm.get_last_training_build()
        except Exception:
            self.lbl_unused.setText("")
            return
        if last:
            built_at = last.get("built_at", "")[:16].replace("T", " ")
            self.lbl_unused.setText(
                f"{len(unused)} unused correction(s) since last build ({built_at})"
            )
        else:
            self.lbl_unused.setText(
                f"{len(unused)} corrected session(s) — no prior build"
            )

    def _review_top(self) -> None:
        if self.al_table.rowCount() == 0:
            return
        cell = self.al_table.item(0, 0)
        if cell:
            sid = cell.data(Qt.UserRole)
            if sid:
                self.review_requested.emit(sid)

    # ------------------------------------------------------------------
    # Build dispatch (PRD J2 / J4 / J4b)
    # ------------------------------------------------------------------

    def _build_dataset_clicked(self) -> None:
        mode = self.build_mode.currentText()
        if mode == "All corrected":
            self._build_dataset(restrict_to=None)
        elif mode == "Unused since last build":
            self._build_dataset_unused()
        elif mode == "Manual selection":
            self._build_dataset_manual()
        elif mode == "Combine existing datasets":
            self._combine_selected()

    def _build_dataset_unused(self) -> None:
        try:
            unused = self.dm.list_unused_corrections()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Couldn't list unused corrections: {e}")
            return
        if not unused:
            QMessageBox.information(
                self, "Nothing to build",
                "No corrections have changed since the last training build.\n"
                "Make new corrections in the Correction tab first."
            )
            return
        session_ids = [s["id"] for s in unused]
        reasons: dict[str, int] = {}
        for s in unused:
            reasons[s["_reason"]] = reasons.get(s["_reason"], 0) + 1
        self._append_log(
            f"Found {len(unused)} unused correction(s): "
            + ", ".join(f"{n} {r}" for r, n in reasons.items())
        )
        self._build_dataset(restrict_to=session_ids)

    def _build_dataset_manual(self) -> None:
        ids = []
        for i in range(self.manual_list.count()):
            item = self.manual_list.item(i)
            if item.checkState() == Qt.Checked:
                ids.append(item.data(Qt.UserRole))
        if not ids:
            QMessageBox.information(self, "Manual selection",
                                    "Tick at least one session to include.")
            return
        self._append_log(f"Manual build: {len(ids)} session(s) selected.")
        self._build_dataset(restrict_to=ids)

    def _build_dataset(self, restrict_to: list[str] | None) -> None:
        out = self.dm.data_root / "training" / dt.datetime.now().strftime("ds_%Y%m%d_%H%M%S")
        self._append_log(f"Building dataset at {out}")
        if restrict_to is not None:
            self._append_log(
                f"  Restricted to {len(restrict_to)} session(s): "
                f"{', '.join(restrict_to[:5])}{'…' if len(restrict_to) > 5 else ''}"
            )
        roi = self.roi_filter.currentData()
        if roi:
            self._append_log(f"  ROI filter active ({roi.get('type', '?')})")
        self.progress.setValue(0)
        self.btn_build.setEnabled(False)

        worker = DatasetBuildWorker(
            self.dm, out,
            sample_every_n=self.sample_n.value(),
            val_split=self.val_split.value(),
            roi=roi,
        )
        worker._restrict_session_ids = restrict_to
        worker.progress.connect(self.progress.setValue)
        worker.finished_ok.connect(self._on_dataset_built)
        worker.failed.connect(self._on_failed)
        self._build_worker = worker
        worker.start()

    def _on_dataset_built(self, stats: dict) -> None:
        self.btn_build.setEnabled(True)
        self.progress.setValue(100)
        self._dataset_yaml = stats["yaml_path"]
        self._last_build_id = Path(stats["yaml_path"]).parent.name
        self._append_log(
            f"Dataset built: {stats['images']} images, {stats['labels']} labels "
            f"from {stats['corrected_sessions']} corrected sessions"
        )
        self._append_log(f"Classes: {stats['classes']}")
        self._append_log(f"data.yaml: {self._dataset_yaml}")
        self._refresh_datasets()
        idx = self.train_dataset.findText(self._last_build_id)
        if idx >= 0:
            self.train_dataset.setCurrentIndex(idx)
        if stats["images"] == 0:
            QMessageBox.warning(
                self, "Empty dataset",
                "No corrected sessions matched. Correct some sessions first or "
                "relax your filters."
            )

    # ------------------------------------------------------------------
    # Combine + delete datasets (PRD J4b)
    # ------------------------------------------------------------------

    def _selected_dataset_paths(self) -> list[Path]:
        return [Path(self.dataset_list.item(i).data(Qt.UserRole))
                for i in range(self.dataset_list.count())
                if self.dataset_list.item(i).isSelected()]

    def _combine_selected(self) -> None:
        paths = self._selected_dataset_paths()
        if len(paths) < 2:
            QMessageBox.information(
                self, "Combine datasets",
                "Select at least two datasets in the list above to combine."
            )
            return
        default_name = "combined_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ok = QInputDialog.getText(self, "Combined dataset name",
                                        "Name (will become ds_<name>):",
                                        text=default_name)
        if not ok or not name.strip():
            return
        name = name.strip()
        if not name.startswith("ds_"):
            name = "ds_" + name
        out = self.dm.data_root / "training" / name
        if out.exists():
            QMessageBox.warning(self, "Already exists", f"{out} already exists.")
            return
        try:
            combine_datasets(paths, out, val_split=self.val_split.value())
        except Exception as e:
            QMessageBox.critical(self, "Combine failed", f"{e}")
            return
        self._append_log(f"Combined {len(paths)} datasets -> {out.name}")
        self._refresh_datasets()
        idx = self.train_dataset.findText(out.name)
        if idx >= 0:
            self.train_dataset.setCurrentIndex(idx)
            self.btn_train.setEnabled(True)
        self._dataset_yaml = str(out / "data.yaml")
        self._last_build_id = out.name

    def _delete_selected_datasets(self) -> None:
        paths = self._selected_dataset_paths()
        if not paths:
            return
        ret = QMessageBox.question(
            self, "Delete datasets?",
            f"Permanently delete {len(paths)} dataset folder(s)?\n\n"
            + "\n".join(p.name for p in paths),
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        for p in paths:
            try:
                shutil.rmtree(p)
                self._append_log(f"Deleted {p.name}")
            except OSError as e:
                self._append_log(f"FAILED to delete {p}: {e}")
        self._refresh_datasets()

    def _open_dataset_folder(self) -> None:
        paths = self._selected_dataset_paths()
        target = paths[0] if paths else (self.dm.data_root / "training")
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            elif sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except OSError as e:
            self._append_log(f"Couldn't open folder: {e}")

    # ------------------------------------------------------------------
    # Training (PRD J4c, J7)
    # ------------------------------------------------------------------

    def _start_training(self) -> None:
        path = self.train_dataset.currentData()
        if not path:
            QMessageBox.warning(self, "No dataset",
                                "Build or select a dataset first.")
            return
        yaml = Path(path) / "data.yaml"
        if not yaml.exists():
            QMessageBox.warning(self, "Missing data.yaml",
                                f"{yaml} does not exist.")
            return
        self._dataset_yaml = str(yaml)
        self._last_build_id = Path(path).name
        self._append_log("=" * 50)
        self._append_log(f"Starting training on {self._last_build_id}")
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

    def _stop_training(self) -> None:
        if self._train_worker:
            self._train_worker.stop()
            self._append_log("Stop requested…")

    def _on_train_done(self, model_path: str) -> None:
        self.btn_build.setEnabled(True)
        self.btn_train.setEnabled(self.train_dataset.count() > 0)
        self.btn_stop.setEnabled(False)
        self._append_log(f"Training complete: {model_path}")
        if self._last_build_id:
            try:
                self.dm.record_training_build(
                    build_id=self._last_build_id,
                    trained_model=Path(model_path).name,
                )
                self._append_log(f"Training history updated (build_id={self._last_build_id}).")
            except Exception as e:
                self._append_log(f"Warning: couldn't record training history: {e}")

        # PRD J7 — promote prompt with mini side-by-side.
        try:
            current = self.dm.get_last_model()
            current_path = self.dm.models_dir / current if current else None
            summary = compare_models_summary(
                current_path,
                Path(model_path),
                new_dataset_dir=Path(self._dataset_yaml).parent if self._dataset_yaml else None,
            )
            dlg = _PromoteDialog(summary, self)
            dlg.exec()
            if dlg.choice == _PromoteDialog.PROMOTE:
                self.dm.set_last_model(Path(model_path).name)
                self._append_log(f"Promoted {Path(model_path).name} to active model.")
            elif dlg.choice == _PromoteDialog.PROMOTE_AFTER_COMPARE:
                self.compare_models_requested.emit(
                    current or "",
                    Path(model_path).name,
                )
                self._append_log("Opened Performance tab for side-by-side comparison.")
            else:
                self._append_log("Kept current active model.")
        except Exception as e:
            self._append_log(f"(promote prompt error: {e})")

        self.refresh()

    def _on_failed(self, msg: str):
        self.btn_build.setEnabled(True)
        self.btn_build_unused.setEnabled(True)
        self.btn_train.setEnabled(self._dataset_yaml is not None)
        self.btn_stop.setEnabled(False)
        self._append_log(f"FAILED: {msg}")

    def _append_log(self, line: str):
        self.log.appendPlainText(line)
