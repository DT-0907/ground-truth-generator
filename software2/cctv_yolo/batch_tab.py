"""
Batch tab — folder-scoped tree view + parallel scheduler.

PRD Part E. The UI is built around one source folder at a time: you point it
at any folder on disk, get a ``QTreeView`` (lazy QFileSystemModel) of every
video in it, and the scheduler chews through them in parallel using a
``QThreadPool`` you can size on the fly.

Per-folder UI state (active folder + expanded nodes) is persisted in
``config/batch_registry.json``. Every batch session_id is also recorded in
``config/batch_session_map.json`` so DataManager can find the underlying
video regardless of where it lives on disk.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cctv_yolo.batch_queue import BatchQueueManager, VIDEO_EXTS
from cctv_yolo.batch_tree import BatchTree
from cctv_yolo.theme import (
    BORDER,
    ERROR,
    INDIGO,
    OFFWHITE,
    PANEL,
    PANEL_HI,
    PINK,
    PURPLE,
    RADIUS,
    TEXT_MUTED,
    TYPE_HINT,
    TYPE_SECTION,
    TYPE_TITLE,
)
from cctv_yolo.widgets.open_location_bar import OpenLocationBar


# ---------------------------------------------------------------------------
# Stylesheets (all colors sourced from theme.py)
# ---------------------------------------------------------------------------

PRIMARY_BTN = f"""
QPushButton {{
    background-color: {PURPLE};
    color: {OFFWHITE};
    border: none;
    border-radius: {RADIUS}px;
    padding: 7px 16px;
    font-weight: 600;
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
    padding: 6px 14px;
    font-weight: 500;
    font-size: 12px;
}}
QPushButton:hover {{ border-color: {PINK}; color: {PINK}; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; border-color: {BORDER}; }}
"""

DANGER_BTN = f"""
QPushButton {{
    background-color: transparent;
    color: {ERROR};
    border: 1px solid {ERROR};
    border-radius: {RADIUS}px;
    padding: 6px 14px;
    font-weight: 600;
    font-size: 12px;
}}
QPushButton:hover {{ background-color: {ERROR}; color: {OFFWHITE}; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; border-color: {BORDER}; }}
"""

PANEL_QSS = f"""
QFrame#card {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
}}
QLabel {{ color: {OFFWHITE}; }}
QLabel#hint {{ color: {TEXT_MUTED}; font-size: {TYPE_HINT}px; }}
QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {INDIGO};
    color: {OFFWHITE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS - 2}px;
    padding: 3px 6px;
    min-height: 24px;
}}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {PINK};
}}
QComboBox::drop-down {{ border: 0; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {PANEL_HI};
    color: {OFFWHITE};
    border: 1px solid {BORDER};
    selection-background-color: {PURPLE};
}}
QCheckBox {{ color: {OFFWHITE}; }}
"""


# ---------------------------------------------------------------------------
# BatchTab
# ---------------------------------------------------------------------------

class BatchTab(QWidget):
    """Tree-driven batch processor with parallel workers and per-folder state."""

    review_requested = Signal(str)

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.queue = BatchQueueManager(data_manager, self)
        self.queue.queue_changed.connect(self._on_queue_changed)
        self.queue.item_progress.connect(self._on_item_progress)
        self.queue.stats_changed.connect(self._update_stats)

        # Per-folder UI state (PRD E5-5)
        self._registry = self.dm.load_batch_registry()
        self._active_folder: Optional[Path] = None
        ap = self._registry.get("active_folder")
        if ap and Path(ap).exists():
            self._active_folder = Path(ap)

        self._setup_ui()

        # Restore active folder + its expanded tree state.
        if self._active_folder is not None:
            self._activate_folder(self._active_folder, persist=False)

        # Throttle status repaints — we don't want every progress tick to
        # trigger a tree refresh.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(250)
        self._refresh_timer.timeout.connect(self.tree.refresh_statuses)

        self._update_stats(self._initial_stats())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self):
        self.setStyleSheet(PANEL_QSS)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # ---- Title row -------------------------------------------------
        title_row = QHBoxLayout()
        title = QLabel("Batch Processing")
        title.setStyleSheet(f"color: {PINK}; font-size: {TYPE_TITLE}px; font-weight: 600;")
        title_row.addWidget(title)
        title_row.addStretch(1)

        # OpenLocationBar (top-right)
        ol = OpenLocationBar(self)
        ol.add_folder("Active Folder", lambda: self._active_folder)
        ol.add_folder("Tracks Output", self.dm.tracks_dir)
        from cctv_yolo.paths import get_log_file
        ol.add_file("Errors Log", get_log_file)
        title_row.addWidget(ol)
        root.addLayout(title_row)

        # ---- Top control card ------------------------------------------
        top_card = QFrame()
        top_card.setObjectName("card")
        top_lay = QVBoxLayout(top_card)
        top_lay.setContentsMargins(12, 10, 12, 10)
        top_lay.setSpacing(8)

        # Folder picker + active path
        fp_row = QHBoxLayout()
        self.btn_pick = QPushButton("📁 Pick Folder…")
        self.btn_pick.setStyleSheet(PRIMARY_BTN)
        self.btn_pick.clicked.connect(self._pick_folder)
        fp_row.addWidget(self.btn_pick)

        self.lbl_active = QLabel("")
        self.lbl_active.setObjectName("hint")
        self.lbl_active.setStyleSheet(f"color: {TEXT_MUTED};")
        fp_row.addWidget(self.lbl_active, stretch=1)

        self.btn_rescan = QPushButton("⟳ Rescan")
        self.btn_rescan.setStyleSheet(GHOST_BTN)
        self.btn_rescan.clicked.connect(self._rescan)
        fp_row.addWidget(self.btn_rescan)
        top_lay.addLayout(fp_row)

        # Run controls row
        run_row = QHBoxLayout()
        run_row.setSpacing(10)

        run_row.addWidget(QLabel("Parallel workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 100)
        self.workers_spin.setValue(self.queue.max_workers())
        self.workers_spin.valueChanged.connect(self._on_workers_changed)
        self.workers_spin.setFixedWidth(70)
        run_row.addWidget(self.workers_spin)

        # Recommend ≤ os.cpu_count()*2
        cpu = max(1, os.cpu_count() or 1)
        rec = QLabel(f"  (recommend ≤ {cpu * 2})")
        rec.setStyleSheet(f"color: {TEXT_MUTED}; font-size: {TYPE_HINT}px;")
        run_row.addWidget(rec)

        run_row.addSpacing(18)

        run_row.addWidget(QLabel("Stop after:"))
        self.stop_spin = QSpinBox()
        self.stop_spin.setRange(0, 100000)
        self.stop_spin.setValue(0)
        self.stop_spin.setSpecialValueText("All")
        self.stop_spin.setFixedWidth(80)
        self.stop_spin.valueChanged.connect(self._on_stop_after_changed)
        run_row.addWidget(self.stop_spin)

        self.chk_count_failed = QCheckBox("Count failed toward limit")
        self.chk_count_failed.setChecked(True)
        self.chk_count_failed.toggled.connect(self.queue.set_count_failed_toward_limit)
        run_row.addWidget(self.chk_count_failed)

        run_row.addStretch(1)
        top_lay.addLayout(run_row)

        # Model + conf row
        cfg_row = QHBoxLayout()
        cfg_row.setSpacing(10)
        cfg_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self._reload_models()
        self.model_combo.setMinimumWidth(180)
        cfg_row.addWidget(self.model_combo)

        cfg_row.addSpacing(12)
        cfg_row.addWidget(QLabel("Conf:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(self.dm.get_last_confidence())
        self.conf_spin.setFixedWidth(80)
        cfg_row.addWidget(self.conf_spin)
        cfg_row.addStretch(1)
        top_lay.addLayout(cfg_row)

        root.addWidget(top_card)

        # ---- Stats strip ----------------------------------------------
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet(
            f"color: {OFFWHITE}; font-size: {TYPE_HINT + 1}px; padding: 2px 6px;"
        )
        root.addWidget(self.stats_label)

        # ---- Tree ------------------------------------------------------
        self.tree = BatchTree(status_provider=self._status_for_path)
        self.tree.file_activated.connect(self._on_file_activated)
        self.tree.expansion_changed.connect(self._on_expansion_changed)
        root.addWidget(self.tree, stretch=1)

        # ---- Bottom action row ----------------------------------------
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.btn_start = QPushButton("▶ Start All")
        self.btn_start.setStyleSheet(PRIMARY_BTN)
        self.btn_start.clicked.connect(self._start_all)
        action_row.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setStyleSheet(GHOST_BTN)
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self._toggle_pause)
        action_row.addWidget(self.btn_pause)

        self.btn_cancel = QPushButton("✕ Cancel")
        self.btn_cancel.setStyleSheet(DANGER_BTN)
        self.btn_cancel.clicked.connect(self._cancel_all)
        action_row.addWidget(self.btn_cancel)

        self.btn_clear = QPushButton("🧹 Clear Done")
        self.btn_clear.setStyleSheet(GHOST_BTN)
        self.btn_clear.clicked.connect(self.queue.clear_finished)
        action_row.addWidget(self.btn_clear)

        action_row.addStretch(1)

        self.btn_review = QPushButton("Open Review for Selected")
        self.btn_review.setStyleSheet(GHOST_BTN)
        self.btn_review.clicked.connect(self._review_selected)
        action_row.addWidget(self.btn_review)

        root.addLayout(action_row)

        self._update_active_label()

    # ------------------------------------------------------------------
    # Folder management
    # ------------------------------------------------------------------
    def _pick_folder(self):
        start = str(self._active_folder) if self._active_folder else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Pick a video folder", start)
        if not folder:
            return
        self._activate_folder(Path(folder), persist=True)
        # Auto-ingest videos so the user can hit Start immediately.
        self._ingest_folder(Path(folder))

    def _activate_folder(self, folder: Path, *, persist: bool):
        self._active_folder = folder
        self.tree.set_root(folder)
        self._update_active_label()

        folders = self._registry.setdefault("folders", {})
        entry = folders.setdefault(str(folder), {})
        entry.setdefault("added_at", _now_iso())
        # Restore previously-expanded nodes for this folder
        QTimer.singleShot(0, lambda: self.tree.restore_expanded_paths(
            entry.get("expanded_paths", [])
        ))

        if persist:
            self._registry["active_folder"] = str(folder)
            self.dm.save_batch_registry(self._registry)
        else:
            self._registry["active_folder"] = str(folder)

    def _update_active_label(self):
        if self._active_folder:
            self.lbl_active.setText(f"Active: {self._active_folder}")
        else:
            self.lbl_active.setText("Active: (no folder selected — Pick Folder… to start)")

    def _rescan(self):
        if self._active_folder is None:
            return
        # Queue any video under the folder that isn't already enqueued.
        added = self._ingest_folder(self._active_folder)
        self._registry.setdefault("folders", {}).setdefault(
            str(self._active_folder), {}
        )["last_scan_at"] = _now_iso()
        self.dm.save_batch_registry(self._registry)
        self._flash_stats(f"Rescan: added {added}")

    def _ingest_folder(self, folder: Path) -> int:
        """Walk `folder` and add every video to the queue (skipping dups)."""
        added = 0
        for p in folder.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in VIDEO_EXTS:
                continue
            sid = self.queue.add(
                str(p),
                model=self.model_combo.currentText(),
                conf=self.conf_spin.value(),
                source_folder=str(folder),
            )
            if sid:
                added += 1
        return added

    # ------------------------------------------------------------------
    # Run controls
    # ------------------------------------------------------------------
    def _on_workers_changed(self, n: int):
        cpu = max(1, os.cpu_count() or 1)
        self.queue.set_max_workers(n)
        if n > cpu * 2:
            self._flash_stats(
                f"⚠ {n} workers exceeds recommended {cpu * 2} — system may slow"
            )

    def _on_stop_after_changed(self, n: int):
        self.queue.set_stop_after_n(n if n > 0 else None)

    def _start_all(self):
        if self._active_folder is None:
            QMessageBox.information(self, "Batch", "Pick a folder first.")
            return
        # Ensure conf + model are up-to-date for any items that were added
        # before the user tweaked these controls — refresh only queued items.
        model = self.model_combo.currentText()
        conf = float(self.conf_spin.value())
        for it in self.queue.items:
            if it["status"] == "queued":
                it["model"] = model
                it["conf"] = conf
        self.queue.start_all()
        self.dm.set_last_model(model)
        self.dm.set_last_confidence(conf)

    def _toggle_pause(self, checked: bool):
        if checked:
            self.queue.pause_all()
            self.btn_pause.setText("▶ Resume")
        else:
            self.queue.resume_all()
            self.btn_pause.setText("⏸ Pause")

    def _cancel_all(self):
        confirm = QMessageBox.question(
            self, "Cancel batch",
            "Cancel everything (queued + running)? Partial outputs will be deleted.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        self.queue.cancel_all()

    # ------------------------------------------------------------------
    # Tree integration
    # ------------------------------------------------------------------
    def _status_for_path(self, abs_path: str) -> str:
        st = self.queue.status_for_path(abs_path)
        return st or "queued"

    def _on_file_activated(self, abs_path: str):
        # Double-click on a video row → open review for it (if it has tracks).
        for it in self.queue.items:
            if it.get("video_path") == abs_path and it.get("status") == "done":
                self.review_requested.emit(it["session_id"])
                return

    def _on_expansion_changed(self, expanded: list[str]):
        if self._active_folder is None:
            return
        folders = self._registry.setdefault("folders", {})
        entry = folders.setdefault(str(self._active_folder), {})
        entry["expanded_paths"] = list(expanded)
        self.dm.save_batch_registry(self._registry)

    # ------------------------------------------------------------------
    # Queue events
    # ------------------------------------------------------------------
    def _on_queue_changed(self):
        # Coalesce status repaints
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _on_item_progress(self, session_id: str, pct: int):
        # Progress tick — don't repaint the whole tree, just nudge once in a while.
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _initial_stats(self) -> dict:
        items = self.queue.get_items()
        return {
            "total": len(items),
            "processed": sum(1 for it in items if it["status"] == "done"),
            "processing": sum(1 for it in items if it["status"] == "processing"),
            "queued": sum(1 for it in items if it["status"] in ("queued", "paused")),
            "errors": sum(1 for it in items if it["status"] == "error"),
            "cancelled": sum(1 for it in items if it["status"] == "cancelled"),
        }

    def _update_stats(self, stats: dict):
        total = stats.get("total", 0)
        processed = stats.get("processed", 0)
        processing = stats.get("processing", 0)
        queued = stats.get("queued", 0)
        errors = stats.get("errors", 0)
        text = (
            f"{total} videos · "
            f"{processed} processed · "
            f"{processing} processing · "
            f"{queued} queued · "
            f"{errors} errors"
        )
        self.stats_label.setText(text)

    def _flash_stats(self, msg: str):
        """Temporarily replace the stats strip with a toast-style message."""
        prev = self.stats_label.text()
        self.stats_label.setText(msg)
        QTimer.singleShot(2500, lambda: self.stats_label.setText(prev))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _reload_models(self):
        self.model_combo.clear()
        models = self.dm.list_models()
        if not models:
            models = ["yolov8m.pt"]
        self.model_combo.addItems(models)
        last = self.dm.get_last_model()
        if last and last in models:
            self.model_combo.setCurrentText(last)

    def _review_selected(self):
        paths = self.tree.selected_video_paths()
        if not paths:
            return
        target = paths[0]
        for it in self.queue.items:
            if it.get("video_path") == target and it.get("status") == "done":
                self.review_requested.emit(it["session_id"])
                return
        QMessageBox.information(
            self, "Review",
            "Selected video isn't processed yet. Start the batch first.",
        )

    def refresh(self):
        self._reload_models()
        self._on_queue_changed()

    def shutdown(self):
        self.queue.shutdown()
        if self._active_folder is not None:
            self._registry["active_folder"] = str(self._active_folder)
            self.dm.save_batch_registry(self._registry)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# small helper
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")
