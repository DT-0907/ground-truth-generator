"""
Cross-session search dialog — find tracks across all sessions by class,
ROI membership, time-of-day, confidence range, or track length.

Pure-data scan over track JSON, no video reads. Results link back to a
review window.
"""
from __future__ import annotations
import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QCheckBox,
)

from cctv_yolo.analytics import bbox_in_roi
from cctv_yolo.visual_search import VisualIndex


from cctv_yolo.theme import (
    INDIGO as BG, PANEL, BORDER, PURPLE as ACCENT, OFFWHITE as TEXT,
    PINK,
    INDIGO,
    OFFWHITE,
    PURPLE,
)

ACTION_BTN = f"""
QPushButton {{
    background-color: {ACCENT};
    color: ;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton:hover {{ background-color: ; }}
"""


def search_tracks(
    data_manager,
    cls: str | None = None,
    roi_name: str | None = None,
    min_conf: float | None = None,
    max_conf: float | None = None,
    min_length_frames: int | None = None,
    needs_review_only: bool = False,
) -> list[dict]:
    """Return a flat list of matching tracks across all sessions."""
    out = []
    for s in data_manager.get_sessions():
        sid = s["id"]
        data = data_manager.load_session_data(sid)
        if not data:
            continue
        rois = data.get("rois", []) or []
        roi_lookup = {r.get("name"): r for r in rois if r.get("name")}
        target_roi = roi_lookup.get(roi_name) if roi_name else None

        fps = float(data.get("fps", 30.0)) or 30.0

        for tr in data.get("tracks", []):
            tcls = tr.get("class")
            if cls and cls != "any" and tcls != cls:
                continue
            avg = tr.get("avg_confidence", 0.0)
            if min_conf is not None and avg < min_conf:
                continue
            if max_conf is not None and avg > max_conf:
                continue
            length = (tr.get("end_frame", 0) - tr.get("start_frame", 0))
            if min_length_frames is not None and length < min_length_frames:
                continue
            if needs_review_only and not tr.get("needs_review"):
                continue
            if target_roi:
                if not any(bbox_in_roi(f["bbox"], target_roi) for f in tr.get("frames", [])):
                    continue
            start_sec = tr.get("start_frame", 0) / fps
            out.append({
                "session_id": sid,
                "video_name": s["video_name"],
                "track_id": tr.get("track_id"),
                "class": tcls,
                "subclass": tr.get("subclass") or "",
                "avg_conf": round(avg, 3),
                "length": length,
                "start_sec": round(start_sec, 2),
                "needs_review": bool(tr.get("needs_review")),
            })
    return out


class CrossSessionSearchDialog(QDialog):
    """Modal search dialog. Emits *open_review(session_id)* via the
    callback when the user clicks a row's "open" button."""

    open_review = Signal(str)

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self._visual_index = None  # lazy
        self.setWindowTitle("Cross-session search")
        self.resize(900, 640)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Filters
        f_box = QGroupBox("Filters")
        f_layout = QHBoxLayout(f_box)

        f_layout.addWidget(QLabel("Class:"))
        self.cls = QComboBox()
        from cctv_yolo import classes as class_registry
        self.cls.addItems(["any"] + list(class_registry.class_names()))
        f_layout.addWidget(self.cls)

        f_layout.addWidget(QLabel("ROI name:"))
        self.roi_name = QLineEdit()
        self.roi_name.setPlaceholderText("(any)")
        f_layout.addWidget(self.roi_name)

        f_layout.addWidget(QLabel("Min conf:"))
        self.min_conf = QDoubleSpinBox()
        self.min_conf.setRange(0.0, 1.0)
        self.min_conf.setSingleStep(0.05)
        self.min_conf.setValue(0.0)
        f_layout.addWidget(self.min_conf)

        f_layout.addWidget(QLabel("Max conf:"))
        self.max_conf = QDoubleSpinBox()
        self.max_conf.setRange(0.0, 1.0)
        self.max_conf.setSingleStep(0.05)
        self.max_conf.setValue(1.0)
        f_layout.addWidget(self.max_conf)

        f_layout.addWidget(QLabel("Min length (frames):"))
        self.min_len = QSpinBox()
        self.min_len.setRange(0, 100000)
        self.min_len.setValue(0)
        f_layout.addWidget(self.min_len)

        self.needs_review = QCheckBox("Only needs-review")
        f_layout.addWidget(self.needs_review)

        f_layout.addStretch()
        btn_search = QPushButton("Search")
        btn_search.setStyleSheet(ACTION_BTN)
        btn_search.clicked.connect(self._do_search)
        f_layout.addWidget(btn_search)

        layout.addWidget(f_box)

        # Visual search panel (uses CLIP / ResNet embedding index)
        v_box = QGroupBox("Visual search (text or by track)")
        v_layout = QHBoxLayout(v_box)
        self.visual_query = QLineEdit()
        self.visual_query.setPlaceholderText(
            "e.g. 'red truck' (text-only available with open_clip)"
        )
        v_layout.addWidget(self.visual_query, stretch=1)
        btn_visual = QPushButton("Visual Search")
        btn_visual.setStyleSheet(ACTION_BTN)
        btn_visual.clicked.connect(self._do_visual_search)
        v_layout.addWidget(btn_visual)
        btn_rebuild = QPushButton("Rebuild Index")
        btn_rebuild.clicked.connect(self._rebuild_index)
        v_layout.addWidget(btn_rebuild)
        self.visual_status = QLabel("")
        self.visual_status.setStyleSheet("color:#aaa;")
        v_layout.addWidget(self.visual_status)
        layout.addWidget(v_box)

        # Results
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Session", "Track", "Class", "Subclass",
            "Avg conf", "Length", "Start (s)", "Needs review",
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 8):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemDoubleClicked.connect(self._on_double)
        self.table.setStyleSheet(
            f"QTableWidget {{ background-color: {PANEL}; color: {TEXT}; "
            f"gridline-color: {BORDER}; border: 1px solid {BORDER}; }}"
            f"QHeaderView::section {{ background-color: {PANEL}; color: {ACCENT}; "
            f"border: 0; padding: 4px; }}"
        )
        layout.addWidget(self.table, stretch=1)

        # Actions
        btn_row = QHBoxLayout()
        btn_open = QPushButton("Open Review for Selected")
        btn_open.setStyleSheet(ACTION_BTN)
        btn_open.clicked.connect(self._open_selected)
        btn_row.addWidget(btn_open)

        self.count_label = QLabel("0 matches")
        self.count_label.setStyleSheet("color:#aaa;")
        btn_row.addWidget(self.count_label)
        btn_row.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _do_search(self):
        results = search_tracks(
            self.dm,
            cls=self.cls.currentText(),
            roi_name=self.roi_name.text().strip() or None,
            min_conf=self.min_conf.value(),
            max_conf=self.max_conf.value() if self.max_conf.value() < 1.0 else None,
            min_length_frames=self.min_len.value() or None,
            needs_review_only=self.needs_review.isChecked(),
        )
        self.table.setRowCount(len(results))
        for i, r in enumerate(results):
            sess = QTableWidgetItem(r["video_name"])
            sess.setData(Qt.UserRole, r["session_id"])
            self.table.setItem(i, 0, sess)
            self.table.setItem(i, 1, QTableWidgetItem(str(r["track_id"])))
            self.table.setItem(i, 2, QTableWidgetItem(r["class"]))
            self.table.setItem(i, 3, QTableWidgetItem(r["subclass"]))
            self.table.setItem(i, 4, QTableWidgetItem(f"{r['avg_conf']:.3f}"))
            self.table.setItem(i, 5, QTableWidgetItem(str(r["length"])))
            self.table.setItem(i, 6, QTableWidgetItem(f"{r['start_sec']:.2f}"))
            yes_no = "✓" if r["needs_review"] else ""
            self.table.setItem(i, 7, QTableWidgetItem(yes_no))
        self.count_label.setText(f"{len(results)} matches")

    def _open_selected(self):
        row = self.table.currentRow()
        if row < 0:
            return
        cell = self.table.item(row, 0)
        if cell:
            sid = cell.data(Qt.UserRole)
            if sid:
                self.open_review.emit(sid)

    def _on_double(self, item):
        row = item.row()
        cell = self.table.item(row, 0)
        if cell:
            sid = cell.data(Qt.UserRole)
            if sid:
                self.open_review.emit(sid)

    # ----- Visual search -----
    def _ensure_index(self) -> VisualIndex:
        if self._visual_index is None:
            self._visual_index = VisualIndex(self.dm)
        return self._visual_index

    def _rebuild_index(self):
        from PySide6.QtWidgets import QProgressDialog
        idx = self._ensure_index()
        prog = QProgressDialog("Building visual index…", "Cancel", 0, 100, self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setValue(0)

        def cb(p):
            prog.setValue(p)

        try:
            stats = idx.build(progress_callback=cb)
        finally:
            prog.close()
        if stats.get("error"):
            self.visual_status.setText(stats["error"])
            return
        self.visual_status.setText(
            f"index: {stats['entries']} crops · backend: {stats['backend']}"
        )

    def _do_visual_search(self):
        idx = self._ensure_index()
        if idx.is_empty():
            self.visual_status.setText(
                "Index empty — click 'Rebuild Index' first."
            )
            return
        q = self.visual_query.text().strip()
        if not q:
            self.visual_status.setText("Enter a text query (e.g. 'red truck').")
            return
        if not idx.backend.supports_text:
            self.visual_status.setText(
                "Current backend (" + idx.backend.name + ") does not support "
                "text queries. Install open-clip-torch for CLIP support."
            )
            return
        results = idx.query_text(q, k=50)
        self._populate_visual_results(results)

    def _populate_visual_results(self, results):
        # Reuse the main results table layout
        self.table.setRowCount(len(results))
        for i, (score, e) in enumerate(results):
            sess = QTableWidgetItem(e.session_id)
            sess.setData(Qt.UserRole, e.session_id)
            self.table.setItem(i, 0, sess)
            self.table.setItem(i, 1, QTableWidgetItem(str(e.track_id)))
            self.table.setItem(i, 2, QTableWidgetItem(e.cls))
            self.table.setItem(i, 3, QTableWidgetItem(e.subclass))
            self.table.setItem(i, 4, QTableWidgetItem(f"{score:.3f}"))
            self.table.setItem(i, 5, QTableWidgetItem(""))
            self.table.setItem(i, 6, QTableWidgetItem(str(e.frame)))
            self.table.setItem(i, 7, QTableWidgetItem(""))
        self.count_label.setText(f"{len(results)} visual matches")
