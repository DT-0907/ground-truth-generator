"""
Occlusion-suggestions dialog — shows ranked merge candidates produced
by ``occlusion.find_gap_candidates``. The user accepts / rejects each
one; accepted pairs route through the existing merge logic so gap
interpolation and frame dedup happen exactly as for a manual merge.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QMessageBox,
)

from cctv_yolo.occlusion import find_gap_candidates


ACCENT = "#4ecca3"
PANEL = "#16213e"
BORDER = "#2d3a5a"
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


class OcclusionSuggestionsDialog(QDialog):
    """Modal dialog. Emits ``apply(track_a, track_b)`` for each pair the
    user accepts.
    """

    apply_pair = Signal(int, int)
    show_pair = Signal(int)  # focus on a track in the review window

    def __init__(self, track_data: dict, parent=None):
        super().__init__(parent)
        self.track_data = track_data
        self.setWindowTitle("Occlusion / track-gap suggestions")
        self.resize(760, 520)
        self._setup_ui()
        self._scan()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Filters
        f_box = QGroupBox("Detection thresholds")
        f_lay = QHBoxLayout(f_box)
        f_lay.addWidget(QLabel("Max gap (frames):"))
        self.max_gap = QSpinBox()
        self.max_gap.setRange(1, 600)
        self.max_gap.setValue(90)
        f_lay.addWidget(self.max_gap)
        f_lay.addWidget(QLabel("Max prediction offset (px):"))
        self.max_offset = QDoubleSpinBox()
        self.max_offset.setRange(5.0, 1000.0)
        self.max_offset.setSingleStep(10.0)
        self.max_offset.setValue(120.0)
        f_lay.addWidget(self.max_offset)
        f_lay.addWidget(QLabel("Min score:"))
        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(0.0, 1.0)
        self.min_score.setSingleStep(0.05)
        self.min_score.setValue(0.35)
        f_lay.addWidget(self.min_score)
        self.same_class = QCheckBox("Require same class")
        self.same_class.setChecked(True)
        f_lay.addWidget(self.same_class)
        btn_rescan = QPushButton("Rescan")
        btn_rescan.setStyleSheet(ACTION_BTN)
        btn_rescan.clicked.connect(self._scan)
        f_lay.addWidget(btn_rescan)
        layout.addWidget(f_box)

        # Table
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Score", "A → B", "Same class", "Gap (frames)",
            "Predicted offset", "Spatial dist", "End / start frames", "Action",
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for c in range(2, 8):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setStyleSheet(
            f"QTableWidget {{ background-color: {PANEL}; color: {TEXT}; "
            f"gridline-color: {BORDER}; border: 1px solid {BORDER}; }}"
            f"QHeaderView::section {{ background-color: {PANEL}; color: {ACCENT}; "
            f"border: 0; padding: 4px; font-weight: bold; }}"
        )
        layout.addWidget(self.table, stretch=1)

        # Help
        hint = QLabel(
            "Click <b>Merge</b> to accept a suggestion (interpolates the gap "
            "and marks those frames as <i>occluded</i>). Click a row to focus "
            "on track A in the review window."
        )
        hint.setStyleSheet("color:#aaa;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Footer
        foot = QHBoxLayout()
        self.summary = QLabel("0 candidates")
        self.summary.setStyleSheet("color:#aaa;")
        foot.addWidget(self.summary)
        foot.addStretch()

        btn_apply_all = QPushButton("Apply all visible")
        btn_apply_all.setStyleSheet(ACTION_BTN)
        btn_apply_all.clicked.connect(self._apply_all)
        foot.addWidget(btn_apply_all)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        foot.addWidget(btn_close)
        layout.addLayout(foot)

        self.table.itemSelectionChanged.connect(self._on_select)

    def _scan(self):
        self._candidates = find_gap_candidates(
            self.track_data,
            max_gap_frames=self.max_gap.value(),
            max_predicted_offset=self.max_offset.value(),
            min_score=self.min_score.value(),
            require_same_class=self.same_class.isChecked(),
        )
        self.table.setRowCount(len(self._candidates))
        for r, c in enumerate(self._candidates):
            self.table.setItem(r, 0, QTableWidgetItem(f"{c.score:.3f}"))
            it = QTableWidgetItem(f"#{c.track_a} → #{c.track_b}")
            it.setData(Qt.UserRole, (c.track_a, c.track_b))
            self.table.setItem(r, 1, it)
            self.table.setItem(r, 2, QTableWidgetItem("yes" if c.same_class else "no"))
            self.table.setItem(r, 3, QTableWidgetItem(str(c.gap_frames)))
            self.table.setItem(r, 4, QTableWidgetItem(f"{c.velocity_distance_px:.0f} px"))
            self.table.setItem(r, 5, QTableWidgetItem(f"{c.spatial_distance_px:.0f} px"))
            self.table.setItem(
                r, 6, QTableWidgetItem(f"{c.a_end[0]} → {c.b_start[0]}"))

            btn = QPushButton("Merge")
            btn.setStyleSheet(ACTION_BTN)
            btn.clicked.connect(
                lambda _checked, a=c.track_a, b=c.track_b, row=r:
                    self._apply_one(a, b, row))
            self.table.setCellWidget(r, 7, btn)

        self.summary.setText(f"{len(self._candidates)} candidates")

    def _apply_one(self, a: int, b: int, row: int):
        self.apply_pair.emit(a, b)
        # Visually mark the row as merged
        for c in range(self.table.columnCount() - 1):
            it = self.table.item(row, c)
            if it:
                it.setForeground(Qt.GlobalColor.darkGray)
        widget = self.table.cellWidget(row, self.table.columnCount() - 1)
        if widget:
            widget.setEnabled(False)
            widget.setText("merged")

    def _apply_all(self):
        if self.table.rowCount() == 0:
            return
        reply = QMessageBox.question(
            self, "Apply all",
            f"Apply all {self.table.rowCount()} suggestions? "
            "Lower-scoring ones may be wrong — review first if unsure.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        # Apply in score-desc order, but keep applying after merges in
        # case multiple suggestions involve the same track.
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 1)
            if it is None:
                continue
            pair = it.data(Qt.UserRole)
            if not pair:
                continue
            self._apply_one(pair[0], pair[1], r)

    def _on_select(self):
        rows = self.table.selectedItems()
        if not rows:
            return
        first = self.table.item(rows[0].row(), 1)
        if first:
            pair = first.data(Qt.UserRole)
            if pair:
                self.show_pair.emit(pair[0])
