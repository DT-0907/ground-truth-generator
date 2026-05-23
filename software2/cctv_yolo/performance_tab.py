"""
Performance tab — traffic counts, detection statistics, A/B compare,
before/after, and confusion matrix. PRD Part G.

Layout
------
- Header row: title + Refresh + OpenLocationBar (PRD C12)
- Single/Group toggle: switch between per-session and aggregate views
- Session/Group dropdown + provenance badge ("Showing: Corrections (N edits)")
- ROI drawing panel (single mode only)
- Stat cards + breakdown tables (hidden when no selection)
- Analyses: three CollapsibleSections (Model A/B Compare, Before/After,
  Confusion Matrix)
- Group membership panel (Group mode only)
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices

from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from cctv_yolo.before_after import BeforeAfterWorker
from cctv_yolo.dialogs import RoiNameDialog
from cctv_yolo.metrics import compute_confusion_matrix
from cctv_yolo.model_compare import ModelCompareWorker
from cctv_yolo.theme import (
    BORDER,
    CLASS_COLORS,
    ERROR,
    INDIGO as BG,
    OFFWHITE as TEXT,
    PANEL,
    PINK,
    PURPLE as ACCENT,
    RADIUS,
    ROI_COLOR_ROTATION,
    TEXT_MUTED,
    INDIGO,
    OFFWHITE,
    PURPLE,
)
from cctv_yolo.video_canvas import VideoCanvas
from cctv_yolo.widgets.collapsible_section import CollapsibleSection
from cctv_yolo.widgets.group_picker_dialog import GroupPickerDialog
from cctv_yolo.widgets.open_location_bar import OpenLocationBar, open_path

logger = logging.getLogger(__name__)

VEHICLE_ORDER = ["car", "truck", "bus", "motorcycle", "bicycle"]


# ---------------------------------------------------------------------------
# Stylesheets (kept compact — theme tokens used everywhere)
# ---------------------------------------------------------------------------

CONTROLS_STYLE = f"""
QComboBox {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS - 2}px;
    padding: 6px 10px;
    min-width: 250px;
    font-size: 13px;
}}
QComboBox::drop-down {{
    border: none;
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: rgba(152, 37, 152, 0.30);
    selection-color: {TEXT};
}}
QSpinBox, QDoubleSpinBox {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS - 2}px;
    padding: 5px 8px;
    font-size: 13px;
    min-width: 70px;
}}
"""

PRIMARY_BTN = f"""
QPushButton {{
    background-color: {ACCENT};
    color: {TEXT};
    border: none;
    border-radius: {RADIUS - 2}px;
    padding: 8px 20px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {PINK};
    color: {BG};
}}
QPushButton:disabled {{
    background-color: {BORDER};
    color: {TEXT_MUTED};
}}
"""

REFRESH_BTN_STYLE = f"""
QPushButton {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS - 2}px;
    padding: 6px 16px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: rgba(228, 145, 201, 0.15);
}}
"""

TABLE_STYLE = f"""
QTableWidget {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    gridline-color: {BORDER};
    font-size: 13px;
}}
QTableWidget::item {{
    padding: 8px 12px;
    border-bottom: 1px solid {BORDER};
}}
QTableWidget::item:selected {{
    background-color: rgba(152, 37, 152, 0.30);
    color: {TEXT};
}}
QHeaderView::section {{
    background-color: {BG};
    color: {ACCENT};
    border: none;
    border-bottom: 2px solid {ACCENT};
    padding: 8px 12px;
    font-weight: bold;
    font-size: 12px;
}}
QTableWidget QTableCornerButton::section {{
    background-color: {BG};
    border: none;
}}
"""

SECTION_LABEL = f"""
QLabel {{
    color: {TEXT};
    font-size: 16px;
    font-weight: bold;
    padding: 8px 0 4px 0;
}}
"""

SECONDARY_BTN = f"""
QPushButton {{
    background-color: transparent;
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: {RADIUS - 2}px;
    padding: 4px 12px;
    font-size: 11px;
}}
QPushButton:hover {{
    background-color: {ACCENT};
    color: {TEXT};
}}
QPushButton:disabled {{
    border-color: {BORDER};
    color: {TEXT_MUTED};
}}
"""

ROI_BTN_ACTIVE = f"""
QPushButton {{
    background-color: {ACCENT};
    color: {TEXT};
    border: 1px solid {ACCENT};
    border-radius: {RADIUS - 2}px;
    padding: 4px 12px;
    font-weight: bold;
    font-size: 11px;
}}
"""

BADGE_INFO = f"""
QLabel {{
    color: {TEXT};
    background-color: rgba(152, 37, 152, 0.20);
    border: 1px solid {ACCENT};
    border-radius: {RADIUS - 2}px;
    padding: 4px 10px;
    font-size: 11px;
}}
"""

BADGE_WARN = f"""
QLabel {{
    color: {TEXT};
    background-color: rgba(255, 107, 122, 0.18);
    border: 1px solid {ERROR};
    border-radius: {RADIUS - 2}px;
    padding: 8px 12px;
    font-size: 12px;
}}
"""

LOG_STYLE = f"""
QTextEdit {{
    background-color: {BG};
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
    border-radius: {RADIUS - 2}px;
    font-family: 'Menlo', 'Consolas', monospace;
    font-size: 11px;
}}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vehicle_color(cls: str) -> str:
    """Look up the CLASS_COLORS dict with a graceful fallback."""
    return CLASS_COLORS.get(cls, ACCENT)


def _ui_state_path(data_manager) -> Path:
    return data_manager.config_dir / "ui_state.json"


def _load_ui_state(data_manager) -> dict:
    p = _ui_state_path(data_manager)
    if not p.exists():
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_ui_state(data_manager, state: dict) -> None:
    p = _ui_state_path(data_manager)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(p, "w") as f:
            json.dump(state, f, indent=2)
    except OSError as e:
        logger.warning("Couldn't persist ui_state.json: %s", e)


# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------

class PerformanceTab(QWidget):
    """Performance tab — traffic counts and stats for processed sessions."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.dm = data_manager  # short alias used throughout
        self._current_stats: dict = {}
        self._performance_rois: list[dict] = []
        self._current_session_id: str | None = None
        self._current_group_id: str | None = None
        self._mode: str = "single"  # "single" or "group"
        self._provenance: str = "tracks"  # "corrections" or "tracks"

        # Workers (held to keep references alive)
        self._compare_worker: ModelCompareWorker | None = None
        self._beforeafter_worker: BeforeAfterWorker | None = None

        self._build_ui()
        self._populate_sessions()

        # Subscribe to data-manager signals (PRD F4-1, Part M)
        try:
            self.dm.corrections_changed.connect(self._on_corrections_changed)
        except Exception:
            pass
        try:
            self.dm.groups_changed.connect(self._on_groups_changed)
        except Exception:
            pass

        # Restore last session
        state = _load_ui_state(self.dm)
        last_sid = state.get("last_performance_session")
        if last_sid:
            for i in range(self.session_combo.count()):
                if self.session_combo.itemData(i) == last_sid:
                    self.session_combo.setCurrentIndex(i)
                    break

    # ------------------------------------------------------------------
    # UI BUILD
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # --- Header row ----------------------------------------------------
        header_row = QHBoxLayout()
        title = QLabel("Performance")
        title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {TEXT};")
        header_row.addWidget(title)
        header_row.addStretch()

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setStyleSheet(REFRESH_BTN_STYLE)
        self.btn_refresh.clicked.connect(self.refresh)
        header_row.addWidget(self.btn_refresh)
        layout.addLayout(header_row)

        # OpenLocationBar (PRD C12) — top-right chip row
        ol = OpenLocationBar(self)
        ol.add_folder("Tracks Folder", lambda: self.dm.tracks_dir)
        ol.add_folder("Corrections Folder", lambda: self.dm.corrections_dir)
        ol.add_folder("Reports", lambda: self.dm.exports_dir / "reports")
        ol.add_folder("Models", lambda: self.dm.models_dir)
        layout.addWidget(ol)

        # --- Single/Group toggle ------------------------------------------
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(16)

        self.radio_single = QRadioButton("Single session")
        self.radio_single.setChecked(True)
        self.radio_single.setStyleSheet(f"color: {TEXT}; font-size: 13px;")
        self.radio_group = QRadioButton("Group")
        self.radio_group.setStyleSheet(f"color: {TEXT}; font-size: 13px;")

        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.radio_single)
        self._mode_group.addButton(self.radio_group)
        self.radio_single.toggled.connect(self._on_mode_toggled)

        toggle_row.addWidget(self.radio_single)
        toggle_row.addWidget(self.radio_group)
        toggle_row.addStretch()

        self.btn_new_group = QPushButton("+ New Group")
        self.btn_new_group.setStyleSheet(SECONDARY_BTN)
        self.btn_new_group.clicked.connect(self._on_new_group)
        toggle_row.addWidget(self.btn_new_group)

        self.btn_edit_group = QPushButton("Edit Group")
        self.btn_edit_group.setStyleSheet(SECONDARY_BTN)
        self.btn_edit_group.clicked.connect(self._on_edit_group)
        self.btn_edit_group.setVisible(False)
        toggle_row.addWidget(self.btn_edit_group)

        layout.addLayout(toggle_row)

        # --- Selector row --------------------------------------------------
        selector_row = QHBoxLayout()
        selector_row.setSpacing(10)

        self.selector_label = QLabel("Session:")
        self.selector_label.setStyleSheet(f"color: {TEXT};")
        selector_row.addWidget(self.selector_label)

        self.session_combo = QComboBox()
        self.session_combo.setStyleSheet(CONTROLS_STYLE)
        self.session_combo.currentIndexChanged.connect(self._on_session_changed)
        selector_row.addWidget(self.session_combo)

        # Group combo (built alongside; hidden initially)
        self.group_combo = QComboBox()
        self.group_combo.setStyleSheet(CONTROLS_STYLE)
        self.group_combo.currentIndexChanged.connect(self._on_group_changed)
        self.group_combo.setVisible(False)
        selector_row.addWidget(self.group_combo)

        selector_row.addStretch()

        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.setStyleSheet(PRIMARY_BTN)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self._export_csv)
        selector_row.addWidget(self.btn_export_csv)

        layout.addLayout(selector_row)

        # Provenance badge
        self.badge_provenance = QLabel("")
        self.badge_provenance.setStyleSheet(BADGE_INFO)
        self.badge_provenance.setVisible(False)
        layout.addWidget(self.badge_provenance)

        # Empty-detection warning banner (G2-2)
        self.banner_empty = QLabel(
            "This session was processed but no vehicles were detected. "
            "Try a lower confidence threshold and reprocess."
        )
        self.banner_empty.setStyleSheet(BADGE_WARN)
        self.banner_empty.setWordWrap(True)
        self.banner_empty.setVisible(False)
        layout.addWidget(self.banner_empty)

        # --- ROI drawing panel --------------------------------------------
        self.roi_panel = self._build_roi_panel()
        self.roi_panel.setVisible(False)
        layout.addWidget(self.roi_panel, stretch=1)

        # --- Scrollable content -------------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background-color: {BG}; border: none; }}")

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(16)

        # Placeholder (G2-1 — shown when nothing selected)
        self.lbl_placeholder = QLabel("Select a session to view performance statistics.")
        self.lbl_placeholder.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 14px; padding: 24px;"
        )
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(self.lbl_placeholder)

        # Stat-cards container (hidden until session selected — G2-1)
        self.cards_container = QWidget()
        cards_layout = QVBoxLayout(self.cards_container)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(12)
        self._stat_cards_row = QHBoxLayout()
        self._stat_cards_row.setSpacing(12)
        cards_layout.addLayout(self._stat_cards_row)

        self._tables_layout = QVBoxLayout()
        self._tables_layout.setSpacing(12)
        cards_layout.addLayout(self._tables_layout)

        self.cards_container.setVisible(False)
        self.content_layout.addWidget(self.cards_container)

        # --- Group membership panel (visible only in group mode) ----------
        self.group_membership_section = CollapsibleSection(
            "Sessions in this group", expanded=False
        )
        self._build_group_membership_body()
        self.group_membership_section.setVisible(False)
        self.content_layout.addWidget(self.group_membership_section)

        # --- Analyses sub-panels ------------------------------------------
        analyses_label = QLabel("Analyses")
        analyses_label.setStyleSheet(SECTION_LABEL)
        self.content_layout.addWidget(analyses_label)

        self.section_compare = CollapsibleSection("Model A/B Compare", expanded=False)
        self._build_compare_body()
        self.content_layout.addWidget(self.section_compare)

        self.section_beforeafter = CollapsibleSection(
            "Before / After Renderer", expanded=False
        )
        self._build_beforeafter_body()
        self.content_layout.addWidget(self.section_beforeafter)

        self.section_confusion = CollapsibleSection("Confusion Matrix", expanded=False)
        self._build_confusion_body()
        self.content_layout.addWidget(self.section_confusion)

        self.content_layout.addStretch()
        scroll.setWidget(self.content_widget)
        layout.addWidget(scroll, stretch=2)

    # ------------------------------------------------------------------
    # ROI panel
    # ------------------------------------------------------------------

    def _build_roi_panel(self) -> QFrame:
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {PANEL};
                border: 1px solid {BORDER};
                border-radius: {RADIUS}px;
            }}
        """)
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("Define ROI for Statistics")
        title.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {TEXT}; border: none;"
        )
        header.addWidget(title)
        header.addStretch()

        self.btn_roi_rect = QPushButton("Draw Rect ROI")
        self.btn_roi_rect.setStyleSheet(SECONDARY_BTN)
        self.btn_roi_rect.setCheckable(True)
        self.btn_roi_rect.clicked.connect(self._on_roi_rect_mode)
        header.addWidget(self.btn_roi_rect)

        self.btn_roi_poly = QPushButton("Draw Polygon ROI")
        self.btn_roi_poly.setStyleSheet(SECONDARY_BTN)
        self.btn_roi_poly.setCheckable(True)
        self.btn_roi_poly.clicked.connect(self._on_roi_poly_mode)
        header.addWidget(self.btn_roi_poly)

        self.btn_roi_clear = QPushButton("Clear All ROIs")
        self.btn_roi_clear.setStyleSheet(SECONDARY_BTN)
        self.btn_roi_clear.clicked.connect(self._on_roi_clear_all)
        header.addWidget(self.btn_roi_clear)
        v.addLayout(header)

        self.perf_canvas = VideoCanvas()
        self.perf_canvas.setMinimumHeight(350)
        self.perf_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.perf_canvas.roi_rect_drawn.connect(self._on_perf_roi_rect_drawn)
        self.perf_canvas.roi_polygon_drawn.connect(self._on_perf_roi_polygon_drawn)
        v.addWidget(self.perf_canvas, stretch=1)

        self.roi_status = QLabel("Draw ROIs on the frame to calculate per-region statistics.")
        self.roi_status.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; border: none;"
        )
        v.addWidget(self.roi_status)
        return panel

    # ------------------------------------------------------------------
    # Stat card factory
    # ------------------------------------------------------------------

    def _make_stat_card(self, label_text: str, value_text: str,
                        color: str | None = None) -> dict:
        accent = color or ACCENT
        frame = QFrame()
        frame.setObjectName("statCard")
        frame.setStyleSheet(f"""
        QFrame#statCard {{
            background-color: {PANEL};
            border: 1px solid {BORDER};
            border-top: 2px solid {accent};
            border-radius: {RADIUS}px;
            padding: 12px;
        }}
        QFrame#statCard QLabel {{
            background: transparent;
            border: none;
            padding: 0;
        }}
        """)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        frame.setFixedHeight(110)

        v = QVBoxLayout(frame)
        v.setContentsMargins(16, 10, 16, 14)
        v.setSpacing(2)

        lbl = QLabel(label_text.upper())
        lbl.setStyleSheet(f"color: {TEXT}; font-size: 11px; letter-spacing: 1px;")
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lbl.setFixedHeight(18)

        val = QLabel(value_text)
        val.setStyleSheet(f"color: {accent}; font-size: 32px; font-weight: bold;")
        val.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        v.addWidget(lbl)
        v.addWidget(val, stretch=1)
        return {"frame": frame, "value_label": val}

    # ------------------------------------------------------------------
    # Analyses panels — bodies
    # ------------------------------------------------------------------

    def _build_compare_body(self) -> None:
        """G3-1 Model A/B Compare body."""
        body = self.section_compare.body_layout()

        controls = QGridLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("Model A:"), 0, 0)
        self.compare_model_a = QComboBox()
        self.compare_model_a.setStyleSheet(CONTROLS_STYLE)
        controls.addWidget(self.compare_model_a, 0, 1)

        controls.addWidget(QLabel("Model B:"), 0, 2)
        self.compare_model_b = QComboBox()
        self.compare_model_b.setStyleSheet(CONTROLS_STYLE)
        controls.addWidget(self.compare_model_b, 0, 3)

        controls.addWidget(QLabel("Session:"), 1, 0)
        self.compare_session = QComboBox()
        self.compare_session.setStyleSheet(CONTROLS_STYLE)
        controls.addWidget(self.compare_session, 1, 1)

        controls.addWidget(QLabel("Stride:"), 1, 2)
        self.compare_stride = QSpinBox()
        self.compare_stride.setRange(1, 10)
        self.compare_stride.setValue(1)
        self.compare_stride.setStyleSheet(CONTROLS_STYLE)
        controls.addWidget(self.compare_stride, 1, 3)

        controls.addWidget(QLabel("Confidence:"), 2, 0)
        self.compare_conf = QDoubleSpinBox()
        self.compare_conf.setRange(0.05, 0.95)
        self.compare_conf.setSingleStep(0.05)
        self.compare_conf.setValue(0.25)
        self.compare_conf.setStyleSheet(CONTROLS_STYLE)
        controls.addWidget(self.compare_conf, 2, 1)

        self.btn_compare_run = QPushButton("Run Compare")
        self.btn_compare_run.setStyleSheet(PRIMARY_BTN)
        self.btn_compare_run.clicked.connect(self._on_run_compare)
        controls.addWidget(self.btn_compare_run, 2, 3)

        body.addLayout(controls)

        self.compare_status = QLabel("Idle.")
        self.compare_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        body.addWidget(self.compare_status)

        self.compare_log = QTextEdit()
        self.compare_log.setReadOnly(True)
        self.compare_log.setStyleSheet(LOG_STYLE)
        self.compare_log.setFixedHeight(80)
        body.addWidget(self.compare_log)

        # Results table
        self.compare_table = QTableWidget()
        self.compare_table.setStyleSheet(TABLE_STYLE)
        self.compare_table.setColumnCount(4)
        self.compare_table.setHorizontalHeaderLabels(["Metric", "A", "B", "Delta (B - A)"])
        self.compare_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.compare_table.verticalHeader().setVisible(False)
        self.compare_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        body.addWidget(self.compare_table)

        # Bottom buttons
        btn_row = QHBoxLayout()
        self.btn_export_compare = QPushButton("Export Compare CSV")
        self.btn_export_compare.setStyleSheet(SECONDARY_BTN)
        self.btn_export_compare.setEnabled(False)
        self.btn_export_compare.clicked.connect(self._export_compare_csv)
        btn_row.addWidget(self.btn_export_compare)

        self.btn_promote_b = QPushButton("Promote Model B to Active")
        self.btn_promote_b.setStyleSheet(SECONDARY_BTN)
        self.btn_promote_b.setEnabled(False)
        self.btn_promote_b.clicked.connect(self._on_promote_b)
        btn_row.addWidget(self.btn_promote_b)
        btn_row.addStretch()
        body.addLayout(btn_row)

        # Cached result
        self._last_compare_result: dict | None = None

        self._refresh_compare_pickers()

    def _build_beforeafter_body(self) -> None:
        """G3-2 Before/After renderer body."""
        body = self.section_beforeafter.body_layout()

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Filter to ROI:"))
        self.beforeafter_roi = QComboBox()
        self.beforeafter_roi.setStyleSheet(CONTROLS_STYLE)
        self.beforeafter_roi.addItem("(no filter)", None)
        controls.addWidget(self.beforeafter_roi)
        controls.addStretch()

        self.btn_render_ba = QPushButton("Render Side-by-Side")
        self.btn_render_ba.setStyleSheet(PRIMARY_BTN)
        self.btn_render_ba.clicked.connect(self._on_render_beforeafter)
        controls.addWidget(self.btn_render_ba)
        body.addLayout(controls)

        self.beforeafter_status = QLabel("Idle.")
        self.beforeafter_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        body.addWidget(self.beforeafter_status)

        out_row = QHBoxLayout()
        self.btn_ba_open = QPushButton("Open in Finder")
        self.btn_ba_open.setStyleSheet(SECONDARY_BTN)
        self.btn_ba_open.setEnabled(False)
        self.btn_ba_open.clicked.connect(self._on_open_ba_output)
        out_row.addWidget(self.btn_ba_open)

        self.btn_ba_play = QPushButton("Play")
        self.btn_ba_play.setStyleSheet(SECONDARY_BTN)
        self.btn_ba_play.setEnabled(False)
        self.btn_ba_play.clicked.connect(self._on_play_ba_output)
        out_row.addWidget(self.btn_ba_play)
        out_row.addStretch()
        body.addLayout(out_row)

        self._last_ba_path: Path | None = None

    def _build_confusion_body(self) -> None:
        """G3-3 Confusion matrix body."""
        body = self.section_confusion.body_layout()

        controls = QHBoxLayout()
        controls.addWidget(QLabel("ROI filter:"))
        self.confusion_roi = QComboBox()
        self.confusion_roi.setStyleSheet(CONTROLS_STYLE)
        self.confusion_roi.addItem("(no filter)", None)
        controls.addWidget(self.confusion_roi)
        controls.addStretch()

        self.btn_compute_confusion = QPushButton("Compute")
        self.btn_compute_confusion.setStyleSheet(PRIMARY_BTN)
        self.btn_compute_confusion.clicked.connect(self._on_compute_confusion)
        controls.addWidget(self.btn_compute_confusion)
        body.addLayout(controls)

        self.confusion_status = QLabel(
            "Uses corrections as ground truth, raw tracks as predictions."
        )
        self.confusion_status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.confusion_status.setWordWrap(True)
        body.addWidget(self.confusion_status)

        self.confusion_matrix_widget = QTableWidget()
        self.confusion_matrix_widget.setStyleSheet(TABLE_STYLE)
        self.confusion_matrix_widget.verticalHeader().setVisible(True)
        self.confusion_matrix_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        body.addWidget(self.confusion_matrix_widget)

        self.confusion_metrics_table = QTableWidget()
        self.confusion_metrics_table.setStyleSheet(TABLE_STYLE)
        self.confusion_metrics_table.setColumnCount(5)
        self.confusion_metrics_table.setHorizontalHeaderLabels(
            ["Class", "TP", "FP", "FN", "Precision / Recall / F1"]
        )
        self.confusion_metrics_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.confusion_metrics_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.Stretch
        )
        self.confusion_metrics_table.verticalHeader().setVisible(False)
        self.confusion_metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        body.addWidget(self.confusion_metrics_table)

    # ------------------------------------------------------------------
    # Group membership body
    # ------------------------------------------------------------------

    def _build_group_membership_body(self) -> None:
        body = self.group_membership_section.body_layout()

        btn_row = QHBoxLayout()
        self.btn_add_sessions = QPushButton("+ Add Sessions")
        self.btn_add_sessions.setStyleSheet(SECONDARY_BTN)
        self.btn_add_sessions.clicked.connect(self._on_add_sessions_to_group)
        btn_row.addWidget(self.btn_add_sessions)

        self.btn_remove_selected = QPushButton("- Remove Selected")
        self.btn_remove_selected.setStyleSheet(SECONDARY_BTN)
        self.btn_remove_selected.clicked.connect(self._on_remove_sessions_from_group)
        btn_row.addWidget(self.btn_remove_selected)
        btn_row.addStretch()
        body.addLayout(btn_row)

        self.group_membership_list = QListWidget()
        self.group_membership_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {PANEL};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: {RADIUS}px;
                padding: 4px;
            }}
            QListWidget::item {{
                padding: 4px 8px;
            }}
            QListWidget::item:hover {{
                background-color: rgba(228, 145, 201, 0.15);
            }}
            QListWidget::item:selected {{
                background-color: rgba(152, 37, 152, 0.30);
                color: {TEXT};
            }}
        """)
        self.group_membership_list.setSelectionMode(QAbstractItemView.MultiSelection)
        body.addWidget(self.group_membership_list)

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def _on_mode_toggled(self, _checked: bool) -> None:
        self._mode = "single" if self.radio_single.isChecked() else "group"
        is_group = self._mode == "group"
        self.session_combo.setVisible(not is_group)
        self.group_combo.setVisible(is_group)
        self.selector_label.setText("Group:" if is_group else "Session:")
        self.btn_edit_group.setVisible(is_group)
        self.group_membership_section.setVisible(is_group)
        self.roi_panel.setVisible(False)

        if is_group:
            self._populate_groups()
        else:
            sid = self.session_combo.currentData()
            if sid:
                self._on_session_changed(self.session_combo.currentIndex())
            else:
                self._show_placeholder()

    # ------------------------------------------------------------------
    # Session combo population
    # ------------------------------------------------------------------

    def _populate_sessions(self) -> None:
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        self.session_combo.addItem("-- Select Session --", "")
        sessions = self.dm.get_sessions()
        for s in sessions:
            sid = s.get("id", "")
            display = s.get("video_name", sid)
            data = self.dm.load_session_data(sid)
            track_count = len(data.get("tracks", [])) if data else s.get("track_count", 0)
            suffix = "  (corrected)" if s.get("has_corrections") else ""
            self.session_combo.addItem(f"{display}  ({track_count} tracks{suffix})", sid)
        self.session_combo.blockSignals(False)
        if hasattr(self, "compare_session"):
            self._refresh_compare_pickers()

    def _populate_groups(self) -> None:
        self.group_combo.blockSignals(True)
        prev_id = self._current_group_id
        self.group_combo.clear()
        self.group_combo.addItem("-- Select Group --", "")
        for g in self.dm.list_groups():
            gid = g.get("id", "")
            name = g.get("name", gid)
            n = len(g.get("session_ids", []))
            self.group_combo.addItem(f"{name}  ({n} sessions)", gid)
        self.group_combo.blockSignals(False)

        if prev_id:
            for i in range(self.group_combo.count()):
                if self.group_combo.itemData(i) == prev_id:
                    self.group_combo.setCurrentIndex(i)
                    return
        self._show_placeholder()

    # ------------------------------------------------------------------
    # Session changed
    # ------------------------------------------------------------------

    def _on_session_changed(self, _index: int) -> None:
        sid = self.session_combo.currentData()
        if not sid:
            self._current_session_id = None
            self._show_placeholder()
            self.btn_export_csv.setEnabled(False)
            self.roi_panel.setVisible(False)
            self.badge_provenance.setVisible(False)
            self.banner_empty.setVisible(False)
            return

        self._current_session_id = sid

        # Persist (G2-5)
        state = _load_ui_state(self.dm)
        state["last_performance_session"] = sid
        _save_ui_state(self.dm, state)

        # Determine provenance (G2-4)
        if self.dm.has_corrections(sid):
            self._provenance = "corrections"
            corr_data = self.dm.load_corrections(sid) or {}
            edit_count = self._estimate_edit_count(corr_data)
            self.badge_provenance.setText(f"Showing: Corrections ({edit_count} edits)")
        else:
            self._provenance = "tracks"
            self.badge_provenance.setText("Showing: Raw tracks")
        self.badge_provenance.setStyleSheet(BADGE_INFO)
        self.badge_provenance.setVisible(True)

        # Load + display session
        try:
            data = self.dm.load_session_data(sid)
        except json.JSONDecodeError:
            # G2-3 — corrupt JSON
            QMessageBox.warning(
                self,
                "Couldn't read tracks for this session.",
                "The tracks/corrections file looks corrupt. Open the file "
                "location and inspect the JSON manually.",
            )
            self.dm.open_folder("tracks")
            self._show_placeholder()
            return

        if data is None:
            self._show_placeholder("No data found for this session.")
            self.btn_export_csv.setEnabled(False)
            return

        self._performance_rois = data.get("rois", []) or []

        # Open video in ROI canvas
        video_path = self.dm.get_video_path(sid)
        if video_path and video_path.exists():
            self.perf_canvas.open_video(str(video_path))
            self.perf_canvas.set_frame(0)
            self.perf_canvas.drawing_mode = "select"
            self._update_roi_display()
            self.roi_panel.setVisible(True)
        else:
            self.roi_panel.setVisible(False)

        self._refresh_roi_pickers()
        self._load_stats(sid)

    def _estimate_edit_count(self, corr_data: dict) -> int:
        if "edit_count" in corr_data:
            return int(corr_data["edit_count"])
        return len(corr_data.get("tracks", []))

    # ------------------------------------------------------------------
    # Group changed
    # ------------------------------------------------------------------

    def _on_group_changed(self, _index: int) -> None:
        gid = self.group_combo.currentData()
        if not gid:
            self._current_group_id = None
            self._show_placeholder()
            self.btn_export_csv.setEnabled(False)
            return
        self._current_group_id = gid
        self._render_group_stats(gid)
        self._refresh_group_membership()

    def _refresh_group_membership(self) -> None:
        self.group_membership_list.clear()
        gid = self._current_group_id
        if not gid:
            return
        sessions = self.dm.get_sessions_in_group(gid)
        for s in sessions:
            sid = s.get("id", "")
            label = s.get("video_name") or sid
            n = s.get("track_count", 0)
            item = QListWidgetItem(f"{label}  ({n} tracks)")
            item.setData(Qt.UserRole, sid)
            self.group_membership_list.addItem(item)
        self.group_membership_section.set_subtitle(f"{len(sessions)} sessions")

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate_group_stats(self, sessions: list[dict]) -> dict:
        """Aggregate stats across multiple session dicts.

        Per-class counts: SUM.
        Mean confidence:  WEIGHTED by total_tracks.
        """
        total_tracks = 0
        class_counts: dict[str, int] = defaultdict(int)
        weighted_conf_num = 0.0
        per_session_rows: list[dict] = []
        n_corrected = 0

        for s in sessions:
            sid = s.get("id", "")
            data = self.dm.load_session_data(sid)
            if data is None:
                continue
            tracks = data.get("tracks", [])
            n = len(tracks)
            total_tracks += n
            cls_count: dict[str, int] = defaultdict(int)
            conf_sum = 0.0
            for tr in tracks:
                cls = tr.get("class", "vehicle")
                class_counts[cls] += 1
                cls_count[cls] += 1
                conf_sum += float(tr.get("avg_confidence", 0.0))
            mean_conf = (conf_sum / n) if n else 0.0
            weighted_conf_num += n * mean_conf
            if s.get("has_corrections"):
                n_corrected += 1
            per_session_rows.append({
                "id": sid,
                "video_name": s.get("video_name", sid),
                "total_tracks": n,
                "by_class": dict(cls_count),
                "mean_conf": round(mean_conf, 3),
                "has_corrections": bool(s.get("has_corrections")),
            })

        weighted_conf = weighted_conf_num / max(1, total_tracks)
        return {
            "total_tracks": total_tracks,
            "class_counts": dict(class_counts),
            "mean_conf": round(weighted_conf, 3),
            "n_sessions": len(sessions),
            "n_corrected": n_corrected,
            "rows": per_session_rows,
        }

    # ------------------------------------------------------------------
    # Stats loading + rendering — SINGLE session
    # ------------------------------------------------------------------

    def _load_stats(self, sid: str) -> None:
        data = self.dm.load_session_data(sid)
        if data is None:
            self._show_placeholder("No data found for this session.")
            self.btn_export_csv.setEnabled(False)
            return

        tracks = data.get("tracks", [])
        rois = self._performance_rois if self._performance_rois else data.get("rois", []) or []

        class_counts: dict[str, int] = defaultdict(int)
        for tr in tracks:
            cls = tr.get("class", "vehicle")
            class_counts[cls] += 1

        roi_counts: list[dict] = []
        for roi in rois:
            r_name = roi.get("name", "Unknown")
            r_type = roi.get("type", "rect")
            pts = roi.get("points", [])
            result = self._count_tracks_in_roi(tracks, r_type, pts)
            roi_counts.append({
                "name": r_name,
                "type": r_type,
                "count": result["total"],
                "by_class": result["by_class"],
            })

        self._current_stats = {
            "session_id": sid,
            "total_tracks": len(tracks),
            "class_counts": dict(class_counts),
            "roi_counts": roi_counts,
        }
        self.btn_export_csv.setEnabled(True)

        # G2-2 — zero-detection banner
        self.banner_empty.setVisible(len(tracks) == 0)

        self._render_single_stats()

    def _render_single_stats(self) -> None:
        self.lbl_placeholder.setVisible(False)
        self.cards_container.setVisible(True)
        self._clear_stat_cards_row()
        self._clear_tables_layout()

        stats = self._current_stats
        class_counts = stats.get("class_counts", {})

        # Total card
        total_card = self._make_stat_card(
            "Total Vehicles", str(stats.get("total_tracks", 0)),
            color=CLASS_COLORS.get("total", ACCENT),
        )
        self._stat_cards_row.addWidget(total_card["frame"])

        for cls in VEHICLE_ORDER:
            count = class_counts.get(cls, 0)
            card = self._make_stat_card(cls.capitalize(), str(count),
                                        color=_vehicle_color(cls))
            self._stat_cards_row.addWidget(card["frame"])
        for cls, count in sorted(class_counts.items()):
            if cls not in VEHICLE_ORDER:
                card = self._make_stat_card(cls.capitalize(), str(count))
                self._stat_cards_row.addWidget(card["frame"])

        # --- Vehicle type breakdown table ---
        sec1 = QLabel("Vehicle Type Breakdown")
        sec1.setStyleSheet(SECTION_LABEL)
        self._tables_layout.addWidget(sec1)

        type_table = QTableWidget()
        type_table.setStyleSheet(TABLE_STYLE)
        type_table.setColumnCount(3)
        type_table.setHorizontalHeaderLabels(["Vehicle Type", "Count", "Percentage"])
        type_table.horizontalHeader().setStretchLastSection(True)
        type_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        type_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        type_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        type_table.verticalHeader().setVisible(False)
        type_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        total = stats.get("total_tracks", 0) or 1
        all_classes: list[tuple[str, int]] = []
        for cls in VEHICLE_ORDER:
            if cls in class_counts:
                all_classes.append((cls, class_counts[cls]))
        for cls in sorted(class_counts.keys()):
            if cls not in VEHICLE_ORDER:
                all_classes.append((cls, class_counts[cls]))

        type_table.setRowCount(len(all_classes))
        for row, (cls, count) in enumerate(all_classes):
            pct = (count / total) * 100.0
            t = QTableWidgetItem(cls.capitalize())
            c = QTableWidgetItem(str(count))
            c.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            p = QTableWidgetItem(f"{pct:.1f}%")
            p.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            type_table.setItem(row, 0, t)
            type_table.setItem(row, 1, c)
            type_table.setItem(row, 2, p)
        type_table.setFixedHeight(max(50, 30 + len(all_classes) * 35))
        self._tables_layout.addWidget(type_table)

        # --- ROI counts ---
        roi_counts = stats.get("roi_counts", [])
        if roi_counts:
            sec2 = QLabel("ROI Counts  (select rows to filter CSV export)")
            sec2.setStyleSheet(SECTION_LABEL)
            self._tables_layout.addWidget(sec2)

            cols = ["ROI Name", "Type", "Total"] + [v.capitalize() for v in VEHICLE_ORDER]
            self.roi_table = QTableWidget()
            self.roi_table.setStyleSheet(TABLE_STYLE)
            self.roi_table.setColumnCount(len(cols))
            self.roi_table.setHorizontalHeaderLabels(cols)
            self.roi_table.horizontalHeader().setStretchLastSection(True)
            self.roi_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            for c in range(1, len(cols)):
                self.roi_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
            self.roi_table.verticalHeader().setVisible(False)
            self.roi_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.roi_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.roi_table.setSelectionMode(QAbstractItemView.MultiSelection)
            self.roi_table.setRowCount(len(roi_counts))
            for row, roi in enumerate(roi_counts):
                self.roi_table.setItem(row, 0, QTableWidgetItem(roi["name"]))
                self.roi_table.setItem(row, 1, QTableWidgetItem(roi["type"].capitalize()))
                t = QTableWidgetItem(str(roi["count"]))
                t.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.roi_table.setItem(row, 2, t)
                by_class = roi.get("by_class", {})
                for ci, v in enumerate(VEHICLE_ORDER):
                    vc = by_class.get(v, 0)
                    item = QTableWidgetItem(str(vc) if vc else "-")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.roi_table.setItem(row, 3 + ci, item)
            self.roi_table.setFixedHeight(max(50, 30 + len(roi_counts) * 35))
            self._tables_layout.addWidget(self.roi_table)
        else:
            self.roi_table = None

    # ------------------------------------------------------------------
    # Group rendering
    # ------------------------------------------------------------------

    def _render_group_stats(self, gid: str) -> None:
        group = self.dm.get_group(gid)
        if not group:
            self._show_placeholder("Group not found.")
            return

        sessions = self.dm.get_sessions_in_group(gid)
        agg = self._aggregate_group_stats(sessions)

        self.badge_provenance.setText(
            f"Group: {group.get('name', gid)} | "
            f"{agg['n_sessions']} sessions | "
            f"{agg['n_corrected']} corrected"
        )
        self.badge_provenance.setStyleSheet(BADGE_INFO)
        self.badge_provenance.setVisible(True)

        self.banner_empty.setVisible(agg["total_tracks"] == 0 and agg["n_sessions"] > 0)
        self.lbl_placeholder.setVisible(False)
        self.cards_container.setVisible(True)
        self._clear_stat_cards_row()
        self._clear_tables_layout()

        # Cards
        total_card = self._make_stat_card(
            "Total Vehicles", str(agg["total_tracks"]),
            color=CLASS_COLORS.get("total", ACCENT),
        )
        self._stat_cards_row.addWidget(total_card["frame"])

        for cls in VEHICLE_ORDER:
            count = agg["class_counts"].get(cls, 0)
            card = self._make_stat_card(cls.capitalize(), str(count),
                                        color=_vehicle_color(cls))
            self._stat_cards_row.addWidget(card["frame"])

        mc_card = self._make_stat_card(
            "Mean Conf (weighted)", f"{agg['mean_conf']:.3f}", color=ACCENT,
        )
        self._stat_cards_row.addWidget(mc_card["frame"])

        # Per-session breakdown table
        sec1 = QLabel("Per-Session Breakdown")
        sec1.setStyleSheet(SECTION_LABEL)
        self._tables_layout.addWidget(sec1)

        cols = ["Session", "Tracks", "Mean Conf", "Corrected?"] + \
               [v.capitalize() for v in VEHICLE_ORDER]
        tbl = QTableWidget()
        tbl.setStyleSheet(TABLE_STYLE)
        tbl.setColumnCount(len(cols))
        tbl.setHorizontalHeaderLabels(cols)
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(cols)):
            tbl.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.setRowCount(len(agg["rows"]))
        for r, row in enumerate(agg["rows"]):
            tbl.setItem(r, 0, QTableWidgetItem(row["video_name"]))
            nt = QTableWidgetItem(str(row["total_tracks"]))
            nt.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tbl.setItem(r, 1, nt)
            mc = QTableWidgetItem(f"{row['mean_conf']:.3f}")
            mc.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tbl.setItem(r, 2, mc)
            tbl.setItem(r, 3, QTableWidgetItem("Yes" if row["has_corrections"] else "-"))
            for ci, v in enumerate(VEHICLE_ORDER):
                vc = row["by_class"].get(v, 0)
                cell = QTableWidgetItem(str(vc) if vc else "-")
                cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                tbl.setItem(r, 4 + ci, cell)
        tbl.setFixedHeight(max(50, 30 + len(agg["rows"]) * 35))
        self._tables_layout.addWidget(tbl)

        self._current_stats = {
            "group_id": gid,
            "group_name": group.get("name", gid),
            "total_tracks": agg["total_tracks"],
            "class_counts": agg["class_counts"],
            "rows": agg["rows"],
        }
        self.btn_export_csv.setEnabled(True)

    # ------------------------------------------------------------------
    # ROI counting helpers
    # ------------------------------------------------------------------

    def _count_tracks_in_roi(self, tracks, r_type, pts):
        if not pts:
            return {"total": 0, "by_class": {}}
        total = 0
        by_class: dict[str, int] = {}
        for tr in tracks:
            for fd in tr.get("frames", []):
                bbox = fd.get("bbox", [0, 0, 0, 0])
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                if self._point_in_roi(cx, cy, r_type, pts):
                    total += 1
                    cls = tr.get("class", "vehicle")
                    by_class[cls] = by_class.get(cls, 0) + 1
                    break
        return {"total": total, "by_class": by_class}

    def _point_in_roi(self, x, y, r_type, pts):
        def pt_xy(p):
            if isinstance(p, dict):
                return (p.get("x", 0), p.get("y", 0))
            return p
        if r_type == "rect" and len(pts) == 2:
            x1, y1 = pt_xy(pts[0])
            x2, y2 = pt_xy(pts[1])
            return (min(x1, x2) <= x <= max(x1, x2)
                    and min(y1, y2) <= y <= max(y1, y2))
        if r_type == "polygon" and len(pts) >= 3:
            n = len(pts)
            norm = [pt_xy(p) for p in pts]
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = norm[i]
                xj, yj = norm[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(1e-9, (yj - yi)) + xi):
                    inside = not inside
                j = i
            return inside
        return False

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _clear_stat_cards_row(self) -> None:
        while self._stat_cards_row.count():
            item = self._stat_cards_row.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.deleteLater()

    def _clear_tables_layout(self) -> None:
        while self._tables_layout.count():
            item = self._tables_layout.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.deleteLater()

    def _show_placeholder(self, message: str | None = None) -> None:
        # G2-1 — HIDE the stat cards entirely
        self.cards_container.setVisible(False)
        self._clear_stat_cards_row()
        self._clear_tables_layout()
        self.banner_empty.setVisible(False)
        self.lbl_placeholder.setVisible(True)
        self.lbl_placeholder.setText(
            message or "Select a session to view performance statistics."
        )
        self._current_stats = {}

    # ------------------------------------------------------------------
    # ROI drawing handlers
    # ------------------------------------------------------------------

    def _on_roi_rect_mode(self, checked: bool) -> None:
        if checked:
            self.btn_roi_poly.setChecked(False)
            self.perf_canvas.drawing_mode = "roi_rect"
            self.perf_canvas.set_cursor_for_mode()
            self.btn_roi_rect.setStyleSheet(ROI_BTN_ACTIVE)
            self.btn_roi_poly.setStyleSheet(SECONDARY_BTN)
        else:
            self.perf_canvas.drawing_mode = "select"
            self.perf_canvas.set_cursor_for_mode()
            self.btn_roi_rect.setStyleSheet(SECONDARY_BTN)

    def _on_roi_poly_mode(self, checked: bool) -> None:
        if checked:
            self.btn_roi_rect.setChecked(False)
            self.perf_canvas.drawing_mode = "roi_polygon"
            self.perf_canvas.set_cursor_for_mode()
            self.btn_roi_poly.setStyleSheet(ROI_BTN_ACTIVE)
            self.btn_roi_rect.setStyleSheet(SECONDARY_BTN)
        else:
            self.perf_canvas.drawing_mode = "select"
            self.perf_canvas.set_cursor_for_mode()
            self.btn_roi_poly.setStyleSheet(SECONDARY_BTN)

    def _on_perf_roi_rect_drawn(self, p1, p2) -> None:
        dlg = RoiNameDialog(f"ROI {len(self._performance_rois) + 1}", self)
        if dlg.exec() != RoiNameDialog.Accepted:
            return
        roi = {
            "type": "rect", "name": dlg.roi_name(),
            "points": [p1, p2],
            "color": ROI_COLOR_ROTATION[
                len(self._performance_rois) % len(ROI_COLOR_ROTATION)],
        }
        self._performance_rois.append(roi)
        self._update_roi_display()
        self.btn_roi_rect.setChecked(False)
        self.perf_canvas.drawing_mode = "select"
        self.perf_canvas.set_cursor_for_mode()
        self.btn_roi_rect.setStyleSheet(SECONDARY_BTN)
        if self._current_session_id:
            self._load_stats(self._current_session_id)
        self._refresh_roi_pickers()

    def _on_perf_roi_polygon_drawn(self, points) -> None:
        if len(points) < 3:
            return
        dlg = RoiNameDialog(f"ROI {len(self._performance_rois) + 1}", self)
        if dlg.exec() != RoiNameDialog.Accepted:
            return
        roi = {
            "type": "polygon", "name": dlg.roi_name(),
            "points": points,
            "color": ROI_COLOR_ROTATION[
                len(self._performance_rois) % len(ROI_COLOR_ROTATION)],
        }
        self._performance_rois.append(roi)
        self._update_roi_display()
        self.btn_roi_poly.setChecked(False)
        self.perf_canvas.drawing_mode = "select"
        self.perf_canvas.set_cursor_for_mode()
        self.btn_roi_poly.setStyleSheet(SECONDARY_BTN)
        if self._current_session_id:
            self._load_stats(self._current_session_id)
        self._refresh_roi_pickers()

    def _on_roi_clear_all(self) -> None:
        self._performance_rois = []
        self._update_roi_display()
        if self._current_session_id:
            self._load_stats(self._current_session_id)
        self._refresh_roi_pickers()

    def _update_roi_display(self) -> None:
        self.perf_canvas.rois = self._performance_rois
        self.perf_canvas.update()
        count = len(self._performance_rois)
        if count:
            self.roi_status.setText(
                f"{count} ROI(s) defined. Statistics will include per-ROI counts."
            )
        else:
            self.roi_status.setText(
                "Draw ROIs on the frame to calculate per-region statistics."
            )
        if self._current_session_id:
            self._save_rois_to_session()

    def _save_rois_to_session(self) -> None:
        sid = self._current_session_id
        if not sid:
            return
        data = self.dm.load_session_data(sid)
        if data is None:
            return
        data["rois"] = self._performance_rois
        self.dm.save_corrections(sid, data)

    def _refresh_roi_pickers(self) -> None:
        for combo in (getattr(self, "beforeafter_roi", None),
                      getattr(self, "confusion_roi", None)):
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(no filter)", None)
            for i, roi in enumerate(self._performance_rois):
                combo.addItem(roi.get("name", f"ROI {i+1}"), roi)
            combo.blockSignals(False)

    # ------------------------------------------------------------------
    # CSV export (single + group)
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        if not self._current_stats:
            return
        if self._mode == "group":
            self._export_group_csv()
        else:
            self._export_single_csv()

    def _export_single_csv(self) -> None:
        sid = self._current_stats.get("session_id", "session")
        roi_indices = self._get_selected_roi_indices()

        data = self.dm.load_session_data(sid)
        all_tracks = data.get("tracks", []) if data else []
        if roi_indices:
            filtered = self._get_tracks_in_rois(all_tracks, roi_indices)
            roi_names = [self._performance_rois[i].get("name", f"ROI {i+1}")
                         for i in roi_indices if i < len(self._performance_rois)]
            filter_desc = f"Filtered by ROIs: {', '.join(roi_names)}"
        else:
            filtered = all_tracks
            filter_desc = "All vehicles (no ROI filter)"

        class_counts: dict[str, int] = defaultdict(int)
        for tr in filtered:
            class_counts[tr.get("class", "vehicle")] += 1
        total = len(filtered)

        default = f"{sid}_stats.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Stats as CSV",
            str(self.dm.exports_dir / default),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["CCTV-YOLO Performance Report"])
                w.writerow(["Session", sid])
                w.writerow(["Provenance", self._provenance])
                w.writerow(["Filter", filter_desc])
                w.writerow([])
                w.writerow(["Vehicle Type Breakdown"])
                w.writerow(["Type", "Count", "Percentage"])
                denom = total or 1
                for cls, count in sorted(class_counts.items()):
                    w.writerow([cls.capitalize(), count,
                                f"{(count / denom) * 100.0:.1f}%"])
                w.writerow(["Total", total, "100.0%"])
                w.writerow([])
                roi_counts = self._current_stats.get("roi_counts", [])
                if roi_counts:
                    w.writerow(["ROI Counts"])
                    w.writerow(["ROI Name", "Type", "Total"] +
                               [v.capitalize() for v in VEHICLE_ORDER])
                    for roi in roi_counts:
                        by_class = roi.get("by_class", {})
                        row = [roi["name"], roi["type"].capitalize(), roi["count"]]
                        row += [by_class.get(v, 0) for v in VEHICLE_ORDER]
                        w.writerow(row)
            QMessageBox.information(
                self, "Export Complete",
                f"Stats exported to:\n{path}\n\n{filter_desc}\nTotal: {total} vehicles",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{e}")

    def _export_group_csv(self) -> None:
        gid = self._current_stats.get("group_id", "group")
        gname = self._current_stats.get("group_name", gid)
        rows = self._current_stats.get("rows", [])

        default = f"{gname}_group_stats.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Group Stats as CSV",
            str(self.dm.exports_dir / default),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["CCTV-YOLO Performance Report (Group)"])
                w.writerow(["Group", gname])
                w.writerow(["Total Tracks", self._current_stats.get("total_tracks", 0)])
                w.writerow([])
                w.writerow(["Aggregate Class Counts"])
                w.writerow(["Type", "Count"])
                for cls, count in sorted(self._current_stats.get("class_counts", {}).items()):
                    w.writerow([cls.capitalize(), count])
                w.writerow([])
                w.writerow(["Per-Session Breakdown"])
                w.writerow(["Session", "Tracks", "Mean Conf", "Corrected?"] +
                           [v.capitalize() for v in VEHICLE_ORDER])
                for r in rows:
                    w.writerow([r["video_name"], r["total_tracks"],
                                f"{r['mean_conf']:.3f}",
                                "Yes" if r["has_corrections"] else "No"] +
                               [r["by_class"].get(v, 0) for v in VEHICLE_ORDER])
            QMessageBox.information(
                self, "Export Complete",
                f"Group stats exported to:\n{path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{e}")

    def _get_selected_roi_indices(self):
        if getattr(self, "roi_table", None) is None or self.roi_table.rowCount() == 0:
            return []
        return sorted(set(idx.row() for idx in self.roi_table.selectedIndexes()))

    def _get_tracks_in_rois(self, tracks, roi_indices):
        rois = self._performance_rois
        if not roi_indices or not rois:
            return tracks
        active = [rois[i] for i in roi_indices if i < len(rois)]
        if not active:
            return tracks
        out = []
        for tr in tracks:
            for roi in active:
                found = False
                for fd in tr.get("frames", []):
                    b = fd.get("bbox", [0, 0, 0, 0])
                    cx = (b[0] + b[2]) / 2.0
                    cy = (b[1] + b[3]) / 2.0
                    if self._point_in_roi(cx, cy, roi.get("type", "rect"),
                                          roi.get("points", [])):
                        found = True
                        break
                if found:
                    out.append(tr)
                    break
        return out

    # ------------------------------------------------------------------
    # Model A/B Compare
    # ------------------------------------------------------------------

    def prefill_compare(self, model_a: str, model_b: str) -> None:
        """Pre-fill the Compare panel's Model A / Model B combos.

        Called by main_window when Training emits compare_models_requested
        (PRD J7 — "Promote after comparison" flow). Refreshes the pickers
        first so a newly-trained model that wasn't in the list yet appears.
        """
        self._refresh_compare_pickers()
        for combo, want in ((self.compare_model_a, model_a),
                             (self.compare_model_b, model_b)):
            idx = combo.findText(want)
            if idx < 0:
                combo.addItem(want, want)
                idx = combo.findText(want)
            combo.setCurrentIndex(idx)
        # If the Compare section is collapsible and collapsed, expand it so
        # the pre-filled selection is visible.
        sec = getattr(self, "compare_section", None)
        if sec is not None and hasattr(sec, "set_expanded"):
            try:
                sec.set_expanded(True)
            except Exception:
                pass

    def _refresh_compare_pickers(self) -> None:
        models = self.dm.list_models()
        for combo in (self.compare_model_a, self.compare_model_b):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            for m in models:
                combo.addItem(m, m)
            if cur:
                idx = combo.findText(cur)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        sessions = self.dm.get_sessions()
        cur = self.compare_session.currentData()
        self.compare_session.blockSignals(True)
        self.compare_session.clear()
        for s in sessions:
            self.compare_session.addItem(s.get("video_name", s.get("id", "")), s.get("id", ""))
        if cur:
            for i in range(self.compare_session.count()):
                if self.compare_session.itemData(i) == cur:
                    self.compare_session.setCurrentIndex(i)
                    break
        self.compare_session.blockSignals(False)

    def _on_run_compare(self) -> None:
        sid = self.compare_session.currentData()
        ma = self.compare_model_a.currentData()
        mb = self.compare_model_b.currentData()
        if not sid or not ma or not mb:
            QMessageBox.warning(self, "Missing Selection",
                                "Pick Model A, Model B, and a session before running.")
            return
        video_path = self.dm.get_video_path(sid)
        if not video_path or not video_path.exists():
            QMessageBox.warning(self, "Missing Video",
                                f"Video for session '{sid}' not found.")
            return

        self.compare_log.clear()
        self.compare_status.setText("Running comparison...")
        self.btn_compare_run.setEnabled(False)
        self.btn_export_compare.setEnabled(False)
        self.btn_promote_b.setEnabled(False)

        roi = self.dm.get_processing_roi(sid) or self.dm.get_global_processing_roi()
        self._compare_worker = ModelCompareWorker(
            video_path=video_path,
            model_a=ma, model_b=mb,
            models_dir=self.dm.models_dir,
            conf=float(self.compare_conf.value()),
            stride=int(self.compare_stride.value()),
            processing_roi=roi,
        )
        self._compare_worker.log_line.connect(self._on_compare_log)
        self._compare_worker.progress.connect(self._on_compare_progress)
        self._compare_worker.finished_ok.connect(self._on_compare_done)
        self._compare_worker.failed.connect(self._on_compare_failed)
        self._compare_worker.start()

    def _on_compare_log(self, line: str) -> None:
        self.compare_log.append(line)

    def _on_compare_progress(self, which: str, pct: int) -> None:
        self.compare_status.setText(f"[{which}] {pct}%")

    def _on_compare_done(self, result: dict) -> None:
        self._last_compare_result = result
        self.btn_compare_run.setEnabled(True)
        self.btn_export_compare.setEnabled(True)
        self.btn_promote_b.setEnabled(True)
        self.compare_status.setText("Done.")

        a = result.get("a", {})
        b = result.get("b", {})
        delta = result.get("delta", {})

        rows: list[tuple[str, str, str, str]] = []
        rows.append(("Total Tracks", str(a.get("total_tracks", 0)),
                     str(b.get("total_tracks", 0)),
                     self._signed(delta.get("total_tracks", 0))))
        rows.append(("Mean Confidence", f"{a.get('mean_conf', 0):.3f}",
                     f"{b.get('mean_conf', 0):.3f}",
                     self._signed(delta.get("mean_conf", 0), fmt=".3f")))
        rows.append(("Median Track Length", str(a.get("median_track_length", 0)),
                     str(b.get("median_track_length", 0)),
                     self._signed(delta.get("median_track_length", 0))))
        rows.append(("Total Detections", str(a.get("total_detections", 0)),
                     str(b.get("total_detections", 0)),
                     self._signed(delta.get("total_detections", 0))))
        classes = sorted(set(a.get("by_class", {})) | set(b.get("by_class", {})))
        for c in classes:
            rows.append((
                c.capitalize(),
                str(a.get("by_class", {}).get(c, 0)),
                str(b.get("by_class", {}).get(c, 0)),
                self._signed(delta.get("by_class", {}).get(c, 0)),
            ))

        self.compare_table.setRowCount(len(rows))
        for r, (m, av, bv, dv) in enumerate(rows):
            self.compare_table.setItem(r, 0, QTableWidgetItem(m))
            self.compare_table.setItem(r, 1, QTableWidgetItem(av))
            self.compare_table.setItem(r, 2, QTableWidgetItem(bv))
            d_item = QTableWidgetItem(dv)
            if dv.startswith("+"):
                d_item.setForeground(Qt.green)
            elif dv.startswith("-"):
                d_item.setForeground(Qt.red)
            self.compare_table.setItem(r, 3, d_item)
        self.compare_table.resizeColumnsToContents()

    def _signed(self, v, fmt: str = "d") -> str:
        try:
            n = float(v)
        except (TypeError, ValueError):
            return str(v)
        if fmt == ".3f":
            return f"{n:+.3f}" if n != 0 else "0.000"
        return f"{int(n):+d}" if n != 0 else "0"

    def _on_compare_failed(self, msg: str) -> None:
        self.btn_compare_run.setEnabled(True)
        self.compare_status.setText("Failed.")
        QMessageBox.critical(self, "Compare Failed", msg)

    def _export_compare_csv(self) -> None:
        if not self._last_compare_result:
            return
        default = "model_compare.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Compare CSV",
            str(self.dm.exports_dir / default),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["CCTV-YOLO Model Comparison"])
                w.writerow(["Model A", self._last_compare_result.get("model_a", "")])
                w.writerow(["Model B", self._last_compare_result.get("model_b", "")])
                w.writerow([])
                w.writerow(["Metric", "A", "B", "Delta"])
                for r in range(self.compare_table.rowCount()):
                    w.writerow([
                        self.compare_table.item(r, 0).text() if self.compare_table.item(r, 0) else "",
                        self.compare_table.item(r, 1).text() if self.compare_table.item(r, 1) else "",
                        self.compare_table.item(r, 2).text() if self.compare_table.item(r, 2) else "",
                        self.compare_table.item(r, 3).text() if self.compare_table.item(r, 3) else "",
                    ])
            QMessageBox.information(self, "Export Complete", f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_promote_b(self) -> None:
        if not self._last_compare_result:
            return
        mb = self._last_compare_result.get("model_b")
        if not mb:
            return
        confirm = QMessageBox.question(
            self, "Promote Model B",
            f"Set '{mb}' as the active model? "
            "This updates the last-used model preference."
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            self.dm.set_last_model(mb)
            QMessageBox.information(self, "Model Promoted",
                                    f"Active model is now '{mb}'.")
        except Exception as e:
            QMessageBox.critical(self, "Promote Failed", str(e))

    # ------------------------------------------------------------------
    # Before / After renderer
    # ------------------------------------------------------------------

    def _on_render_beforeafter(self) -> None:
        sid = self._current_session_id
        if not sid:
            QMessageBox.warning(self, "No Session",
                                "Select a session first.")
            return
        if not self.dm.has_corrections(sid):
            QMessageBox.warning(self, "No Corrections",
                                "This session has no corrections to compare against.")
            return

        out_dir = self.dm.exports_dir / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "before_after.mp4"

        roi = self.beforeafter_roi.currentData()

        self.btn_render_ba.setEnabled(False)
        self.btn_ba_open.setEnabled(False)
        self.btn_ba_play.setEnabled(False)
        self.beforeafter_status.setText("Rendering...")

        self._beforeafter_worker = BeforeAfterWorker(
            self.dm, sid, output_path=out_path, processing_roi=roi
        )
        self._beforeafter_worker.progress.connect(self._on_ba_progress)
        self._beforeafter_worker.finished_ok.connect(self._on_ba_done)
        self._beforeafter_worker.failed.connect(self._on_ba_failed)
        self._beforeafter_worker.start()

    def _on_ba_progress(self, _sid: str, pct: int) -> None:
        self.beforeafter_status.setText(f"Rendering... {pct}%")

    def _on_ba_done(self, _sid: str, out_path: str, stats: dict) -> None:
        self.btn_render_ba.setEnabled(True)
        self.btn_ba_open.setEnabled(True)
        self.btn_ba_play.setEnabled(True)
        self._last_ba_path = Path(out_path)
        written = stats.get("frames_written", 0)
        diff = stats.get("frames_diff", 0)
        pct = (diff / max(1, written)) * 100.0
        self.beforeafter_status.setText(
            f"Rendered {diff}/{written} frames | {diff} differ ({pct:.1f}%)"
        )

    def _on_ba_failed(self, _sid: str, msg: str) -> None:
        self.btn_render_ba.setEnabled(True)
        self.beforeafter_status.setText("Failed.")
        QMessageBox.critical(self, "Before/After Failed", msg)

    def _on_open_ba_output(self) -> None:
        if not self._last_ba_path or not self._last_ba_path.exists():
            return
        open_path(self._last_ba_path, select=True)

    def _on_play_ba_output(self) -> None:
        if not self._last_ba_path or not self._last_ba_path.exists():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_ba_path)))

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def _on_compute_confusion(self) -> None:
        sid = self._current_session_id
        if not sid:
            QMessageBox.warning(self, "No Session", "Select a session first.")
            return
        if not self.dm.has_corrections(sid):
            QMessageBox.warning(
                self, "No Corrections",
                "Confusion matrix needs corrections as ground truth. "
                "Open the correction tab and save some edits, then come back."
            )
            return
        raw = self.dm.load_tracks(sid)
        if not raw:
            QMessageBox.warning(self, "Missing Tracks",
                                "Raw tracks are missing for this session.")
            return
        corr = self.dm.load_corrections(sid)
        roi = self.confusion_roi.currentData()

        self.confusion_status.setText("Computing...")
        QApplication.processEvents()
        try:
            classes = VEHICLE_ORDER
            result = compute_confusion_matrix(
                predictions=raw,
                ground_truth=corr,
                classes=classes,
                iou_threshold=0.5,
                roi_filter=roi,
            )
        except Exception as e:
            self.confusion_status.setText("Failed.")
            QMessageBox.critical(self, "Compute Failed", str(e))
            return

        self._render_confusion(result)

    def _render_confusion(self, result: dict) -> None:
        axis = result.get("axis", [])
        matrix = result.get("matrix", {})
        per_class = result.get("per_class", {})
        agg = result.get("aggregate", {})

        self.confusion_matrix_widget.clear()
        self.confusion_matrix_widget.setColumnCount(len(axis))
        self.confusion_matrix_widget.setRowCount(len(axis))
        self.confusion_matrix_widget.setHorizontalHeaderLabels(
            [a.capitalize() for a in axis]
        )
        self.confusion_matrix_widget.setVerticalHeaderLabels(
            [a.capitalize() for a in axis]
        )
        for r, gt in enumerate(axis):
            for c, pr in enumerate(axis):
                v = int(matrix.get((gt, pr), 0))
                item = QTableWidgetItem(str(v))
                item.setTextAlignment(Qt.AlignCenter)
                if r == c and gt != "background":
                    item.setForeground(Qt.green)
                self.confusion_matrix_widget.setItem(r, c, item)
        self.confusion_matrix_widget.resizeColumnsToContents()

        classes = [c for c in axis if c != "background"]
        self.confusion_metrics_table.setRowCount(len(classes) + 1)
        for r, cls in enumerate(classes):
            m = per_class.get(cls, {})
            self.confusion_metrics_table.setItem(r, 0, QTableWidgetItem(cls.capitalize()))
            self.confusion_metrics_table.setItem(r, 1, QTableWidgetItem(str(m.get("tp", 0))))
            self.confusion_metrics_table.setItem(r, 2, QTableWidgetItem(str(m.get("fp", 0))))
            self.confusion_metrics_table.setItem(r, 3, QTableWidgetItem(str(m.get("fn", 0))))
            self.confusion_metrics_table.setItem(
                r, 4,
                QTableWidgetItem(
                    f"P={m.get('precision', 0):.3f}  "
                    f"R={m.get('recall', 0):.3f}  "
                    f"F1={m.get('f1', 0):.3f}"
                ),
            )
        last = len(classes)
        self.confusion_metrics_table.setItem(last, 0, QTableWidgetItem("AGGREGATE"))
        self.confusion_metrics_table.setItem(last, 1, QTableWidgetItem(""))
        self.confusion_metrics_table.setItem(last, 2, QTableWidgetItem(""))
        self.confusion_metrics_table.setItem(last, 3, QTableWidgetItem(""))
        self.confusion_metrics_table.setItem(
            last, 4,
            QTableWidgetItem(
                f"P={agg.get('precision', 0):.3f}  "
                f"R={agg.get('recall', 0):.3f}  "
                f"F1={agg.get('f1', 0):.3f}  "
                f"mAP={agg.get('mAP', 0):.3f}"
            ),
        )
        self.confusion_status.setText(
            f"P={agg.get('precision', 0):.3f} | R={agg.get('recall', 0):.3f} | "
            f"F1={agg.get('f1', 0):.3f} | mAP={agg.get('mAP', 0):.3f}"
        )

    # ------------------------------------------------------------------
    # Group create / edit / membership
    # ------------------------------------------------------------------

    def _on_new_group(self) -> None:
        name, ok = QInputDialog.getText(self, "New Group", "Group name:")
        if not ok or not name.strip():
            return
        gid = self.dm.create_group(name.strip())
        self.radio_group.setChecked(True)
        self._populate_groups()
        for i in range(self.group_combo.count()):
            if self.group_combo.itemData(i) == gid:
                self.group_combo.setCurrentIndex(i)
                break

    def _on_edit_group(self) -> None:
        gid = self._current_group_id
        if not gid:
            return
        group = self.dm.get_group(gid)
        if not group:
            return
        name, ok = QInputDialog.getText(
            self, "Edit Group", "Group name:", text=group.get("name", "")
        )
        if not ok or not name.strip():
            return
        self.dm.rename_group(gid, name.strip())

    def _on_add_sessions_to_group(self) -> None:
        gid = self._current_group_id
        if not gid:
            QMessageBox.warning(self, "No Group", "Select a group first.")
            return
        group = self.dm.get_group(gid)
        if not group:
            return
        all_sessions = self.dm.get_sessions()
        dlg = GroupPickerDialog(
            sessions=all_sessions,
            already_in=group.get("session_ids", []),
            title=f"Add Sessions to: {group.get('name', gid)}",
            parent=self,
        )
        if dlg.exec() != GroupPickerDialog.Accepted:
            return
        picks = dlg.selected_session_ids()
        if not picks:
            return
        added = self.dm.add_to_group(gid, picks)
        QMessageBox.information(self, "Sessions Added",
                                f"Added {added} session(s) to the group.")

    def _on_remove_sessions_from_group(self) -> None:
        gid = self._current_group_id
        if not gid:
            return
        selected = self.group_membership_list.selectedItems()
        if not selected:
            QMessageBox.information(self, "Nothing Selected",
                                    "Tick rows in the list above to remove.")
            return
        sids = [item.data(Qt.UserRole) for item in selected]
        removed = self.dm.remove_from_group(gid, sids)
        QMessageBox.information(self, "Sessions Removed",
                                f"Removed {removed} session(s).")

    # ------------------------------------------------------------------
    # Signals from data_manager
    # ------------------------------------------------------------------

    def _on_corrections_changed(self, sid: str) -> None:
        if self._mode == "single" and sid == self._current_session_id:
            self._on_session_changed(self.session_combo.currentIndex())
        elif self._mode == "group" and self._current_group_id:
            group = self.dm.get_group(self._current_group_id)
            if group and sid in group.get("session_ids", []):
                self._render_group_stats(self._current_group_id)

    def _on_groups_changed(self) -> None:
        if self._mode == "group":
            self._populate_groups()
            if self._current_group_id:
                self._refresh_group_membership()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        if self._mode == "group":
            self._populate_groups()
            if self._current_group_id:
                self._render_group_stats(self._current_group_id)
            return

        current_id = self.session_combo.currentData()
        self._populate_sessions()
        if current_id:
            for i in range(self.session_combo.count()):
                if self.session_combo.itemData(i) == current_id:
                    self.session_combo.setCurrentIndex(i)
                    return
        self._show_placeholder()
        self.btn_export_csv.setEnabled(False)
