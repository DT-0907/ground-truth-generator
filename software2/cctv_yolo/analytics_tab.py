"""
Analytics tab — heatmap, OD matrix, time-series CSV, speed estimation,
direction-of-travel, annotated MP4, event clips, before/after, HTML report.

PRD Part H: every result is rendered INLINE — no orphaned status labels,
no "saved to disk and now invisible" outputs. Each section also gets a
hint before its first run ("Click button above to compute…").

Three modes (PRD H4):
- Single session (combo picker)
- Group (run aggregate variants over every session in a chosen group)
- Multi-select (placeholder — defers to Single for now)
"""
from __future__ import annotations

import csv as _csv
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QPixmap, QColor, QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QFrame,
    QRadioButton,
    QButtonGroup,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QFileDialog,
)

from cctv_yolo import analytics as A
from cctv_yolo import calibration
from cctv_yolo.annotated_export import AnnotatedVideoWorker
from cctv_yolo.clips import ClipExtractWorker, ClipExtractConfig, EVENT_TYPES
from cctv_yolo.before_after import BeforeAfterWorker
from cctv_yolo.report import render_html_report, render_group_html_report
from cctv_yolo.widgets.mini_chart import MiniChart
from cctv_yolo.widgets.open_location_bar import OpenLocationBar, open_path

from cctv_yolo.theme import (
    INDIGO as BG,
    PANEL,
    PANEL_HI,
    BORDER,
    PURPLE as ACCENT,
    PINK,
    OFFWHITE as TEXT,
    TEXT_MUTED,
    ERROR,
    INDIGO,
    OFFWHITE,
    PURPLE,
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
QPushButton:hover {{ background-color: {PANEL_HI}; }}
QPushButton:disabled {{ background-color: {BORDER}; color: {TEXT_MUTED}; }}
"""

SECONDARY_BTN = f"""
QPushButton {{
    background-color: transparent;
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 12px;
    font-size: 11px;
}}
QPushButton:hover {{ color: {PINK}; border-color: {PINK}; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; border-color: {BORDER}; }}
"""

GROUP_TITLE_STYLE = f"""
QGroupBox {{
    color: {TEXT};
    font-weight: bold;
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding: 14px 10px 10px 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {PINK};
    border-bottom: 2px solid {ACCENT};
}}
"""

HINT_STYLE = f"color: {TEXT_MUTED}; font-size: 11px; font-style: italic;"
STATUS_OK = f"color: {PINK}; font-size: 11px;"
STATUS_ERR = f"color: {ERROR}; font-size: 11px;"
STATUS_MUTED = f"color: {TEXT_MUTED}; font-size: 11px;"


class AnalyticsWorker(QThread):
    """Run blocking analytics ops off the UI thread."""
    finished = Signal(str, dict)
    error = Signal(str, str)

    def __init__(self, op, kwargs, parent=None):
        super().__init__(parent)
        self.op = op
        self.kwargs = kwargs

    def run(self):
        try:
            if self.op == "heatmap":
                p = A.render_heatmap(**self.kwargs)
                self.finished.emit("heatmap", {"path": str(p)})
            elif self.op == "group_heatmap":
                p = A.aggregate_group_heatmap(**self.kwargs)
                self.finished.emit("heatmap", {"path": str(p)})
            elif self.op == "od":
                od = A.origin_destination_matrix(self.kwargs["track_data"])
                csv_path = A.write_od_matrix_csv(od, self.kwargs["output_path"])
                self.finished.emit("od", {"od": od, "csv": str(csv_path)})
            elif self.op == "group_od":
                od = A.aggregate_group_od_matrix(
                    self.kwargs["data_manager"], self.kwargs["session_ids"]
                )
                csv_path = A.write_od_matrix_csv(od, self.kwargs["output_path"])
                self.finished.emit("od", {"od": od, "csv": str(csv_path)})
            elif self.op == "timeseries":
                p = A.time_series_csv(**self.kwargs)
                self.finished.emit("timeseries", {"path": str(p)})
            elif self.op == "group_timeseries":
                p = A.aggregate_group_time_series(**self.kwargs)
                self.finished.emit("timeseries", {"path": str(p)})
            elif self.op == "speeds":
                speeds = A.estimate_speeds(
                    self.kwargs["track_data"],
                    self.kwargs["pixels_per_meter"],
                )
                csv_path = A.write_speeds_csv(speeds, self.kwargs["output_path"])
                self.finished.emit(
                    "speeds", {"speeds": speeds, "csv": str(csv_path)}
                )
            elif self.op == "group_speeds":
                speeds = A.aggregate_group_speeds(
                    self.kwargs["data_manager"],
                    self.kwargs["session_ids"],
                    self.kwargs["pixels_per_meter"],
                )
                csv_path = A.write_group_speeds_csv(speeds, self.kwargs["output_path"])
                self.finished.emit(
                    "speeds", {"speeds": speeds, "csv": str(csv_path)}
                )
            elif self.op == "direction":
                d = A.direction_of_travel(self.kwargs["track_data"])
                self.finished.emit("direction", {"direction": d})
            elif self.op == "group_direction":
                d = A.aggregate_group_direction(
                    self.kwargs["data_manager"], self.kwargs["session_ids"]
                )
                self.finished.emit("direction", {"direction": d})
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.error.emit(self.op, str(e))


class AnalyticsTab(QWidget):
    """Generate analytics for any processed session or group."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self._workers = []
        self._mode = "single"     # single | group | multi
        self._last_heatmap_path = None
        self._last_av_path = None
        self._last_ba_path = None
        self._last_report_path = None
        self._last_report_time = None
        self._last_supercut_path = None
        self._setup_ui()
        self.refresh()
        if hasattr(self.dm, "groups_changed"):
            self.dm.groups_changed.connect(self._refresh_groups)

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------
    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Scroll area — Analytics is 9 vertical sections.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(12, 12, 12, 12)

        # Header row (title + OpenLocationBar)
        header_row = QHBoxLayout()
        header = QLabel("Analytics")
        header.setStyleSheet(
            f"color: {PINK}; font-size: 18px; font-weight: bold;"
        )
        header_row.addWidget(header)
        header_row.addStretch()
        self.open_bar = OpenLocationBar(self)
        self.open_bar.add_folder("Session Exports",
                                 lambda: self._session_export_dir())
        self.open_bar.add_folder("All Exports", self.dm.exports_dir)
        self.open_bar.add_folder("Group Exports",
                                 lambda: self.dm.exports_dir / "groups")
        header_row.addWidget(self.open_bar)
        layout.addLayout(header_row)

        # Mode toggle (PRD H4)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_group = QButtonGroup(self)
        self.rb_single = QRadioButton("Single session")
        self.rb_group = QRadioButton("Group")
        self.rb_multi = QRadioButton("Multi-select")
        self.rb_single.setChecked(True)
        for rb in (self.rb_single, self.rb_group, self.rb_multi):
            mode_row.addWidget(rb)
            self.mode_group.addButton(rb)
            rb.toggled.connect(self._on_mode_change)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Session picker (single mode)
        self.single_row = QWidget()
        sr = QHBoxLayout(self.single_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.addWidget(QLabel("Session:"))
        self.session_combo = QComboBox()
        self.session_combo.setMinimumWidth(360)
        sr.addWidget(self.session_combo)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.setStyleSheet(SECONDARY_BTN)
        btn_refresh.clicked.connect(self.refresh)
        sr.addWidget(btn_refresh)
        sr.addStretch()
        layout.addWidget(self.single_row)

        # Group picker (group mode)
        self.group_row = QWidget()
        gr = QHBoxLayout(self.group_row)
        gr.setContentsMargins(0, 0, 0, 0)
        gr.addWidget(QLabel("Group:"))
        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(360)
        gr.addWidget(self.group_combo)
        self.group_count_lbl = QLabel("")
        self.group_count_lbl.setStyleSheet(STATUS_MUTED)
        gr.addWidget(self.group_count_lbl)
        gr.addStretch()
        layout.addWidget(self.group_row)
        self.group_row.setVisible(False)
        self.group_combo.currentIndexChanged.connect(self._on_group_pick)

        # --- Section builders ---
        self._build_heatmap(layout)
        self._build_od(layout)
        self._build_timeseries(layout)
        self._build_speed(layout)
        self._build_direction(layout)
        self._build_annotated(layout)
        self._build_clips(layout)
        self._build_before_after(layout)
        self._build_report(layout)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Section builders — each section follows the same shape:
    #   [QGroupBox]
    #     row of controls
    #     hint label  (visible until result lands)
    #     inline result widget(s)
    # ------------------------------------------------------------------
    def _make_box(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setStyleSheet(GROUP_TITLE_STYLE)
        return box

    def _make_hint(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(HINT_STYLE)
        return lbl

    def _build_heatmap(self, layout):
        hm_box = self._make_box("Path-density heatmap")
        hm_v = QVBoxLayout(hm_box)

        row = QHBoxLayout()
        self.hm_sigma = QDoubleSpinBox()
        self.hm_sigma.setRange(2.0, 60.0)
        self.hm_sigma.setSingleStep(2.0)
        self.hm_sigma.setValue(12.0)
        row.addWidget(QLabel("Sigma:"))
        row.addWidget(self.hm_sigma)
        btn = QPushButton("Render Heatmap")
        btn.setStyleSheet(ACTION_BTN)
        btn.clicked.connect(self._run_heatmap)
        row.addWidget(btn)
        self.btn_hm_open = QPushButton("Open in Preview")
        self.btn_hm_open.setStyleSheet(SECONDARY_BTN)
        self.btn_hm_open.setEnabled(False)
        self.btn_hm_open.clicked.connect(self._open_heatmap)
        row.addWidget(self.btn_hm_open)
        row.addStretch()
        self.hm_status = QLabel("")
        self.hm_status.setStyleSheet(STATUS_MUTED)
        row.addWidget(self.hm_status)
        hm_v.addLayout(row)

        self.hm_hint = self._make_hint(
            "Click button above to compute. Output will appear here."
        )
        hm_v.addWidget(self.hm_hint)

        self.hm_image = QLabel()
        self.hm_image.setAlignment(Qt.AlignCenter)
        self.hm_image.setStyleSheet("background: transparent; border: none;")
        self.hm_image.setVisible(False)
        hm_v.addWidget(self.hm_image)

        layout.addWidget(hm_box)

    def _build_od(self, layout):
        od_box = self._make_box("Origin-destination matrix (uses ROIs)")
        v = QVBoxLayout(od_box)

        row = QHBoxLayout()
        btn = QPushButton("Compute OD Matrix")
        btn.setStyleSheet(ACTION_BTN)
        btn.clicked.connect(self._run_od)
        row.addWidget(btn)
        self.btn_od_csv = QPushButton("Export CSV")
        self.btn_od_csv.setStyleSheet(SECONDARY_BTN)
        self.btn_od_csv.setEnabled(False)
        self.btn_od_csv.clicked.connect(self._export_od_csv)
        row.addWidget(self.btn_od_csv)
        row.addStretch()
        self.od_status = QLabel("")
        self.od_status.setStyleSheet(STATUS_MUTED)
        row.addWidget(self.od_status)
        v.addLayout(row)

        self.od_hint = self._make_hint(
            "Click button above to compute. Output will appear here."
        )
        v.addWidget(self.od_hint)

        self.od_table = QTableWidget(0, 0)
        self.od_table.setStyleSheet(
            f"QTableWidget {{ background-color: {PANEL}; color: {TEXT}; "
            f"gridline-color: {BORDER}; border: 1px solid {BORDER}; }}"
            f"QHeaderView::section {{ background-color: {PANEL}; color: {PINK}; "
            f"border: 0; padding: 4px; }}"
        )
        self.od_table.setMaximumHeight(220)
        self.od_table.setVisible(False)
        v.addWidget(self.od_table)
        layout.addWidget(od_box)

    def _build_timeseries(self, layout):
        ts_box = self._make_box("Time-series CSV")
        v = QVBoxLayout(ts_box)

        row = QHBoxLayout()
        self.ts_bucket = QSpinBox()
        self.ts_bucket.setRange(1, 3600)
        self.ts_bucket.setValue(60)
        self.ts_bucket.setSuffix(" s/bucket")
        row.addWidget(self.ts_bucket)
        self.ts_class = QCheckBox("Per class")
        self.ts_class.setChecked(True)
        row.addWidget(self.ts_class)
        self.ts_roi = QCheckBox("Per ROI")
        self.ts_roi.setChecked(True)
        row.addWidget(self.ts_roi)
        btn = QPushButton("Export Time-Series")
        btn.setStyleSheet(ACTION_BTN)
        btn.clicked.connect(self._run_timeseries)
        row.addWidget(btn)
        row.addStretch()
        self.ts_status = QLabel("")
        self.ts_status.setStyleSheet(STATUS_MUTED)
        row.addWidget(self.ts_status)
        v.addLayout(row)

        self.ts_hint = self._make_hint(
            "Click button above to compute. Output will appear here."
        )
        v.addWidget(self.ts_hint)

        # Inline preview table (first 50 rows) + chart
        self.ts_table = QTableWidget(0, 0)
        self.ts_table.setStyleSheet(
            f"QTableWidget {{ background-color: {PANEL}; color: {TEXT}; "
            f"gridline-color: {BORDER}; border: 1px solid {BORDER}; }}"
            f"QHeaderView::section {{ background-color: {PANEL}; color: {PINK}; "
            f"border: 0; padding: 4px; }}"
        )
        self.ts_table.setMaximumHeight(180)
        self.ts_table.setVisible(False)
        v.addWidget(self.ts_table)

        self.ts_chart = MiniChart("line")
        self.ts_chart.set_title("total per bucket")
        self.ts_chart.setVisible(False)
        v.addWidget(self.ts_chart)

        layout.addWidget(ts_box)

    def _build_speed(self, layout):
        sp_box = self._make_box("Speed estimation")
        v = QVBoxLayout(sp_box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Pixels per meter:"))
        self.sp_ppm = QDoubleSpinBox()
        self.sp_ppm.setRange(0.5, 1000.0)
        self.sp_ppm.setValue(20.0)
        self.sp_ppm.setSingleStep(1.0)
        row.addWidget(self.sp_ppm)
        btn_auto = QPushButton("Auto")
        btn_auto.setStyleSheet(SECONDARY_BTN)
        btn_auto.setToolTip(
            "Estimate pixels-per-meter from track velocities and bbox heights."
        )
        btn_auto.clicked.connect(self._auto_calibrate)
        row.addWidget(btn_auto)
        btn = QPushButton("Compute & Export Speeds")
        btn.setStyleSheet(ACTION_BTN)
        btn.clicked.connect(self._run_speeds)
        row.addWidget(btn)
        row.addStretch()
        self.sp_status = QLabel("")
        self.sp_status.setStyleSheet(STATUS_MUTED)
        row.addWidget(self.sp_status)
        v.addLayout(row)

        self.sp_hint = self._make_hint(
            "Click button above to compute. Output will appear here."
        )
        v.addWidget(self.sp_hint)

        self.sp_summary = QLabel("")
        self.sp_summary.setStyleSheet(f"color: {TEXT}; font-size: 12px;")
        self.sp_summary.setVisible(False)
        v.addWidget(self.sp_summary)

        self.sp_chart = MiniChart("histogram")
        self.sp_chart.set_title("speed distribution (mph)")
        self.sp_chart.setVisible(False)
        v.addWidget(self.sp_chart)

        layout.addWidget(sp_box)

    def _build_direction(self, layout):
        dt_box = self._make_box("Direction of travel per ROI")
        v = QVBoxLayout(dt_box)

        row = QHBoxLayout()
        btn = QPushButton("Compute Direction")
        btn.setStyleSheet(ACTION_BTN)
        btn.clicked.connect(self._run_direction)
        row.addWidget(btn)
        row.addStretch()
        self.dt_status = QLabel("")
        self.dt_status.setStyleSheet(STATUS_MUTED)
        row.addWidget(self.dt_status)
        v.addLayout(row)

        self.dt_hint = self._make_hint(
            "Click button above to compute. Output will appear here."
        )
        v.addWidget(self.dt_hint)

        # Container for per-ROI mini bar-charts
        self.dt_container = QWidget()
        self.dt_container_layout = QVBoxLayout(self.dt_container)
        self.dt_container_layout.setContentsMargins(0, 0, 0, 0)
        self.dt_container.setVisible(False)
        v.addWidget(self.dt_container)

        layout.addWidget(dt_box)

    def _build_annotated(self, layout):
        av_box = self._make_box("Annotated video export (MP4)")
        v = QVBoxLayout(av_box)
        row = QHBoxLayout()
        self.av_blur = QCheckBox("Blur license plates")
        row.addWidget(self.av_blur)
        btn = QPushButton("Render Annotated Video")
        btn.setStyleSheet(ACTION_BTN)
        btn.clicked.connect(self._run_annotated)
        row.addWidget(btn)
        self.btn_av_open = QPushButton("Open Result")
        self.btn_av_open.setStyleSheet(SECONDARY_BTN)
        self.btn_av_open.setEnabled(False)
        self.btn_av_open.clicked.connect(self._open_annotated)
        row.addWidget(self.btn_av_open)
        self.av_progress = QProgressBar()
        self.av_progress.setRange(0, 100)
        self.av_progress.setVisible(False)
        row.addWidget(self.av_progress, stretch=1)
        self.av_status = QLabel("")
        self.av_status.setStyleSheet(STATUS_MUTED)
        row.addWidget(self.av_status)
        v.addLayout(row)
        self.av_hint = self._make_hint(
            "Click button above to compute. Output will appear here."
        )
        v.addWidget(self.av_hint)
        layout.addWidget(av_box)

    def _build_clips(self, layout):
        clip_box = self._make_box("Auto-clip events + supercut")
        v = QVBoxLayout(clip_box)

        # Event-type checkboxes
        ev_row = QHBoxLayout()
        ev_row.addWidget(QLabel("Event types:"))
        self.ev_checks = {}
        defaults_on = {"roi_entry", "anomaly", "long_dwell",
                       "speed_outlier", "user_flag"}
        pretty = {
            "roi_entry": "ROI entry",
            "roi_exit": "ROI exit",
            "anomaly": "Anomalies",
            "long_dwell": "Long dwell",
            "speed_outlier": "Speed outlier",
            "user_flag": "User-flagged",
        }
        for k in EVENT_TYPES:
            cb = QCheckBox(pretty.get(k, k))
            cb.setChecked(k in defaults_on)
            self.ev_checks[k] = cb
            ev_row.addWidget(cb)
        ev_row.addStretch()
        v.addLayout(ev_row)

        # Thresholds row
        th_row = QHBoxLayout()
        th_row.addWidget(QLabel("Dwell (sec):"))
        self.clip_dwell = QDoubleSpinBox()
        self.clip_dwell.setRange(1.0, 600.0)
        self.clip_dwell.setValue(8.0)
        th_row.addWidget(self.clip_dwell)

        th_row.addWidget(QLabel("Speed pct:"))
        self.clip_speed_pct = QDoubleSpinBox()
        self.clip_speed_pct.setRange(0.5, 50.0)
        self.clip_speed_pct.setValue(5.0)
        self.clip_speed_pct.setSuffix(" %")
        th_row.addWidget(self.clip_speed_pct)

        th_row.addWidget(QLabel("Z threshold:"))
        self.clip_z = QDoubleSpinBox()
        self.clip_z.setRange(0.5, 10.0)
        self.clip_z.setValue(2.0)
        th_row.addWidget(self.clip_z)

        self.clip_supercut = QCheckBox("Build supercut")
        self.clip_supercut.setChecked(True)
        th_row.addWidget(self.clip_supercut)
        th_row.addStretch()
        v.addLayout(th_row)

        # Pre/post + button row
        pp_row = QHBoxLayout()
        pp_row.addWidget(QLabel("Pre/post (sec):"))
        self.clip_pre = QDoubleSpinBox()
        self.clip_pre.setRange(0.5, 30.0)
        self.clip_pre.setValue(2.0)
        pp_row.addWidget(self.clip_pre)
        self.clip_post = QDoubleSpinBox()
        self.clip_post.setRange(0.5, 30.0)
        self.clip_post.setValue(4.0)
        pp_row.addWidget(self.clip_post)

        btn = QPushButton("Extract Event Clips")
        btn.setStyleSheet(ACTION_BTN)
        btn.clicked.connect(self._run_clips)
        pp_row.addWidget(btn)
        self.btn_supercut_open = QPushButton("Play Supercut")
        self.btn_supercut_open.setStyleSheet(SECONDARY_BTN)
        self.btn_supercut_open.setEnabled(False)
        self.btn_supercut_open.clicked.connect(self._open_supercut)
        pp_row.addWidget(self.btn_supercut_open)
        self.clip_progress = QProgressBar()
        self.clip_progress.setRange(0, 100)
        self.clip_progress.setVisible(False)
        pp_row.addWidget(self.clip_progress, stretch=1)
        self.clip_status = QLabel("")
        self.clip_status.setStyleSheet(STATUS_MUTED)
        pp_row.addWidget(self.clip_status)
        v.addLayout(pp_row)

        self.clip_hint = self._make_hint(
            "Click button above to compute. Output will appear here."
        )
        v.addWidget(self.clip_hint)

        # Thumbnail list
        self.clip_list = QListWidget()
        self.clip_list.setViewMode(QListWidget.IconMode)
        self.clip_list.setIconSize(self.clip_list.iconSize())
        from PySide6.QtCore import QSize
        self.clip_list.setIconSize(QSize(160, 90))
        self.clip_list.setResizeMode(QListWidget.Adjust)
        self.clip_list.setSpacing(8)
        self.clip_list.setMovement(QListWidget.Static)
        self.clip_list.setStyleSheet(
            f"QListWidget {{ background: {PANEL}; color: {TEXT}; "
            f"border: 1px solid {BORDER}; }}"
        )
        self.clip_list.setMinimumHeight(180)
        self.clip_list.setVisible(False)
        self.clip_list.itemDoubleClicked.connect(self._on_clip_double_clicked)
        v.addWidget(self.clip_list)

        layout.addWidget(clip_box)

    def _build_before_after(self, layout):
        ba_box = self._make_box("Before/after side-by-side video")
        v = QVBoxLayout(ba_box)
        row = QHBoxLayout()
        btn = QPushButton("Render Before/After")
        btn.setStyleSheet(ACTION_BTN)
        btn.clicked.connect(self._run_before_after)
        row.addWidget(btn)
        self.btn_ba_play = QPushButton("Play")
        self.btn_ba_play.setStyleSheet(SECONDARY_BTN)
        self.btn_ba_play.setEnabled(False)
        self.btn_ba_play.clicked.connect(self._play_before_after)
        row.addWidget(self.btn_ba_play)
        self.ba_progress = QProgressBar()
        self.ba_progress.setRange(0, 100)
        self.ba_progress.setVisible(False)
        row.addWidget(self.ba_progress, stretch=1)
        self.ba_status = QLabel("")
        self.ba_status.setStyleSheet(STATUS_MUTED)
        row.addWidget(self.ba_status)
        v.addLayout(row)
        self.ba_hint = self._make_hint(
            "Click button above to compute. Output will appear here."
        )
        v.addWidget(self.ba_hint)
        layout.addWidget(ba_box)

    def _build_report(self, layout):
        rp_box = self._make_box("HTML session report")
        v = QVBoxLayout(rp_box)
        row = QHBoxLayout()
        self.rp_embed = QCheckBox("Embed video (large file)")
        row.addWidget(self.rp_embed)
        btn = QPushButton("Generate Report")
        btn.setStyleSheet(ACTION_BTN)
        btn.clicked.connect(self._run_report)
        row.addWidget(btn)
        self.btn_rp_open = QPushButton("Open Report")
        self.btn_rp_open.setStyleSheet(SECONDARY_BTN)
        self.btn_rp_open.setEnabled(False)
        self.btn_rp_open.clicked.connect(self._open_report)
        row.addWidget(self.btn_rp_open)
        row.addStretch()
        self.rp_status = QLabel("")
        self.rp_status.setStyleSheet(STATUS_MUTED)
        row.addWidget(self.rp_status)
        v.addLayout(row)
        self.rp_hint = self._make_hint(
            "Click button above to compute. Output will appear here."
        )
        v.addWidget(self.rp_hint)
        self.rp_age_lbl = QLabel("")
        self.rp_age_lbl.setStyleSheet(STATUS_MUTED)
        self.rp_age_lbl.setVisible(False)
        v.addWidget(self.rp_age_lbl)
        layout.addWidget(rp_box)

    # ------------------------------------------------------------------
    # Mode handling
    # ------------------------------------------------------------------
    def _on_mode_change(self):
        if self.rb_single.isChecked():
            self._mode = "single"
            self.single_row.setVisible(True)
            self.group_row.setVisible(False)
        elif self.rb_group.isChecked():
            self._mode = "group"
            self.single_row.setVisible(False)
            self.group_row.setVisible(True)
            self._refresh_groups()
        else:
            self._mode = "multi"
            self.single_row.setVisible(True)
            self.group_row.setVisible(False)

    def _refresh_groups(self):
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        try:
            for g in self.dm.list_groups():
                self.group_combo.addItem(
                    f"{g.get('name')} ({len(g.get('session_ids', []))})",
                    g.get("id"),
                )
        except Exception:
            pass
        self.group_combo.blockSignals(False)
        self._on_group_pick()

    def _on_group_pick(self):
        gid = self.group_combo.currentData()
        if not gid:
            self.group_count_lbl.setText("")
            return
        try:
            sessions = self.dm.get_sessions_in_group(gid)
            self.group_count_lbl.setText(f"{len(sessions)} sessions")
        except Exception:
            self.group_count_lbl.setText("")

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def refresh(self):
        cur = self.session_combo.currentData() if self.session_combo.count() else None
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        for s in self.dm.get_sessions():
            self.session_combo.addItem(s["video_name"], s["id"])
        if cur:
            idx = self.session_combo.findData(cur)
            if idx >= 0:
                self.session_combo.setCurrentIndex(idx)
        self.session_combo.blockSignals(False)
        self._refresh_groups()

    def _current_session_id(self) -> str | None:
        # Safe to call before session_combo is constructed (OpenLocationBar
        # tooltips resolve eagerly during __init__).
        combo = getattr(self, "session_combo", None)
        if combo is None or combo.count() == 0:
            return None
        return combo.currentData()

    def _current_group_id(self) -> str | None:
        combo = getattr(self, "group_combo", None)
        if combo is None or combo.count() == 0:
            return None
        return combo.currentData()

    def _current_group_sessions(self) -> list:
        gid = self._current_group_id()
        if not gid:
            return []
        return [s["id"] for s in self.dm.get_sessions_in_group(gid)]

    def _session_export_dir(self) -> Path:
        sid = self._current_session_id()
        if sid:
            return self.dm.exports_dir / sid
        return self.dm.exports_dir

    def _group_export_dir(self) -> Path:
        gid = self._current_group_id() or "_unspecified"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = self.dm.exports_dir / "groups" / gid / ts
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_track_data(self, sid: str) -> dict | None:
        return self.dm.load_session_data(sid)

    def _spawn(self, op: str, kwargs: dict, on_finish=None):
        worker = AnalyticsWorker(op, kwargs, parent=self)
        if on_finish:
            worker.finished.connect(on_finish)
        worker.error.connect(self._on_error)
        worker.finished.connect(lambda *_: self._workers.remove(worker)
                                if worker in self._workers else None)
        self._workers.append(worker)
        worker.start()

    def _on_error(self, op: str, msg: str):
        QMessageBox.critical(self, f"{op} failed", msg)
        # Bubble to the matching status label
        status = {
            "heatmap": self.hm_status, "group_heatmap": self.hm_status,
            "od": self.od_status, "group_od": self.od_status,
            "timeseries": self.ts_status, "group_timeseries": self.ts_status,
            "speeds": self.sp_status, "group_speeds": self.sp_status,
            "direction": self.dt_status, "group_direction": self.dt_status,
        }.get(op)
        if status:
            status.setStyleSheet(STATUS_ERR)
            status.setText(f"error: {msg[:80]}")

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------
    def _run_heatmap(self):
        self.hm_status.setStyleSheet(STATUS_MUTED)
        if self._mode == "group":
            sids = self._current_group_sessions()
            if not sids:
                QMessageBox.warning(self, "No sessions", "Group is empty.")
                return
            out = self._group_export_dir() / "group_heatmap.png"
            self.hm_status.setText("rendering group heatmap…")
            self._spawn("group_heatmap", {
                "data_manager": self.dm,
                "session_ids": sids,
                "output_path": out,
                "sigma": self.hm_sigma.value(),
            }, lambda op, r: self._on_heatmap(op, r))
            return

        sid = self._current_session_id()
        if not sid:
            return
        data = self._load_track_data(sid)
        if not data:
            QMessageBox.warning(self, "No data", "Session has no tracks/corrections.")
            return
        video_path = self.dm.get_video_path(sid)
        if not video_path:
            QMessageBox.warning(self, "No video", "Video file missing for session.")
            return
        out = self.dm.exports_dir / sid / "heatmap.png"
        self.hm_status.setText("rendering…")
        self._spawn("heatmap", {
            "video_path": video_path,
            "track_data": data,
            "output_path": out,
            "sigma": self.hm_sigma.value(),
        }, lambda op, r: self._on_heatmap(op, r))

    def _on_heatmap(self, op, r):
        path = Path(r["path"])
        self._last_heatmap_path = path
        self.hm_status.setStyleSheet(STATUS_OK)
        self.hm_status.setText(f"saved: {path.name}")
        try:
            pix = QPixmap(str(path))
            if not pix.isNull():
                pix = pix.scaledToHeight(360, Qt.SmoothTransformation)
                self.hm_image.setPixmap(pix)
                self.hm_image.setVisible(True)
                self.hm_hint.setVisible(False)
                self.btn_hm_open.setEnabled(True)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Couldn't load heatmap pixmap: %s", e)

    def _open_heatmap(self):
        if self._last_heatmap_path and self._last_heatmap_path.exists():
            open_path(self._last_heatmap_path)

    # ------------------------------------------------------------------
    # OD matrix
    # ------------------------------------------------------------------
    def _run_od(self):
        self.od_status.setStyleSheet(STATUS_MUTED)
        if self._mode == "group":
            sids = self._current_group_sessions()
            if not sids:
                QMessageBox.warning(self, "No sessions", "Group is empty.")
                return
            out = self._group_export_dir() / "group_od_matrix.csv"
            self.od_status.setText("computing…")
            self._spawn("group_od", {
                "data_manager": self.dm,
                "session_ids": sids,
                "output_path": out,
            }, lambda op, r: self._on_od(op, r))
            return

        sid = self._current_session_id()
        if not sid:
            return
        data = self._load_track_data(sid)
        if not data:
            return
        out = self.dm.exports_dir / sid / "od_matrix.csv"
        self.od_status.setText("computing…")
        self._spawn("od", {"track_data": data, "output_path": out},
                    lambda op, r: self._on_od(op, r))

    def _on_od(self, op, r):
        od = r["od"]
        self._last_od_csv = Path(r["csv"])
        self.od_status.setStyleSheet(STATUS_OK)
        self.od_status.setText(f"saved: {Path(r['csv']).name}")
        rois = od["rois"]
        self.od_table.setRowCount(len(rois))
        self.od_table.setColumnCount(len(rois) + 1)
        self.od_table.setHorizontalHeaderLabels(["origin\\dest"] + rois)
        # Find max for gradient
        max_v = 1
        for o in rois:
            for d in rois:
                v = od["matrix"].get(o, {}).get(d, 0)
                if v > max_v:
                    max_v = v
        for i, o in enumerate(rois):
            self.od_table.setItem(i, 0, QTableWidgetItem(o))
            for j, d in enumerate(rois):
                v = od["matrix"].get(o, {}).get(d, 0)
                item = QTableWidgetItem(str(v))
                # Gradient PANEL -> PURPLE based on value
                ratio = (v / max_v) if max_v else 0
                p = QColor(PANEL)
                a = QColor(ACCENT)
                r_ = int(p.red() + (a.red() - p.red()) * ratio)
                g_ = int(p.green() + (a.green() - p.green()) * ratio)
                b_ = int(p.blue() + (a.blue() - p.blue()) * ratio)
                item.setBackground(QColor(r_, g_, b_))
                item.setForeground(QColor(TEXT))
                self.od_table.setItem(i, j + 1, item)
        for c in range(self.od_table.columnCount()):
            self.od_table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeToContents
            )
        self.od_table.setVisible(True)
        self.od_hint.setVisible(False)
        self.btn_od_csv.setEnabled(True)

    def _export_od_csv(self):
        if not getattr(self, "_last_od_csv", None) or not self._last_od_csv.exists():
            return
        # Start at the session's exports/ folder so the file lands somewhere
        # the user can find later via the OpenLocationBar.
        default = self.dm.exports_dir / self._last_od_csv.name
        target, _ = QFileDialog.getSaveFileName(
            self, "Save OD matrix CSV", str(default),
            "CSV files (*.csv)",
        )
        if not target:
            return
        try:
            target = Path(target)
            target.write_bytes(self._last_od_csv.read_bytes())
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    # ------------------------------------------------------------------
    # Time-series
    # ------------------------------------------------------------------
    def _run_timeseries(self):
        self.ts_status.setStyleSheet(STATUS_MUTED)
        if self._mode == "group":
            sids = self._current_group_sessions()
            if not sids:
                QMessageBox.warning(self, "No sessions", "Group is empty.")
                return
            out = self._group_export_dir() / "group_timeseries.csv"
            self.ts_status.setText("computing…")
            self._spawn("group_timeseries", {
                "data_manager": self.dm,
                "session_ids": sids,
                "output_path": out,
                "bucket_seconds": self.ts_bucket.value(),
            }, lambda op, r: self._on_timeseries(op, r))
            return

        sid = self._current_session_id()
        if not sid:
            return
        data = self._load_track_data(sid)
        if not data:
            return
        out = self.dm.exports_dir / sid / "timeseries.csv"
        self.ts_status.setText("computing…")
        self._spawn("timeseries", {
            "track_data": data,
            "output_path": out,
            "bucket_seconds": self.ts_bucket.value(),
            "per_class": self.ts_class.isChecked(),
            "per_roi": self.ts_roi.isChecked(),
        }, lambda op, r: self._on_timeseries(op, r))

    def _on_timeseries(self, op, r):
        path = Path(r["path"])
        self.ts_status.setStyleSheet(STATUS_OK)
        self.ts_status.setText(f"saved: {path.name}")
        # Load + preview first 50 rows
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = _csv.reader(f)
                rows = list(reader)
        except Exception:
            return
        if not rows:
            return
        header = rows[0]
        body = rows[1:51]
        self.ts_table.setRowCount(len(body))
        self.ts_table.setColumnCount(len(header))
        self.ts_table.setHorizontalHeaderLabels(header)
        for i, row in enumerate(body):
            for j, cell in enumerate(row):
                self.ts_table.setItem(i, j, QTableWidgetItem(cell))
        for c in range(self.ts_table.columnCount()):
            self.ts_table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeToContents
            )
        self.ts_table.setVisible(True)
        self.ts_hint.setVisible(False)

        # Chart 'total' column (find by header name)
        try:
            total_idx = header.index("total")
            vals = []
            for row in rows[1:]:
                if len(row) > total_idx:
                    try:
                        vals.append(float(row[total_idx]))
                    except ValueError:
                        pass
            self.ts_chart.set_data(vals)
            self.ts_chart.setVisible(True)
        except (ValueError, IndexError):
            pass

    # ------------------------------------------------------------------
    # Speeds
    # ------------------------------------------------------------------
    def _run_speeds(self):
        self.sp_status.setStyleSheet(STATUS_MUTED)
        if self._mode == "group":
            sids = self._current_group_sessions()
            if not sids:
                QMessageBox.warning(self, "No sessions", "Group is empty.")
                return
            out = self._group_export_dir() / "group_speeds.csv"
            self.sp_status.setText("computing…")
            self._spawn("group_speeds", {
                "data_manager": self.dm,
                "session_ids": sids,
                "pixels_per_meter": self.sp_ppm.value(),
                "output_path": out,
            }, lambda op, r: self._on_speeds(op, r))
            return

        sid = self._current_session_id()
        if not sid:
            return
        data = self._load_track_data(sid)
        if not data:
            return
        out = self.dm.exports_dir / sid / "speeds.csv"
        self.sp_status.setText("computing…")
        self._spawn("speeds", {
            "track_data": data,
            "pixels_per_meter": self.sp_ppm.value(),
            "output_path": out,
        }, lambda op, r: self._on_speeds(op, r))

    def _on_speeds(self, op, r):
        speeds = r.get("speeds", [])
        csv_path = Path(r["csv"])
        self.sp_status.setStyleSheet(STATUS_OK)
        self.sp_status.setText(f"saved: {csv_path.name}")
        if not speeds:
            self.sp_summary.setText("0 tracks with usable speed")
            self.sp_summary.setVisible(True)
            self.sp_hint.setVisible(False)
            return
        avg = sum(s["avg_speed_mph"] for s in speeds) / len(speeds)
        peak = max(s["peak_speed_mph"] for s in speeds)
        self.sp_summary.setText(
            f"Avg {avg:.1f} mph · Peak {peak:.1f} mph · {len(speeds)} tracks"
        )
        self.sp_summary.setVisible(True)
        self.sp_chart.set_data([s["avg_speed_mph"] for s in speeds])
        self.sp_chart.setVisible(True)
        self.sp_hint.setVisible(False)

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------
    def _run_direction(self):
        self.dt_status.setStyleSheet(STATUS_MUTED)
        if self._mode == "group":
            sids = self._current_group_sessions()
            if not sids:
                QMessageBox.warning(self, "No sessions", "Group is empty.")
                return
            self.dt_status.setText("computing…")
            self._spawn("group_direction", {
                "data_manager": self.dm,
                "session_ids": sids,
            }, lambda op, r: self._on_direction(r))
            return

        sid = self._current_session_id()
        if not sid:
            return
        data = self._load_track_data(sid)
        if not data:
            return
        self.dt_status.setText("computing…")
        self._spawn("direction", {"track_data": data},
                    lambda op, r: self._on_direction(r))

    def _on_direction(self, r):
        d = r["direction"]
        # Clear container
        while self.dt_container_layout.count():
            item = self.dt_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not d:
            self.dt_status.setStyleSheet(STATUS_MUTED)
            self.dt_status.setText("no ROIs defined")
            return
        self.dt_status.setStyleSheet(STATUS_OK)
        self.dt_status.setText(f"computed {len(d)} ROIs")
        for name, counts in d.items():
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(f"{name} (total {counts['total']})")
            lbl.setStyleSheet(f"color: {TEXT}; font-weight: bold; min-width: 160px;")
            rl.addWidget(lbl)
            chart = MiniChart("bar")
            chart.setMinimumHeight(110)
            chart.set_data([
                ("N", counts["N"]),
                ("S", counts["S"]),
                ("E", counts["E"]),
                ("W", counts["W"]),
            ])
            rl.addWidget(chart, stretch=1)
            self.dt_container_layout.addWidget(row)
        self.dt_container.setVisible(True)
        self.dt_hint.setVisible(False)

    # ------------------------------------------------------------------
    # Annotated video
    # ------------------------------------------------------------------
    def _run_annotated(self):
        self.av_status.setStyleSheet(STATUS_MUTED)
        if self._mode == "group":
            QMessageBox.information(
                self, "Group mode",
                "Annotated MP4 export runs per-session.\n"
                "Switch to 'Single session' or use the Batch tab to render "
                "annotated videos for every session in this group."
            )
            return
        sid = self._current_session_id()
        if not sid:
            return
        worker = AnnotatedVideoWorker(self.dm, sid,
                                      blur_lp=self.av_blur.isChecked(),
                                      parent=self)
        worker.progress.connect(lambda _, p: self.av_progress.setValue(p))
        worker.finished.connect(self._on_annotated_done)
        worker.error.connect(lambda _, m: self._av_error(m))
        self._workers.append(worker)
        self.av_progress.setVisible(True)
        self.av_progress.setValue(0)
        self.av_status.setText("rendering…")
        worker.start()

    def _av_error(self, m):
        self.av_progress.setVisible(False)
        self.av_status.setStyleSheet(STATUS_ERR)
        self.av_status.setText("error")
        QMessageBox.critical(self, "Render failed", m)

    def _on_annotated_done(self, sid, path, stats):
        self.av_progress.setVisible(False)
        self.av_status.setStyleSheet(STATUS_OK)
        self.av_status.setText(
            f"saved: {Path(path).name} ({stats.get('frames_written', '?')} frames)"
        )
        self._last_av_path = Path(path)
        self.btn_av_open.setEnabled(True)
        self.av_hint.setVisible(False)

    def _open_annotated(self):
        if self._last_av_path and self._last_av_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_av_path)))

    # ------------------------------------------------------------------
    # Auto-calibration
    # ------------------------------------------------------------------
    def _auto_calibrate(self):
        sid = self._current_session_id()
        if not sid:
            return
        data = self._load_track_data(sid)
        video_path = self.dm.get_video_path(sid)
        if not data or not video_path:
            QMessageBox.warning(self, "Missing", "Need both video and tracks.")
            return
        result = calibration.auto_calibrate(video_path, data)
        msgs = []
        chosen = None
        for label in ("by_track", "by_scene"):
            r = result.get(label)
            if not r:
                continue
            ppm = r.get("pixels_per_meter")
            if ppm:
                msgs.append(f"{r['method']}: {ppm} px/m")
                if chosen is None:
                    chosen = ppm
        if chosen is None:
            QMessageBox.information(
                self, "Calibration",
                "Could not estimate pixels-per-meter automatically. "
                "Most likely cause: too few cars or too short tracks."
            )
            return
        self.sp_ppm.setValue(chosen)
        self.sp_status.setStyleSheet(STATUS_OK)
        self.sp_status.setText(
            "auto: " + " | ".join(msgs) + f"  → using {chosen}"
        )

    # ------------------------------------------------------------------
    # Auto-clip
    # ------------------------------------------------------------------
    def _run_clips(self):
        self.clip_status.setStyleSheet(STATUS_MUTED)
        if self._mode == "group":
            QMessageBox.information(
                self, "Group mode",
                "Auto-clip extraction runs per-session.\n"
                "Switch to 'Single session' to extract clips."
            )
            return
        sid = self._current_session_id()
        if not sid:
            return

        config = ClipExtractConfig(
            types={k for k, cb in self.ev_checks.items() if cb.isChecked()},
            dwell_seconds=self.clip_dwell.value(),
            speed_outlier_pct=self.clip_speed_pct.value(),
            z_threshold=self.clip_z.value(),
            pixels_per_meter=self.sp_ppm.value(),
        )
        if not config.types:
            QMessageBox.information(self, "No event types",
                                    "Tick at least one event type.")
            return
        worker = ClipExtractWorker(
            self.dm, sid,
            pre_seconds=self.clip_pre.value(),
            post_seconds=self.clip_post.value(),
            make_supercut=self.clip_supercut.isChecked(),
            config=config,
            parent=self,
        )
        worker.progress.connect(lambda _, p: self.clip_progress.setValue(p))
        worker.finished_ok.connect(self._on_clips_done)
        worker.failed.connect(lambda _, m: self._clips_failed(m))
        self._workers.append(worker)
        self.clip_progress.setVisible(True)
        self.clip_progress.setValue(0)
        self.clip_status.setText("extracting…")
        worker.start()

    def _clips_failed(self, m):
        self.clip_progress.setVisible(False)
        self.clip_status.setStyleSheet(STATUS_ERR)
        self.clip_status.setText("error")
        QMessageBox.critical(self, "Clips failed", m)

    def _on_clips_done(self, sid, results, supercut):
        self.clip_progress.setVisible(False)
        if not results:
            self.clip_status.setStyleSheet(STATUS_MUTED)
            self.clip_status.setText("no events found")
            self.clip_hint.setVisible(False)
            return
        self.clip_status.setStyleSheet(STATUS_OK)
        msg = f"{len(results)} clips"
        if supercut:
            msg += " + supercut"
        self.clip_status.setText(msg)

        self.clip_list.clear()
        for r in results:
            item = QListWidgetItem()
            item.setText(r["label"])
            thumb = r.get("thumb")
            if thumb and Path(thumb).exists():
                item.setIcon(QIcon(thumb))
            item.setData(Qt.UserRole, r["path"])
            self.clip_list.addItem(item)
        self.clip_list.setVisible(True)
        self.clip_hint.setVisible(False)
        if supercut:
            self._last_supercut_path = Path(supercut)
            self.btn_supercut_open.setEnabled(True)

    def _on_clip_double_clicked(self, item):
        p = item.data(Qt.UserRole)
        if p:
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def _open_supercut(self):
        if self._last_supercut_path and self._last_supercut_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_supercut_path)))

    # ------------------------------------------------------------------
    # Before/after
    # ------------------------------------------------------------------
    def _run_before_after(self):
        self.ba_status.setStyleSheet(STATUS_MUTED)
        if self._mode == "group":
            QMessageBox.information(
                self, "Group mode",
                "Before/after rendering runs per-session.\n"
                "Switch to 'Single session' to render."
            )
            return
        sid = self._current_session_id()
        if not sid:
            return
        worker = BeforeAfterWorker(self.dm, sid, parent=self)
        worker.progress.connect(lambda _, p: self.ba_progress.setValue(p))
        worker.finished_ok.connect(self._on_before_after_done)
        worker.failed.connect(lambda _, m: self._ba_failed(m))
        self._workers.append(worker)
        self.ba_progress.setVisible(True)
        self.ba_progress.setValue(0)
        self.ba_status.setText("rendering…")
        worker.start()

    def _ba_failed(self, m):
        self.ba_progress.setVisible(False)
        self.ba_status.setStyleSheet(STATUS_ERR)
        self.ba_status.setText("error")
        QMessageBox.critical(self, "Render failed", m)

    def _on_before_after_done(self, sid, path, stats):
        self.ba_progress.setVisible(False)
        self.ba_status.setStyleSheet(STATUS_OK)
        self.ba_status.setText(
            f"saved: {Path(path).name} · {stats.get('frames_diff', '?')} frames differ"
        )
        self._last_ba_path = Path(path)
        self.btn_ba_play.setEnabled(True)
        self.ba_hint.setVisible(False)

    def _play_before_after(self):
        if self._last_ba_path and self._last_ba_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_ba_path)))

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------
    def _run_report(self):
        self.rp_status.setStyleSheet(STATUS_MUTED)
        try:
            if self._mode == "group":
                gid = self._current_group_id()
                if not gid:
                    QMessageBox.warning(self, "No group", "Pick a group first.")
                    return
                p = render_group_html_report(self.dm, gid)
            else:
                sid = self._current_session_id()
                if not sid:
                    return
                p = render_html_report(self.dm, sid,
                                       embed_video=self.rp_embed.isChecked())
            self._last_report_path = Path(p)
            self._last_report_time = time.time()
            self.rp_status.setStyleSheet(STATUS_OK)
            self.rp_status.setText(f"saved: {Path(p).name}")
            self.btn_rp_open.setEnabled(True)
            self._update_report_age()
            self.rp_hint.setVisible(False)
            # Auto-open in browser
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        except Exception as e:
            self.rp_status.setStyleSheet(STATUS_ERR)
            self.rp_status.setText("error")
            QMessageBox.critical(self, "Report failed", str(e))

    def _update_report_age(self):
        if not self._last_report_time:
            self.rp_age_lbl.setVisible(False)
            return
        delta = int(time.time() - self._last_report_time)
        if delta < 60:
            txt = f"Last generated: {delta}s ago"
        elif delta < 3600:
            txt = f"Last generated: {delta // 60}m ago"
        else:
            txt = f"Last generated: {delta // 3600}h ago"
        self.rp_age_lbl.setText(txt)
        self.rp_age_lbl.setVisible(True)

    def _open_report(self):
        if self._last_report_path and self._last_report_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_report_path)))
            self._update_report_age()
