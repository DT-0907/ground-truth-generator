"""
Analytics tab — heatmap, OD matrix, time-series CSV, speed estimation,
direction-of-travel. Plus annotated-video export.
"""
import json
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
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
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QFrame,
)

from cctv_yolo import analytics as A
from cctv_yolo import calibration
from cctv_yolo.annotated_export import AnnotatedVideoWorker
from cctv_yolo.clips import ClipExtractWorker
from cctv_yolo.before_after import BeforeAfterWorker
from cctv_yolo.report import render_html_report

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


class AnalyticsWorker(QThread):
    """Run blocking analytics ops off the UI thread."""
    finished = Signal(str, dict)  # op_name, result
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
            elif self.op == "od":
                od = A.origin_destination_matrix(self.kwargs["track_data"])
                csv_path = A.write_od_matrix_csv(od, self.kwargs["output_path"])
                self.finished.emit("od", {"od": od, "csv": str(csv_path)})
            elif self.op == "timeseries":
                p = A.time_series_csv(**self.kwargs)
                self.finished.emit("timeseries", {"path": str(p)})
            elif self.op == "speeds":
                speeds = A.estimate_speeds(
                    self.kwargs["track_data"],
                    self.kwargs["pixels_per_meter"],
                )
                csv_path = A.write_speeds_csv(speeds, self.kwargs["output_path"])
                self.finished.emit(
                    "speeds", {"count": len(speeds), "csv": str(csv_path)}
                )
            elif self.op == "direction":
                d = A.direction_of_travel(self.kwargs["track_data"])
                self.finished.emit("direction", {"direction": d})
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.error.emit(self.op, str(e))


class AnalyticsTab(QWidget):
    """Generate analytics for any processed session."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self._workers = []
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # PRD H3 fix: Analytics has 9 vertical sections — wrap in a scroll area
        # so the window doesn't squash everything when it's short.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QLabel("Analytics")
        header.setStyleSheet(f"color: {ACCENT}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        # Session picker
        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("Session:"))
        self.session_combo = QComboBox()
        self.session_combo.setMinimumWidth(360)
        pick_row.addWidget(self.session_combo)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh)
        pick_row.addWidget(btn_refresh)
        pick_row.addStretch()
        layout.addLayout(pick_row)

        # Heatmap (PRD H2: render result inline so the section isn't empty)
        hm_box = QGroupBox("Path-density heatmap")
        hm_outer = QVBoxLayout(hm_box)
        hm_row = QHBoxLayout()
        self.hm_sigma = QDoubleSpinBox()
        self.hm_sigma.setRange(2.0, 60.0)
        self.hm_sigma.setSingleStep(2.0)
        self.hm_sigma.setValue(12.0)
        hm_row.addWidget(QLabel("Sigma:"))
        hm_row.addWidget(self.hm_sigma)
        btn_hm = QPushButton("Render Heatmap")
        btn_hm.setStyleSheet(ACTION_BTN)
        btn_hm.clicked.connect(self._run_heatmap)
        hm_row.addWidget(btn_hm)
        hm_row.addStretch()
        self.hm_status = QLabel("Click \"Render Heatmap\" — result will appear below.")
        self.hm_status.setStyleSheet("color:#A89BA8; font-size:11px;")
        hm_row.addWidget(self.hm_status)
        hm_outer.addLayout(hm_row)
        self.hm_image = QLabel()
        self.hm_image.setAlignment(Qt.AlignCenter)
        self.hm_image.setMinimumHeight(0)
        self.hm_image.setStyleSheet("background: transparent; border: none;")
        hm_outer.addWidget(self.hm_image)
        layout.addWidget(hm_box)

        # OD matrix
        od_box = QGroupBox("Origin-destination matrix (uses ROIs)")
        od_layout = QHBoxLayout(od_box)
        btn_od = QPushButton("Compute OD Matrix")
        btn_od.setStyleSheet(ACTION_BTN)
        btn_od.clicked.connect(self._run_od)
        od_layout.addWidget(btn_od)
        self.od_status = QLabel("")
        self.od_status.setStyleSheet("color:#aaa;")
        od_layout.addWidget(self.od_status)
        od_layout.addStretch()
        layout.addWidget(od_box)

        self.od_table = QTableWidget(0, 0)
        self.od_table.setStyleSheet(
            f"QTableWidget {{ background-color: {PANEL}; color: {TEXT}; "
            f"gridline-color: {BORDER}; border: 1px solid {BORDER}; }}"
            f"QHeaderView::section {{ background-color: {PANEL}; color: {ACCENT}; "
            f"border: 0; padding: 4px; }}"
        )
        self.od_table.setMaximumHeight(180)
        layout.addWidget(self.od_table)

        # Time-series
        ts_box = QGroupBox("Time-series CSV")
        ts_layout = QHBoxLayout(ts_box)
        self.ts_bucket = QSpinBox()
        self.ts_bucket.setRange(1, 3600)
        self.ts_bucket.setValue(60)
        self.ts_bucket.setSuffix(" s/bucket")
        ts_layout.addWidget(self.ts_bucket)
        self.ts_class = QCheckBox("Per class")
        self.ts_class.setChecked(True)
        ts_layout.addWidget(self.ts_class)
        self.ts_roi = QCheckBox("Per ROI")
        self.ts_roi.setChecked(True)
        ts_layout.addWidget(self.ts_roi)
        btn_ts = QPushButton("Export Time-Series")
        btn_ts.setStyleSheet(ACTION_BTN)
        btn_ts.clicked.connect(self._run_timeseries)
        ts_layout.addWidget(btn_ts)
        ts_layout.addStretch()
        self.ts_status = QLabel("")
        self.ts_status.setStyleSheet("color:#aaa;")
        ts_layout.addWidget(self.ts_status)
        layout.addWidget(ts_box)

        # Speed
        sp_box = QGroupBox("Speed estimation")
        sp_layout = QHBoxLayout(sp_box)
        sp_layout.addWidget(QLabel("Pixels per meter:"))
        self.sp_ppm = QDoubleSpinBox()
        self.sp_ppm.setRange(0.5, 1000.0)
        self.sp_ppm.setValue(20.0)
        self.sp_ppm.setSingleStep(1.0)
        sp_layout.addWidget(self.sp_ppm)
        btn_auto = QPushButton("Auto")
        btn_auto.setToolTip(
            "Estimate pixels-per-meter from track velocities and bbox heights. "
            "Pick the better-fitting result for your scene."
        )
        btn_auto.clicked.connect(self._auto_calibrate)
        sp_layout.addWidget(btn_auto)
        btn_sp = QPushButton("Compute & Export Speeds")
        btn_sp.setStyleSheet(ACTION_BTN)
        btn_sp.clicked.connect(self._run_speeds)
        sp_layout.addWidget(btn_sp)
        sp_layout.addStretch()
        self.sp_status = QLabel("")
        self.sp_status.setStyleSheet("color:#aaa;")
        sp_layout.addWidget(self.sp_status)
        layout.addWidget(sp_box)

        # Direction-of-travel
        dt_box = QGroupBox("Direction of travel per ROI")
        dt_layout = QHBoxLayout(dt_box)
        btn_dt = QPushButton("Compute Direction")
        btn_dt.setStyleSheet(ACTION_BTN)
        btn_dt.clicked.connect(self._run_direction)
        dt_layout.addWidget(btn_dt)
        self.dt_status = QLabel("")
        self.dt_status.setStyleSheet("color:#aaa;")
        dt_layout.addWidget(self.dt_status)
        dt_layout.addStretch()
        layout.addWidget(dt_box)

        # Annotated video export
        av_box = QGroupBox("Annotated video export (MP4)")
        av_layout = QHBoxLayout(av_box)
        self.av_blur = QCheckBox("Blur license plates")
        av_layout.addWidget(self.av_blur)
        btn_av = QPushButton("Render Annotated Video")
        btn_av.setStyleSheet(ACTION_BTN)
        btn_av.clicked.connect(self._run_annotated)
        av_layout.addWidget(btn_av)
        self.av_progress = QProgressBar()
        self.av_progress.setRange(0, 100)
        self.av_progress.setVisible(False)
        av_layout.addWidget(self.av_progress, stretch=1)
        self.av_status = QLabel("")
        self.av_status.setStyleSheet("color:#aaa;")
        av_layout.addWidget(self.av_status)
        layout.addWidget(av_box)

        # Auto-clip extraction + supercut
        clip_box = QGroupBox("Auto-clip events + supercut")
        clip_layout = QHBoxLayout(clip_box)
        self.clip_supercut = QCheckBox("Build supercut")
        self.clip_supercut.setChecked(True)
        clip_layout.addWidget(self.clip_supercut)
        clip_layout.addWidget(QLabel("Pre/post (sec):"))
        self.clip_pre = QDoubleSpinBox()
        self.clip_pre.setRange(0.5, 30.0)
        self.clip_pre.setValue(2.0)
        clip_layout.addWidget(self.clip_pre)
        self.clip_post = QDoubleSpinBox()
        self.clip_post.setRange(0.5, 30.0)
        self.clip_post.setValue(4.0)
        clip_layout.addWidget(self.clip_post)
        btn_clip = QPushButton("Extract Event Clips")
        btn_clip.setStyleSheet(ACTION_BTN)
        btn_clip.clicked.connect(self._run_clips)
        clip_layout.addWidget(btn_clip)
        self.clip_progress = QProgressBar()
        self.clip_progress.setRange(0, 100)
        self.clip_progress.setVisible(False)
        clip_layout.addWidget(self.clip_progress, stretch=1)
        self.clip_status = QLabel("")
        self.clip_status.setStyleSheet("color:#aaa;")
        clip_layout.addWidget(self.clip_status)
        layout.addWidget(clip_box)

        # Before/after corrections playback
        ba_box = QGroupBox("Before/after side-by-side video")
        ba_layout = QHBoxLayout(ba_box)
        btn_ba = QPushButton("Render Before/After")
        btn_ba.setStyleSheet(ACTION_BTN)
        btn_ba.clicked.connect(self._run_before_after)
        ba_layout.addWidget(btn_ba)
        self.ba_progress = QProgressBar()
        self.ba_progress.setRange(0, 100)
        self.ba_progress.setVisible(False)
        ba_layout.addWidget(self.ba_progress, stretch=1)
        self.ba_status = QLabel("")
        self.ba_status.setStyleSheet("color:#aaa;")
        ba_layout.addWidget(self.ba_status)
        layout.addWidget(ba_box)

        # HTML report
        rp_box = QGroupBox("HTML session report")
        rp_layout = QHBoxLayout(rp_box)
        self.rp_embed = QCheckBox("Embed video (large file)")
        rp_layout.addWidget(self.rp_embed)
        btn_rp = QPushButton("Generate Report")
        btn_rp.setStyleSheet(ACTION_BTN)
        btn_rp.clicked.connect(self._run_report)
        rp_layout.addWidget(btn_rp)
        rp_layout.addStretch()
        self.rp_status = QLabel("")
        self.rp_status.setStyleSheet("color:#aaa;")
        rp_layout.addWidget(self.rp_status)
        layout.addWidget(rp_box)

        layout.addStretch()

    # ----- Common -----
    def refresh(self):
        self.session_combo.clear()
        for s in self.dm.get_sessions():
            self.session_combo.addItem(s["video_name"], s["id"])

    def _current_session_id(self) -> str | None:
        if self.session_combo.count() == 0:
            return None
        return self.session_combo.currentData()

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

    # ----- Heatmap -----
    def _run_heatmap(self):
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
        out = self.dm.exports_dir / sid / f"{sid}_heatmap.png"
        self.hm_status.setText("rendering…")
        self._spawn(
            "heatmap",
            {
                "video_path": video_path,
                "track_data": data,
                "output_path": out,
                "sigma": self.hm_sigma.value(),
            },
            lambda op, r: self._on_heatmap(op, r),
        )

    def _on_heatmap(self, op, r):
        # PRD H2: load + display the rendered PNG inline
        path = Path(r["path"])
        self.hm_status.setText(f"saved: {path.name}")
        try:
            pix = QPixmap(str(path))
            if not pix.isNull():
                # Cap height so the section doesn't take over the tab
                pix = pix.scaledToHeight(360, Qt.SmoothTransformation)
                self.hm_image.setPixmap(pix)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Couldn't load heatmap pixmap: %s", e)

    # ----- OD -----
    def _run_od(self):
        sid = self._current_session_id()
        if not sid:
            return
        data = self._load_track_data(sid)
        if not data:
            return
        out = self.dm.exports_dir / sid / f"{sid}_od_matrix.csv"
        self.od_status.setText("computing…")
        self._spawn("od", {"track_data": data, "output_path": out},
                    lambda op, r: self._on_od(op, r))

    def _on_od(self, op, r):
        od = r["od"]
        self.od_status.setText(f"saved: {Path(r['csv']).name}")
        rois = od["rois"]
        self.od_table.setRowCount(len(rois))
        self.od_table.setColumnCount(len(rois) + 1)
        self.od_table.setHorizontalHeaderLabels(["origin\\dest"] + rois)
        for i, o in enumerate(rois):
            self.od_table.setItem(i, 0, QTableWidgetItem(o))
            for j, d in enumerate(rois):
                self.od_table.setItem(i, j + 1, QTableWidgetItem(
                    str(od["matrix"].get(o, {}).get(d, 0))))
        for c in range(self.od_table.columnCount()):
            self.od_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)

    # ----- Time-series -----
    def _run_timeseries(self):
        sid = self._current_session_id()
        if not sid:
            return
        data = self._load_track_data(sid)
        if not data:
            return
        out = self.dm.exports_dir / sid / f"{sid}_timeseries.csv"
        self.ts_status.setText("computing…")
        self._spawn(
            "timeseries",
            {
                "track_data": data,
                "output_path": out,
                "bucket_seconds": self.ts_bucket.value(),
                "per_class": self.ts_class.isChecked(),
                "per_roi": self.ts_roi.isChecked(),
            },
            lambda op, r: self.ts_status.setText(f"saved: {Path(r['path']).name}"),
        )

    # ----- Speeds -----
    def _run_speeds(self):
        sid = self._current_session_id()
        if not sid:
            return
        data = self._load_track_data(sid)
        if not data:
            return
        out = self.dm.exports_dir / sid / f"{sid}_speeds.csv"
        self.sp_status.setText("computing…")
        self._spawn(
            "speeds",
            {
                "track_data": data,
                "pixels_per_meter": self.sp_ppm.value(),
                "output_path": out,
            },
            lambda op, r: self.sp_status.setText(
                f"{r['count']} speeds saved → {Path(r['csv']).name}"),
        )

    # ----- Direction -----
    def _run_direction(self):
        sid = self._current_session_id()
        if not sid:
            return
        data = self._load_track_data(sid)
        if not data:
            return
        self.dt_status.setText("computing…")
        self._spawn(
            "direction",
            {"track_data": data},
            lambda op, r: self._on_direction(r),
        )

    def _on_direction(self, r):
        d = r["direction"]
        if not d:
            self.dt_status.setText("no ROIs defined")
            return
        parts = []
        for name, counts in d.items():
            parts.append(
                f"{name}: N{counts['N']} S{counts['S']} E{counts['E']} W{counts['W']} "
                f"(total {counts['total']})"
            )
        self.dt_status.setText(" | ".join(parts))

    # ----- Annotated video -----
    def _run_annotated(self):
        sid = self._current_session_id()
        if not sid:
            return
        worker = AnnotatedVideoWorker(self.dm, sid, blur_lp=self.av_blur.isChecked(),
                                      parent=self)
        worker.progress.connect(lambda _, p: self.av_progress.setValue(p))
        worker.finished.connect(self._on_annotated_done)
        worker.error.connect(lambda _, m: QMessageBox.critical(self, "Render failed", m))
        self._workers.append(worker)
        self.av_progress.setVisible(True)
        self.av_progress.setValue(0)
        self.av_status.setText("rendering…")
        worker.start()

    def _on_annotated_done(self, sid, path, stats):
        self.av_progress.setVisible(False)
        self.av_status.setText(f"saved: {Path(path).name} ({stats['frames_written']} frames)")

    # ----- Auto-calibration -----
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
                "Most likely cause: too few cars or too short tracks. "
                "You can keep the manual value."
            )
            return
        self.sp_ppm.setValue(chosen)
        self.sp_status.setText("auto: " + " | ".join(msgs) +
                               f"  → using {chosen}")

    # ----- Auto-clip -----
    def _run_clips(self):
        sid = self._current_session_id()
        if not sid:
            return
        worker = ClipExtractWorker(
            self.dm, sid,
            pre_seconds=self.clip_pre.value(),
            post_seconds=self.clip_post.value(),
            make_supercut=self.clip_supercut.isChecked(),
            parent=self,
        )
        worker.progress.connect(lambda _, p: self.clip_progress.setValue(p))
        worker.finished_ok.connect(self._on_clips_done)
        worker.failed.connect(lambda _, m: QMessageBox.critical(self, "Clips failed", m))
        self._workers.append(worker)
        self.clip_progress.setVisible(True)
        self.clip_progress.setValue(0)
        self.clip_status.setText("extracting…")
        worker.start()

    def _on_clips_done(self, sid, paths, supercut):
        self.clip_progress.setVisible(False)
        if not paths:
            self.clip_status.setText("no events found")
            return
        msg = f"{len(paths)} clips"
        if supercut:
            msg += " + supercut"
        self.clip_status.setText(msg)

    # ----- Before/after -----
    def _run_before_after(self):
        sid = self._current_session_id()
        if not sid:
            return
        worker = BeforeAfterWorker(self.dm, sid, parent=self)
        worker.progress.connect(lambda _, p: self.ba_progress.setValue(p))
        worker.finished_ok.connect(self._on_before_after_done)
        worker.failed.connect(lambda _, m: QMessageBox.critical(self, "Render failed", m))
        self._workers.append(worker)
        self.ba_progress.setVisible(True)
        self.ba_progress.setValue(0)
        self.ba_status.setText("rendering…")
        worker.start()

    def _on_before_after_done(self, sid, path, stats):
        self.ba_progress.setVisible(False)
        self.ba_status.setText(
            f"saved: {Path(path).name} · {stats['frames_diff']} frames differ"
        )

    # ----- HTML report -----
    def _run_report(self):
        sid = self._current_session_id()
        if not sid:
            return
        try:
            p = render_html_report(self.dm, sid, embed_video=self.rp_embed.isChecked())
            self.rp_status.setText(f"saved: {p.name}")
        except Exception as e:
            QMessageBox.critical(self, "Report failed", str(e))
