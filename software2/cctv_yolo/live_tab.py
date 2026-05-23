"""
Live tab — RTSP / webcam stream with detection, A/B model compare,
per-ROI live counts, recording, FPS overlay, RTSP reconnect, and
per-source persistence (last URL/index + last ROIs).

PRD Part L (P1):
  L2  A/B model compare on live
  L3  ROI on live + entry/exit/dwell alerts
  L4  Recording (manual + on-event)
  L6-1  FPS / inference overlay
  L6-2  Source persistence
  L6-3  RTSP connect timeout toast
  L6-4  Auto-reconnect banner

Personal/research grade — minimalistic + elegant.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import (
    QPixmap, QFont, QPainter, QPen, QBrush, QColor, QPolygonF,
)
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QPlainTextEdit,
    QMessageBox,
    QButtonGroup,
    QRadioButton,
    QCheckBox,
    QFrame,
)

from cctv_yolo.live_stream import LiveStreamWorker, source_hash
from cctv_yolo.widgets.open_location_bar import OpenLocationBar
from cctv_yolo.theme import (
    INDIGO, PANEL, BORDER, PURPLE, PINK, OFFWHITE, ERROR, TEXT_MUTED,
)


# ---------------------------------------------------------------------------
# Buttons / styles
# ---------------------------------------------------------------------------

ACTION_BTN = f"""
QPushButton {{
    background-color: {PURPLE};
    color: {OFFWHITE};
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
    color: {OFFWHITE};
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton:hover {{ background-color: ; }}
QPushButton:disabled {{ background-color: {BORDER}; color: {TEXT_MUTED}; }}
"""

SECONDARY_BTN = f"""
QPushButton {{
    background-color: transparent;
    color: {OFFWHITE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 12px;
    font-size: 12px;
}}
QPushButton:hover {{ border-color: {PINK}; color: {PINK}; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; border-color: {BORDER}; }}
"""

BANNER_STYLE = f"""
QLabel {{
    background-color: {PURPLE};
    color: {OFFWHITE};
    padding: 6px 10px;
    border-radius: 4px;
    font-weight: 600;
}}
"""


# ---------------------------------------------------------------------------
# LiveCanvas — frame display + ROI drawing overlay
# ---------------------------------------------------------------------------

class LiveCanvas(QLabel):
    """QLabel that shows live frames and lets the user draw ROIs on top.

    Modes:
      - "select"  : default, no drawing
      - "roi_rect": click + drag to make a rectangle
      - "roi_poly": click to add points, double-click to finish

    Signals out:
      - roi_rect_drawn(dict, dict)   p1, p2 in video coords
      - roi_polygon_drawn(list)      list of {x, y} dicts in video coords
    """

    roi_rect_drawn    = Signal(dict, dict)
    roi_polygon_drawn = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color: {INDIGO}; border: 1px solid {BORDER};")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._pixmap: Optional[QPixmap] = None
        self._frame_size: tuple[int, int] = (0, 0)  # video pixels
        self._display_rect = QRect()

        self.drawing_mode = "select"
        self.rois: list[dict] = []

        self._draw_start: Optional[QPoint] = None
        self._draw_end:   Optional[QPoint] = None
        self._polygon_pts: list[QPoint] = []

    # ---- public API ----

    def set_frame_pixmap(self, pix: QPixmap, frame_w: int, frame_h: int):
        self._pixmap = pix
        self._frame_size = (frame_w, frame_h)
        self._recompute_rect()
        self.update()

    def set_drawing_mode(self, mode: str):
        self.drawing_mode = mode
        self._draw_start = None
        self._draw_end = None
        self._polygon_pts = []
        self.update()

    def set_rois(self, rois: list[dict]):
        self.rois = list(rois or [])
        self.update()

    # ---- geometry ----

    def resizeEvent(self, e):
        self._recompute_rect()
        super().resizeEvent(e)

    def _recompute_rect(self):
        if not self._pixmap or self._pixmap.isNull():
            self._display_rect = QRect()
            return
        ww, wh = self.width(), self.height()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if iw == 0 or ih == 0:
            return
        scale = min(ww / iw, wh / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        x = (ww - dw) // 2
        y = (wh - dh) // 2
        self._display_rect = QRect(x, y, dw, dh)

    def _canvas_to_video(self, p: QPoint) -> tuple[float, float]:
        dr = self._display_rect
        if dr.width() == 0 or dr.height() == 0:
            return 0.0, 0.0
        fw, fh = self._frame_size
        vx = (p.x() - dr.x()) / dr.width() * fw
        vy = (p.y() - dr.y()) / dr.height() * fh
        return float(vx), float(vy)

    def _video_to_canvas(self, vx: float, vy: float) -> QPoint:
        dr = self._display_rect
        fw, fh = self._frame_size
        if fw == 0 or fh == 0:
            return QPoint(0, 0)
        cx = dr.x() + vx / fw * dr.width()
        cy = dr.y() + vy / fh * dr.height()
        return QPoint(int(cx), int(cy))

    # ---- paint ----

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(INDIGO))
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self._display_rect.size(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            painter.drawPixmap(self._display_rect.topLeft(), scaled)

        # Existing ROIs (the worker also burns them onto the frame, but we
        # paint them client-side too so editing is responsive even if a
        # frame stalls).
        painter.setRenderHint(QPainter.Antialiasing, True)
        for roi in self.rois:
            self._paint_roi(painter, roi)

        # In-progress drawing
        if self.drawing_mode == "roi_rect" and self._draw_start and self._draw_end:
            pen = QPen(QColor(PINK))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            r = QRect(self._draw_start, self._draw_end).normalized()
            painter.drawRect(r)

        if self.drawing_mode == "roi_poly" and self._polygon_pts:
            pen = QPen(QColor(PINK))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            poly = QPolygonF([QPoint(p.x(), p.y()) for p in self._polygon_pts])
            painter.drawPolyline(poly)
            for p in self._polygon_pts:
                painter.fillRect(p.x() - 3, p.y() - 3, 6, 6, QColor(PURPLE))

    def _paint_roi(self, painter: QPainter, roi: dict):
        pen = QPen(QColor(PINK))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(228, 145, 201, 35)))
        if roi["type"] == "rect" and len(roi["points"]) >= 2:
            p1 = self._video_to_canvas(roi["points"][0]["x"], roi["points"][0]["y"])
            p2 = self._video_to_canvas(roi["points"][1]["x"], roi["points"][1]["y"])
            painter.drawRect(QRect(p1, p2).normalized())
        elif roi["type"] == "polygon" and len(roi["points"]) >= 3:
            poly = QPolygonF([
                self._video_to_canvas(p["x"], p["y"]) for p in roi["points"]
            ])
            painter.drawPolygon(poly)

    # ---- mouse ----

    def mousePressEvent(self, e):
        if not self._display_rect.contains(e.pos()):
            return
        if self.drawing_mode == "roi_rect":
            self._draw_start = e.pos()
            self._draw_end = e.pos()
            self.update()
        elif self.drawing_mode == "roi_poly":
            self._polygon_pts.append(e.pos())
            self.update()

    def mouseMoveEvent(self, e):
        if self.drawing_mode == "roi_rect" and self._draw_start:
            self._draw_end = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if self.drawing_mode == "roi_rect" and self._draw_start and self._draw_end:
            r = QRect(self._draw_start, self._draw_end).normalized()
            if r.width() > 5 and r.height() > 5:
                vx1, vy1 = self._canvas_to_video(r.topLeft())
                vx2, vy2 = self._canvas_to_video(r.bottomRight())
                self.roi_rect_drawn.emit(
                    {"x": vx1, "y": vy1}, {"x": vx2, "y": vy2}
                )
            self._draw_start = None
            self._draw_end = None
            self.update()

    def mouseDoubleClickEvent(self, e):
        if self.drawing_mode == "roi_poly" and len(self._polygon_pts) >= 3:
            # Drop the duplicate point queued by the preceding mousedown
            # (matches the existing web UI behaviour).
            if (len(self._polygon_pts) >= 2 and
                    (self._polygon_pts[-1] - self._polygon_pts[-2]).manhattanLength() < 8):
                self._polygon_pts.pop()
            pts = [
                {"x": v[0], "y": v[1]}
                for v in (self._canvas_to_video(p) for p in self._polygon_pts)
            ]
            self.roi_polygon_drawn.emit(pts)
            self._polygon_pts = []
            self.update()


# ---------------------------------------------------------------------------
# LiveTab
# ---------------------------------------------------------------------------

class LiveTab(QWidget):
    """RTSP/webcam viewer with detection, A/B compare, ROIs, recording."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self._worker: Optional[LiveStreamWorker] = None

        self._live_dir = self.dm.data_dir / "live"
        self._snapshots_dir = self._live_dir / "snapshots"
        self._recordings_dir = self._live_dir / "recordings"
        self._live_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._recordings_dir.mkdir(parents=True, exist_ok=True)

        self._ui_state_path = self.dm.config_dir / "ui_state.json"
        self._rois_store_path = self._live_dir / "rois.json"

        self._current_source: str = ""
        self._rois: list[dict] = []
        self._recording: bool = False

        self._setup_ui()
        self.refresh()
        self._restore_source()

    # ----------------------------------------------------------------- UI ---

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ---- Header row: title + OpenLocationBar
        header_row = QHBoxLayout()
        header = QLabel("Live Stream")
        header.setStyleSheet(
            f"color: {PURPLE}; font-size: 18px; font-weight: bold;"
        )
        header_row.addWidget(header)
        header_row.addStretch()

        self.open_bar = OpenLocationBar(self)
        self.open_bar.add_folder("Snapshots", lambda: self._snapshots_dir)
        self.open_bar.add_folder("Recordings", lambda: self._recordings_dir)
        header_row.addWidget(self.open_bar)
        layout.addLayout(header_row)

        # ---- Reconnect banner (hidden by default)
        self.banner = QLabel("")
        self.banner.setStyleSheet(BANNER_STYLE)
        self.banner.setVisible(False)
        layout.addWidget(self.banner)

        # ---- Source group
        src_box = QGroupBox("Source")
        src_layout = QHBoxLayout(src_box)
        src_layout.addWidget(QLabel("URL or webcam index:"))
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText(
            "rtsp://user:pass@192.168.1.10:554/stream  •  or 0 for webcam"
        )
        self.source_edit.editingFinished.connect(self._on_source_changed)
        src_layout.addWidget(self.source_edit, stretch=1)

        src_layout.addWidget(QLabel("Conf:"))
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.3)
        src_layout.addWidget(self.conf)

        src_layout.addWidget(QLabel("Max FPS:"))
        self.max_fps = QSpinBox()
        self.max_fps.setRange(1, 60)
        self.max_fps.setValue(15)
        src_layout.addWidget(self.max_fps)
        layout.addWidget(src_box)

        # ---- Mode + model selection
        mode_box = QGroupBox("Detection")
        mode_layout = QHBoxLayout(mode_box)

        self.mode_group = QButtonGroup(self)
        self.radio_single = QRadioButton("Single model")
        self.radio_compare = QRadioButton("A/B compare")
        self.radio_single.setChecked(True)
        self.mode_group.addButton(self.radio_single)
        self.mode_group.addButton(self.radio_compare)
        self.radio_single.toggled.connect(self._update_mode_visibility)
        mode_layout.addWidget(self.radio_single)
        mode_layout.addWidget(self.radio_compare)

        mode_layout.addSpacing(20)
        mode_layout.addWidget(QLabel("Model A:"))
        self.model_combo = QComboBox()
        mode_layout.addWidget(self.model_combo)

        self.model_b_label = QLabel("Model B:")
        mode_layout.addWidget(self.model_b_label)
        self.model_combo_b = QComboBox()
        mode_layout.addWidget(self.model_combo_b)

        self.b_every_n_label = QLabel("B every Nth:")
        mode_layout.addWidget(self.b_every_n_label)
        self.b_every_n = QSpinBox()
        self.b_every_n.setRange(1, 30)
        self.b_every_n.setValue(1)
        mode_layout.addWidget(self.b_every_n)

        mode_layout.addStretch()
        layout.addWidget(mode_box)
        self._update_mode_visibility()

        # ---- Alert + ROI config
        alert_box = QGroupBox("Alert rules")
        alert_layout = QHBoxLayout(alert_box)
        alert_layout.addWidget(QLabel("Loiter ≥"))
        self.loiter = QSpinBox()
        self.loiter.setRange(1, 1800)
        self.loiter.setValue(30)
        self.loiter.setSuffix(" s")
        alert_layout.addWidget(self.loiter)

        alert_layout.addWidget(QLabel("Wrong-way dx ≤"))
        self.wrong_way = QSpinBox()
        self.wrong_way.setRange(-200, 200)
        self.wrong_way.setValue(-10)
        alert_layout.addWidget(self.wrong_way)

        alert_layout.addWidget(QLabel("ROI dwell ≥"))
        self.roi_dwell = QSpinBox()
        self.roi_dwell.setRange(1, 600)
        self.roi_dwell.setValue(5)
        self.roi_dwell.setSuffix(" s")
        alert_layout.addWidget(self.roi_dwell)
        alert_layout.addStretch()
        layout.addWidget(alert_box)

        # ---- Top control row
        ctrl_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.setStyleSheet(ACTION_BTN)
        self.btn_start.clicked.connect(self._start)
        ctrl_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet(DANGER_BTN)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        ctrl_row.addWidget(self.btn_stop)

        ctrl_row.addSpacing(20)

        # ROI tools
        self.btn_roi_rect = QPushButton("ROI Rect")
        self.btn_roi_rect.setStyleSheet(SECONDARY_BTN)
        self.btn_roi_rect.setCheckable(True)
        self.btn_roi_rect.clicked.connect(self._toggle_roi_rect)
        ctrl_row.addWidget(self.btn_roi_rect)

        self.btn_roi_poly = QPushButton("ROI Polygon")
        self.btn_roi_poly.setStyleSheet(SECONDARY_BTN)
        self.btn_roi_poly.setCheckable(True)
        self.btn_roi_poly.clicked.connect(self._toggle_roi_poly)
        ctrl_row.addWidget(self.btn_roi_poly)

        self.btn_roi_clear = QPushButton("Clear ROIs")
        self.btn_roi_clear.setStyleSheet(SECONDARY_BTN)
        self.btn_roi_clear.clicked.connect(self._clear_rois)
        ctrl_row.addWidget(self.btn_roi_clear)

        ctrl_row.addSpacing(20)

        # Recording
        self.btn_record = QPushButton("Start Recording")
        self.btn_record.setStyleSheet(DANGER_BTN)
        self.btn_record.clicked.connect(self._toggle_recording)
        ctrl_row.addWidget(self.btn_record)

        self.chk_record_on_event = QCheckBox("Record on ROI event")
        ctrl_row.addWidget(self.chk_record_on_event)

        # Snapshot compare
        self.btn_snapshot = QPushButton("Snapshot Compare")
        self.btn_snapshot.setStyleSheet(SECONDARY_BTN)
        self.btn_snapshot.clicked.connect(self._snapshot_compare)
        ctrl_row.addWidget(self.btn_snapshot)

        ctrl_row.addStretch()
        self.status = QLabel("Idle")
        self.status.setStyleSheet(f"color:{TEXT_MUTED};")
        ctrl_row.addWidget(self.status)
        layout.addLayout(ctrl_row)

        # ---- Body: canvases + side panel
        body = QHBoxLayout()

        canvases_row = QHBoxLayout()
        self.canvas_a = LiveCanvas()
        self.canvas_a.roi_rect_drawn.connect(self._on_roi_rect)
        self.canvas_a.roi_polygon_drawn.connect(self._on_roi_poly)
        canvases_row.addWidget(self.canvas_a, stretch=1)

        self.canvas_b = LiveCanvas()
        self.canvas_b.roi_rect_drawn.connect(self._on_roi_rect)
        self.canvas_b.roi_polygon_drawn.connect(self._on_roi_poly)
        self.canvas_b.setVisible(False)
        canvases_row.addWidget(self.canvas_b, stretch=1)
        body.addLayout(canvases_row, stretch=3)

        # Side panel
        side = QVBoxLayout()

        # Counts (per-ROI, per-class)
        side.addWidget(self._section_label("ROI counts"))
        self.roi_counts_label = QLabel("(none)")
        self.roi_counts_label.setStyleSheet(
            f"color:{OFFWHITE}; font-family: Menlo;"
        )
        self.roi_counts_label.setWordWrap(True)
        side.addWidget(self.roi_counts_label)

        side.addWidget(self._section_label("Model A stats"))
        self.stats_a_label = QLabel("—")
        self.stats_a_label.setStyleSheet(
            f"color:{PURPLE}; font-family: Menlo;"
        )
        side.addWidget(self.stats_a_label)

        side.addWidget(self._section_label("Model B stats"))
        self.stats_b_label = QLabel("—")
        self.stats_b_label.setStyleSheet(
            f"color:{PINK}; font-family: Menlo;"
        )
        side.addWidget(self.stats_b_label)

        side.addWidget(self._section_label("Alerts"))
        self.alert_log = QPlainTextEdit()
        self.alert_log.setReadOnly(True)
        self.alert_log.setMaximumBlockCount(500)
        self.alert_log.setFont(QFont("Menlo", 11))
        self.alert_log.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {PANEL}; "
            f"color: {OFFWHITE}; border: 1px solid {BORDER}; }}"
        )
        side.addWidget(self.alert_log, stretch=1)

        body.addLayout(side, stretch=1)
        layout.addLayout(body, stretch=1)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{TEXT_MUTED}; font-weight:600; "
            f"margin-top:6px; font-size:11px; text-transform:uppercase;"
        )
        return lbl

    # ----------------------------------------------------------- refresh ---

    def refresh(self):
        self.model_combo.clear()
        models = self.dm.list_models()
        if not models:
            models = ["yolov8m.pt"]
        self.model_combo.addItems(models)
        self.model_combo_b.clear()
        self.model_combo_b.addItems(models)

        last = self.dm.get_last_model()
        if last and last in models:
            self.model_combo.setCurrentText(last)
        # Default B to a different one if possible
        if len(models) > 1:
            self.model_combo_b.setCurrentIndex(1)

    def _update_mode_visibility(self):
        compare = self.radio_compare.isChecked()
        # All of these may be missing during early __init__ — guard each.
        for name in ("model_b_label", "model_combo_b", "b_every_n_label",
                     "b_every_n", "canvas_b", "stats_b_label"):
            w = getattr(self, name, None)
            if w is not None:
                w.setVisible(compare)

    # ------------------------------------------------------- persistence ---

    def _load_ui_state(self) -> dict:
        try:
            if self._ui_state_path.exists():
                with open(self._ui_state_path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            pass
        return {}

    def _save_ui_state(self, patch: dict):
        state = self._load_ui_state()
        state.update(patch)
        try:
            self._ui_state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._ui_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def _restore_source(self):
        state = self._load_ui_state()
        live = state.get("live", {})
        last_src = live.get("last_source", "")
        if last_src:
            self.source_edit.setText(last_src)
            self._current_source = last_src
            self._load_rois_for_source(last_src)

    def _on_source_changed(self):
        new_src = self.source_edit.text().strip()
        if new_src == self._current_source:
            return
        self._current_source = new_src
        if new_src:
            self._save_ui_state({"live": {"last_source": new_src}})
            self._load_rois_for_source(new_src)

    # ----------------------------------------------------------- ROI I/O ---

    def _load_all_rois(self) -> dict:
        if not self._rois_store_path.exists():
            return {}
        try:
            with open(self._rois_store_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _save_all_rois(self, all_rois: dict):
        try:
            self._rois_store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._rois_store_path, "w", encoding="utf-8") as f:
                json.dump(all_rois, f, indent=2)
        except Exception:
            pass

    def _source_key(self, source: str) -> str:
        return source_hash(source) if source else ""

    def _load_rois_for_source(self, source: str):
        key = self._source_key(source)
        if not key:
            self._rois = []
        else:
            all_rois = self._load_all_rois()
            self._rois = list(all_rois.get(key, []))
        self.canvas_a.set_rois(self._rois)
        self.canvas_b.set_rois(self._rois)
        if self._worker:
            self._worker.update_rois(self._rois)

    def _persist_rois(self):
        key = self._source_key(self._current_source)
        if not key:
            return
        all_rois = self._load_all_rois()
        all_rois[key] = self._rois
        self._save_all_rois(all_rois)

    # ----------------------------------------------------- ROI tool state ---

    def _toggle_roi_rect(self):
        on = self.btn_roi_rect.isChecked()
        if on:
            self.btn_roi_poly.setChecked(False)
            self.canvas_a.set_drawing_mode("roi_rect")
            self.canvas_b.set_drawing_mode("roi_rect")
        else:
            self.canvas_a.set_drawing_mode("select")
            self.canvas_b.set_drawing_mode("select")

    def _toggle_roi_poly(self):
        on = self.btn_roi_poly.isChecked()
        if on:
            self.btn_roi_rect.setChecked(False)
            self.canvas_a.set_drawing_mode("roi_poly")
            self.canvas_b.set_drawing_mode("roi_poly")
        else:
            self.canvas_a.set_drawing_mode("select")
            self.canvas_b.set_drawing_mode("select")

    def _clear_rois(self):
        if not self._rois:
            return
        self._rois = []
        self.canvas_a.set_rois(self._rois)
        self.canvas_b.set_rois(self._rois)
        if self._worker:
            self._worker.update_rois(self._rois)
        self._persist_rois()

    def _on_roi_rect(self, p1: dict, p2: dict):
        name = f"ROI {len(self._rois) + 1}"
        self._rois.append({"type": "rect", "name": name, "points": [p1, p2]})
        self.canvas_a.set_rois(self._rois)
        self.canvas_b.set_rois(self._rois)
        if self._worker:
            self._worker.update_rois(self._rois)
        self._persist_rois()
        # Exit drawing mode
        self.btn_roi_rect.setChecked(False)
        self._toggle_roi_rect()

    def _on_roi_poly(self, points: list):
        name = f"ROI {len(self._rois) + 1}"
        self._rois.append({"type": "polygon", "name": name, "points": points})
        self.canvas_a.set_rois(self._rois)
        self.canvas_b.set_rois(self._rois)
        if self._worker:
            self._worker.update_rois(self._rois)
        self._persist_rois()
        self.btn_roi_poly.setChecked(False)
        self._toggle_roi_poly()

    # ------------------------------------------------------------ start ---

    def _start(self):
        source = self.source_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "No source", "Provide an RTSP URL or webcam index.")
            return
        if source != self._current_source:
            self._current_source = source
            self._load_rois_for_source(source)
        self._save_ui_state({"live": {"last_source": source}})

        model_a = self.model_combo.currentText()
        model_b = self.model_combo_b.currentText() if self.radio_compare.isChecked() else None

        worker = LiveStreamWorker(
            source=source,
            model_path=model_a,
            model_b_path=model_b,
            b_every_n=self.b_every_n.value(),
            models_dir=self.dm.models_dir,
            conf=self.conf.value(),
            loiter_seconds=self.loiter.value(),
            wrong_way_dx_threshold=self.wrong_way.value(),
            max_fps=self.max_fps.value(),
            rois=self._rois,
            roi_dwell_seconds=float(self.roi_dwell.value()),
            record_dir=self._recordings_dir,
            record_on_event=self.chk_record_on_event.isChecked(),
            connect_timeout_seconds=10.0,
            reconnect=True,
        )
        worker.frame_ready.connect(self._on_frame_a)
        worker.frame_ready_b.connect(self._on_frame_b)
        worker.alert.connect(self._on_alert)
        worker.failed.connect(self._on_failed)
        worker.stopped.connect(self._on_stopped)
        worker.reconnecting.connect(self._on_reconnecting)
        worker.reconnected.connect(self._on_reconnected)
        worker.recording_started.connect(self._on_recording_started)
        worker.recording_stopped.connect(self._on_recording_stopped)
        self._worker = worker

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText(f"connecting to {source}…")
        worker.start()

    def _stop(self):
        if self._worker:
            self._worker.stop()
            self.status.setText("stopping…")

    # ----------------------------------------------------------- frames ---

    def _on_frame_a(self, image, stats):
        pix = QPixmap.fromImage(image)
        self.canvas_a.set_frame_pixmap(pix, image.width(), image.height())
        self._update_stats(self.stats_a_label, stats)
        self._update_roi_panel(stats.get("roi_counts", {}))
        self.status.setText("streaming")

    def _on_frame_b(self, image, stats):
        pix = QPixmap.fromImage(image)
        self.canvas_b.set_frame_pixmap(pix, image.width(), image.height())
        self._update_stats(self.stats_b_label, stats)

    def _update_stats(self, label: QLabel, stats: dict):
        parts = [
            f"frame {stats.get('frame', 0)}",
            f"det {stats.get('detections', 0)}",
            f"{stats.get('fps', 0):.1f}fps",
            f"{stats.get('inference_ms', 0):.0f}ms",
        ]
        by_class = stats.get("by_class") or {}
        for k, v in by_class.items():
            parts.append(f"{k}:{v}")
        label.setText("  ·  ".join(parts))

    def _update_roi_panel(self, roi_counts: dict):
        if not roi_counts:
            self.roi_counts_label.setText("(no ROIs)")
            return
        lines = [f"{name}: {n}" for name, n in roi_counts.items()]
        self.roi_counts_label.setText("\n".join(lines))

    # ----------------------------------------------------------- alerts ---

    def _on_alert(self, payload):
        ts = dt.datetime.fromtimestamp(payload["timestamp"]).strftime("%H:%M:%S")
        rule = payload.get("rule", "alert")
        msg = payload.get("message", "")
        roi = payload.get("roi")
        prefix = f"[{ts}] {rule}"
        if roi:
            prefix += f" ({roi})"
        self.alert_log.appendPlainText(f"{prefix}: {msg}")

    # ----------------------------------------------------------- state ----

    def _on_failed(self, msg: str):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("error")
        self.banner.setVisible(False)
        QMessageBox.critical(self, "Stream error", msg)

    def _on_stopped(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("stopped")
        self.banner.setVisible(False)
        self._recording = False
        self.btn_record.setText("Start Recording")

    def _on_reconnecting(self, msg: str):
        self.banner.setText(f"Reconnecting…  {msg}")
        self.banner.setVisible(True)
        self.status.setText("reconnecting")

    def _on_reconnected(self):
        self.banner.setVisible(False)
        self.status.setText("streaming")

    # -------------------------------------------------------- recording ---

    def _toggle_recording(self):
        if not self._worker:
            QMessageBox.information(
                self, "Not streaming",
                "Start the stream before recording."
            )
            return
        if self._recording:
            self._worker.stop_recording()
        else:
            self._worker.start_recording()

    def _on_recording_started(self, path: str):
        self._recording = True
        self.btn_record.setText("Stop Recording")
        self.status.setText(f"recording → {Path(path).name}")

    def _on_recording_stopped(self, path: str):
        self._recording = False
        self.btn_record.setText("Start Recording")
        self.status.setText(f"saved {Path(path).name}")

    # -------------------------------------------------------- snapshot ---

    def _snapshot_compare(self):
        if not self._worker:
            QMessageBox.information(
                self, "Not streaming",
                "Start the stream before taking a snapshot."
            )
            return
        self._worker.request_snapshot(self._snapshots_dir)
        self.status.setText("snapshot saved")
