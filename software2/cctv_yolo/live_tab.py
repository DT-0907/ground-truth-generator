"""
Live tab — RTSP / webcam stream with detection + alerts.
"""
import datetime as dt
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QFont
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
)

from cctv_yolo.live_stream import LiveStreamWorker


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


class LiveTab(QWidget):
    """RTSP/webcam viewer with detection and alerts."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self._worker = None
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QLabel("Live Stream")
        header.setStyleSheet(f"color: {ACCENT}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        # Source group
        src_box = QGroupBox("Source")
        src_layout = QHBoxLayout(src_box)
        src_layout.addWidget(QLabel("URL or webcam index:"))
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText(
            "rtsp://user:pass@192.168.1.10:554/stream  •  or 0 for webcam"
        )
        src_layout.addWidget(self.source_edit, stretch=1)

        src_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        src_layout.addWidget(self.model_combo)

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

        # Alert config
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
        self.wrong_way.setToolTip(
            "Negative number: tracks moving leftward by at least this many "
            "pixels per smoothing window are flagged."
        )
        alert_layout.addWidget(self.wrong_way)
        alert_layout.addStretch()
        layout.addWidget(alert_box)

        # Controls
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

        ctrl_row.addStretch()
        self.status = QLabel("Idle")
        self.status.setStyleSheet("color:#aaa;")
        ctrl_row.addWidget(self.status)
        layout.addLayout(ctrl_row)

        # Video display
        body = QHBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            f"background-color: #000; border: 1px solid {BORDER};"
        )
        body.addWidget(self.video_label, stretch=2)

        # Alert log
        log_col = QVBoxLayout()
        log_col.addWidget(QLabel("Alerts"))
        self.alert_log = QPlainTextEdit()
        self.alert_log.setReadOnly(True)
        self.alert_log.setMaximumBlockCount(500)
        self.alert_log.setFont(QFont("Menlo", 11))
        self.alert_log.setStyleSheet(
            f"QPlainTextEdit {{ background-color: #0e1424; color: {TEXT}; "
            f"border: 1px solid {BORDER}; }}"
        )
        log_col.addWidget(self.alert_log, stretch=1)

        log_col.addWidget(QLabel("Stats"))
        self.stats_label = QLabel("—")
        self.stats_label.setStyleSheet(f"color: {ACCENT}; font-family: Menlo;")
        log_col.addWidget(self.stats_label)
        body.addLayout(log_col, stretch=1)

        layout.addLayout(body, stretch=1)

    def refresh(self):
        self.model_combo.clear()
        models = self.dm.list_models()
        if not models:
            models = ["yolov8m.pt"]
        self.model_combo.addItems(models)
        last = self.dm.get_last_model()
        if last and last in models:
            self.model_combo.setCurrentText(last)

    def _start(self):
        source = self.source_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "No source", "Provide an RTSP URL or webcam index.")
            return
        worker = LiveStreamWorker(
            source=source,
            model_path=self.model_combo.currentText(),
            models_dir=self.dm.models_dir,
            conf=self.conf.value(),
            loiter_seconds=self.loiter.value(),
            wrong_way_dx_threshold=self.wrong_way.value(),
            max_fps=self.max_fps.value(),
        )
        worker.frame_ready.connect(self._on_frame)
        worker.alert.connect(self._on_alert)
        worker.failed.connect(self._on_failed)
        worker.stopped.connect(self._on_stopped)
        self._worker = worker
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText(f"connecting to {source}…")
        worker.start()

    def _stop(self):
        if self._worker:
            self._worker.stop()
            self.status.setText("stopping…")

    def _on_frame(self, image, stats):
        pix = QPixmap.fromImage(image).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)
        parts = [f"frame {stats['frame']}", f"detections {stats['detections']}"]
        for k, v in stats["by_class"].items():
            parts.append(f"{k}:{v}")
        self.stats_label.setText("  ·  ".join(parts))
        self.status.setText("streaming")

    def _on_alert(self, payload):
        ts = dt.datetime.fromtimestamp(payload["timestamp"]).strftime("%H:%M:%S")
        self.alert_log.appendPlainText(f"[{ts}] {payload['rule']}: {payload['message']}")

    def _on_failed(self, msg: str):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("error")
        QMessageBox.critical(self, "Stream error", msg)

    def _on_stopped(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("stopped")
