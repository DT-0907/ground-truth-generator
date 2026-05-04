"""
Insights tab — combines dataset health, anomaly detection, and the
confusion-matrix evaluator. Read-mostly; the heavy lifting is in
``dataset_health.py``, ``anomaly.py``, ``confusion.py``.
"""
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPlainTextEdit,
    QProgressBar,
    QMessageBox,
    QScrollArea,
)

from cctv_yolo import dataset_health, anomaly
from cctv_yolo.confusion import ConfusionMatrixWorker


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


class InsightsTab(QWidget):
    review_requested = Signal(str)

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self._cm_worker = None
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QLabel("Insights")
        header.setStyleSheet(f"color: {ACCENT}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        scroll.setWidget(body)
        layout.addWidget(scroll, stretch=1)

        # ---------- Dataset health ----------
        dh_box = QGroupBox("Dataset health")
        dh_layout = QVBoxLayout(dh_box)
        self.dh_summary = QLabel("(not loaded)")
        self.dh_summary.setStyleSheet(f"color: {TEXT};")
        self.dh_summary.setWordWrap(True)
        dh_layout.addWidget(self.dh_summary)

        self.dh_warnings = QPlainTextEdit()
        self.dh_warnings.setReadOnly(True)
        self.dh_warnings.setMaximumBlockCount(200)
        self.dh_warnings.setMaximumHeight(140)
        self.dh_warnings.setStyleSheet(
            f"QPlainTextEdit {{ background-color: #2d1808; color: #ffd07a; "
            f"border: 1px solid {BORDER}; }}"
        )
        dh_layout.addWidget(self.dh_warnings)

        self.dh_table = QTableWidget(0, 4)
        self.dh_table.setHorizontalHeaderLabels(
            ["Class", "Count", "Subclass", "Count"]
        )
        for c in range(4):
            self.dh_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.dh_table.setMaximumHeight(180)
        self.dh_table.setStyleSheet(
            f"QTableWidget {{ background-color: {PANEL}; color: {TEXT}; "
            f"gridline-color: {BORDER}; border: 1px solid {BORDER}; }}"
            f"QHeaderView::section {{ background-color: {PANEL}; color: {ACCENT}; "
            f"border: 0; padding: 4px; }}"
        )
        dh_layout.addWidget(self.dh_table)

        btn_dh = QPushButton("Refresh dataset health")
        btn_dh.setStyleSheet(ACTION_BTN)
        btn_dh.clicked.connect(self._refresh_health)
        dh_layout.addWidget(btn_dh)
        body_layout.addWidget(dh_box)

        # ---------- Anomaly detection ----------
        an_box = QGroupBox("Anomaly detection (z-score vs baseline)")
        an_layout = QVBoxLayout(an_box)
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Session:"))
        self.an_session = QComboBox()
        self.an_session.setMinimumWidth(320)
        ctl.addWidget(self.an_session, stretch=1)

        ctl.addWidget(QLabel("Z >="))
        self.an_z = QDoubleSpinBox()
        self.an_z.setRange(0.5, 6.0)
        self.an_z.setSingleStep(0.5)
        self.an_z.setValue(2.0)
        ctl.addWidget(self.an_z)

        btn_run = QPushButton("Detect anomalies")
        btn_run.setStyleSheet(ACTION_BTN)
        btn_run.clicked.connect(self._run_anomalies)
        ctl.addWidget(btn_run)
        an_layout.addLayout(ctl)

        self.an_table = QTableWidget(0, 6)
        self.an_table.setHorizontalHeaderLabels(
            ["Metric", "ROI", "Hour", "Value", "Baseline", "Z"]
        )
        for c in range(6):
            self.an_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.an_table.setMaximumHeight(220)
        self.an_table.setStyleSheet(self.dh_table.styleSheet())
        an_layout.addWidget(self.an_table)

        body_layout.addWidget(an_box)

        # ---------- Confusion matrix ----------
        cm_box = QGroupBox(
            "Confusion matrix (run a model against held-out corrections)"
        )
        cm_layout = QVBoxLayout(cm_box)
        cm_ctl = QHBoxLayout()
        cm_ctl.addWidget(QLabel("Session (must have corrections):"))
        self.cm_session = QComboBox()
        self.cm_session.setMinimumWidth(280)
        cm_ctl.addWidget(self.cm_session, stretch=1)

        cm_ctl.addWidget(QLabel("Model:"))
        self.cm_model = QComboBox()
        cm_ctl.addWidget(self.cm_model)

        cm_ctl.addWidget(QLabel("Stride:"))
        self.cm_stride = QSpinBox()
        self.cm_stride.setRange(1, 30)
        self.cm_stride.setValue(2)
        cm_ctl.addWidget(self.cm_stride)

        btn_cm = QPushButton("Evaluate")
        btn_cm.setStyleSheet(ACTION_BTN)
        btn_cm.clicked.connect(self._run_confusion)
        cm_ctl.addWidget(btn_cm)
        cm_layout.addLayout(cm_ctl)

        self.cm_progress = QProgressBar()
        self.cm_progress.setRange(0, 100)
        cm_layout.addWidget(self.cm_progress)

        self.cm_metrics = QTableWidget(0, 5)
        self.cm_metrics.setHorizontalHeaderLabels(
            ["Class", "TP", "FP", "FN", "P/R/F1"]
        )
        for c in range(5):
            self.cm_metrics.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.cm_metrics.setMaximumHeight(180)
        self.cm_metrics.setStyleSheet(self.dh_table.styleSheet())
        cm_layout.addWidget(self.cm_metrics)

        self.cm_image = QLabel()
        self.cm_image.setAlignment(Qt.AlignCenter)
        self.cm_image.setMinimumHeight(280)
        self.cm_image.setStyleSheet(
            f"background:#0e1424; border:1px solid {BORDER};"
        )
        cm_layout.addWidget(self.cm_image)

        self.cm_log = QPlainTextEdit()
        self.cm_log.setReadOnly(True)
        self.cm_log.setMaximumBlockCount(400)
        self.cm_log.setMaximumHeight(120)
        self.cm_log.setFont(QFont("Menlo", 10))
        self.cm_log.setStyleSheet(
            f"QPlainTextEdit {{ background-color: #0e1424; color: {TEXT}; "
            f"border: 1px solid {BORDER}; }}"
        )
        cm_layout.addWidget(self.cm_log)

        body_layout.addWidget(cm_box)

    def refresh(self):
        sessions = self.dm.get_sessions()
        self.an_session.clear()
        self.cm_session.clear()
        for s in sessions:
            self.an_session.addItem(s["video_name"], s["id"])
            if s.get("has_corrections"):
                self.cm_session.addItem(s["video_name"], s["id"])

        self.cm_model.clear()
        models = self.dm.list_models() or ["yolov8m.pt"]
        self.cm_model.addItems(models)
        last = self.dm.get_last_model()
        if last and last in models:
            self.cm_model.setCurrentText(last)

        self._refresh_health()

    # ----- Dataset health -----
    def _refresh_health(self):
        report = dataset_health.collect_health(self.dm)
        warns = dataset_health.health_warnings(report)
        summary = (
            f"<b>{report['n_sessions_corrected']}</b> corrected sessions "
            f"(of {report['n_sessions_total']}); "
            f"<b>{report['n_train']}</b> train / <b>{report['n_val']}</b> val.<br>"
            f"<b>{report['n_bboxes']}</b> bboxes total, "
            f"size buckets: small={report['size_buckets'].get('small', 0)}, "
            f"medium={report['size_buckets'].get('medium', 0)}, "
            f"large={report['size_buckets'].get('large', 0)}. "
            f"Median area: {report['median_area_px']} px². "
            f"Frame coverage: {report['frame_coverage']*100:.1f}%."
        )
        self.dh_summary.setText(summary)
        self.dh_warnings.clear()
        if warns:
            for w in warns:
                self.dh_warnings.appendPlainText("• " + w)
        else:
            self.dh_warnings.appendPlainText("• No warnings.")

        # Class + subclass table side-by-side
        classes = sorted(report["classes"].items(), key=lambda kv: -kv[1])
        subs = sorted(report["subclasses"].items(), key=lambda kv: -kv[1])
        n = max(len(classes), len(subs))
        self.dh_table.setRowCount(n)
        for i in range(n):
            if i < len(classes):
                k, v = classes[i]
                self.dh_table.setItem(i, 0, QTableWidgetItem(k))
                self.dh_table.setItem(i, 1, QTableWidgetItem(str(v)))
            if i < len(subs):
                k, v = subs[i]
                self.dh_table.setItem(i, 2, QTableWidgetItem(k))
                self.dh_table.setItem(i, 3, QTableWidgetItem(str(v)))

    # ----- Anomalies -----
    def _run_anomalies(self):
        sid = self.an_session.currentData()
        if not sid:
            return
        anomalies = anomaly.detect_anomalies(
            self.dm, sid, z_threshold=self.an_z.value()
        )
        self.an_table.setRowCount(len(anomalies))
        for i, a in enumerate(anomalies):
            self.an_table.setItem(i, 0, QTableWidgetItem(a.metric))
            self.an_table.setItem(i, 1, QTableWidgetItem(a.roi))
            self.an_table.setItem(i, 2, QTableWidgetItem(str(a.hour)))
            self.an_table.setItem(i, 3, QTableWidgetItem(str(a.value)))
            self.an_table.setItem(
                i, 4, QTableWidgetItem(f"{a.baseline_mean} ± {a.baseline_std}")
            )
            zi = QTableWidgetItem(str(a.z_score))
            zi.setForeground(
                Qt.GlobalColor.red if abs(a.z_score) >= 3 else Qt.GlobalColor.yellow
            )
            self.an_table.setItem(i, 5, zi)

    # ----- Confusion matrix -----
    def _run_confusion(self):
        sid = self.cm_session.currentData()
        if not sid:
            QMessageBox.warning(self, "No session", "Pick a corrected session.")
            return
        model = self.cm_model.currentText()
        worker = ConfusionMatrixWorker(
            self.dm, sid, model_path=model, stride=self.cm_stride.value(),
        )
        worker.progress.connect(self.cm_progress.setValue)
        worker.log_line.connect(self.cm_log.appendPlainText)
        worker.finished_ok.connect(self._on_cm_done)
        worker.failed.connect(lambda m: QMessageBox.critical(self, "Eval failed", m))
        self._cm_worker = worker
        self.cm_log.clear()
        self.cm_progress.setValue(0)
        worker.start()

    def _on_cm_done(self, result, png_path):
        self.cm_progress.setValue(100)
        metrics = result["metrics"]
        self.cm_metrics.setRowCount(len(metrics))
        for i, (cls, m) in enumerate(metrics.items()):
            self.cm_metrics.setItem(i, 0, QTableWidgetItem(cls))
            self.cm_metrics.setItem(i, 1, QTableWidgetItem(str(m["tp"])))
            self.cm_metrics.setItem(i, 2, QTableWidgetItem(str(m["fp"])))
            self.cm_metrics.setItem(i, 3, QTableWidgetItem(str(m["fn"])))
            self.cm_metrics.setItem(
                i, 4, QTableWidgetItem(
                    f"P {m['precision']} · R {m['recall']} · F1 {m['f1']}"
                )
            )

        if png_path:
            pix = QPixmap(png_path)
            if not pix.isNull():
                self.cm_image.setPixmap(pix.scaled(
                    self.cm_image.width() or 800,
                    self.cm_image.height() or 380,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                ))
