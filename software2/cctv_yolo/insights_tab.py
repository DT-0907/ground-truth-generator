"""
Insights tab — Session / Group / Dataset / Multi sub-tabs.

Each sub-tab combines the same three panels (Dataset Health, Anomaly
Detection, Confusion Matrix) scoped to its own data source:

  * Session — one chosen processed session.
  * Group   — every session inside a named DataManager group.
  * Dataset — a built YOLO dataset folder under data/training/.
  * Multi   — an ad-hoc multi-select of sessions for one-off comparisons.

Heavy lifting lives in ``dataset_health.py``, ``anomaly.py``,
``confusion.py``, and ``metrics.py``. Results are persisted per-sub-tab
under ``data/exports/insights/...`` and re-loadable via the History
dropdown that sits beside each panel (PRD I3).
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QColor, QFont, QPixmap
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
    QTabWidget,
    QFileDialog,
    QFrame,
)

from cctv_yolo import anomaly, confusion as confusion_mod, dataset_health
from cctv_yolo.confusion import ConfusionMatrixWorker
from cctv_yolo.metrics import (
    aggregate_confusion_matrices,
    compute_confusion_matrix,
)
from cctv_yolo.theme import (
    BORDER,
    ERROR,
    INDIGO,
    OFFWHITE,
    PANEL,
    PINK,
    PURPLE,
    TEXT_MUTED,
    YELLOW,
)
from cctv_yolo.widgets.open_location_bar import OpenLocationBar


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

GHOST_BTN = f"""
QPushButton {{
    background-color: transparent;
    color: {OFFWHITE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11px;
}}
QPushButton:hover {{ color: {PINK}; border-color: {PINK}; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; }}
"""

TABLE_STYLE = (
    f"QTableWidget {{ background-color: {PANEL}; color: {OFFWHITE}; "
    f"gridline-color: {BORDER}; border: 1px solid {BORDER}; }}"
    f"QHeaderView::section {{ background-color: {PANEL}; color: {PURPLE}; "
    f"border: 0; padding: 4px; font-weight: bold; }}"
)


def _section_underline(text: str) -> QLabel:
    """Purple-underlined section label (PRD C11)."""
    lab = QLabel(text)
    lab.setStyleSheet(
        f"color: {OFFWHITE}; font-size: 13px; font-weight: bold; "
        f"padding-bottom: 2px; border-bottom: 2px solid {PURPLE};"
    )
    return lab


def _zscore_color(z: float) -> QColor:
    """PRD C11 — color z-score by magnitude."""
    az = abs(float(z))
    if az >= 3:
        return QColor(ERROR)
    if az >= 2:
        return QColor(PINK)
    return QColor(TEXT_MUTED)


# ---------------------------------------------------------------------------
# Confusion worker for a YOLO dataset val split (PRD I2 Dataset sub-tab)
# ---------------------------------------------------------------------------

class DatasetEvalWorker(QThread):
    """Run a model against a built dataset's val split and compute a
    confusion matrix using cctv_yolo.metrics.compute_confusion_matrix.

    The val split is the directory ``<dataset_root>/images/val`` + its
    sibling ``labels/val`` (YOLO-formatted .txt files).
    """

    progress = Signal(int)
    log_line = Signal(str)
    finished_ok = Signal(dict, str)  # eval-result, png path
    failed = Signal(str)

    def __init__(self, dataset_root: Path, model_path: str, models_dir: Path,
                 conf: float = 0.25, parent=None):
        super().__init__(parent)
        self.dataset_root = Path(dataset_root)
        self.model_path = model_path
        self.models_dir = Path(models_dir)
        self.conf = conf

    def run(self):
        try:
            import cv2

            from ultralytics import YOLO

            images_val = self.dataset_root / "images" / "val"
            labels_val = self.dataset_root / "labels" / "val"
            if not images_val.exists() or not labels_val.exists():
                raise FileNotFoundError(
                    f"Dataset missing val split: {self.dataset_root}"
                )

            # Read class names from data.yaml
            class_names: list[str] = []
            yaml_path = self.dataset_root / "data.yaml"
            if yaml_path.exists():
                with open(yaml_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("names:"):
                            raw = line.split("names:", 1)[1].strip().strip("[]")
                            class_names = [
                                p.strip().strip("'").strip('"')
                                for p in raw.split(",") if p.strip()
                            ]
                            break

            local = self.models_dir / self.model_path
            model = YOLO(str(local) if local.exists() else self.model_path)

            self.log_line.emit(f"Loading val images from {images_val}...")
            image_paths = sorted(images_val.glob("*.jpg")) + sorted(images_val.glob("*.png"))
            if not image_paths:
                raise FileNotFoundError("No val images found.")

            gt_per_image: dict[str, list[dict]] = {}
            for img_p in image_paths:
                lbl_p = labels_val / (img_p.stem + ".txt")
                if not lbl_p.exists():
                    continue
                img = cv2.imread(str(img_p))
                if img is None:
                    continue
                h, w = img.shape[:2]
                boxes: list[dict] = []
                with open(lbl_p, "r", encoding="utf-8") as lf:
                    for line in lf:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            ci = int(parts[0])
                            cx = float(parts[1]) * w
                            cy = float(parts[2]) * h
                            bw = float(parts[3]) * w
                            bh = float(parts[4]) * h
                        except ValueError:
                            continue
                        cname = (class_names[ci] if 0 <= ci < len(class_names)
                                 else f"class_{ci}")
                        boxes.append({
                            "bbox": [cx - bw / 2, cy - bh / 2,
                                     cx + bw / 2, cy + bh / 2],
                            "class": cname,
                        })
                gt_per_image[img_p.name] = boxes

            pred_per_image: dict[str, list[dict]] = {}
            total = len(image_paths)
            for i, img_p in enumerate(image_paths):
                try:
                    res = model.predict(
                        source=str(img_p),
                        conf=self.conf,
                        verbose=False,
                    )
                except Exception as e:
                    self.log_line.emit(f"predict failed for {img_p.name}: {e}")
                    continue
                preds: list[dict] = []
                for r in res:
                    if r.boxes is None:
                        continue
                    names = r.names or {}
                    for j in range(len(r.boxes)):
                        cid = int(r.boxes.cls[j].item())
                        cname = names.get(cid, f"class_{cid}")
                        bbox = r.boxes.xyxy[j].cpu().numpy().tolist()
                        preds.append({"bbox": bbox, "class": cname})
                pred_per_image[img_p.name] = preds
                self.progress.emit(int((i + 1) / total * 99))

            self.log_line.emit("Computing confusion matrix...")
            keys = sorted(gt_per_image.keys() | pred_per_image.keys())
            key_to_idx = {k: i for i, k in enumerate(keys)}
            gt_idx = {key_to_idx[k]: v for k, v in gt_per_image.items()}
            pr_idx = {key_to_idx[k]: v for k, v in pred_per_image.items()}

            metrics_result = compute_confusion_matrix(pr_idx, gt_idx)

            # Convert metrics.compute_confusion_matrix output -> confusion.evaluate
            # format so render_confusion_png works.
            axis = list(metrics_result.get("axis", []))
            n = len(axis)
            import numpy as np
            cm = np.zeros((n, n), dtype=np.int64)
            idx_of = {c: i for i, c in enumerate(axis)}
            for (gt_cls, pr_cls), count in metrics_result.get("matrix", {}).items():
                if gt_cls in idx_of and pr_cls in idx_of:
                    cm[idx_of[gt_cls], idx_of[pr_cls]] += int(count)
            per_class = metrics_result.get("per_class", {})
            metrics_block: dict[str, dict] = {}
            for cls in axis:
                if cls == "background":
                    continue
                m = per_class.get(cls, {})
                metrics_block[cls] = {
                    "tp": int(m.get("tp", 0)),
                    "fp": int(m.get("fp", 0)),
                    "fn": int(m.get("fn", 0)),
                    "precision": round(m.get("precision", 0.0), 3),
                    "recall": round(m.get("recall", 0.0), 3),
                    "f1": round(m.get("f1", 0.0), 3),
                }
            evaluator_payload = {
                "axis": axis,
                "confusion": cm.tolist(),
                "metrics": metrics_block,
                "aggregate": metrics_result.get("aggregate", {}),
            }
            png_path = self.dataset_root / "confusion_matrix.png"
            from cctv_yolo.confusion import render_confusion_png
            render_confusion_png(evaluator_payload, png_path)
            self.finished_ok.emit(evaluator_payload, str(png_path))
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.failed.emit(str(e))


# ---------------------------------------------------------------------------
# Shared panel widgets — each sub-tab instantiates its own set
# ---------------------------------------------------------------------------

class _PanelTrio(QWidget):
    """Three vertically-stacked panels: dataset-health, anomaly, confusion."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ---------- Dataset health ----------
        self.dh_box = QGroupBox()
        dh_layout = QVBoxLayout(self.dh_box)
        head = QHBoxLayout()
        head.addWidget(_section_underline("Dataset health"))
        head.addStretch(1)
        head.addWidget(QLabel("History:"))
        self.dh_history = QComboBox()
        self.dh_history.setMinimumWidth(220)
        head.addWidget(self.dh_history)
        dh_layout.addLayout(head)

        self.dh_summary = QLabel("(not loaded)")
        self.dh_summary.setStyleSheet(f"color: {OFFWHITE};")
        self.dh_summary.setWordWrap(True)
        dh_layout.addWidget(self.dh_summary)

        self.dh_warnings = QPlainTextEdit()
        self.dh_warnings.setReadOnly(True)
        self.dh_warnings.setMaximumBlockCount(200)
        self.dh_warnings.setMaximumHeight(140)
        self.dh_warnings.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {PANEL}; color: {YELLOW}; "
            f"border: 1px solid {BORDER}; }}"
        )
        dh_layout.addWidget(self.dh_warnings)

        self.dh_table = QTableWidget(0, 4)
        self.dh_table.setHorizontalHeaderLabels(
            ["Class", "Count", "Subclass", "Count"]
        )
        for c in range(4):
            self.dh_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.dh_table.setMaximumHeight(200)
        self.dh_table.setStyleSheet(TABLE_STYLE)
        dh_layout.addWidget(self.dh_table)

        dh_btns = QHBoxLayout()
        self.dh_refresh = QPushButton("Refresh")
        self.dh_refresh.setStyleSheet(ACTION_BTN)
        dh_btns.addWidget(self.dh_refresh)
        self.dh_save = QPushButton("Save snapshot")
        self.dh_save.setStyleSheet(GHOST_BTN)
        dh_btns.addWidget(self.dh_save)
        dh_btns.addStretch(1)
        dh_layout.addLayout(dh_btns)
        layout.addWidget(self.dh_box)

        # ---------- Anomaly detection ----------
        self.an_box = QGroupBox()
        an_layout = QVBoxLayout(self.an_box)
        an_head = QHBoxLayout()
        an_head.addWidget(_section_underline("Anomaly detection"))
        an_head.addStretch(1)
        an_head.addWidget(QLabel("History:"))
        self.an_history = QComboBox()
        self.an_history.setMinimumWidth(220)
        an_head.addWidget(self.an_history)
        an_layout.addLayout(an_head)

        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Z >="))
        self.an_z = QDoubleSpinBox()
        self.an_z.setRange(0.5, 6.0)
        self.an_z.setSingleStep(0.5)
        self.an_z.setValue(2.0)
        ctl.addWidget(self.an_z)
        self.an_run = QPushButton("Detect anomalies")
        self.an_run.setStyleSheet(ACTION_BTN)
        ctl.addWidget(self.an_run)
        ctl.addStretch(1)
        an_layout.addLayout(ctl)

        # 7 cols — first is session_id (used by Group / Multi / Dataset views).
        self.an_table = QTableWidget(0, 7)
        self.an_table.setHorizontalHeaderLabels(
            ["Session", "Metric", "ROI", "Hour", "Value", "Baseline", "Z"]
        )
        for c in range(7):
            self.an_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.an_table.setMaximumHeight(240)
        self.an_table.setStyleSheet(TABLE_STYLE)
        an_layout.addWidget(self.an_table)
        layout.addWidget(self.an_box)

        # ---------- Confusion matrix ----------
        self.cm_box = QGroupBox()
        cm_layout = QVBoxLayout(self.cm_box)
        cm_head = QHBoxLayout()
        cm_head.addWidget(_section_underline("Confusion matrix"))
        cm_head.addStretch(1)
        cm_head.addWidget(QLabel("History:"))
        self.cm_history = QComboBox()
        self.cm_history.setMinimumWidth(220)
        cm_head.addWidget(self.cm_history)
        cm_layout.addLayout(cm_head)

        cm_ctl = QHBoxLayout()
        cm_ctl.addWidget(QLabel("Model:"))
        self.cm_model = QComboBox()
        cm_ctl.addWidget(self.cm_model)

        cm_ctl.addWidget(QLabel("Stride:"))
        self.cm_stride = QSpinBox()
        self.cm_stride.setRange(1, 30)
        self.cm_stride.setValue(2)
        cm_ctl.addWidget(self.cm_stride)

        self.cm_run = QPushButton("Evaluate")
        self.cm_run.setStyleSheet(ACTION_BTN)
        cm_ctl.addWidget(self.cm_run)
        cm_ctl.addStretch(1)
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
        self.cm_metrics.setMaximumHeight(200)
        self.cm_metrics.setStyleSheet(TABLE_STYLE)
        cm_layout.addWidget(self.cm_metrics)

        self.cm_image = QLabel()
        self.cm_image.setAlignment(Qt.AlignCenter)
        self.cm_image.setMinimumHeight(280)
        self.cm_image.setStyleSheet(
            f"background:{INDIGO}; border:1px solid {BORDER};"
        )
        cm_layout.addWidget(self.cm_image)

        self.cm_log = QPlainTextEdit()
        self.cm_log.setReadOnly(True)
        self.cm_log.setMaximumBlockCount(400)
        self.cm_log.setMaximumHeight(120)
        self.cm_log.setFont(QFont("Menlo", 10))
        self.cm_log.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {INDIGO}; color: {OFFWHITE}; "
            f"border: 1px solid {BORDER}; }}"
        )
        cm_layout.addWidget(self.cm_log)

        layout.addWidget(self.cm_box)

    # Convenience render helpers used by every sub-tab ---------------------

    def render_health(self, report: dict, source_label: str = ""):
        warns = dataset_health.health_warnings(report)
        roi_suffix = (f" (ROI: {report.get('roi_id')})"
                      if report.get('roi_id') else "")
        prefix = f"<b>{source_label}</b><br>" if source_label else ""
        summary = (
            f"{prefix}"
            f"<b>{report.get('n_sessions_corrected', 0)}</b> corrected sessions "
            f"(of {report.get('n_sessions_total', 0)}); "
            f"<b>{report.get('n_train', 0)}</b> train / "
            f"<b>{report.get('n_val', 0)}</b> val.<br>"
            f"<b>{report.get('n_bboxes', 0)}</b> bboxes total, "
            f"size buckets: small={report.get('size_buckets', {}).get('small', 0)}, "
            f"medium={report.get('size_buckets', {}).get('medium', 0)}, "
            f"large={report.get('size_buckets', {}).get('large', 0)}. "
            f"Median area: {report.get('median_area_px', 0)} px^2. "
            f"Frame coverage: {report.get('frame_coverage', 0)*100:.1f}%."
            f"{roi_suffix}"
        )
        self.dh_summary.setText(summary)
        self.dh_warnings.clear()
        if warns:
            for w in warns:
                self.dh_warnings.appendPlainText("- " + w)
        else:
            self.dh_warnings.appendPlainText("- No warnings.")

        classes = sorted(report.get("classes", {}).items(), key=lambda kv: -kv[1])
        subs = sorted(report.get("subclasses", {}).items(), key=lambda kv: -kv[1])
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

    def render_dataset_health(self, report: dict, source_label: str = ""):
        """Variant for built-YOLO-dataset stats (PRD I2 Dataset sub-tab)."""
        splits = report.get("splits", {})
        prefix = f"<b>{source_label}</b><br>" if source_label else ""
        summary = (
            f"{prefix}"
            f"Train: {splits.get('train', {}).get('images', 0)} images, "
            f"{splits.get('train', {}).get('bboxes', 0)} bboxes. "
            f"Val: {splits.get('val', {}).get('images', 0)} images, "
            f"{splits.get('val', {}).get('bboxes', 0)} bboxes.<br>"
            f"Total: <b>{report.get('n_labels', 0)}</b> label files, "
            f"<b>{report.get('n_bboxes', 0)}</b> bboxes. "
            f"Classes: {', '.join(report.get('class_names') or []) or '(none)'}."
        )
        self.dh_summary.setText(summary)
        self.dh_warnings.clear()
        if report.get("n_bboxes", 0) < 200:
            self.dh_warnings.appendPlainText(
                "- Only a few hundred labels - train a small head only."
            )
        else:
            self.dh_warnings.appendPlainText("- No warnings.")

        classes = sorted(report.get("classes", {}).items(), key=lambda kv: -kv[1])
        self.dh_table.setRowCount(len(classes))
        for i, (k, v) in enumerate(classes):
            self.dh_table.setItem(i, 0, QTableWidgetItem(k))
            self.dh_table.setItem(i, 1, QTableWidgetItem(str(v)))
            self.dh_table.setItem(i, 2, QTableWidgetItem(""))
            self.dh_table.setItem(i, 3, QTableWidgetItem(""))

    def render_anomalies(self, anomalies, *, show_session: bool = False):
        self.an_table.setRowCount(len(anomalies))
        for i, a in enumerate(anomalies):
            sid_item = QTableWidgetItem(a.session_id if show_session else "")
            sid_item.setForeground(QColor(TEXT_MUTED))
            self.an_table.setItem(i, 0, sid_item)
            self.an_table.setItem(i, 1, QTableWidgetItem(a.metric))
            self.an_table.setItem(i, 2, QTableWidgetItem(a.roi))
            self.an_table.setItem(i, 3, QTableWidgetItem(str(a.hour)))
            self.an_table.setItem(i, 4, QTableWidgetItem(str(a.value)))
            self.an_table.setItem(
                i, 5,
                QTableWidgetItem(f"{a.baseline_mean} +/- {a.baseline_std}"),
            )
            zi = QTableWidgetItem(str(a.z_score))
            zi.setForeground(_zscore_color(a.z_score))
            self.an_table.setItem(i, 6, zi)

    def render_confusion(self, result: dict, png_path: Optional[str]):
        self.cm_progress.setValue(100)
        metrics = result.get("metrics", {})
        self.cm_metrics.setRowCount(len(metrics))
        for i, (cls, m) in enumerate(metrics.items()):
            self.cm_metrics.setItem(i, 0, QTableWidgetItem(cls))
            self.cm_metrics.setItem(i, 1, QTableWidgetItem(str(m.get("tp", 0))))
            self.cm_metrics.setItem(i, 2, QTableWidgetItem(str(m.get("fp", 0))))
            self.cm_metrics.setItem(i, 3, QTableWidgetItem(str(m.get("fn", 0))))
            self.cm_metrics.setItem(
                i, 4, QTableWidgetItem(
                    f"P {m.get('precision', 0)} - "
                    f"R {m.get('recall', 0)} - "
                    f"F1 {m.get('f1', 0)}"
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


# ---------------------------------------------------------------------------
# Sub-tabs
# ---------------------------------------------------------------------------

class _SubTabBase(QWidget):
    """Common scaffolding: scroll area + panel trio + OpenLocationBar."""

    def __init__(self, dm, parent=None):
        super().__init__(parent)
        self.dm = dm
        self._cm_worker = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # Selector row sits at the top of every sub-tab; subclasses fill it
        self.header_row = QHBoxLayout()
        outer.addLayout(self.header_row)

        # OpenLocationBar (PRD C12)
        self.olb = OpenLocationBar(self)
        outer.addWidget(self.olb)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)
        self.panels = _PanelTrio(body)
        body_layout.addWidget(self.panels)
        body_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, stretch=1)

        # Populate model dropdown
        self._refresh_models()

    def _refresh_models(self):
        self.panels.cm_model.clear()
        models = self.dm.list_models() or ["yolov8m.pt"]
        self.panels.cm_model.addItems(models)
        last = self.dm.get_last_model()
        if last and last in models:
            self.panels.cm_model.setCurrentText(last)

    # History helpers ------------------------------------------------------

    def _refresh_health_history(self, folder: Path):
        cb = self.panels.dh_history
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("-- current --", None)
        for p in dataset_health.list_health_history(folder):
            cb.addItem(p.stem.replace("dataset_health_", ""), str(p))
        cb.blockSignals(False)

    def _refresh_anom_history(self, folder: Path):
        cb = self.panels.an_history
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("-- current --", None)
        for p in anomaly.list_anomaly_history(folder):
            cb.addItem(p.stem.replace("anomalies_", ""), str(p))
        cb.blockSignals(False)

    def _refresh_cm_history(self, folder: Path):
        cb = self.panels.cm_history
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("-- current --", None)
        for p in confusion_mod.list_confusion_history(folder):
            cb.addItem(p.stem.replace("confusion_", ""), str(p))
        cb.blockSignals(False)


# --- Session ----------------------------------------------------------------

class _SessionSubTab(_SubTabBase):
    """Single session (existing UX, now persisted and themed)."""

    review_requested = Signal(str)

    def __init__(self, dm, parent=None):
        super().__init__(dm, parent)
        self._current_sid: Optional[str] = None

        self.header_row.addWidget(QLabel("Session:"))
        self.session_cb = QComboBox()
        self.session_cb.setMinimumWidth(320)
        self.session_cb.currentIndexChanged.connect(self._on_session_changed)
        self.header_row.addWidget(self.session_cb, stretch=1)
        self.header_row.addStretch(1)

        self.olb.add_folder(
            "Insights Exports",
            lambda: self.dm.exports_dir / "insights" / (self._current_sid or ""),
        )
        self.olb.add_folder("Corrections", self.dm.corrections_dir)

        # Wire panel actions
        self.panels.dh_refresh.clicked.connect(self._refresh_health)
        self.panels.dh_save.clicked.connect(self._save_health_snapshot)
        self.panels.an_run.clicked.connect(self._run_anomalies)
        self.panels.cm_run.clicked.connect(self._run_confusion)
        self.panels.dh_history.currentIndexChanged.connect(self._load_dh_history)
        self.panels.an_history.currentIndexChanged.connect(self._load_an_history)
        self.panels.cm_history.currentIndexChanged.connect(self._load_cm_history)

        self.refresh()

    def refresh(self):
        sessions = self.dm.get_sessions()
        self.session_cb.blockSignals(True)
        self.session_cb.clear()
        for s in sessions:
            self.session_cb.addItem(s["video_name"], s["id"])
        self.session_cb.blockSignals(False)
        self._refresh_models()
        self._on_session_changed()

    def _on_session_changed(self):
        sid = self.session_cb.currentData()
        self._current_sid = sid
        folder = self._folder()
        self._refresh_health_history(folder)
        self._refresh_anom_history(folder)
        self._refresh_cm_history(folder)
        self._refresh_health()

    def _folder(self) -> Path:
        return self.dm.exports_dir / "insights" / (self._current_sid or "_no_session")

    def _refresh_health(self):
        if not self._current_sid:
            return
        report = dataset_health.collect_health(self.dm, session_ids=[self._current_sid])
        sid_label = self.session_cb.currentText()
        self.panels.render_health(report, source_label=f"Session: {sid_label}")

    def _save_health_snapshot(self):
        if not self._current_sid:
            return
        report = dataset_health.collect_health(self.dm, session_ids=[self._current_sid])
        dataset_health.save_health(report, self._folder())
        self._refresh_anom_history(self._folder())
        self._refresh_health_history(self._folder())
        QMessageBox.information(self, "Saved", "Dataset health snapshot saved.")

    def _run_anomalies(self):
        if not self._current_sid:
            return
        anomalies = anomaly.detect_anomalies(
            self.dm, self._current_sid, z_threshold=self.panels.an_z.value()
        )
        self.panels.render_anomalies(anomalies, show_session=False)
        if anomalies:
            anomaly.save_anomalies(anomalies, self._folder())
            self._refresh_anom_history(self._folder())

    def _run_confusion(self):
        if not self._current_sid:
            QMessageBox.warning(self, "No session", "Pick a session.")
            return
        if not self.dm.has_corrections(self._current_sid):
            QMessageBox.warning(self, "No corrections",
                                "This session has no corrections.")
            return
        model = self.panels.cm_model.currentText()
        worker = ConfusionMatrixWorker(
            self.dm, self._current_sid, model_path=model,
            stride=self.panels.cm_stride.value(),
        )
        worker.progress.connect(self.panels.cm_progress.setValue)
        worker.log_line.connect(self.panels.cm_log.appendPlainText)
        worker.finished_ok.connect(self._on_cm_done)
        worker.failed.connect(lambda m: QMessageBox.critical(self, "Eval failed", m))
        self._cm_worker = worker
        self.panels.cm_log.clear()
        self.panels.cm_progress.setValue(0)
        worker.start()

    def _on_cm_done(self, result, png_path):
        self.panels.render_confusion(result, png_path)
        if self._current_sid:
            confusion_mod.save_confusion(result, self._folder())
            self._refresh_cm_history(self._folder())

    def _load_dh_history(self):
        path = self.panels.dh_history.currentData()
        if not path:
            return
        report = dataset_health.load_health(Path(path))
        self.panels.render_health(report,
                                  source_label="(loaded from history)")

    def _load_an_history(self):
        path = self.panels.an_history.currentData()
        if not path:
            return
        anomalies = anomaly.load_anomalies(Path(path))
        self.panels.render_anomalies(anomalies, show_session=False)

    def _load_cm_history(self):
        path = self.panels.cm_history.currentData()
        if not path:
            return
        result = confusion_mod.load_confusion(Path(path))
        png = str(Path(path).with_suffix(".png"))
        self.panels.render_confusion(result, png if Path(png).exists() else None)


# --- Group ------------------------------------------------------------------

class _GroupSubTab(_SubTabBase):
    """Aggregate across every session in a named DataManager group."""

    def __init__(self, dm, parent=None):
        super().__init__(dm, parent)
        self._current_gid: Optional[str] = None

        self.header_row.addWidget(QLabel("Group:"))
        self.group_cb = QComboBox()
        self.group_cb.setMinimumWidth(280)
        self.group_cb.currentIndexChanged.connect(self._on_group_changed)
        self.header_row.addWidget(self.group_cb, stretch=1)
        self.header_row.addStretch(1)

        self.olb.add_folder(
            "Group Insights",
            lambda: (self.dm.exports_dir / "insights" / "groups"
                     / (self._current_gid or "")),
        )
        self.olb.add_folder("Corrections", self.dm.corrections_dir)

        self.panels.dh_refresh.clicked.connect(self._refresh_health)
        self.panels.dh_save.clicked.connect(self._save_health_snapshot)
        self.panels.an_run.clicked.connect(self._run_anomalies)
        self.panels.cm_run.clicked.connect(self._run_confusion)
        self.panels.dh_history.currentIndexChanged.connect(self._load_dh_history)
        self.panels.an_history.currentIndexChanged.connect(self._load_an_history)
        self.panels.cm_history.currentIndexChanged.connect(self._load_cm_history)

        # Auto-refresh subscriptions
        try:
            dm.groups_changed.connect(self.refresh_groups)
            dm.corrections_changed.connect(lambda _sid: self._refresh_health())
        except Exception:
            pass

        self.refresh_groups()

    def refresh_groups(self):
        gid_before = self.group_cb.currentData()
        self.group_cb.blockSignals(True)
        self.group_cb.clear()
        for g in self.dm.list_groups():
            self.group_cb.addItem(f"{g['name']} ({len(g.get('session_ids', []))})",
                                  g["id"])
        if gid_before is not None:
            idx = self.group_cb.findData(gid_before)
            if idx >= 0:
                self.group_cb.setCurrentIndex(idx)
        self.group_cb.blockSignals(False)
        self._refresh_models()
        self._on_group_changed()

    def _on_group_changed(self):
        self._current_gid = self.group_cb.currentData()
        folder = self._folder()
        self._refresh_health_history(folder)
        self._refresh_anom_history(folder)
        self._refresh_cm_history(folder)
        self._refresh_health()

    def _folder(self) -> Path:
        return (self.dm.exports_dir / "insights" / "groups"
                / (self._current_gid or "_no_group"))

    def _session_ids(self) -> list[str]:
        if not self._current_gid:
            return []
        return [s["id"] for s in self.dm.get_sessions_in_group(self._current_gid)]

    def _refresh_health(self):
        sids = self._session_ids()
        if not sids:
            self.panels.render_health({}, source_label="(empty group)")
            return
        report = dataset_health.collect_health(self.dm, session_ids=sids)
        self.panels.render_health(
            report,
            source_label=f"Group: {self.group_cb.currentText()}",
        )

    def _save_health_snapshot(self):
        sids = self._session_ids()
        if not sids:
            return
        report = dataset_health.collect_health(self.dm, session_ids=sids)
        dataset_health.save_health(report, self._folder())
        self._refresh_health_history(self._folder())
        QMessageBox.information(self, "Saved", "Group health snapshot saved.")

    def _run_anomalies(self):
        sids = self._session_ids()
        if not sids:
            QMessageBox.warning(self, "Empty group", "Group has no sessions.")
            return
        anomalies = anomaly.detect_anomalies_batch(
            self.dm, sids, z_threshold=self.panels.an_z.value()
        )
        self.panels.render_anomalies(anomalies, show_session=True)
        if anomalies:
            anomaly.save_anomalies(anomalies, self._folder())
            self._refresh_anom_history(self._folder())

    def _run_confusion(self):
        sids = [s for s in self._session_ids()
                if self.dm.has_corrections(s)]
        if not sids:
            QMessageBox.warning(self, "No corrected sessions",
                                "No sessions in this group have corrections.")
            return
        model = self.panels.cm_model.currentText()
        self.panels.cm_log.clear()
        self.panels.cm_progress.setValue(0)
        self._cm_queue = list(sids)
        self._cm_partials: list[dict] = []
        self._cm_total = len(sids)
        self._cm_completed = 0
        self._cm_model = model
        self._start_next_group_eval()

    def _start_next_group_eval(self):
        if not self._cm_queue:
            self._finalize_group_eval()
            return
        sid = self._cm_queue.pop(0)
        self.panels.cm_log.appendPlainText(f"--- Session {sid} ---")
        worker = ConfusionMatrixWorker(
            self.dm, sid, model_path=self._cm_model,
            stride=self.panels.cm_stride.value(),
        )
        worker.progress.connect(self._cm_progress)
        worker.log_line.connect(self.panels.cm_log.appendPlainText)
        worker.finished_ok.connect(self._cm_session_done)
        worker.failed.connect(self._cm_session_failed)
        self._cm_worker = worker
        worker.start()

    def _cm_progress(self, pct: int):
        overall = int(((self._cm_completed * 100) + pct) / max(1, self._cm_total))
        self.panels.cm_progress.setValue(min(99, overall))

    def _cm_session_done(self, result: dict, png_path: str):
        self._cm_partials.append(result)
        self._cm_completed += 1
        self._start_next_group_eval()

    def _cm_session_failed(self, msg: str):
        self.panels.cm_log.appendPlainText(f"[failed] {msg}")
        self._cm_completed += 1
        self._start_next_group_eval()

    def _finalize_group_eval(self):
        if not self._cm_partials:
            QMessageBox.warning(self, "No results",
                                "All per-session evaluations failed.")
            return
        # Aggregate via cell-sum (confusion.aggregate_results) plus a
        # sanity-checked aggregate via metrics.aggregate_confusion_matrices.
        agg = confusion_mod.aggregate_results(self._cm_partials)

        try:
            shaped = []
            for res in self._cm_partials:
                axis = res.get("axis", [])
                cm = res.get("confusion", [])
                matrix = {}
                for ri, gt_cls in enumerate(axis):
                    for ci, pr_cls in enumerate(axis):
                        try:
                            matrix[(gt_cls, pr_cls)] = int(cm[ri][ci])
                        except (IndexError, TypeError, ValueError):
                            pass
                shaped.append({
                    "matrix": matrix,
                    "axis": list(axis),
                    "per_class": {
                        cls: m for cls, m in (res.get("metrics", {}) or {}).items()
                    },
                    "aggregate": res.get("aggregate", {}),
                })
            metrics_agg = aggregate_confusion_matrices(shaped)
            ag = metrics_agg.get("aggregate", {})
            self.panels.cm_log.appendPlainText(
                f"Aggregate P {ag.get('precision', 0)} - "
                f"R {ag.get('recall', 0)} - "
                f"F1 {ag.get('f1', 0)} - "
                f"mAP {ag.get('mAP', 0)}"
            )
        except Exception:
            pass

        json_path, png_path = confusion_mod.save_confusion(agg, self._folder())
        self.panels.render_confusion(agg, str(png_path))
        self._refresh_cm_history(self._folder())

    # History loaders ------------------------------------------------------

    def _load_dh_history(self):
        path = self.panels.dh_history.currentData()
        if not path:
            return
        self.panels.render_health(dataset_health.load_health(Path(path)),
                                  source_label="(loaded from history)")

    def _load_an_history(self):
        path = self.panels.an_history.currentData()
        if not path:
            return
        self.panels.render_anomalies(
            anomaly.load_anomalies(Path(path)), show_session=True
        )

    def _load_cm_history(self):
        path = self.panels.cm_history.currentData()
        if not path:
            return
        result = confusion_mod.load_confusion(Path(path))
        png = str(Path(path).with_suffix(".png"))
        self.panels.render_confusion(result, png if Path(png).exists() else None)


# --- Dataset ----------------------------------------------------------------

class _DatasetSubTab(_SubTabBase):
    """Stats for a built YOLO dataset (data/training/<run>)."""

    def __init__(self, dm, parent=None):
        super().__init__(dm, parent)
        self._current_ds_id: Optional[str] = None
        self._current_root: Optional[Path] = None

        self.header_row.addWidget(QLabel("Dataset:"))
        self.dataset_cb = QComboBox()
        self.dataset_cb.setMinimumWidth(320)
        self.dataset_cb.currentIndexChanged.connect(self._on_dataset_changed)
        self.header_row.addWidget(self.dataset_cb, stretch=1)
        browse = QPushButton("Browse...")
        browse.setStyleSheet(GHOST_BTN)
        browse.clicked.connect(self._browse)
        self.header_row.addWidget(browse)
        self.header_row.addStretch(1)

        self.olb.add_folder(
            "Dataset Insights",
            lambda: (self.dm.exports_dir / "insights" / "datasets"
                     / (self._current_ds_id or "")),
        )
        self.olb.add_folder("Datasets", lambda: self.dm.data_dir / "training")

        self.panels.dh_refresh.clicked.connect(self._refresh_health)
        self.panels.dh_save.clicked.connect(self._save_health_snapshot)
        self.panels.an_run.clicked.connect(self._run_anomalies)
        self.panels.cm_run.clicked.connect(self._run_confusion)
        self.panels.dh_history.currentIndexChanged.connect(self._load_dh_history)
        self.panels.an_history.currentIndexChanged.connect(self._load_an_history)
        self.panels.cm_history.currentIndexChanged.connect(self._load_cm_history)

        self.refresh_datasets()

    def refresh_datasets(self):
        self.dataset_cb.blockSignals(True)
        prev = self.dataset_cb.currentData()
        self.dataset_cb.clear()
        training_root = self.dm.data_dir / "training"
        if training_root.exists():
            for ds in sorted(training_root.iterdir()):
                if not ds.is_dir():
                    continue
                if not (ds / "data.yaml").exists():
                    continue
                self.dataset_cb.addItem(ds.name, str(ds))
        if prev:
            idx = self.dataset_cb.findData(prev)
            if idx >= 0:
                self.dataset_cb.setCurrentIndex(idx)
        self.dataset_cb.blockSignals(False)
        self._refresh_models()
        self._on_dataset_changed()

    def _browse(self):
        start = str(self.dm.data_dir / "training")
        path = QFileDialog.getExistingDirectory(self, "Pick a YOLO dataset", start)
        if not path:
            return
        self.dataset_cb.addItem(Path(path).name, path)
        self.dataset_cb.setCurrentIndex(self.dataset_cb.count() - 1)

    def _on_dataset_changed(self):
        path = self.dataset_cb.currentData()
        if not path:
            self._current_ds_id = None
            self._current_root = None
            return
        self._current_root = Path(path)
        self._current_ds_id = self._current_root.name
        folder = self._folder()
        self._refresh_health_history(folder)
        self._refresh_anom_history(folder)
        self._refresh_cm_history(folder)
        self._refresh_health()

    def _folder(self) -> Path:
        return (self.dm.exports_dir / "insights" / "datasets"
                / (self._current_ds_id or "_no_dataset"))

    def _refresh_health(self):
        if not self._current_root:
            return
        report = dataset_health.collect_health_for_dataset(self._current_root)
        self.panels.render_dataset_health(
            report,
            source_label=f"Dataset: {self._current_ds_id}",
        )

    def _save_health_snapshot(self):
        if not self._current_root:
            return
        report = dataset_health.collect_health_for_dataset(self._current_root)
        dataset_health.save_health(report, self._folder())
        self._refresh_health_history(self._folder())
        QMessageBox.information(self, "Saved", "Dataset health snapshot saved.")

    def _run_anomalies(self):
        """For datasets, anomalies = images where the active model is most
        wrong on the val split. Rank by (FP + FN) per image after a quick
        prediction pass.
        """
        if not self._current_root:
            return
        try:
            from ultralytics import YOLO
            import cv2
        except Exception as e:  # pragma: no cover
            QMessageBox.critical(self, "Missing deps", str(e))
            return

        images_val = self._current_root / "images" / "val"
        labels_val = self._current_root / "labels" / "val"
        if not images_val.exists() or not labels_val.exists():
            QMessageBox.warning(self, "No val split",
                                "Dataset has no val split to analyse.")
            return

        class_names: list[str] = []
        yaml_path = self._current_root / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("names:"):
                        raw = line.split("names:", 1)[1].strip().strip("[]")
                        class_names = [
                            p.strip().strip("'").strip('"')
                            for p in raw.split(",") if p.strip()
                        ]
                        break

        model_path = self.panels.cm_model.currentText()
        local = self.dm.models_dir / model_path
        model = YOLO(str(local) if local.exists() else model_path)

        rows = []
        image_paths = (sorted(images_val.glob("*.jpg"))
                       + sorted(images_val.glob("*.png")))
        for img_p in image_paths[:200]:  # cap to keep UI snappy
            lbl_p = labels_val / (img_p.stem + ".txt")
            img = cv2.imread(str(img_p))
            if img is None or not lbl_p.exists():
                continue
            h, w = img.shape[:2]
            gt: list[dict] = []
            with open(lbl_p, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        ci = int(parts[0])
                        cx = float(parts[1]) * w
                        cy = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h
                    except ValueError:
                        continue
                    cname = (class_names[ci] if 0 <= ci < len(class_names)
                             else f"class_{ci}")
                    gt.append({"bbox": [cx - bw / 2, cy - bh / 2,
                                        cx + bw / 2, cy + bh / 2],
                               "class": cname})
            try:
                res = model.predict(source=str(img_p), conf=0.25, verbose=False)
            except Exception:
                continue
            preds: list[dict] = []
            for r in res:
                if r.boxes is None:
                    continue
                names = r.names or {}
                for j in range(len(r.boxes)):
                    cid = int(r.boxes.cls[j].item())
                    cname = names.get(cid, f"class_{cid}")
                    bbox = r.boxes.xyxy[j].cpu().numpy().tolist()
                    preds.append({"bbox": bbox, "class": cname})
            res = compute_confusion_matrix({0: preds}, {0: gt})
            tp_total = sum(v.get("tp", 0) for v in res.get("per_class", {}).values())
            fp_total = sum(v.get("fp", 0) for v in res.get("per_class", {}).values())
            fn_total = sum(v.get("fn", 0) for v in res.get("per_class", {}).values())
            wrongness = fp_total + fn_total
            rows.append((wrongness, img_p, tp_total, fp_total, fn_total))

        rows.sort(key=lambda r: -r[0])
        flagged = []
        for wrongness, img_p, tp_total, fp_total, fn_total in rows[:50]:
            flagged.append(anomaly.Anomaly(
                session_id=img_p.name,
                hour=0,
                roi="val",
                metric=f"errors (FP {fp_total} + FN {fn_total})",
                value=float(wrongness),
                baseline_mean=float(tp_total),
                baseline_std=0.0,
                z_score=float(wrongness),
            ))
        self.panels.render_anomalies(flagged, show_session=True)
        if flagged:
            anomaly.save_anomalies(flagged, self._folder())
            self._refresh_anom_history(self._folder())

    def _run_confusion(self):
        if not self._current_root:
            QMessageBox.warning(self, "No dataset", "Pick a dataset.")
            return
        worker = DatasetEvalWorker(
            self._current_root,
            self.panels.cm_model.currentText(),
            self.dm.models_dir,
        )
        worker.progress.connect(self.panels.cm_progress.setValue)
        worker.log_line.connect(self.panels.cm_log.appendPlainText)
        worker.finished_ok.connect(self._on_cm_done)
        worker.failed.connect(lambda m: QMessageBox.critical(self, "Eval failed", m))
        self._cm_worker = worker
        self.panels.cm_log.clear()
        self.panels.cm_progress.setValue(0)
        worker.start()

    def _on_cm_done(self, result: dict, png_path: str):
        confusion_mod.save_confusion(result, self._folder())
        self.panels.render_confusion(result, png_path)
        self._refresh_cm_history(self._folder())

    def _load_dh_history(self):
        path = self.panels.dh_history.currentData()
        if not path:
            return
        report = dataset_health.load_health(Path(path))
        if "splits" in report:
            self.panels.render_dataset_health(report, source_label="(history)")
        else:
            self.panels.render_health(report, source_label="(history)")

    def _load_an_history(self):
        path = self.panels.an_history.currentData()
        if not path:
            return
        self.panels.render_anomalies(
            anomaly.load_anomalies(Path(path)), show_session=True
        )

    def _load_cm_history(self):
        path = self.panels.cm_history.currentData()
        if not path:
            return
        result = confusion_mod.load_confusion(Path(path))
        png = str(Path(path).with_suffix(".png"))
        self.panels.render_confusion(result, png if Path(png).exists() else None)


# --- Multi ------------------------------------------------------------------

class _MultiSubTab(_SubTabBase):
    """Ad-hoc multi-select via :class:`GroupPickerDialog`."""

    def __init__(self, dm, parent=None):
        super().__init__(dm, parent)
        self._selected: list[str] = []

        self.header_row.addWidget(QLabel("Sessions:"))
        self.sel_label = QLabel("(none selected)")
        self.sel_label.setStyleSheet(f"color: {TEXT_MUTED};")
        self.header_row.addWidget(self.sel_label, stretch=1)
        pick = QPushButton("Pick sessions...")
        pick.setStyleSheet(ACTION_BTN)
        pick.clicked.connect(self._pick)
        self.header_row.addWidget(pick)
        self.header_row.addStretch(1)

        self.olb.add_folder(
            "Multi Insights",
            lambda: self.dm.exports_dir / "insights" / "multi",
        )
        self.olb.add_folder("Corrections", self.dm.corrections_dir)

        self.panels.dh_refresh.clicked.connect(self._refresh_health)
        self.panels.dh_save.clicked.connect(self._save_health_snapshot)
        self.panels.an_run.clicked.connect(self._run_anomalies)
        self.panels.cm_run.clicked.connect(self._run_confusion)
        self.panels.dh_history.currentIndexChanged.connect(self._load_dh_history)
        self.panels.an_history.currentIndexChanged.connect(self._load_an_history)
        self.panels.cm_history.currentIndexChanged.connect(self._load_cm_history)

        folder = self._folder()
        self._refresh_health_history(folder)
        self._refresh_anom_history(folder)
        self._refresh_cm_history(folder)

    def _folder(self) -> Path:
        return self.dm.exports_dir / "insights" / "multi"

    def _pick(self):
        try:
            from cctv_yolo.widgets.group_picker_dialog import GroupPickerDialog
        except Exception:
            QMessageBox.critical(self, "Missing widget",
                                 "GroupPickerDialog not available.")
            return
        dlg = GroupPickerDialog(
            self.dm.get_sessions(),
            already_in=[],
            title="Pick sessions for ad-hoc Insights",
            parent=self,
        )
        if dlg.exec():
            self._selected = list(dlg.selected_session_ids())
            self.sel_label.setText(
                f"{len(self._selected)} sessions: "
                + ", ".join(self._selected[:3])
                + ("..." if len(self._selected) > 3 else "")
            )
            self._refresh_models()
            self._refresh_health()

    def _refresh_health(self):
        if not self._selected:
            return
        report = dataset_health.collect_health(self.dm, session_ids=self._selected)
        self.panels.render_health(
            report, source_label=f"Multi: {len(self._selected)} sessions"
        )

    def _save_health_snapshot(self):
        if not self._selected:
            return
        report = dataset_health.collect_health(self.dm, session_ids=self._selected)
        dataset_health.save_health(report, self._folder())
        self._refresh_health_history(self._folder())
        QMessageBox.information(self, "Saved", "Multi-session health snapshot saved.")

    def _run_anomalies(self):
        if not self._selected:
            QMessageBox.warning(self, "No selection", "Pick at least one session.")
            return
        anomalies = anomaly.detect_anomalies_batch(
            self.dm, self._selected, z_threshold=self.panels.an_z.value()
        )
        self.panels.render_anomalies(anomalies, show_session=True)
        if anomalies:
            anomaly.save_anomalies(anomalies, self._folder())
            self._refresh_anom_history(self._folder())

    def _run_confusion(self):
        sids = [s for s in self._selected if self.dm.has_corrections(s)]
        if not sids:
            QMessageBox.warning(self, "No corrections",
                                "Pick sessions that have corrections.")
            return
        self.panels.cm_log.clear()
        self.panels.cm_progress.setValue(0)
        self._cm_queue = list(sids)
        self._cm_partials: list[dict] = []
        self._cm_total = len(sids)
        self._cm_completed = 0
        self._cm_model = self.panels.cm_model.currentText()
        self._start_next_eval()

    def _start_next_eval(self):
        if not self._cm_queue:
            self._finalize_eval()
            return
        sid = self._cm_queue.pop(0)
        self.panels.cm_log.appendPlainText(f"--- Session {sid} ---")
        worker = ConfusionMatrixWorker(
            self.dm, sid, model_path=self._cm_model,
            stride=self.panels.cm_stride.value(),
        )
        worker.progress.connect(self._cm_progress)
        worker.log_line.connect(self.panels.cm_log.appendPlainText)
        worker.finished_ok.connect(self._cm_session_done)
        worker.failed.connect(self._cm_session_failed)
        self._cm_worker = worker
        worker.start()

    def _cm_progress(self, pct: int):
        overall = int(((self._cm_completed * 100) + pct) / max(1, self._cm_total))
        self.panels.cm_progress.setValue(min(99, overall))

    def _cm_session_done(self, result, _png):
        self._cm_partials.append(result)
        self._cm_completed += 1
        self._start_next_eval()

    def _cm_session_failed(self, msg: str):
        self.panels.cm_log.appendPlainText(f"[failed] {msg}")
        self._cm_completed += 1
        self._start_next_eval()

    def _finalize_eval(self):
        if not self._cm_partials:
            QMessageBox.warning(self, "No results",
                                "All per-session evaluations failed.")
            return
        agg = confusion_mod.aggregate_results(self._cm_partials)
        _, png_path = confusion_mod.save_confusion(agg, self._folder())
        self.panels.render_confusion(agg, str(png_path))
        self._refresh_cm_history(self._folder())

    def _load_dh_history(self):
        path = self.panels.dh_history.currentData()
        if not path:
            return
        self.panels.render_health(dataset_health.load_health(Path(path)),
                                  source_label="(history)")

    def _load_an_history(self):
        path = self.panels.an_history.currentData()
        if not path:
            return
        self.panels.render_anomalies(
            anomaly.load_anomalies(Path(path)), show_session=True
        )

    def _load_cm_history(self):
        path = self.panels.cm_history.currentData()
        if not path:
            return
        result = confusion_mod.load_confusion(Path(path))
        png = str(Path(path).with_suffix(".png"))
        self.panels.render_confusion(result, png if Path(png).exists() else None)


# ---------------------------------------------------------------------------
# Public widget — InsightsTab
# ---------------------------------------------------------------------------

class InsightsTab(QWidget):
    """Top-level tab that hosts the four sub-tabs.

    The legacy ``review_requested`` signal is kept for compatibility with
    callers (main_window wires it to the review window).
    """

    review_requested = Signal(str)

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QLabel("Insights")
        header.setStyleSheet(
            f"color: {PURPLE}; font-size: 18px; font-weight: bold; "
            f"padding-bottom: 4px; border-bottom: 2px solid {PURPLE};"
        )
        layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"QTabBar::tab {{ background: transparent; color: {OFFWHITE}; "
            f"padding: 6px 12px; border: 1px solid {BORDER}; "
            f"border-bottom: none; }}"
            f"QTabBar::tab:selected {{ background: {PANEL}; color: {PINK}; "
            f"border-bottom: 2px solid {PURPLE}; }}"
            f"QTabWidget::pane {{ border: 1px solid {BORDER}; top: -1px; }}"
        )

        self.session_tab = _SessionSubTab(self.dm)
        self.group_tab = _GroupSubTab(self.dm)
        self.dataset_tab = _DatasetSubTab(self.dm)
        self.multi_tab = _MultiSubTab(self.dm)

        # Plumb the legacy review_requested signal up through the Session tab
        try:
            self.session_tab.review_requested.connect(self.review_requested)
        except Exception:
            pass

        self.tabs.addTab(self.session_tab, "(i) Session")
        self.tabs.addTab(self.group_tab, "Group")
        self.tabs.addTab(self.dataset_tab, "Dataset")
        self.tabs.addTab(self.multi_tab, "Multi")

        layout.addWidget(self.tabs, stretch=1)

    def refresh(self):
        """Refresh every sub-tab. Called by main_window on tab switch."""
        try:
            self.session_tab.refresh()
        except Exception:
            pass
        try:
            self.group_tab.refresh_groups()
        except Exception:
            pass
        try:
            self.dataset_tab.refresh_datasets()
        except Exception:
            pass
