"""
Confusion matrix + per-class precision/recall/F1.

Method
------
1. Pick a held-out session (one whose corrections are kept aside).
2. Run the trained model fresh on that video to get raw predictions.
3. Match predictions to corrections per-frame using IoU; the closest
   match (>= 0.5) becomes a TP, mismatched class becomes a confusion,
   unmatched corrections become FN, unmatched predictions become FP.
4. Aggregate into a confusion matrix and per-class metrics.

PNG render uses pure OpenCV so we don't need matplotlib.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal


from cctv_yolo import classes as class_registry


# ---------------------------------------------------------------------------
# Colormap — PRD C11. We replace OpenCV's COLORMAP_VIRIDIS with a 3-stop
# INDIGO → PURPLE → PINK ramp so the matrix art harmonises with the rest of
# the tab. Stop colors are pulled from cctv_yolo.theme.
# ---------------------------------------------------------------------------

def _hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _theme_colormap_bgr(t: float) -> tuple[int, int, int]:
    """t in [0, 1] → BGR. INDIGO → PURPLE → PINK."""
    from cctv_yolo.theme import INDIGO, PURPLE, PINK
    c0 = _hex_to_bgr(INDIGO)
    c1 = _hex_to_bgr(PURPLE)
    c2 = _hex_to_bgr(PINK)
    t = max(0.0, min(1.0, float(t)))
    if t <= 0.5:
        k = t / 0.5
        a, b = c0, c1
    else:
        k = (t - 0.5) / 0.5
        a, b = c1, c2
    return (
        int(a[0] + (b[0] - a[0]) * k),
        int(a[1] + (b[1] - a[1]) * k),
        int(a[2] + (b[2] - a[2]) * k),
    )


def _device():
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aarea = (ax2 - ax1) * (ay2 - ay1)
    barea = (bx2 - bx1) * (by2 - by1)
    union = aarea + barea - inter
    return inter / union if union > 0 else 0.0


def _per_frame_index(track_data: dict) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for tr in track_data.get("tracks", []):
        cls = tr.get("class", "vehicle")
        for fd in tr.get("frames", []):
            if fd.get("interpolated"):
                continue
            out[fd["frame"]].append({"bbox": fd["bbox"], "class": cls})
    return out


def run_predictions(
    video_path: Path,
    model_path: str,
    models_dir: Path,
    conf: float = 0.25,
    stride: int = 1,
    progress_callback=None,
) -> dict[int, list[dict]]:
    """Run a fresh detection pass and return per-frame predictions."""
    from ultralytics import YOLO

    local = Path(models_dir) / model_path
    model = YOLO(str(local) if local.exists() else model_path)
    model.to(_device())
    _filt, _names = class_registry.detect_mapping_for_model(model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    results = model.predict(
        source=str(video_path),
        conf=conf,
        classes=_filt,
        stream=True,
        verbose=False,
        vid_stride=stride,
    )
    out: dict[int, list[dict]] = defaultdict(list)
    last_pct = -1
    for idx, r in enumerate(results):
        if r.boxes is None:
            continue
        frame_idx = idx * stride
        for i in range(len(r.boxes)):
            cid = int(r.boxes.cls[i].item())
            cname = _names.get(cid, "unknown")
            bbox = r.boxes.xyxy[i].cpu().numpy().tolist()
            out[frame_idx].append({"bbox": bbox, "class": cname})
        if progress_callback and total > 0:
            pct = int((idx + 1) / max(1, total / max(1, stride)) * 100)
            pct = min(pct, 99)
            if pct != last_pct:
                progress_callback(pct)
                last_pct = pct
    return out


def evaluate(
    gt_per_frame: dict[int, list[dict]],
    pred_per_frame: dict[int, list[dict]],
    iou_threshold: float = 0.5,
    classes: Optional[list[str]] = None,
) -> dict:
    """Match predictions to ground-truth via greedy IoU and produce a
    confusion matrix + per-class metrics.

    Adds a special "background" pseudo-class on each axis to capture
    FP (predicted but no GT) and FN (GT but no prediction).
    """
    if classes is None:
        classes = sorted({d["class"] for boxes in gt_per_frame.values() for d in boxes}
                         | {d["class"] for boxes in pred_per_frame.values() for d in boxes})

    bg = "background"
    axis = list(classes) + [bg]
    cm = np.zeros((len(axis), len(axis)), dtype=np.int64)
    idx_of = {c: i for i, c in enumerate(axis)}

    all_frames = set(gt_per_frame.keys()) | set(pred_per_frame.keys())
    for f in all_frames:
        gts = list(gt_per_frame.get(f, []))
        preds = list(pred_per_frame.get(f, []))
        used_pred = set()
        # Greedy match each gt to its highest-IoU pred
        for gi, g in enumerate(gts):
            best, best_iou = -1, 0.0
            for pi, p in enumerate(preds):
                if pi in used_pred:
                    continue
                iou = _iou(g["bbox"], p["bbox"])
                if iou > best_iou:
                    best, best_iou = pi, iou
            if best >= 0 and best_iou >= iou_threshold:
                used_pred.add(best)
                cm[idx_of.get(g["class"], -1), idx_of.get(preds[best]["class"], -1)] += 1
            else:
                cm[idx_of.get(g["class"], -1), idx_of[bg]] += 1
        for pi, p in enumerate(preds):
            if pi in used_pred:
                continue
            cm[idx_of[bg], idx_of.get(p["class"], -1)] += 1

    # Per-class precision/recall/F1
    metrics = {}
    for cls in classes:
        i = idx_of[cls]
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
        }
    return {"axis": axis, "confusion": cm.tolist(), "metrics": metrics}


def render_confusion_png(eval_result: dict, output_path: Path) -> Path:
    """Render the confusion matrix as a PNG using pure OpenCV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    axis = eval_result["axis"]
    cm = np.array(eval_result["confusion"])
    n = len(axis)
    cell = 80
    label_left = 200
    label_top = 60
    title_h = 40
    margin = 24

    w = label_left + cell * n + margin
    h = title_h + label_top + cell * n + margin
    img = np.full((h, w, 3), 28, dtype=np.uint8)

    cv2.putText(img, "Confusion Matrix (rows=GT, cols=Pred)",
                (margin, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

    # Color cells by normalized count — uses theme colormap (PRD C11):
    # INDIGO -> PURPLE -> PINK gradient instead of OpenCV's default.
    max_v = max(1, int(cm.max()))
    for r in range(n):
        for c in range(n):
            x = label_left + c * cell
            y = title_h + label_top + r * cell
            v = cm[r, c]
            t = v / max_v
            color = _theme_colormap_bgr(t)
            cv2.rectangle(img, (x, y), (x + cell - 2, y + cell - 2), color, -1)
            text = str(int(v))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(img, text,
                        (x + cell // 2 - tw // 2, y + cell // 2 + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)

    # Row labels
    for r, cls in enumerate(axis):
        y = title_h + label_top + r * cell + cell // 2 + 6
        cv2.putText(img, cls, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1, cv2.LINE_AA)
    # Column labels
    for c, cls in enumerate(axis):
        x = label_left + c * cell + 8
        cv2.putText(img, cls, (x, title_h + label_top - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                    cv2.LINE_AA)

    cv2.imwrite(str(output_path), img)
    return output_path


# ---------------------------------------------------------------------------
# Save / load — PRD I3 (history dropdowns in every sub-tab)
# ---------------------------------------------------------------------------

def save_confusion(result: dict, out_dir: Path, ts: Optional[str] = None) -> tuple[Path, Path]:
    """Persist a confusion result + its rendered PNG to ``out_dir``.

    Returns ``(json_path, png_path)``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"confusion_{ts}.json"
    png_path = out_dir / f"confusion_{ts}.png"
    payload = dict(result)
    payload.setdefault("saved_at", datetime.now().isoformat())
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    render_confusion_png(result, png_path)
    return json_path, png_path


def load_confusion(path: Path) -> dict:
    """Load a saved confusion result."""
    with open(Path(path), "r") as f:
        return json.load(f)


def list_confusion_history(folder: Path) -> list[Path]:
    """Return saved confusion runs under ``folder`` (newest first)."""
    p = Path(folder)
    if not p.exists():
        return []
    return sorted(p.glob("confusion_*.json"), reverse=True)


def aggregate_results(results: Iterable[dict]) -> dict:
    """Sum cells across multiple :func:`evaluate` results.

    Used by the Group / Multi sub-tabs in Insights — adds per-session matrices
    cell-wise then recomputes precision/recall/F1.
    """
    results = list(results)
    if not results:
        return {"axis": ["background"], "confusion": [[0]], "metrics": {}}

    axis_set: set[str] = set()
    for r in results:
        axis_set.update(r.get("axis", []))
    if "background" in axis_set:
        axis_set.discard("background")
    classes = sorted(axis_set)
    axis = list(classes) + ["background"]
    idx_of = {c: i for i, c in enumerate(axis)}

    n = len(axis)
    cm = np.zeros((n, n), dtype=np.int64)
    for r in results:
        r_axis = r.get("axis", [])
        r_cm = np.array(r.get("confusion", []), dtype=np.int64)
        if r_cm.size == 0:
            continue
        for ri, gt_cls in enumerate(r_axis):
            for ci, pr_cls in enumerate(r_axis):
                if gt_cls in idx_of and pr_cls in idx_of:
                    cm[idx_of[gt_cls], idx_of[pr_cls]] += int(r_cm[ri, ci])

    metrics: dict[str, dict] = {}
    for cls in classes:
        i = idx_of[cls]
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
        }
    return {"axis": axis, "confusion": cm.tolist(), "metrics": metrics}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class ConfusionMatrixWorker(QThread):
    """Run predictions, evaluate against corrections, render PNG."""

    progress = Signal(int)
    log_line = Signal(str)
    finished_ok = Signal(dict, str)  # {axis, confusion, metrics}, png path
    failed = Signal(str)

    def __init__(self, data_manager, session_id: str, model_path: str,
                 conf: float = 0.25, stride: int = 1, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.session_id = session_id
        self.model_path = model_path
        self.conf = conf
        self.stride = stride

    def run(self):
        try:
            video_path = self.dm.get_video_path(self.session_id)
            if not video_path or not video_path.exists():
                raise FileNotFoundError(f"Video missing for {self.session_id}")
            corrections = self.dm.load_corrections(self.session_id)
            if not corrections:
                raise FileNotFoundError(
                    "No corrections — pick a session that has been reviewed."
                )

            self.log_line.emit(f"Running predictions with {self.model_path}…")
            preds = run_predictions(
                video_path, self.model_path, self.dm.models_dir,
                conf=self.conf, stride=self.stride,
                progress_callback=self.progress.emit,
            )
            self.log_line.emit("Matching predictions to corrections (IoU >= 0.5)…")
            gt = _per_frame_index(corrections)
            result = evaluate(gt, preds)

            png = self.dm.exports_dir / self.session_id / "confusion_matrix.png"
            render_confusion_png(result, png)
            self.log_line.emit(f"Saved {png}")
            self.finished_ok.emit(result, str(png))
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.failed.emit(str(e))
