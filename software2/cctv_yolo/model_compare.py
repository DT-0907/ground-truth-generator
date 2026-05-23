"""
Model comparison — run two YOLO models on the same video and produce
a diff: counts, mean confidence, per-class breakdown, side-by-side
visualization.

PRD K2-5 — every successful comparison is also persisted as JSON under
``data/exports/model_compare/<a>_vs_<b>_<sid>_<ts>.json`` so the Models tab
can offer a "History" picker.
"""
from __future__ import annotations
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal


VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}


def _device():
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _bbox_center_in_roi(bbox, roi: Optional[dict]) -> bool:
    """Test whether the bbox center falls inside a processing ROI dict.
    Returns True if no ROI is supplied (no spatial filter)."""
    if not roi:
        return True
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rtype = roi.get("type", "rect")
    pts = roi.get("points", [])

    def pt_xy(p):
        if isinstance(p, dict):
            return (p.get("x", 0), p.get("y", 0))
        return p

    if rtype == "rect" and len(pts) == 2:
        a, b = pt_xy(pts[0]), pt_xy(pts[1])
        return (min(a[0], b[0]) <= cx <= max(a[0], b[0])
                and min(a[1], b[1]) <= cy <= max(a[1], b[1]))
    if rtype == "polygon" and len(pts) >= 3:
        norm = [pt_xy(p) for p in pts]
        n = len(norm)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = norm[i]
            xj, yj = norm[j]
            if ((yi > cy) != (yj > cy)) and (cx < (xj - xi) * (cy - yi) / max(1e-9, (yj - yi)) + xi):
                inside = not inside
            j = i
        return inside
    return True


def run_model_track(
    video_path: Path,
    model_path: str,
    conf: float,
    models_dir: Path,
    progress_callback=None,
    stride: int = 1,
    processing_roi: Optional[dict] = None,
) -> dict:
    """Run YOLO detect+track on a video. Returns track summary stats
    suitable for comparison."""
    from ultralytics import YOLO

    local = Path(models_dir) / model_path
    model = YOLO(str(local) if local.exists() else model_path)
    model.to(_device())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    results = model.track(
        source=str(video_path),
        conf=conf,
        classes=list(VEHICLE_CLASSES.keys()),
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
        vid_stride=stride,
    )

    tracks = defaultdict(lambda: {"frames": 0, "confs": [], "class": None})
    last_pct = -1

    for frame_idx, result in enumerate(results):
        if progress_callback and total_frames > 0:
            pct = int((frame_idx + 1) / (total_frames / max(1, stride)) * 100)
            pct = min(pct, 99)
            if pct != last_pct:
                progress_callback(pct)
                last_pct = pct

        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes
        if boxes.id is None:
            continue
        for i in range(len(boxes)):
            tid = int(boxes.id[i].item())
            conf_v = float(boxes.conf[i].item())
            cid = int(boxes.cls[i].item())
            cname = VEHICLE_CLASSES.get(cid, "unknown")
            if processing_roi is not None:
                bbox_xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                if not _bbox_center_in_roi(bbox_xyxy, processing_roi):
                    continue
            tracks[tid]["frames"] += 1
            tracks[tid]["confs"].append(conf_v)
            tracks[tid]["class"] = cname

    stats = {
        "total_tracks": len(tracks),
        "by_class": defaultdict(int),
        "mean_conf": 0.0,
        "median_track_length": 0.0,
        "total_detections": 0,
    }
    confs = []
    lengths = []
    for tid, t in tracks.items():
        stats["by_class"][t["class"]] += 1
        confs.extend(t["confs"])
        lengths.append(t["frames"])
        stats["total_detections"] += t["frames"]

    if confs:
        stats["mean_conf"] = round(sum(confs) / len(confs), 3)
    if lengths:
        lengths.sort()
        stats["median_track_length"] = lengths[len(lengths) // 2]
    stats["by_class"] = dict(stats["by_class"])
    return stats


def diff_stats(a: dict, b: dict) -> dict:
    """Diff two stats dicts. Returns delta (b - a)."""
    delta = {
        "total_tracks": b["total_tracks"] - a["total_tracks"],
        "mean_conf": round(b["mean_conf"] - a["mean_conf"], 3),
        "total_detections": b["total_detections"] - a["total_detections"],
        "median_track_length": b["median_track_length"] - a["median_track_length"],
        "by_class": {},
    }
    classes = set(a["by_class"].keys()) | set(b["by_class"].keys())
    for c in sorted(classes):
        delta["by_class"][c] = b["by_class"].get(c, 0) - a["by_class"].get(c, 0)
    return delta


def _safe_stem(name: str) -> str:
    """Turn a model filename like ``yolov8m.pt`` into a path-safe stem."""
    return Path(name).stem.replace(" ", "_")


def save_comparison_result(
    payload: dict,
    exports_root: Path,
    *,
    session_id: Optional[str] = None,
    extra_suffix: Optional[str] = None,
) -> Path:
    """Persist a comparison ``payload`` to ``exports_root/model_compare/``.

    Returns the path written.
    """
    out_dir = Path(exports_root) / "model_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    a = _safe_stem(payload.get("model_a", "A"))
    b = _safe_stem(payload.get("model_b", "B"))
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [a, "vs", b]
    if session_id:
        parts.append(session_id)
    if extra_suffix:
        parts.append(extra_suffix)
    parts.append(ts)
    fname = "_".join(parts) + ".json"
    out = out_dir / fname
    enriched = dict(payload)
    enriched.setdefault("saved_at", dt.datetime.now().isoformat(timespec="seconds"))
    enriched.setdefault("session_id", session_id)
    with open(out, "w") as f:
        json.dump(enriched, f, indent=2, default=str)
    return out


class ModelCompareWorker(QThread):
    """Run two models sequentially and report the diff."""

    log_line = Signal(str)
    progress = Signal(str, int)  # which model (A/B), pct
    finished_ok = Signal(dict)   # {"a": stats, "b": stats, "delta": stats}
    failed = Signal(str)

    def __init__(
        self,
        video_path: Path,
        model_a: str,
        model_b: str,
        models_dir: Path,
        conf: float = 0.25,
        stride: int = 1,
        processing_roi: Optional[dict] = None,
        exports_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.model_a = model_a
        self.model_b = model_b
        self.models_dir = Path(models_dir)
        self.conf = conf
        self.stride = stride
        self.processing_roi = processing_roi
        self.exports_dir = Path(exports_dir) if exports_dir else None
        self.session_id = session_id

    def run(self):
        try:
            self.log_line.emit(f"[A] running {self.model_a}…")
            stats_a = run_model_track(
                self.video_path, self.model_a, self.conf, self.models_dir,
                progress_callback=lambda p: self.progress.emit("A", p),
                stride=self.stride,
                processing_roi=self.processing_roi,
            )
            self.log_line.emit(f"[A] tracks={stats_a['total_tracks']} "
                               f"mean_conf={stats_a['mean_conf']}")
            self.log_line.emit(f"[B] running {self.model_b}…")
            stats_b = run_model_track(
                self.video_path, self.model_b, self.conf, self.models_dir,
                progress_callback=lambda p: self.progress.emit("B", p),
                stride=self.stride,
                processing_roi=self.processing_roi,
            )
            self.log_line.emit(f"[B] tracks={stats_b['total_tracks']} "
                               f"mean_conf={stats_b['mean_conf']}")
            payload = {
                "mode": "video",
                "a": stats_a, "b": stats_b, "delta": diff_stats(stats_a, stats_b),
                "model_a": self.model_a, "model_b": self.model_b,
                "video": str(self.video_path),
                "session_id": self.session_id,
                "conf": self.conf, "stride": self.stride,
            }
            # PRD K2-5 — save to disk for the History picker
            if self.exports_dir is not None:
                try:
                    out = save_comparison_result(
                        payload, self.exports_dir, session_id=self.session_id,
                    )
                    payload["saved_path"] = str(out)
                    self.log_line.emit(f"Saved comparison: {out.name}")
                except Exception as e:
                    self.log_line.emit(f"Warning: couldn't save result: {e}")
            self.finished_ok.emit(payload)
        except Exception as e:
            import traceback
            self.log_line.emit(traceback.format_exc())
            self.failed.emit(str(e))


# ---------------------------------------------------------------------------
# Dataset val-split comparison (PRD J4c)
# ---------------------------------------------------------------------------

def run_model_on_val_split(
    dataset_root: Path,
    model_path: str,
    conf: float,
    models_dir: Path,
    progress_callback=None,
) -> dict:
    """Run a YOLO model across every image in ``<dataset_root>/images/val/``
    and return predictions + GT in a per-frame index suitable for
    :func:`cctv_yolo.metrics.compute_confusion_matrix`."""
    from ultralytics import YOLO

    val_images = Path(dataset_root) / "images" / "val"
    val_labels = Path(dataset_root) / "labels" / "val"
    if not val_images.exists():
        raise RuntimeError(f"No val images dir at {val_images}")

    # Parse class names from data.yaml
    yaml_path = Path(dataset_root) / "data.yaml"
    class_names: list[str] = []
    if yaml_path.exists():
        for line in yaml_path.read_text().splitlines():
            if line.strip().startswith("names:"):
                bracket = line[line.find("["):line.rfind("]") + 1] if "[" in line else ""
                if bracket:
                    class_names = [
                        c.strip().strip("'").strip('"')
                        for c in bracket.strip("[]").split(",") if c.strip()
                    ]
                break

    local = Path(models_dir) / model_path
    model = YOLO(str(local) if local.exists() else model_path)
    model.to(_device())

    images = sorted(val_images.glob("*.jpg")) + sorted(val_images.glob("*.png"))
    total = max(1, len(images))

    predictions: dict[int, list[dict]] = {}
    ground_truth: dict[int, list[dict]] = {}

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Ground truth from YOLO txt label
        lbl = val_labels / (img_path.stem + ".txt")
        gt_list: list[dict] = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                ci = int(parts[0])
                cx, cy, bw, bh = (float(p) for p in parts[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                name = class_names[ci] if 0 <= ci < len(class_names) else f"cls_{ci}"
                gt_list.append({"class": name, "bbox": [x1, y1, x2, y2]})

        # Predictions
        results = model.predict(source=img, conf=conf, verbose=False)
        pred_list: list[dict] = []
        if results and len(results) > 0:
            r0 = results[0]
            if r0.boxes is not None and len(r0.boxes) > 0:
                names_map = r0.names if hasattr(r0, "names") else {}
                for j in range(len(r0.boxes)):
                    cid = int(r0.boxes.cls[j].item())
                    name = names_map.get(cid, f"cls_{cid}")
                    # Map COCO ids to our short names where applicable
                    if name in ("car",) or name in VEHICLE_CLASSES.values():
                        pass
                    bbox = r0.boxes.xyxy[j].cpu().numpy().tolist()
                    pred_list.append({"class": name, "bbox": bbox})

        predictions[i] = pred_list
        ground_truth[i] = gt_list

        if progress_callback and i % max(1, total // 100) == 0:
            progress_callback(min(99, int((i + 1) / total * 100)))

    return {"predictions": predictions, "ground_truth": ground_truth,
            "image_count": len(images), "class_names": class_names}


class DatasetCompareWorker(QThread):
    """Compare two models against a dataset's val split (PRD J4c)."""

    log_line = Signal(str)
    progress = Signal(str, int)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        dataset_root: Path,
        model_a: str,
        model_b: str,
        models_dir: Path,
        conf: float = 0.25,
        iou_threshold: float = 0.5,
        exports_dir: Optional[Path] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.dataset_root = Path(dataset_root)
        self.model_a = model_a
        self.model_b = model_b
        self.models_dir = Path(models_dir)
        self.conf = conf
        self.iou_threshold = iou_threshold
        self.exports_dir = Path(exports_dir) if exports_dir else None

    def run(self):
        try:
            from cctv_yolo.metrics import compute_confusion_matrix

            self.log_line.emit(f"[A] {self.model_a} → val split…")
            a_data = run_model_on_val_split(
                self.dataset_root, self.model_a, self.conf, self.models_dir,
                progress_callback=lambda p: self.progress.emit("A", p),
            )
            self.log_line.emit(f"[B] {self.model_b} → val split…")
            b_data = run_model_on_val_split(
                self.dataset_root, self.model_b, self.conf, self.models_dir,
                progress_callback=lambda p: self.progress.emit("B", p),
            )

            cm_a = compute_confusion_matrix(
                a_data["predictions"], a_data["ground_truth"],
                iou_threshold=self.iou_threshold,
            )
            cm_b = compute_confusion_matrix(
                b_data["predictions"], b_data["ground_truth"],
                iou_threshold=self.iou_threshold,
            )

            self.log_line.emit(
                f"[A] P={cm_a['aggregate']['precision']:.3f} "
                f"R={cm_a['aggregate']['recall']:.3f} "
                f"F1={cm_a['aggregate']['f1']:.3f} "
                f"mAP={cm_a['aggregate']['mAP']:.3f}"
            )
            self.log_line.emit(
                f"[B] P={cm_b['aggregate']['precision']:.3f} "
                f"R={cm_b['aggregate']['recall']:.3f} "
                f"F1={cm_b['aggregate']['f1']:.3f} "
                f"mAP={cm_b['aggregate']['mAP']:.3f}"
            )

            payload = {
                "mode": "dataset",
                "dataset_id": self.dataset_root.name,
                "image_count": a_data["image_count"],
                "model_a": self.model_a,
                "model_b": self.model_b,
                "conf": self.conf,
                "iou_threshold": self.iou_threshold,
                "a": cm_a,
                "b": cm_b,
            }
            if self.exports_dir is not None:
                try:
                    out = save_comparison_result(
                        payload, self.exports_dir,
                        extra_suffix=f"on_{self.dataset_root.name}",
                    )
                    payload["saved_path"] = str(out)
                    self.log_line.emit(f"Saved comparison: {out.name}")
                except Exception as e:
                    self.log_line.emit(f"Warning: couldn't save result: {e}")
            self.finished_ok.emit(payload)
        except Exception as e:
            import traceback
            self.log_line.emit(traceback.format_exc())
            self.failed.emit(str(e))
