"""IoU-based detection metrics — Precision/Recall/F1/Confusion Matrix.

Used by Performance G3-3 (single session) and Insights I2 (dataset val-split).
Ground truth = corrections; prediction = raw tracks (or active model run output).

All bboxes are [x1, y1, x2, y2] (xyxy format).

Predictions / ground_truth payloads accept either:
  - per-frame dicts: {frame_idx: [{'bbox': [...], 'class': 'car'}, ...]}
  - track-list dicts: {'tracks': [{'class': ..., 'frames': [{'frame': N, 'bbox': [...]}, ...]}, ...]}
The helper :func:`_flatten` normalises both into per-frame index form.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_iou(box1, box2) -> float:
    """[x1,y1,x2,y2] — Returns IoU in [0,1]."""
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def _bbox_center(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_in_roi(x: float, y: float, roi: dict) -> bool:
    """Test whether (x, y) falls inside a rect or polygon ROI dict."""
    if not roi:
        return True

    def pt_xy(p):
        if isinstance(p, dict):
            return (p.get("x", 0), p.get("y", 0))
        return p

    rtype = roi.get("type", "rect")
    points = roi.get("points", [])
    if rtype == "rect" and len(points) == 2:
        x1, y1 = pt_xy(points[0])
        x2, y2 = pt_xy(points[1])
        return (min(x1, x2) <= x <= max(x1, x2)
                and min(y1, y2) <= y <= max(y1, y2))
    if rtype == "polygon" and len(points) >= 3:
        norm = [pt_xy(p) for p in points]
        n = len(norm)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = norm[i]
            xj, yj = norm[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(1e-9, (yj - yi)) + xi):
                inside = not inside
            j = i
        return inside
    return True


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_detections(gt_boxes, pred_boxes, iou_threshold: float = 0.5):
    """Greedy IoU matching.

    Args:
        gt_boxes: list of [x1, y1, x2, y2]
        pred_boxes: list of [x1, y1, x2, y2]
        iou_threshold: minimum IoU to be considered a match (default 0.5)

    Returns:
        list of (gt_idx, pred_idx) pairs for matched detections. Unmatched
        gt/pred indices are inferable from the input lengths.
    """
    matches: list[tuple[int, int]] = []
    used_pred: set[int] = set()
    for gi, g in enumerate(gt_boxes):
        best, best_iou = -1, 0.0
        for pi, p in enumerate(pred_boxes):
            if pi in used_pred:
                continue
            iou = bbox_iou(g, p)
            if iou > best_iou:
                best, best_iou = pi, iou
        if best >= 0 and best_iou >= iou_threshold:
            matches.append((gi, best))
            used_pred.add(best)
    return matches


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _flatten(payload) -> dict[int, list[dict]]:
    """Normalise either a per-frame dict or a track-list dict into per-frame.

    Each entry is {'bbox': [...], 'class': '...'}.
    """
    if not payload:
        return {}

    # Already per-frame
    if isinstance(payload, dict) and "tracks" not in payload:
        # Heuristic: keys look like ints
        sample_k = next(iter(payload.keys()), None)
        if sample_k is None or isinstance(sample_k, int):
            return {int(k): list(v) for k, v in payload.items()}

    out: dict[int, list[dict]] = defaultdict(list)
    tracks = payload.get("tracks", []) if isinstance(payload, dict) else []
    for tr in tracks:
        cls = tr.get("class", "vehicle")
        for fd in tr.get("frames", []):
            if fd.get("interpolated"):
                continue
            out[int(fd["frame"])].append({
                "bbox": fd["bbox"],
                "class": cls,
            })
    return dict(out)


def _filter_by_roi(frame_index: dict[int, list[dict]],
                   roi: Optional[dict]) -> dict[int, list[dict]]:
    """Drop detections whose bbox center is outside ``roi``."""
    if not roi:
        return frame_index
    filtered: dict[int, list[dict]] = {}
    for f, dets in frame_index.items():
        kept = []
        for d in dets:
            cx, cy = _bbox_center(d["bbox"])
            if _point_in_roi(cx, cy, roi):
                kept.append(d)
        if kept:
            filtered[f] = kept
    return filtered


# ---------------------------------------------------------------------------
# Confusion matrix + metrics
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    predictions,
    ground_truth,
    classes: Optional[Iterable[str]] = None,
    iou_threshold: float = 0.5,
    roi_filter: Optional[dict] = None,
) -> dict:
    """Compute confusion matrix and per-class metrics.

    Args:
        predictions: per-frame index OR track-list dict (raw tracks).
        ground_truth: per-frame index OR track-list dict (corrections).
        classes: optional explicit class list. Defaults to union across data.
        iou_threshold: matching threshold (default 0.5).
        roi_filter: optional ROI dict {type, points} — only detections inside
            this ROI (by bbox center) participate in matching.

    Returns:
        dict with keys:
          - ``matrix``: {(gt_class, pred_class): count} including 'background'
            for false positives / false negatives.
          - ``axis``: ordered class list including 'background' (last).
          - ``per_class``: {class: {tp, fp, fn, precision, recall, f1}}
          - ``aggregate``: {precision, recall, f1, mAP}
    """
    gt_index = _filter_by_roi(_flatten(ground_truth), roi_filter)
    pr_index = _filter_by_roi(_flatten(predictions), roi_filter)

    if classes is None:
        cls_set: set[str] = set()
        for dets in gt_index.values():
            for d in dets:
                cls_set.add(d["class"])
        for dets in pr_index.values():
            for d in dets:
                cls_set.add(d["class"])
        classes = sorted(cls_set)
    else:
        classes = list(classes)

    bg = "background"
    axis = list(classes) + [bg]
    matrix: dict[tuple[str, str], int] = defaultdict(int)

    all_frames = set(gt_index.keys()) | set(pr_index.keys())
    for f in all_frames:
        gts = gt_index.get(f, [])
        preds = pr_index.get(f, [])

        gt_boxes = [g["bbox"] for g in gts]
        pred_boxes = [p["bbox"] for p in preds]
        matches = match_detections(gt_boxes, pred_boxes, iou_threshold)
        matched_gt = {gi for gi, _ in matches}
        matched_pr = {pi for _, pi in matches}

        for gi, pi in matches:
            matrix[(gts[gi]["class"], preds[pi]["class"])] += 1

        # Unmatched GT → background prediction (false negative)
        for gi, g in enumerate(gts):
            if gi not in matched_gt:
                matrix[(g["class"], bg)] += 1
        # Unmatched prediction → background GT (false positive)
        for pi, p in enumerate(preds):
            if pi not in matched_pr:
                matrix[(bg, p["class"])] += 1

    # Per-class precision/recall/F1
    per_class: dict[str, dict] = {}
    for cls in classes:
        tp = matrix.get((cls, cls), 0)
        fp = sum(matrix.get((other, cls), 0) for other in axis if other != cls)
        fn = sum(matrix.get((cls, other), 0) for other in axis if other != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    # Aggregate (micro-averaged + simple mAP proxy: mean of per-class P*R)
    tp_total = sum(per_class[c]["tp"] for c in classes)
    fp_total = sum(per_class[c]["fp"] for c in classes)
    fn_total = sum(per_class[c]["fn"] for c in classes)
    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    if classes:
        m_ap = sum(per_class[c]["precision"] * per_class[c]["recall"]
                   for c in classes) / len(classes)
    else:
        m_ap = 0.0

    aggregate = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mAP": round(m_ap, 4),
    }

    return {
        "matrix": dict(matrix),
        "axis": axis,
        "per_class": per_class,
        "aggregate": aggregate,
    }


def aggregate_confusion_matrices(matrices: list[dict]) -> dict:
    """Sum cells across multiple confusion matrices.

    Args:
        matrices: list of dicts from :func:`compute_confusion_matrix`.

    Returns:
        dict shaped like compute_confusion_matrix output but with summed
        counts and recomputed per-class / aggregate metrics.
    """
    if not matrices:
        return {"matrix": {}, "axis": ["background"], "per_class": {},
                "aggregate": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mAP": 0.0}}

    merged: dict[tuple[str, str], int] = defaultdict(int)
    axis_set: set[str] = set()
    for m in matrices:
        for cell, count in m.get("matrix", {}).items():
            merged[cell] += count
        for c in m.get("axis", []):
            axis_set.add(c)

    if "background" in axis_set:
        axis_set.discard("background")
    classes = sorted(axis_set)
    axis = classes + ["background"]

    per_class: dict[str, dict] = {}
    for cls in classes:
        tp = merged.get((cls, cls), 0)
        fp = sum(merged.get((other, cls), 0) for other in axis if other != cls)
        fn = sum(merged.get((cls, other), 0) for other in axis if other != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    tp_total = sum(per_class[c]["tp"] for c in classes)
    fp_total = sum(per_class[c]["fp"] for c in classes)
    fn_total = sum(per_class[c]["fn"] for c in classes)
    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    if classes:
        m_ap = sum(per_class[c]["precision"] * per_class[c]["recall"]
                   for c in classes) / len(classes)
    else:
        m_ap = 0.0

    return {
        "matrix": dict(merged),
        "axis": axis,
        "per_class": per_class,
        "aggregate": {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "mAP": round(m_ap, 4),
        },
    }
