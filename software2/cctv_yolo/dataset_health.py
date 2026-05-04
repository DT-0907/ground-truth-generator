"""
Dataset-health diagnostics.

Walks every corrected session and reports:
  * Class balance (counts + percentages)
  * Subclass balance
  * Bbox-size distribution (small / medium / large via COCO conventions)
  * Frame coverage (fraction of frames that have any annotations)
  * Train/val split counts (approximate — every 10th session is val)
  * Per-session annotation density

Useful before kicking off a training run — catches "95% cars, 1% bus"
imbalance and "bbox 5px wide" outliers that will tank a model.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


# COCO bbox-size buckets, in pixels^2 of bbox area
SMALL_AREA_PX = 32 * 32
MEDIUM_AREA_PX = 96 * 96


def collect_health(data_manager) -> dict:
    sessions = data_manager.get_sessions()
    corrected = [s for s in sessions if s.get("has_corrections")]

    classes = Counter()
    subclasses = Counter()
    sizes = Counter({"small": 0, "medium": 0, "large": 0})
    sample_dims = Counter()
    bbox_areas = []
    bbox_aspects = []

    annotated_frames = 0
    total_video_frames = 0

    per_session = []
    for s in corrected:
        sid = s["id"]
        data = data_manager.load_corrections(sid)
        if not data:
            continue
        ann_frames_this = set()
        bbox_count = 0
        for tr in data.get("tracks", []):
            cls = tr.get("class", "vehicle")
            classes[cls] += 1
            sub = tr.get("subclass") or "(none)"
            subclasses[sub] += 1
            for fd in tr.get("frames", []):
                if fd.get("interpolated"):
                    continue
                ann_frames_this.add(fd["frame"])
                x1, y1, x2, y2 = fd["bbox"]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                area = w * h
                bbox_areas.append(area)
                if h > 0:
                    bbox_aspects.append(w / h)
                if area < SMALL_AREA_PX:
                    sizes["small"] += 1
                elif area < MEDIUM_AREA_PX:
                    sizes["medium"] += 1
                else:
                    sizes["large"] += 1
                bbox_count += 1
        annotated_frames += len(ann_frames_this)
        total_video_frames += int(data.get("total_frames", 0))
        sample_dims[data.get("resolution", "?")] += 1
        per_session.append({
            "session_id": sid,
            "video_name": s["video_name"],
            "tracks": len(data.get("tracks", [])),
            "annotated_frames": len(ann_frames_this),
            "bboxes": bbox_count,
            "total_frames": int(data.get("total_frames", 0)),
            "coverage": (len(ann_frames_this) / max(1, int(data.get("total_frames", 1)))),
        })

    # Train/val approx split — same as build_yolo_dataset
    train_count = sum(1 for i, _ in enumerate(corrected) if i % 10 != 0)
    val_count = len(corrected) - train_count

    bbox_areas.sort()
    bbox_aspects.sort()
    n_box = len(bbox_areas)

    def pct(a):
        if not a:
            return 0
        return a[int(len(a) * 0.5)]

    return {
        "n_sessions_total": len(sessions),
        "n_sessions_corrected": len(corrected),
        "n_train": train_count,
        "n_val": val_count,
        "classes": dict(classes),
        "subclasses": dict(subclasses),
        "size_buckets": dict(sizes),
        "n_bboxes": n_box,
        "median_area_px": int(pct(bbox_areas)) if bbox_areas else 0,
        "median_aspect": round(pct(bbox_aspects), 2) if bbox_aspects else 0.0,
        "annotated_frames": annotated_frames,
        "total_video_frames": total_video_frames,
        "frame_coverage": round(annotated_frames / max(1, total_video_frames), 4),
        "resolutions": dict(sample_dims),
        "per_session": per_session,
    }


def health_warnings(report: dict) -> list[str]:
    """Heuristic warnings to surface in the UI."""
    warns = []
    classes = report.get("classes", {})
    if classes:
        total = sum(classes.values())
        if total:
            shares = {k: v / total for k, v in classes.items()}
            for k, v in shares.items():
                if v < 0.02:
                    warns.append(
                        f"Class '{k}' is only {v*100:.1f}% of the dataset "
                        f"— the trained model will likely under-perform on it."
                    )
            if max(shares.values()) > 0.85:
                top = max(shares, key=shares.get)
                warns.append(
                    f"Heavily imbalanced — '{top}' is {shares[top]*100:.0f}% "
                    "of the dataset. Consider class-balanced sampling."
                )
    if report.get("n_bboxes", 0) < 200:
        warns.append(
            f"Only {report['n_bboxes']} bboxes total — too small for serious "
            "fine-tuning. Aim for 1000+ per class."
        )
    sizes = report.get("size_buckets", {})
    if sizes.get("small", 0) > 0.5 * sum(sizes.values()):
        warns.append(
            "Over half the bboxes are 'small' (<32×32 px). YOLOv8 detects "
            "small objects but performance suffers — consider a higher imgsz."
        )
    if report.get("n_val", 0) < 1:
        warns.append(
            "No validation sessions — every 10th corrected session goes to "
            "val, so you need at least 10 corrected sessions to get one."
        )
    if report.get("n_sessions_corrected", 0) == 0:
        warns.append(
            "No corrected sessions yet — correct some in the Correction tab "
            "before training."
        )
    return warns
