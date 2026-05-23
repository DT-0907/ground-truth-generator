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

PRD I4: ``collect_health()`` accepts an optional ``roi_id``; when set,
only tracks intersecting that ROI count toward the report.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


# COCO bbox-size buckets, in pixels^2 of bbox area
SMALL_AREA_PX = 32 * 32
MEDIUM_AREA_PX = 96 * 96


def _filter_tracks_by_roi(tracks: list, rois: list, roi_id: str) -> list:
    """Return tracks that pass through the given ROI (by id or name)."""
    try:
        from cctv_yolo.exports import filter_tracks_by_roi
        return filter_tracks_by_roi(tracks, rois, roi_id)
    except Exception:
        # Fallback: bbox-center test
        from cctv_yolo.analytics import bbox_in_roi
        roi = next(
            (r for r in (rois or [])
             if r.get("id") == roi_id or r.get("name") == roi_id),
            None,
        )
        if not roi:
            return list(tracks)
        kept = []
        for tr in tracks:
            for fd in tr.get("frames", []):
                if bbox_in_roi(fd["bbox"], roi):
                    kept.append(tr)
                    break
        return kept


def collect_health(
    data_manager,
    session_ids: Optional[Iterable[str]] = None,
    roi_id: Optional[str] = None,
) -> dict:
    """Aggregate dataset health stats.

    Args:
        data_manager: DataManager instance.
        session_ids: optional restrict-set; default is ALL corrected sessions.
        roi_id: optional — filter corrections to tracks intersecting this ROI
            before counting (PRD I4).
    """
    sessions = data_manager.get_sessions()
    corrected = [s for s in sessions if s.get("has_corrections")]
    if session_ids is not None:
        keep = set(session_ids)
        corrected = [s for s in corrected if s["id"] in keep]

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
        tracks = data.get("tracks", [])
        if roi_id:
            tracks = _filter_tracks_by_roi(tracks, data.get("rois", []), roi_id)
        ann_frames_this = set()
        bbox_count = 0
        for tr in tracks:
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
            "tracks": len(tracks),
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
        "roi_id": roi_id,
    }


def collect_health_for_dataset(dataset_root: Path) -> dict:
    """Count actual label files in a built YOLO dataset.

    PRD I2 (Dataset sub-tab): scans ``<root>/labels/train`` and
    ``<root>/labels/val`` for .txt files, parses class indices, and
    cross-references with ``data.yaml`` for class names.
    """
    root = Path(dataset_root)
    label_dirs = {
        "train": root / "labels" / "train",
        "val": root / "labels" / "val",
    }
    image_dirs = {
        "train": root / "images" / "train",
        "val": root / "images" / "val",
    }

    # Try reading class names from data.yaml
    class_names: list[str] = []
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("names:"):
                        raw = line.split("names:", 1)[1].strip()
                        raw = raw.strip("[]")
                        class_names = [
                            p.strip().strip("'").strip('"')
                            for p in raw.split(",") if p.strip()
                        ]
                        break
        except OSError:
            pass

    class_counts: Counter = Counter()
    size_buckets = Counter({"small": 0, "medium": 0, "large": 0})
    n_labels_total = 0
    n_bboxes_total = 0
    counts_per_split = {}

    for split, ldir in label_dirs.items():
        n_labels = 0
        n_bboxes = 0
        if ldir.exists():
            for lbl in ldir.glob("*.txt"):
                n_labels += 1
                try:
                    with open(lbl, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            try:
                                ci = int(parts[0])
                                bw = float(parts[3])
                                bh = float(parts[4])
                            except ValueError:
                                continue
                            cname = (class_names[ci] if 0 <= ci < len(class_names)
                                     else f"class_{ci}")
                            class_counts[cname] += 1
                            n_bboxes += 1
                            area = bw * bh
                            if area < 0.01:
                                size_buckets["small"] += 1
                            elif area < 0.09:
                                size_buckets["medium"] += 1
                            else:
                                size_buckets["large"] += 1
                except OSError:
                    continue
        n_images = 0
        idir = image_dirs.get(split)
        if idir and idir.exists():
            n_images = sum(1 for _ in idir.glob("*.jpg")) + sum(1 for _ in idir.glob("*.png"))
        counts_per_split[split] = {"labels": n_labels, "bboxes": n_bboxes, "images": n_images}
        n_labels_total += n_labels
        n_bboxes_total += n_bboxes

    return {
        "dataset_root": str(root),
        "yaml_path": str(yaml_path) if yaml_path.exists() else None,
        "class_names": class_names,
        "classes": dict(class_counts),
        "size_buckets": dict(size_buckets),
        "splits": counts_per_split,
        "n_labels": n_labels_total,
        "n_bboxes": n_bboxes_total,
        "n_train": counts_per_split.get("train", {}).get("images", 0),
        "n_val": counts_per_split.get("val", {}).get("images", 0),
    }


def save_health(report: dict, out_dir: Path, ts: Optional[str] = None) -> Path:
    """Persist a health report to ``out_dir/dataset_health_<ts>.json``.

    PRD I3 — used by all four insights sub-tabs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"dataset_health_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return out


def load_health(path: Path) -> dict:
    """Load a previously-saved health report."""
    with open(Path(path), "r") as f:
        return json.load(f)


def list_health_history(folder: Path) -> list[Path]:
    """Return saved health reports under ``folder``, newest first."""
    p = Path(folder)
    if not p.exists():
        return []
    return sorted(p.glob("dataset_health_*.json"), reverse=True)


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
    n_bboxes = report.get("n_bboxes", 0)
    if n_bboxes < 200:
        warns.append(
            f"Only {n_bboxes} bboxes total — too small for serious "
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
