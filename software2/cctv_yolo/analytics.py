"""
Advanced analytics — path-density heatmap, ROI origin-destination
matrix, per-minute time-series CSV, and speed estimation.

Pure functions that operate on standard track JSON (corrections or raw
tracks). Output: PNG (heatmap), JSON (OD matrix), CSV (time-series).
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# ROI helpers (kept independent so analytics can be used without the
# rest of the app)
# ---------------------------------------------------------------------------

def _point_in_polygon(px, py, polygon) -> bool:
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def bbox_in_roi(bbox, roi) -> bool:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    if roi.get("type") == "rect":
        pts = roi["points"]
        x1, y1 = pts[0]["x"], pts[0]["y"]
        x2, y2 = pts[1]["x"], pts[1]["y"]
        return min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2)
    poly = [(p["x"], p["y"]) for p in roi.get("points", [])]
    return _point_in_polygon(cx, cy, poly)


def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


# ---------------------------------------------------------------------------
# 1. Heatmap — path density on top of the first frame
# ---------------------------------------------------------------------------

def render_heatmap(
    video_path: Path,
    track_data: dict,
    output_path: Path,
    sigma: float = 12.0,
    alpha: float = 0.6,
) -> Path:
    """Accumulate every track-center as a gaussian blob on a heatmap,
    then blend onto a representative frame."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(total // 4, 60))
    ret, base = cap.read()
    cap.release()
    if not ret or base is None:
        raise RuntimeError("Cannot grab base frame")

    h, w = base.shape[:2]
    accum = np.zeros((h, w), dtype=np.float32)

    for tr in track_data.get("tracks", []):
        for fd in tr.get("frames", []):
            cx, cy = bbox_center(fd["bbox"])
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= ix < w and 0 <= iy < h:
                accum[iy, ix] += 1.0

    if accum.max() == 0:
        # Empty heatmap — still write a copy of the base frame
        cv2.imwrite(str(output_path), base)
        return output_path

    # Smooth with a gaussian to spread point hits into a path-density map
    k = max(3, int(sigma * 3) | 1)
    accum = cv2.GaussianBlur(accum, (k, k), sigmaX=sigma, sigmaY=sigma)
    norm = (accum / accum.max() * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    blended = cv2.addWeighted(base, 1 - alpha, color, alpha, 0)
    # Mute regions with no traffic so original is visible there
    mask = (norm > 8).astype(np.uint8)
    final = base.copy()
    final[mask == 1] = blended[mask == 1]

    cv2.imwrite(str(output_path), final)
    return output_path


# ---------------------------------------------------------------------------
# 2. Origin-destination matrix between ROIs
# ---------------------------------------------------------------------------

def origin_destination_matrix(track_data: dict) -> dict:
    """For each track, pick the first ROI it enters as origin and the
    last ROI it leaves as destination. Returns a square dict keyed by
    ROI name with counts.
    """
    rois = track_data.get("rois", []) or []
    if not rois:
        return {"rois": [], "matrix": {}, "totals": {}}

    roi_names = [r.get("name") or f"ROI {i+1}" for i, r in enumerate(rois)]
    matrix: dict[str, dict[str, int]] = {a: {b: 0 for b in roi_names} for a in roi_names}
    totals = defaultdict(int)

    for tr in track_data.get("tracks", []):
        first_roi = None
        last_roi = None
        for fd in sorted(tr.get("frames", []), key=lambda x: x["frame"]):
            for name, roi in zip(roi_names, rois):
                if bbox_in_roi(fd["bbox"], roi):
                    if first_roi is None:
                        first_roi = name
                    last_roi = name
                    break
        if first_roi and last_roi:
            matrix[first_roi][last_roi] += 1
            totals[first_roi] += 1

    return {
        "rois": roi_names,
        "matrix": matrix,
        "totals": dict(totals),
        "track_count": len(track_data.get("tracks", [])),
    }


def write_od_matrix_csv(od: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rois = od["rois"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["origin\\destination"] + rois)
        for o in rois:
            row = [o] + [od["matrix"].get(o, {}).get(d, 0) for d in rois]
            w.writerow(row)
    return output_path


# ---------------------------------------------------------------------------
# 3. Time-series CSV — per-minute counts (or any bucket)
# ---------------------------------------------------------------------------

def time_series_csv(
    track_data: dict,
    output_path: Path,
    bucket_seconds: int = 60,
    per_class: bool = True,
    per_roi: bool = True,
) -> Path:
    """Per-bucket counts of unique tracks first observed in that bucket."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = float(track_data.get("fps", 30.0)) or 30.0
    rois = track_data.get("rois", []) or []
    roi_names = [r.get("name") or f"ROI {i+1}" for i, r in enumerate(rois)]

    # bucket -> {"total": n, "by_class": {cls: n}, "by_roi": {name: n}}
    buckets: dict[int, dict] = defaultdict(
        lambda: {"total": 0, "by_class": defaultdict(int), "by_roi": defaultdict(int)}
    )

    classes = set()

    for tr in track_data.get("tracks", []):
        frames = tr.get("frames", [])
        if not frames:
            continue
        first_frame = min(f["frame"] for f in frames)
        bucket = int(first_frame / fps // bucket_seconds)
        cls = tr.get("class", "vehicle")
        classes.add(cls)
        buckets[bucket]["total"] += 1
        buckets[bucket]["by_class"][cls] += 1
        # Which ROIs did the track ever enter?
        seen_rois = set()
        for fd in frames:
            for name, roi in zip(roi_names, rois):
                if name in seen_rois:
                    continue
                if bbox_in_roi(fd["bbox"], roi):
                    seen_rois.add(name)
        for name in seen_rois:
            buckets[bucket]["by_roi"][name] += 1

    # Write CSV
    classes = sorted(classes)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["bucket_start_sec", "bucket_label", "total"]
        if per_class:
            header += [f"class:{c}" for c in classes]
        if per_roi:
            header += [f"roi:{n}" for n in roi_names]
        w.writerow(header)

        if not buckets:
            return output_path

        max_bucket = max(buckets.keys())
        for b in range(max_bucket + 1):
            data = buckets.get(b, {"total": 0, "by_class": {}, "by_roi": {}})
            start_sec = b * bucket_seconds
            mm = start_sec // 60
            ss = start_sec % 60
            label = f"{mm:02d}:{ss:02d}"
            row = [start_sec, label, data["total"]]
            if per_class:
                row += [data["by_class"].get(c, 0) for c in classes]
            if per_roi:
                row += [data["by_roi"].get(n, 0) for n in roi_names]
            w.writerow(row)

    return output_path


# ---------------------------------------------------------------------------
# 4. Speed estimation
# ---------------------------------------------------------------------------

def estimate_speeds(
    track_data: dict,
    pixels_per_meter: float,
) -> list[dict]:
    """Estimate avg/peak speed per track in mph, given a pixel-to-meter
    calibration scalar.

    Pixels-per-meter is the number of pixels per real-world meter at the
    *image plane* of the calibration line. Treat as a global average —
    OK for footage where vehicles travel in a roughly fronto-parallel
    plane, less accurate for steep oblique angles.
    """
    if pixels_per_meter <= 0:
        raise ValueError("pixels_per_meter must be > 0")
    fps = float(track_data.get("fps", 30.0)) or 30.0

    out = []
    for tr in track_data.get("tracks", []):
        frames = sorted(tr.get("frames", []), key=lambda x: x["frame"])
        if len(frames) < 3:
            continue

        speeds_mps = []
        prev = None
        prev_frame = None
        for fd in frames:
            cx, cy = bbox_center(fd["bbox"])
            if prev is not None and fd["frame"] != prev_frame:
                dx = cx - prev[0]
                dy = cy - prev[1]
                dist_px = (dx * dx + dy * dy) ** 0.5
                df = fd["frame"] - prev_frame
                if df > 0:
                    px_per_sec = dist_px / (df / fps)
                    m_per_sec = px_per_sec / pixels_per_meter
                    speeds_mps.append(m_per_sec)
            prev = (cx, cy)
            prev_frame = fd["frame"]

        if not speeds_mps:
            continue

        # Trim outliers — top 5% to avoid jitter spikes
        speeds_mps.sort()
        trimmed = speeds_mps[: max(1, int(len(speeds_mps) * 0.95))]
        avg_mps = sum(trimmed) / len(trimmed)
        peak_mps = max(trimmed)

        out.append({
            "track_id": tr.get("track_id"),
            "class": tr.get("class"),
            "subclass": tr.get("subclass"),
            "avg_speed_mph": round(avg_mps * 2.23694, 1),
            "peak_speed_mph": round(peak_mps * 2.23694, 1),
            "avg_speed_kph": round(avg_mps * 3.6, 1),
            "peak_speed_kph": round(peak_mps * 3.6, 1),
            "samples": len(speeds_mps),
        })
    return out


def write_speeds_csv(speeds: list[dict], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["track_id", "class", "subclass",
            "avg_speed_mph", "peak_speed_mph",
            "avg_speed_kph", "peak_speed_kph", "samples"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in speeds:
            w.writerow({k: row.get(k, "") for k in cols})
    return output_path


# ---------------------------------------------------------------------------
# Group-aggregate variants (PRD H4)
# ---------------------------------------------------------------------------

def aggregate_group_heatmap(
    data_manager,
    session_ids: list,
    output_path: Path,
    sigma: float = 12.0,
    alpha: float = 0.6,
) -> Path:
    """Render heatmaps for every session and pixel-sum them into a single
    composite heatmap over the first session's base frame."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not session_ids:
        raise ValueError("No sessions to aggregate")

    # Use the first session's video as base canvas + size reference.
    base_frame = None
    accum = None
    base_shape = None

    for sid in session_ids:
        data = data_manager.load_session_data(sid)
        if not data:
            continue
        video_path = data_manager.get_video_path(sid)
        if not video_path:
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total // 4, 60))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            continue
        h, w = frame.shape[:2]
        if base_frame is None:
            base_frame = frame
            base_shape = (h, w)
            accum = np.zeros(base_shape, dtype=np.float32)
        # Add tracks from this session, scaling to base frame if size differs.
        sx = base_shape[1] / w
        sy = base_shape[0] / h
        for tr in data.get("tracks", []):
            for fd in tr.get("frames", []):
                cx, cy = bbox_center(fd["bbox"])
                ix = int(round(cx * sx))
                iy = int(round(cy * sy))
                if 0 <= ix < base_shape[1] and 0 <= iy < base_shape[0]:
                    accum[iy, ix] += 1.0

    if base_frame is None or accum is None or accum.max() == 0:
        raise RuntimeError("No usable session data for group heatmap")

    k = max(3, int(sigma * 3) | 1)
    accum = cv2.GaussianBlur(accum, (k, k), sigmaX=sigma, sigmaY=sigma)
    norm = (accum / accum.max() * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(base_frame, 1 - alpha, color, alpha, 0)
    mask = (norm > 8).astype(np.uint8)
    final = base_frame.copy()
    final[mask == 1] = blended[mask == 1]
    cv2.imwrite(str(output_path), final)
    return output_path


def aggregate_group_od_matrix(data_manager, session_ids: list) -> dict:
    """Sum per-session OD matrices. Uses the union of ROI names across the
    group; ROIs are matched by *name*."""
    all_names: list = []
    seen = set()
    per_session_ods = []
    for sid in session_ids:
        data = data_manager.load_session_data(sid)
        if not data:
            continue
        od = origin_destination_matrix(data)
        per_session_ods.append(od)
        for n in od.get("rois", []):
            if n not in seen:
                seen.add(n)
                all_names.append(n)
    matrix = {a: {b: 0 for b in all_names} for a in all_names}
    totals: dict = defaultdict(int)
    track_count = 0
    for od in per_session_ods:
        track_count += od.get("track_count", 0)
        for a in od.get("rois", []):
            for b in od.get("rois", []):
                v = od["matrix"].get(a, {}).get(b, 0)
                matrix[a][b] += v
                totals[a] += v
    return {
        "rois": all_names,
        "matrix": matrix,
        "totals": dict(totals),
        "track_count": track_count,
    }


def aggregate_group_time_series(
    data_manager,
    session_ids: list,
    output_path: Path,
    bucket_seconds: int = 60,
) -> Path:
    """Concat per-session time-series CSVs with a leading session_id column."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_classes: set = set()
    all_rois: list = []
    seen = set()
    per_session_buckets = {}

    for sid in session_ids:
        data = data_manager.load_session_data(sid)
        if not data:
            continue
        fps = float(data.get("fps", 30.0)) or 30.0
        rois = data.get("rois", []) or []
        roi_names = [r.get("name") or f"ROI {i+1}" for i, r in enumerate(rois)]
        for n in roi_names:
            if n not in seen:
                seen.add(n)
                all_rois.append(n)

        sess_buckets: dict = defaultdict(
            lambda: {"total": 0, "by_class": defaultdict(int),
                     "by_roi": defaultdict(int)}
        )
        for tr in data.get("tracks", []):
            frames = tr.get("frames", [])
            if not frames:
                continue
            first_frame = min(f["frame"] for f in frames)
            bucket = int(first_frame / fps // bucket_seconds)
            cls = tr.get("class", "vehicle")
            all_classes.add(cls)
            sess_buckets[bucket]["total"] += 1
            sess_buckets[bucket]["by_class"][cls] += 1
            seen_rois = set()
            for fd in frames:
                for name, roi in zip(roi_names, rois):
                    if name in seen_rois:
                        continue
                    if bbox_in_roi(fd["bbox"], roi):
                        seen_rois.add(name)
            for name in seen_rois:
                sess_buckets[bucket]["by_roi"][name] += 1
        per_session_buckets[sid] = sess_buckets

    classes = sorted(all_classes)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["session_id", "bucket_start_sec", "bucket_label", "total"]
        header += [f"class:{c}" for c in classes]
        header += [f"roi:{n}" for n in all_rois]
        w.writerow(header)
        for sid, buckets in per_session_buckets.items():
            if not buckets:
                continue
            max_b = max(buckets.keys())
            for b in range(max_b + 1):
                data = buckets.get(b, {"total": 0, "by_class": {}, "by_roi": {}})
                start_sec = b * bucket_seconds
                mm = start_sec // 60
                ss = start_sec % 60
                label = f"{mm:02d}:{ss:02d}"
                row = [sid, start_sec, label, data["total"]]
                row += [data["by_class"].get(c, 0) for c in classes]
                row += [data["by_roi"].get(n, 0) for n in all_rois]
                w.writerow(row)
    return output_path


def aggregate_group_speeds(
    data_manager,
    session_ids: list,
    pixels_per_meter: float,
) -> list[dict]:
    """All per-track speeds across the group, each row tagged with session_id."""
    out: list[dict] = []
    for sid in session_ids:
        data = data_manager.load_session_data(sid)
        if not data:
            continue
        rows = estimate_speeds(data, pixels_per_meter=pixels_per_meter)
        for r in rows:
            r["session_id"] = sid
        out.extend(rows)
    return out


def write_group_speeds_csv(speeds: list[dict], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["session_id", "track_id", "class", "subclass",
            "avg_speed_mph", "peak_speed_mph",
            "avg_speed_kph", "peak_speed_kph", "samples"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in speeds:
            w.writerow({k: row.get(k, "") for k in cols})
    return output_path


def aggregate_group_direction(data_manager, session_ids: list) -> dict:
    """Sum N/S/E/W counts per ROI name across every session in the group."""
    out: dict = defaultdict(lambda: {"N": 0, "S": 0, "E": 0, "W": 0, "total": 0})
    for sid in session_ids:
        data = data_manager.load_session_data(sid)
        if not data:
            continue
        d = direction_of_travel(data)
        for roi_name, counts in d.items():
            for k in ("N", "S", "E", "W", "total"):
                out[roi_name][k] += counts.get(k, 0)
    return dict(out)


# ---------------------------------------------------------------------------
# 5. Direction-of-travel per ROI
# ---------------------------------------------------------------------------

def direction_of_travel(track_data: dict) -> dict:
    """For each ROI, count tracks by entry vector direction — N/S/E/W.
    Useful as a quick "in vs out" classifier without lane lines.
    """
    rois = track_data.get("rois", []) or []
    roi_names = [r.get("name") or f"ROI {i+1}" for i, r in enumerate(rois)]
    out = {name: {"N": 0, "S": 0, "E": 0, "W": 0, "total": 0} for name in roi_names}

    for tr in track_data.get("tracks", []):
        frames = sorted(tr.get("frames", []), key=lambda x: x["frame"])
        for name, roi in zip(roi_names, rois):
            entry_idx = None
            for i, fd in enumerate(frames):
                if bbox_in_roi(fd["bbox"], roi):
                    entry_idx = i
                    break
            if entry_idx is None:
                continue
            # Use entry frame and ~10 frames later (or the last frame in ROI)
            exit_idx = entry_idx
            for j in range(entry_idx, len(frames)):
                if not bbox_in_roi(frames[j]["bbox"], roi):
                    break
                exit_idx = j
            if exit_idx == entry_idx and entry_idx + 1 < len(frames):
                exit_idx = entry_idx + 1
            if exit_idx == entry_idx:
                continue

            cx0, cy0 = bbox_center(frames[entry_idx]["bbox"])
            cx1, cy1 = bbox_center(frames[exit_idx]["bbox"])
            dx, dy = cx1 - cx0, cy1 - cy0
            if abs(dx) < 1 and abs(dy) < 1:
                continue
            if abs(dx) > abs(dy):
                key = "E" if dx > 0 else "W"
            else:
                key = "S" if dy > 0 else "N"
            out[name][key] += 1
            out[name]["total"] += 1
    return out
