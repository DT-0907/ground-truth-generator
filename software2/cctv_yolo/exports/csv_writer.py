"""Per-track and per-frame CSV writers.

Per-track columns: track_id, class, start_frame, end_frame,
duration_frames, avg_conf, total_detections.

Per-frame columns: frame, track_id, class, x1, y1, x2, y2, conf, plus
one boolean column per ROI named ``in_roi_<name>`` (sanitised).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from . import _bbox_center_in_roi, filter_tracks_by_roi


def _safe_col(name: str) -> str:
    """Sanitise an ROI name for use as a CSV header."""
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_")
    return s or "roi"


def write_per_track_csv(
    output_path: Path,
    track_data: dict,
    roi_id: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = filter_tracks_by_roi(
        track_data.get("tracks", []),
        track_data.get("rois", []),
        roi_id,
    )

    with output_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "track_id", "class", "start_frame", "end_frame",
            "duration_frames", "avg_conf", "total_detections",
        ])
        for tr in tracks:
            frames = sorted(tr.get("frames", []), key=lambda fd: fd["frame"])
            if not frames:
                continue
            confs = [fd.get("conf", 0) for fd in frames if fd.get("conf")]
            avg_conf = (sum(confs) / len(confs)) if confs else 0.0
            w.writerow([
                tr.get("track_id", 0),
                tr.get("class", "unknown"),
                frames[0]["frame"],
                frames[-1]["frame"],
                frames[-1]["frame"] - frames[0]["frame"] + 1,
                round(avg_conf, 4),
                len(frames),
            ])
    return output_path


def write_per_frame_csv(
    output_path: Path,
    track_data: dict,
    roi_id: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rois = track_data.get("rois", []) or []
    tracks = filter_tracks_by_roi(
        track_data.get("tracks", []), rois, roi_id,
    )

    roi_cols = [(roi, f"in_roi_{_safe_col(roi.get('name') or 'roi')}") for roi in rois]
    header = ["frame", "track_id", "class", "x1", "y1", "x2", "y2", "conf"]
    header.extend(col for _, col in roi_cols)

    with output_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        rows: list[list] = []
        for tr in tracks:
            tid = tr.get("track_id", 0)
            cls = tr.get("class", "unknown")
            for fd in tr.get("frames", []):
                x1, y1, x2, y2 = fd["bbox"]
                row = [
                    fd["frame"], tid, cls,
                    round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                    round(fd.get("conf", 0) or 0, 4),
                ]
                for roi, _ in roi_cols:
                    row.append(int(_bbox_center_in_roi(fd["bbox"], roi)))
                rows.append(row)
        rows.sort(key=lambda r: (r[0], r[1]))
        w.writerows(rows)

    return output_path
