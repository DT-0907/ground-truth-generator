"""
Annotated video export — burn bounding boxes, track IDs, ROI overlays,
and a live counter into an MP4.

Optional: blur license-plate bbox regions from a hooked detector
(see ``cctv_yolo.lp_blur``).
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


CLASS_COLORS = {
    "car": (128, 255, 0),       # BGR — cv2 expects BGR
    "truck": (0, 128, 255),
    "bus": (255, 128, 0),
    "motorcycle": (0, 255, 255),
    "bicycle": (255, 0, 128),
}
DEFAULT_COLOR = (200, 200, 200)
ROI_COLOR = (78, 204, 163)


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


def _bbox_in_roi(bbox, roi) -> bool:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    if roi.get("type") == "rect":
        pts = roi["points"]
        x1, y1 = pts[0]["x"], pts[0]["y"]
        x2, y2 = pts[1]["x"], pts[1]["y"]
        return min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2)
    poly = [(p["x"], p["y"]) for p in roi.get("points", [])]
    return _point_in_polygon(cx, cy, poly)


def _draw_rois(frame, rois):
    for roi in rois or []:
        if roi.get("type") == "rect":
            pts = roi["points"]
            x1, y1 = int(pts[0]["x"]), int(pts[0]["y"])
            x2, y2 = int(pts[1]["x"]), int(pts[1]["y"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_COLOR, 2)
            label = roi.get("name") or "ROI"
            cv2.putText(frame, label, (min(x1, x2) + 4, min(y1, y2) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2)
        else:
            poly = np.array(
                [[int(p["x"]), int(p["y"])] for p in roi.get("points", [])],
                dtype=np.int32,
            )
            if len(poly) >= 2:
                cv2.polylines(frame, [poly], True, ROI_COLOR, 2)
                label = roi.get("name") or "ROI"
                cv2.putText(frame, label, tuple(poly[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2)


def _draw_hud(frame, frame_idx, fps, total_in_view, by_class, roi_counts, w):
    """Top-left HUD with frame, time, totals, and per-class breakdown."""
    pad = 8
    line_h = 22
    lines = []
    secs = frame_idx / fps if fps else 0
    lines.append(f"Frame {frame_idx}  t={secs:6.2f}s")
    lines.append(f"In-frame: {total_in_view}  " + "  ".join(
        f"{k}:{v}" for k, v in sorted(by_class.items())))
    if roi_counts:
        lines.append("ROI: " + "  ".join(
            f"{name}={n}" for name, n in roi_counts.items()))

    # Background
    box_h = line_h * len(lines) + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(w, 700), box_h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        y = pad + line_h * (i + 1) - 4
        cv2.putText(frame, line, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_video(
    video_path: Path,
    track_data: dict,
    output_path: Path,
    fourcc: str = "mp4v",
    blur_lp: bool = False,
    progress_callback=None,
    roi_id: str | None = None,
) -> dict:
    """Write an annotated MP4 alongside the originals.

    Parameters
    ----------
    roi_id : str | None
        If given, only render bounding boxes for tracks whose bbox center
        falls inside the named/id'd ROI on at least one frame (PRD F3-2
        export filter). ROIs themselves are still drawn.

    Returns
    -------
    dict
        Stats: ``{"frames_written": int, "duration_sec": float, ...}``
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(output_path), fourcc_code, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer for {output_path}")

    # ROI filter (PRD F3-2): only include tracks that pass through this ROI
    tracks_in = track_data.get("tracks", [])
    rois_for_filter = track_data.get("rois", []) or []
    if roi_id:
        roi_target = None
        for r in rois_for_filter:
            if r.get("id") == roi_id or r.get("name") == roi_id:
                roi_target = r
                break
        if roi_target is not None:
            kept = []
            for tr in tracks_in:
                for fd in tr.get("frames", []):
                    if _bbox_in_roi(fd["bbox"], roi_target):
                        kept.append(tr)
                        break
            tracks_in = kept

    # Index detections by frame
    per_frame = defaultdict(list)
    tracks = tracks_in
    for tr in tracks:
        cls = tr.get("class", "vehicle")
        tid = tr.get("track_id", 0)
        sub = tr.get("subclass")
        for fd in tr.get("frames", []):
            per_frame[fd["frame"]].append({
                "bbox": fd["bbox"],
                "class": cls,
                "subclass": sub,
                "track_id": tid,
                "conf": fd.get("conf", 0),
                "interpolated": fd.get("interpolated", False),
            })

    rois = track_data.get("rois", []) or []

    # LP blur (lazy import; gracefully no-op if not available)
    lp_detector = None
    if blur_lp:
        try:
            from cctv_yolo.lp_blur import LicensePlateBlurrer
            lp_detector = LicensePlateBlurrer()
        except Exception as e:
            print(f"[annotated_export] LP blur unavailable: {e}")

    last_pct = -1
    written = 0
    seen_track_ids: set[int] = set()
    roi_first_seen: dict[str, set[int]] = defaultdict(set)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = per_frame.get(written, [])

        # Aggregate stats
        total_in_view = len(dets)
        by_class: dict[str, int] = defaultdict(int)
        roi_counts: dict[str, int] = {}
        for d in dets:
            by_class[d["class"]] += 1
            seen_track_ids.add(d["track_id"])
        for roi in rois:
            name = roi.get("name") or "ROI"
            count = 0
            for d in dets:
                if _bbox_in_roi(d["bbox"], roi):
                    count += 1
                    roi_first_seen[name].add(d["track_id"])
            roi_counts[name] = count

        # Optional LP blur
        if lp_detector is not None and dets:
            try:
                lp_detector.blur(frame, [d["bbox"] for d in dets])
            except Exception as e:
                print(f"[annotated_export] LP blur error frame {written}: {e}")

        # ROI overlays first so boxes draw on top
        _draw_rois(frame, rois)

        # Boxes
        for d in dets:
            x1, y1, x2, y2 = [int(c) for c in d["bbox"]]
            color = CLASS_COLORS.get(d["class"], DEFAULT_COLOR)
            thickness = 1 if d["interpolated"] else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"#{d['track_id']} {d['class']}"
            if d.get("subclass"):
                label += f"/{d['subclass']}"
            if d["conf"]:
                label += f" {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        _draw_hud(frame, written, fps, total_in_view, dict(by_class), roi_counts, width)

        writer.write(frame)
        written += 1

        if progress_callback and total_frames:
            pct = int(written / total_frames * 100)
            if pct != last_pct:
                progress_callback(pct)
                last_pct = pct

    cap.release()
    writer.release()

    return {
        "frames_written": written,
        "duration_sec": written / fps if fps else 0,
        "unique_tracks": len(seen_track_ids),
        "roi_unique_tracks": {k: len(v) for k, v in roi_first_seen.items()},
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class AnnotatedVideoWorker(QThread):
    """Render annotated MP4 in a background thread."""

    progress = Signal(str, int)        # session_id, percent
    finished = Signal(str, str, dict)  # session_id, output_path, stats
    error = Signal(str, str)

    def __init__(self, data_manager, session_id: str, output_path: Optional[Path] = None,
                 blur_lp: bool = False, roi_id: str | None = None, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.session_id = session_id
        self.output_path = output_path
        self.blur_lp = blur_lp
        self.roi_id = roi_id

    def run(self):
        try:
            video_path = self.dm.get_video_path(self.session_id)
            if video_path is None or not video_path.exists():
                raise FileNotFoundError(f"Video not found for session: {self.session_id}")

            track_data = self.dm.load_session_data(self.session_id)
            if track_data is None:
                raise FileNotFoundError(f"No tracks/corrections for session: {self.session_id}")

            if self.output_path is None:
                outdir = self.dm.exports_dir / self.session_id
                outdir.mkdir(parents=True, exist_ok=True)
                self.output_path = outdir / f"{self.session_id}_annotated.mp4"

            def _on_progress(pct):
                self.progress.emit(self.session_id, min(pct, 99))

            stats = annotate_video(
                video_path=video_path,
                track_data=track_data,
                output_path=self.output_path,
                blur_lp=self.blur_lp,
                progress_callback=_on_progress,
                roi_id=self.roi_id,
            )
            self.progress.emit(self.session_id, 100)
            self.finished.emit(self.session_id, str(self.output_path), stats)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.error.emit(self.session_id, str(e))
