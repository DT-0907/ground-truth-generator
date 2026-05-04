"""
Before/after corrections side-by-side video.

Compares raw tracks vs corrections frame-by-frame, rendering them
left/right with a divider bar. Frames where the bbox set differs are
flagged with a red corner marker so the diff is obvious during scrubbing.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from cctv_yolo.annotated_export import (
    CLASS_COLORS,
    DEFAULT_COLOR,
    _draw_rois,
)


def _index(track_data: dict) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for tr in track_data.get("tracks", []):
        cls = tr.get("class", "vehicle")
        tid = tr.get("track_id")
        for fd in tr.get("frames", []):
            out[fd["frame"]].append({
                "bbox": fd["bbox"], "class": cls, "track_id": tid,
                "interpolated": fd.get("interpolated", False),
                "occluded": fd.get("occluded", False),
            })
    return out


def _draw_panel(panel, dets, label):
    h, w = panel.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = [int(c) for c in d["bbox"]]
        color = CLASS_COLORS.get(d["class"], DEFAULT_COLOR)
        if d.get("occluded"):
            color = (200, 100, 255)
            thickness = 3
        elif d.get("interpolated"):
            thickness = 1
        else:
            thickness = 2
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, thickness)
        text = f"#{d['track_id']} {d['class']}"
        cv2.putText(panel, text, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(panel, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def _detections_differ(a: list[dict], b: list[dict], iou_thresh: float = 0.6) -> bool:
    if len(a) != len(b):
        return True
    if not a:
        return False
    matched = [False] * len(b)
    for da in a:
        best_j, best_iou = -1, 0.0
        for j, db in enumerate(b):
            if matched[j]:
                continue
            iou = _iou(da["bbox"], db["bbox"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j < 0 or best_iou < iou_thresh:
            return True
        if da["class"] != b[best_j]["class"]:
            return True
        matched[best_j] = True
    return False


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


def render_before_after(
    video_path: Path,
    raw_data: dict,
    corrected_data: dict,
    output_path: Path,
    progress_callback=None,
) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Stack horizontally
    out_w = width * 2 + 4  # 4px divider
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer: {output_path}")

    raw_idx = _index(raw_data)
    corr_idx = _index(corrected_data)
    raw_rois = raw_data.get("rois", []) or []
    corr_rois = corrected_data.get("rois", []) or []

    last_pct = -1
    frames_diff = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        left = frame.copy()
        right = frame.copy()
        _draw_rois(left, raw_rois)
        _draw_rois(right, corr_rois)

        raw_dets = raw_idx.get(written, [])
        corr_dets = corr_idx.get(written, [])
        _draw_panel(left, raw_dets, "BEFORE (raw)")
        _draw_panel(right, corr_dets, "AFTER (corrected)")

        if _detections_differ(raw_dets, corr_dets):
            frames_diff += 1
            cv2.rectangle(right, (0, 0), (width - 1, 6), (0, 0, 255), -1)
            cv2.putText(right, "DIFF", (width - 80, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        cv2.LINE_AA)

        canvas = np.zeros((height, out_w, 3), dtype=np.uint8)
        canvas[:, :width] = left
        canvas[:, width + 4:] = right
        canvas[:, width:width + 4] = (78, 204, 163)  # divider

        writer.write(canvas)
        written += 1

        if progress_callback and total:
            pct = int(written / total * 100)
            if pct != last_pct:
                progress_callback(pct)
                last_pct = pct

    cap.release()
    writer.release()
    return {"frames_written": written, "frames_diff": frames_diff,
            "output_path": str(output_path)}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class BeforeAfterWorker(QThread):
    progress = Signal(str, int)
    finished_ok = Signal(str, str, dict)
    failed = Signal(str, str)

    def __init__(self, data_manager, session_id: str,
                 output_path: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.session_id = session_id
        self.output_path = output_path

    def run(self):
        try:
            raw = self.dm.load_tracks(self.session_id)
            corr = self.dm.load_corrections(self.session_id)
            if not raw:
                raise FileNotFoundError("Raw tracks missing.")
            if not corr:
                raise FileNotFoundError(
                    "No corrections — there's nothing to compare to."
                )
            video_path = self.dm.get_video_path(self.session_id)
            if not video_path or not video_path.exists():
                raise FileNotFoundError("Video missing.")

            if self.output_path is None:
                self.output_path = (
                    self.dm.exports_dir / self.session_id /
                    f"{self.session_id}_before_after.mp4"
                )
            stats = render_before_after(
                video_path, raw, corr, self.output_path,
                progress_callback=lambda p: self.progress.emit(self.session_id, p),
            )
            self.finished_ok.emit(self.session_id, str(self.output_path), stats)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.failed.emit(self.session_id, str(e))
