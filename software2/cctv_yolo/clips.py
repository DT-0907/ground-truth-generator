"""
Event clip extraction + supercut.

Given a session, extract a short MP4 clip per "event" (needs-review
track, ROI entry, occluded segment, low-conf moment) and optionally
concatenate them into a single highlight reel.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from cctv_yolo.annotated_export import (
    CLASS_COLORS,
    DEFAULT_COLOR,
    ROI_COLOR,
    _bbox_in_roi,
    _draw_rois,
)


@dataclass
class ClipEvent:
    """One event worth a clip."""
    label: str          # e.g. "needs_review #42 (truck)"
    center_frame: int
    track_ids: list     # tracks involved in this event
    reason: str         # short reason code: needs_review|low_conf|roi_entry|occluded


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def find_events(track_data: dict, low_conf_threshold: float = 0.4) -> list[ClipEvent]:
    """Detect noteworthy events from a session's tracks/corrections.

    Returns a deduplicated list of events sorted by frame.
    """
    events: list[ClipEvent] = []
    rois = track_data.get("rois", []) or []
    roi_lookup = list(zip(
        [r.get("name") or f"ROI {i+1}" for i, r in enumerate(rois)],
        rois,
    ))

    for tr in track_data.get("tracks", []):
        tid = tr.get("track_id")
        cls = tr.get("class", "vehicle")
        frames = sorted(tr.get("frames", []), key=lambda f: f["frame"])
        if not frames:
            continue
        first = frames[0]["frame"]
        last = frames[-1]["frame"]
        center = first + (last - first) // 2

        # needs_review
        if tr.get("needs_review"):
            events.append(ClipEvent(
                label=f"needs-review #{tid} ({cls})",
                center_frame=center,
                track_ids=[tid],
                reason="needs_review",
            ))
            continue  # skip stacking other reasons for same track

        # low confidence
        if tr.get("avg_confidence", 1.0) < low_conf_threshold:
            events.append(ClipEvent(
                label=f"low-conf #{tid} ({cls}, "
                      f"avg={tr.get('avg_confidence', 0):.2f})",
                center_frame=center,
                track_ids=[tid],
                reason="low_conf",
            ))

        # occluded segment in this track
        for fd in frames:
            if fd.get("occluded"):
                events.append(ClipEvent(
                    label=f"occluded #{tid} ({cls}) f{fd['frame']}",
                    center_frame=fd["frame"],
                    track_ids=[tid],
                    reason="occluded",
                ))
                break  # one event per track per kind

        # ROI entry — emit once per (track, roi) at the entry frame
        for roi_name, roi in roi_lookup:
            for fd in frames:
                if _bbox_in_roi(fd["bbox"], roi):
                    events.append(ClipEvent(
                        label=f"ROI '{roi_name}' #{tid} ({cls})",
                        center_frame=fd["frame"],
                        track_ids=[tid],
                        reason="roi_entry",
                    ))
                    break

    # Dedup by (label, center_frame within 5 frames)
    seen = set()
    out = []
    for e in sorted(events, key=lambda x: x.center_frame):
        key = (e.label, e.center_frame // 5)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Clip rendering
# ---------------------------------------------------------------------------

def _draw_box(frame, fd, cls):
    x1, y1, x2, y2 = [int(c) for c in fd["bbox"]]
    color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
    if fd.get("occluded"):
        color = (200, 100, 255)
        thickness = 3
    elif fd.get("interpolated"):
        thickness = 1
    else:
        thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def _draw_event_banner(frame, label, reason, w, h):
    bar_h = 36
    overlay = frame.copy()
    color_map = {
        "needs_review": (0, 0, 200),
        "low_conf": (0, 80, 200),
        "occluded": (200, 100, 255),
        "roi_entry": (78, 204, 163),
    }
    bg = color_map.get(reason, (60, 60, 60))
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), bg, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, label, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def render_clip(
    video_path: Path,
    track_data: dict,
    event: ClipEvent,
    output_path: Path,
    pre_seconds: float = 2.0,
    post_seconds: float = 4.0,
    fourcc: str = "mp4v",
) -> Path:
    """Render a single annotated clip around *event*."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start = max(0, int(event.center_frame - pre_seconds * fps))
    end = min(total - 1, int(event.center_frame + post_seconds * fps))

    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(output_path), fourcc_code, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer: {output_path}")

    # Index detections by frame
    per_frame = defaultdict(list)
    for tr in track_data.get("tracks", []):
        for fd in tr.get("frames", []):
            per_frame[fd["frame"]].append((tr.get("class", "vehicle"), fd, tr.get("track_id")))

    rois = track_data.get("rois", []) or []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for fnum in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        _draw_rois(frame, rois)
        for cls, fd, tid in per_frame.get(fnum, []):
            highlight = tid in event.track_ids
            _draw_box(frame, fd, cls)
            if highlight:
                x1, y1, x2, y2 = [int(c) for c in fd["bbox"]]
                cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4),
                              (0, 255, 255), 2)
        _draw_event_banner(frame, event.label, event.reason, width, height)
        writer.write(frame)

    cap.release()
    writer.release()
    return output_path


def render_supercut(clip_paths: list[Path], output_path: Path) -> Path:
    """Concatenate clips by re-encoding through OpenCV (avoids ffmpeg
    dependency). All clips must have the same resolution + fps.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not clip_paths:
        raise ValueError("No clips to concatenate")

    # Read first clip to get dims
    cap0 = cv2.VideoCapture(str(clip_paths[0]))
    if not cap0.isOpened():
        raise RuntimeError(f"Cannot open clip {clip_paths[0]}")
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open supercut writer: {output_path}")

    for path in clip_paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            continue
        # Optional title card before each clip
        title = path.stem
        for _ in range(int(fps)):  # 1-second title card
            card = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(card, title, (40, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                        cv2.LINE_AA)
            writer.write(card)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()

    writer.release()
    return output_path


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class ClipExtractWorker(QThread):
    """Render every event clip + optional supercut for one session."""

    progress = Signal(str, int)
    finished_ok = Signal(str, list, str)  # session_id, clip paths, supercut path
    failed = Signal(str, str)

    def __init__(
        self,
        data_manager,
        session_id: str,
        pre_seconds: float = 2.0,
        post_seconds: float = 4.0,
        make_supercut: bool = True,
        max_clips: int = 60,
        parent=None,
    ):
        super().__init__(parent)
        self.dm = data_manager
        self.session_id = session_id
        self.pre = pre_seconds
        self.post = post_seconds
        self.make_supercut = make_supercut
        self.max_clips = max_clips

    def run(self):
        try:
            track_data = self.dm.load_session_data(self.session_id)
            if not track_data:
                raise FileNotFoundError(f"No tracks for {self.session_id}")
            video_path = self.dm.get_video_path(self.session_id)
            if not video_path or not video_path.exists():
                raise FileNotFoundError(f"No video for {self.session_id}")

            events = find_events(track_data)
            if not events:
                self.finished_ok.emit(self.session_id, [], "")
                return

            events = events[: self.max_clips]
            outdir = self.dm.exports_dir / self.session_id / "clips"
            outdir.mkdir(parents=True, exist_ok=True)

            paths: list[Path] = []
            for i, e in enumerate(events):
                safe_label = e.label.replace(" ", "_").replace("/", "_")[:48]
                p = outdir / f"{i:03d}_{e.reason}_{safe_label}.mp4"
                render_clip(
                    video_path, track_data, e, p,
                    pre_seconds=self.pre, post_seconds=self.post,
                )
                paths.append(p)
                self.progress.emit(
                    self.session_id,
                    int((i + 1) / len(events) * 90),  # leave 10% for supercut
                )

            supercut_path = ""
            if self.make_supercut and paths:
                supercut = outdir / "supercut.mp4"
                render_supercut(paths, supercut)
                supercut_path = str(supercut)
            self.progress.emit(self.session_id, 100)
            self.finished_ok.emit(
                self.session_id, [str(p) for p in paths], supercut_path
            )
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.failed.emit(self.session_id, str(e))
