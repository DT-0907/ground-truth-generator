"""
Event clip extraction + supercut (PRD H5).

Given a session and a configured set of event-types, extract a short
annotated MP4 clip per event and optionally concatenate them into a
single highlight reel.

Event types (selectable via the Analytics UI):
- roi_entry      — track first crosses into an ROI
- roi_exit       — track leaves an ROI it was inside
- anomaly        — z-score outlier (from cctv_yolo.anomaly)
- long_dwell     — track stays in same ROI > dwell_seconds
- speed_outlier  — track in top/bottom 5% of session speeds
- user_flag      — track.needs_review == True
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

EVENT_TYPES = (
    "roi_entry",
    "roi_exit",
    "anomaly",
    "long_dwell",
    "speed_outlier",
    "user_flag",
)


@dataclass
class ClipEvent:
    """One event worth a clip."""
    label: str                  # human-readable banner text
    center_frame: int
    track_ids: list             # tracks to highlight in the clip
    reason: str                 # one of EVENT_TYPES


@dataclass
class ClipExtractConfig:
    """User-configurable event detection thresholds.

    Defaults match the UI defaults so a user can hit Extract with no tweaks.
    """
    types: set = field(default_factory=lambda: {"roi_entry", "anomaly",
                                                "long_dwell", "speed_outlier",
                                                "user_flag"})
    dwell_seconds: float = 8.0
    speed_outlier_pct: float = 5.0      # top/bottom percentile
    z_threshold: float = 2.0            # for anomalies
    pixels_per_meter: float = 20.0      # used for speed outlier calc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roi_pairs(track_data: dict):
    rois = track_data.get("rois", []) or []
    return list(zip(
        [r.get("name") or f"ROI {i+1}" for i, r in enumerate(rois)],
        rois,
    ))


def _detect_roi_entries_exits(track_data: dict, config: ClipExtractConfig,
                              fps: float) -> list[ClipEvent]:
    """Emit one event per (track, roi, entry) and per (track, roi, exit)."""
    events: list[ClipEvent] = []
    pairs = _roi_pairs(track_data)
    want_entry = "roi_entry" in config.types
    want_exit = "roi_exit" in config.types
    want_dwell = "long_dwell" in config.types
    dwell_frames = config.dwell_seconds * fps

    for tr in track_data.get("tracks", []):
        tid = tr.get("track_id")
        cls = tr.get("class", "vehicle")
        frames = sorted(tr.get("frames", []), key=lambda f: f["frame"])
        if not frames:
            continue

        for roi_name, roi in pairs:
            in_now = False
            entry_frame = None
            for fd in frames:
                inside = _bbox_in_roi(fd["bbox"], roi)
                if inside and not in_now:
                    in_now = True
                    entry_frame = fd["frame"]
                    if want_entry:
                        events.append(ClipEvent(
                            label=f"enter '{roi_name}' #{tid} ({cls})",
                            center_frame=fd["frame"],
                            track_ids=[tid],
                            reason="roi_entry",
                        ))
                elif not inside and in_now:
                    in_now = False
                    duration = fd["frame"] - (entry_frame or fd["frame"])
                    if want_exit:
                        events.append(ClipEvent(
                            label=f"exit '{roi_name}' #{tid} ({cls})",
                            center_frame=fd["frame"],
                            track_ids=[tid],
                            reason="roi_exit",
                        ))
                    if want_dwell and duration >= dwell_frames:
                        mid = (entry_frame or 0) + duration // 2
                        events.append(ClipEvent(
                            label=(f"dwell '{roi_name}' #{tid} ({cls}) "
                                   f"{duration / fps:.1f}s"),
                            center_frame=mid,
                            track_ids=[tid],
                            reason="long_dwell",
                        ))
                    entry_frame = None
            # Still inside at end of track? Possible dwell case.
            if in_now and want_dwell and entry_frame is not None:
                duration = frames[-1]["frame"] - entry_frame
                if duration >= dwell_frames:
                    mid = entry_frame + duration // 2
                    events.append(ClipEvent(
                        label=(f"dwell '{roi_name}' #{tid} ({cls}) "
                               f"{duration / fps:.1f}s"),
                        center_frame=mid,
                        track_ids=[tid],
                        reason="long_dwell",
                    ))
    return events


def _detect_speed_outliers(track_data: dict, config: ClipExtractConfig,
                           fps: float) -> list[ClipEvent]:
    """Top/bottom percentile of per-track average speeds."""
    if "speed_outlier" not in config.types:
        return []
    try:
        from cctv_yolo.analytics import estimate_speeds
        speeds = estimate_speeds(track_data, pixels_per_meter=config.pixels_per_meter)
    except Exception:
        return []
    if not speeds:
        return []
    speeds.sort(key=lambda s: s["avg_speed_mph"])
    cut = max(1, int(len(speeds) * (config.speed_outlier_pct / 100.0)))
    bottom = speeds[:cut]
    top = speeds[-cut:]

    # Build a track_id → first/middle frame index
    tr_frames = {
        tr.get("track_id"): sorted(tr.get("frames", []), key=lambda f: f["frame"])
        for tr in track_data.get("tracks", [])
    }
    out: list[ClipEvent] = []
    for s in bottom + top:
        tid = s["track_id"]
        fs = tr_frames.get(tid) or []
        if not fs:
            continue
        center = fs[len(fs) // 2]["frame"]
        kind = "slow" if s in bottom else "fast"
        out.append(ClipEvent(
            label=f"{kind} #{tid} ({s.get('class') or 'vehicle'}) "
                  f"{s['avg_speed_mph']} mph",
            center_frame=center,
            track_ids=[tid],
            reason="speed_outlier",
        ))
    return out


def _detect_anomalies(data_manager, session_id: str, track_data: dict,
                      config: ClipExtractConfig, fps: float) -> list[ClipEvent]:
    if "anomaly" not in config.types:
        return []
    try:
        from cctv_yolo.anomaly import detect_anomalies
        anomalies = detect_anomalies(data_manager, session_id,
                                     z_threshold=config.z_threshold)
    except Exception:
        return []
    if not anomalies:
        return []
    # Anomalies are session-wide (per ROI / hour). Anchor each to the middle
    # of the session's track set so the clip lands on real content.
    tracks = track_data.get("tracks", [])
    if not tracks:
        return []
    all_frames = sorted(
        f["frame"] for tr in tracks for f in tr.get("frames", [])
    )
    if not all_frames:
        return []
    mid = all_frames[len(all_frames) // 2]
    out = []
    for a in anomalies[:8]:
        out.append(ClipEvent(
            label=f"anomaly {a.metric} z={a.z_score} (ROI {a.roi})",
            center_frame=mid,
            track_ids=[],
            reason="anomaly",
        ))
    return out


def _detect_user_flags(track_data: dict, config: ClipExtractConfig) -> list[ClipEvent]:
    if "user_flag" not in config.types:
        return []
    out: list[ClipEvent] = []
    for tr in track_data.get("tracks", []):
        if not tr.get("needs_review"):
            continue
        frames = sorted(tr.get("frames", []), key=lambda f: f["frame"])
        if not frames:
            continue
        first = frames[0]["frame"]
        last = frames[-1]["frame"]
        center = first + (last - first) // 2
        cls = tr.get("class", "vehicle")
        tid = tr.get("track_id")
        out.append(ClipEvent(
            label=f"needs-review #{tid} ({cls})",
            center_frame=center,
            track_ids=[tid],
            reason="user_flag",
        ))
    return out


def find_events(track_data: dict,
                config: Optional[ClipExtractConfig] = None,
                data_manager=None,
                session_id: Optional[str] = None) -> list[ClipEvent]:
    """Detect noteworthy events for the configured types."""
    config = config or ClipExtractConfig()
    fps = float(track_data.get("fps", 30.0)) or 30.0

    events: list[ClipEvent] = []
    events += _detect_roi_entries_exits(track_data, config, fps)
    events += _detect_speed_outliers(track_data, config, fps)
    if data_manager is not None and session_id:
        events += _detect_anomalies(data_manager, session_id, track_data,
                                    config, fps)
    events += _detect_user_flags(track_data, config)

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
        "user_flag":     (0, 0, 200),
        "anomaly":       (0, 80, 200),
        "long_dwell":    (200, 100, 255),
        "roi_entry":     (78, 204, 163),
        "roi_exit":      (78, 163, 204),
        "speed_outlier": (40, 220, 220),
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
    thumbnail_path: Optional[Path] = None,
) -> Path:
    """Render a single annotated clip around *event*.

    If *thumbnail_path* is provided, also writes a PNG of the centre frame.
    """
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
            per_frame[fd["frame"]].append(
                (tr.get("class", "vehicle"), fd, tr.get("track_id"))
            )

    rois = track_data.get("rois", []) or []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    thumb_frame = None
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
        if fnum == event.center_frame:
            thumb_frame = frame.copy()
        elif thumb_frame is None and fnum >= (start + end) // 2:
            thumb_frame = frame.copy()

    cap.release()
    writer.release()

    if thumbnail_path is not None and thumb_frame is not None:
        thumbnail_path = Path(thumbnail_path)
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        # Downscale to 160x90-ish
        th, tw = thumb_frame.shape[:2]
        if tw > 0 and th > 0:
            scale = 160.0 / tw
            new_w = int(tw * scale)
            new_h = int(th * scale)
            small = cv2.resize(thumb_frame, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(thumbnail_path), small)
    return output_path


def render_supercut(clip_paths: list[Path], output_path: Path) -> Path:
    """Concatenate clips by re-encoding through OpenCV (avoids ffmpeg
    dependency). All clips must have the same resolution + fps.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not clip_paths:
        raise ValueError("No clips to concatenate")

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
    # session_id, [{'path', 'thumb', 'label', 'reason'}], supercut_path
    finished_ok = Signal(str, list, str)
    failed = Signal(str, str)

    def __init__(
        self,
        data_manager,
        session_id: str,
        pre_seconds: float = 2.0,
        post_seconds: float = 4.0,
        make_supercut: bool = True,
        max_clips: int = 60,
        config: Optional[ClipExtractConfig] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.dm = data_manager
        self.session_id = session_id
        self.pre = pre_seconds
        self.post = post_seconds
        self.make_supercut = make_supercut
        self.max_clips = max_clips
        self.config = config or ClipExtractConfig()

    def run(self):
        try:
            track_data = self.dm.load_session_data(self.session_id)
            if not track_data:
                raise FileNotFoundError(f"No tracks for {self.session_id}")
            video_path = self.dm.get_video_path(self.session_id)
            if not video_path or not video_path.exists():
                raise FileNotFoundError(f"No video for {self.session_id}")

            events = find_events(track_data, config=self.config,
                                 data_manager=self.dm,
                                 session_id=self.session_id)
            if not events:
                self.finished_ok.emit(self.session_id, [], "")
                return

            events = events[: self.max_clips]
            outdir = self.dm.exports_dir / self.session_id / "clips"
            outdir.mkdir(parents=True, exist_ok=True)
            thumbdir = outdir / "thumbs"
            thumbdir.mkdir(parents=True, exist_ok=True)

            results: list[dict] = []
            paths: list[Path] = []
            for i, e in enumerate(events):
                safe_label = e.label.replace(" ", "_").replace("/", "_")[:48]
                p = outdir / f"{i:03d}_{e.reason}_{safe_label}.mp4"
                t = thumbdir / f"{i:03d}_{e.reason}.png"
                render_clip(
                    video_path, track_data, e, p,
                    pre_seconds=self.pre, post_seconds=self.post,
                    thumbnail_path=t,
                )
                paths.append(p)
                results.append({
                    "path": str(p),
                    "thumb": str(t) if t.exists() else "",
                    "label": e.label,
                    "reason": e.reason,
                })
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
            self.finished_ok.emit(self.session_id, results, supercut_path)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.failed.emit(self.session_id, str(e))
