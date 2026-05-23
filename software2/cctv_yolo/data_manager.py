"""
Data manager — all file I/O for tracks, corrections, videos, exports.
Replaces Flask routes with direct Python calls.

PRD F4-1: emits Qt signals (``corrections_changed``, ``groups_changed``,
``tracks_changed``) so dependent tabs auto-refresh without manual buttons.
"""
import json
import logging
import os
import re
import sys
import tempfile
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, data: dict, *, indent: int = 2, backup: bool = False, keep_backups: int = 5) -> None:
    """Write JSON atomically. Power loss / disk full mid-write can't corrupt
    the destination — we write to a temp file in the same directory and then
    os.replace() into place (atomic on POSIX and Windows).

    PRD C3 — used by every save in the app.

    Args:
        path: destination path
        data: JSON-serializable dict
        indent: pretty-print indent (default 2)
        backup: if True and `path` already exists, copy it to a timestamped
                .bak file in `<dir>/.bak/<stem>-<timestamp>.json` BEFORE
                writing. Keeps the last ``keep_backups`` per stem.
        keep_backups: how many .bak rotations to retain per stem
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if backup and path.exists():
        bak_dir = path.parent / ".bak"
        bak_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        bak_path = bak_dir / f"{path.stem}-{ts}{path.suffix}"
        try:
            bak_path.write_bytes(path.read_bytes())
        except OSError as e:
            logger.warning("Couldn't write backup %s: %s", bak_path, e)

        # Rotate: keep only the N most recent .bak for this stem
        try:
            siblings = sorted(
                bak_dir.glob(f"{path.stem}-*{path.suffix}"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for stale in siblings[keep_backups:]:
                stale.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Couldn't rotate backups for %s: %s", path.stem, e)

    # Write to a temp file in the same directory (same filesystem → atomic rename)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=f"{path.suffix}.tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        # Clean up the temp file on any failure so we don't leave debris.
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise


class DataManager(QObject):
    """Centralized data access — no HTTP, direct file operations.

    Inherits QObject so it can emit Qt signals (PRD F4-1, Part M).
    """

    # PRD F4-1 — fires after any save_corrections() completes. Training tab,
    # Insights tab, and the Active Learning queue subscribe to auto-refresh.
    corrections_changed = Signal(str)   # emits session_id

    # PRD M — fires after group create/delete/rename/recolor/membership change.
    groups_changed = Signal()

    # Optional: fires after save_tracks() — useful for tabs caching session lists.
    tracks_changed = Signal(str)        # emits session_id

    def __init__(self):
        super().__init__()
        # Data root resolution lives in cctv_yolo.paths so every consumer
        # (logger, first_run, models_tab, training, ...) sees the same
        # location. In dev mode this is the repo root; in a frozen build
        # it's the folder containing the .exe / .app — so the install is
        # portable.
        from cctv_yolo.paths import get_data_root
        self._data_root = get_data_root()
        self._init_dirs()

        # Active directories (switch between local and NAS)
        self._local_videos = self.videos_dir
        self._local_tracks = self.tracks_dir
        self._local_corrections = self.corrections_dir
        self._local_exports = self.exports_dir

        self.active_mode = "local"
        self.nas_mount_point = None
        self.session_map = {}  # session_id -> absolute video Path

        # Job tracking (used by processing workers)
        self.processing_jobs = {}
        self.export_jobs = {}

    @property
    def data_root(self):
        return self._data_root

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------

    def _init_dirs(self):
        self.data_dir = self._data_root / "data"
        self.videos_dir = self.data_dir / "videos"
        self.tracks_dir = self.data_dir / "tracks"
        self.corrections_dir = self.data_dir / "corrections"
        self.exports_dir = self.data_dir / "exports"
        self.config_dir = self._data_root / "config"
        self.models_dir = self._data_root / "models"
        for d in [
            self.videos_dir,
            self.tracks_dir,
            self.corrections_dir,
            self.exports_dir,
            self.config_dir,
            self.models_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Return a list of .pt model files available in the models directory."""
        if not self.models_dir.exists():
            return []
        return sorted(f.name for f in self.models_dir.glob("*.pt"))

    def get_last_model(self) -> str | None:
        """Read the last-used model name from config."""
        config_file = self.config_dir / "model_config.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                return data.get("last_model")
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def set_last_model(self, model_name: str):
        """Save the last-used model name to config."""
        config_file = self.config_dir / "model_config.json"
        data = {}
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        data["last_model"] = model_name
        _atomic_write_json(config_file, data)

    def get_last_confidence(self) -> float:
        """Read the last-used confidence threshold from config."""
        config_file = self.config_dir / "model_config.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                return float(data.get("last_confidence", 0.25))
            except (json.JSONDecodeError, OSError, ValueError):
                pass
        return 0.25

    def set_last_confidence(self, conf: float):
        """Save the last-used confidence threshold to config."""
        config_file = self.config_dir / "model_config.json"
        data = {}
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        data["last_confidence"] = conf
        _atomic_write_json(config_file, data)

    # ------------------------------------------------------------------
    # Mode switching (local <-> NAS)
    # ------------------------------------------------------------------

    def switch_to_nas(self, mount_point: Path):
        """Switch to NAS mode — video dir becomes mount point,
        processing goes to _cctv_processing/."""
        self.videos_dir = mount_point
        nas_proc = mount_point / "_cctv_processing"
        self.tracks_dir = nas_proc / "tracks"
        self.corrections_dir = nas_proc / "corrections"
        self.exports_dir = nas_proc / "exports"
        for d in [self.tracks_dir, self.corrections_dir, self.exports_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.active_mode = "nas"
        self.nas_mount_point = mount_point

    def switch_to_local(self):
        """Switch back to local mode."""
        self.videos_dir = self._local_videos
        self.tracks_dir = self._local_tracks
        self.corrections_dir = self._local_corrections
        self.exports_dir = self._local_exports
        self.active_mode = "local"
        self.nas_mount_point = None
        self.session_map = {}

    # ------------------------------------------------------------------
    # Session ID helpers
    # ------------------------------------------------------------------

    def build_session_id(self, video_path: Path, root: Path) -> str:
        """Build a session ID from a video path relative to root.

        Spaces become underscores, path parts are joined with ``--``,
        and non-alphanumeric characters (except ``_``, ``-``, ``.``)
        are stripped.
        """
        relative = video_path.relative_to(root).with_suffix("")
        parts = list(relative.parts)
        sanitized = "--".join(p.replace(" ", "_") for p in parts)
        sanitized = re.sub(r"[^a-zA-Z0-9_\-.]", "", sanitized)
        return sanitized

    def get_video_path(self, session_id: str) -> Path | None:
        """Find the video file for a session ID."""
        # Fast path: already resolved
        if session_id in self.session_map:
            return self.session_map[session_id]

        # NAS mode: lazy-populate the full map on first miss
        if self.active_mode == "nas" and not self.session_map:
            self.get_videos()
            if session_id in self.session_map:
                return self.session_map[session_id]

        # Local fallback: try common extensions in the videos directory
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV"]:
            p = self.videos_dir / (session_id + ext)
            if p.exists():
                self.session_map[session_id] = p
                return p

        # PRD E3-h: batch-registered sessions can live anywhere on disk —
        # fall back to the persistent batch session map.
        try:
            mapped = self._lookup_batch_session(session_id)
        except Exception:
            mapped = None
        if mapped is not None:
            self.session_map[session_id] = mapped
            return mapped

        return None

    # ------------------------------------------------------------------
    # Session listing (processed tracks)
    # ------------------------------------------------------------------

    def get_sessions(self) -> list:
        """Get list of all sessions (processed videos with tracks)."""
        sessions = []
        track_files = list(self.tracks_dir.glob("*.json"))
        for track_file in sorted(track_files):
            session_id = track_file.stem
            try:
                with open(track_file, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            video_path = self.get_video_path(session_id)
            video_name = data.get("video_name", "")
            correction_file = self.corrections_dir / f"{session_id}.json"

            sessions.append(
                {
                    "id": session_id,
                    "video_name": video_path.name if video_path else video_name,
                    "video_exists": video_path is not None,
                    "track_count": len(data.get("tracks", [])),
                    "needs_review": data.get("stats", {}).get("needs_review", 0),
                    "has_corrections": correction_file.exists(),
                    "processed_at": data.get("processed_at", "Unknown"),
                }
            )
        return sessions

    # ------------------------------------------------------------------
    # Video listing
    # ------------------------------------------------------------------

    def get_videos(self) -> list:
        """Get list of all video files with metadata."""
        videos = []
        VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
        new_map = {}

        if self.active_mode == "nas":
            candidates = sorted(self.videos_dir.rglob("*"))
        else:
            candidates = (
                sorted(self.videos_dir.iterdir())
                if self.videos_dir.exists()
                else []
            )

        for f in candidates:
            if not f.is_file() or f.suffix.lower() not in VIDEO_EXTS:
                continue

            if self.active_mode == "nas":
                session_id = self.build_session_id(f, self.videos_dir)
                display_name = str(f.relative_to(self.videos_dir))
                rel = f.relative_to(self.videos_dir)
                folder = str(rel.parent) if len(rel.parts) > 1 else ""
            else:
                session_id = f.stem
                display_name = f.name
                folder = ""

            new_map[session_id] = f

            track_file = self.tracks_dir / f"{session_id}.json"
            correction_file = self.corrections_dir / f"{session_id}.json"

            size_mb = round(f.stat().st_size / (1024 * 1024), 1)

            # Probe video metadata with OpenCV
            cap = cv2.VideoCapture(str(f))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = round(total_frames / fps, 1) if fps > 0 else 0
            cap.release()

            # Processing status
            status = "unprocessed"
            track_count = 0
            if session_id in self.processing_jobs:
                status = self.processing_jobs[session_id].get(
                    "status", "unprocessed"
                )
            if track_file.exists() and status != "processing":
                status = "processed"
                try:
                    with open(track_file, "r") as tf:
                        track_data = json.load(tf)
                    track_count = len(track_data.get("tracks", []))
                except Exception:
                    pass

            has_corrections = correction_file.exists()

            # Export status
            export_dir = self.exports_dir / session_id
            export_status = "none"
            export_count = 0
            if session_id in self.export_jobs:
                export_status = self.export_jobs[session_id].get("status", "none")
            if export_dir.exists() and export_status != "exporting":
                labeled_dir = export_dir / "labeled"
                if labeled_dir.exists():
                    export_count = len(list(labeled_dir.glob("*.jpg")))
                    if export_count > 0:
                        export_status = "exported"

            videos.append(
                {
                    "name": f.name,
                    "session_id": session_id,
                    "display_name": display_name,
                    "folder": folder,
                    "size_mb": size_mb,
                    "fps": round(fps, 1),
                    "total_frames": total_frames,
                    "resolution": f"{width}x{height}" if width else "Unknown",
                    "duration": duration,
                    "status": status,
                    "track_count": track_count,
                    "has_corrections": has_corrections,
                    "export_status": export_status,
                    "export_count": export_count,
                }
            )

        self.session_map = new_map
        return videos

    # ------------------------------------------------------------------
    # Track / correction data
    # ------------------------------------------------------------------

    def load_tracks(self, session_id: str) -> dict | None:
        """Load raw track data (ignoring corrections)."""
        track_file = self.tracks_dir / f"{session_id}.json"
        if not track_file.exists():
            return None
        with open(track_file, "r") as f:
            data = json.load(f)
        if "rois" not in data:
            data["rois"] = []
        return data

    def load_corrections(self, session_id: str) -> dict | None:
        """Load corrections only (returns None if no corrections exist)."""
        correction_file = self.corrections_dir / f"{session_id}.json"
        if not correction_file.exists():
            return None
        with open(correction_file, "r") as f:
            data = json.load(f)
        if "rois" not in data:
            data["rois"] = []
        return data

    def load_session_data(self, session_id: str) -> dict | None:
        """Load track data for a session (corrections first, then raw tracks).

        This is the main entry point for the review page — corrections
        always take priority so that user edits are preserved.
        """
        correction_file = self.corrections_dir / f"{session_id}.json"
        track_file = self.tracks_dir / f"{session_id}.json"
        if correction_file.exists():
            with open(correction_file, "r") as f:
                data = json.load(f)
        elif track_file.exists():
            with open(track_file, "r") as f:
                data = json.load(f)
        else:
            return None
        if "rois" not in data:
            data["rois"] = []
        return data

    def save_corrections(self, session_id: str, data: dict):
        """Save corrected track data.

        Atomic write + .bak rotation (keeps last 5 versions in
        corrections/.bak/). PRD F2-1a/b. Emits ``corrections_changed`` so
        downstream tabs (Training AL queue, Insights, Analytics, Performance)
        refresh automatically.
        """
        # Schema version field — future migrations land here (PRD F2-1e).
        data.setdefault("_version", 2)
        correction_file = self.corrections_dir / f"{session_id}.json"
        _atomic_write_json(correction_file, data, backup=True)
        self.corrections_changed.emit(session_id)

    def save_tracks(self, session_id: str, data: dict):
        """Save raw track data (used by processor after detection)."""
        track_file = self.tracks_dir / f"{session_id}.json"
        _atomic_write_json(track_file, data)
        self.tracks_changed.emit(session_id)

    def has_corrections(self, session_id: str) -> bool:
        """Check whether corrections exist for a session."""
        return (self.corrections_dir / f"{session_id}.json").exists()

    def has_tracks(self, session_id: str) -> bool:
        """Check whether raw tracks exist for a session."""
        return (self.tracks_dir / f"{session_id}.json").exists()

    def delete_corrections(self, session_id: str):
        """Delete corrections for a session (revert to raw tracks)."""
        correction_file = self.corrections_dir / f"{session_id}.json"
        if correction_file.exists():
            correction_file.unlink()

    def delete_tracks(self, session_id: str):
        """Delete raw tracks for a session."""
        track_file = self.tracks_dir / f"{session_id}.json"
        if track_file.exists():
            track_file.unlink()

    def delete_session(self, session_id: str):
        """Delete all data (tracks + corrections + exports) for a session."""
        self.delete_tracks(session_id)
        self.delete_corrections(session_id)
        export_dir = self.exports_dir / session_id
        if export_dir.exists():
            import shutil

            shutil.rmtree(export_dir)

    # ------------------------------------------------------------------
    # Next review helper
    # ------------------------------------------------------------------

    def get_next_review_session(self, current_session_id: str | None = None) -> str | None:
        """Find the next session that needs review.

        Returns the session_id of the first session with
        ``needs_review > 0`` that is different from *current_session_id*,
        or ``None`` if nothing needs review.
        """
        track_files = sorted(self.tracks_dir.glob("*.json"))
        for track_file in track_files:
            session_id = track_file.stem
            if session_id == current_session_id:
                continue
            try:
                with open(track_file, "r") as f:
                    data = json.load(f)
                if data.get("stats", {}).get("needs_review", 0) > 0:
                    # Only suggest if there are no corrections yet
                    if not self.has_corrections(session_id):
                        return session_id
            except (json.JSONDecodeError, OSError):
                continue
        return None

    # ------------------------------------------------------------------
    # Video frame helpers
    # ------------------------------------------------------------------

    def get_video_thumbnail_frame(self, video_path: Path) -> np.ndarray | None:
        """Read a thumbnail frame from a video (returns BGR numpy array)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total // 10, 30))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        # Resize to thumbnail
        h, w = frame.shape[:2]
        thumb_w = 320
        thumb_h = int(h * thumb_w / w)
        return cv2.resize(frame, (thumb_w, thumb_h))

    def get_video_frame(self, video_path: Path, frame_number: int) -> np.ndarray | None:
        """Read a specific frame from a video (returns BGR numpy array)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return frame

    def get_video_info(self, video_path: Path) -> dict:
        """Return metadata dict for a video file."""
        cap = cv2.VideoCapture(str(video_path))
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS) or 0,
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
        cap.release()
        info["duration"] = (
            round(info["total_frames"] / info["fps"], 1) if info["fps"] > 0 else 0
        )
        info["resolution"] = (
            f"{info['width']}x{info['height']}" if info["width"] else "Unknown"
        )
        return info

    # ------------------------------------------------------------------
    # Folder opening
    # ------------------------------------------------------------------

    def open_folder(self, folder_type: str):
        """Open a data folder in the system file manager."""
        import subprocess

        folder_map = {
            # "data" lands on the inner data/ folder (videos + tracks +
            # corrections + exports) — that's what users mean by "data".
            # config/ and models/ live one level up at _data_root.
            "data": self.data_dir,
            "root": self._data_root,
            "videos": self.videos_dir,
            "tracks": self.tracks_dir,
            "corrections": self.corrections_dir,
            "exports": self.exports_dir,
            "models": self.models_dir,
        }
        path = folder_map.get(folder_type)
        if not path:
            return
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    # ------------------------------------------------------------------
    # App info
    # ------------------------------------------------------------------

    def get_app_info(self) -> dict:
        """Return application information (mirrors /api/app-info)."""
        return {
            "data_dir": str(self._data_root),
            "mode": self.active_mode,
            "nas_mount": str(self.nas_mount_point) if self.nas_mount_point else None,
            "videos_dir": str(self.videos_dir),
            "tracks_dir": str(self.tracks_dir),
            "corrections_dir": str(self.corrections_dir),
            "exports_dir": str(self.exports_dir),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_labeled_images(self, session_id: str, sample_rate: int = 1) -> int:
        """Export labeled images for a session. Returns count of exported images.

        Draws bounding boxes on video frames and saves them as JPEG files
        alongside a JSON annotations file.
        """
        video_path = self.get_video_path(session_id)
        if not video_path or not video_path.exists():
            raise FileNotFoundError(f"Video not found for session: {session_id}")

        # Load best available data (corrections first)
        correction_file = self.corrections_dir / f"{session_id}.json"
        track_file = self.tracks_dir / f"{session_id}.json"
        if correction_file.exists():
            with open(correction_file, "r") as f:
                data = json.load(f)
        elif track_file.exists():
            with open(track_file, "r") as f:
                data = json.load(f)
        else:
            raise FileNotFoundError(f"No tracks found for {session_id}")

        tracks = data.get("tracks", [])

        # Build per-frame detection index
        frame_detections = {}
        for track in tracks:
            cls = track.get("class", "vehicle")
            tid = track.get("track_id", 0)
            for fd in track.get("frames", []):
                fn = fd["frame"]
                if fn not in frame_detections:
                    frame_detections[fn] = []
                frame_detections[fn].append(
                    {
                        "bbox": fd["bbox"],
                        "class": cls,
                        "track_id": tid,
                        "conf": fd.get("conf", 0),
                        "interpolated": fd.get("interpolated", False),
                    }
                )

        if not frame_detections:
            raise ValueError("No detections to export")

        output_dir = self.exports_dir / session_id / "labeled"
        output_dir.mkdir(parents=True, exist_ok=True)

        class_colors = {
            "car": (0, 255, 128),
            "truck": (255, 128, 0),
            "bus": (0, 128, 255),
            "motorcycle": (255, 255, 0),
            "bicycle": (128, 0, 255),
        }
        default_color = (200, 200, 200)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        sorted_frames = sorted(frame_detections.keys())
        if sample_rate > 1:
            sorted_frames = sorted_frames[::sample_rate]

        exported = 0
        total_to_export = len(sorted_frames)
        annotations = []

        for frame_num in sorted_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            dets = frame_detections[frame_num]
            frame_annotations = []

            for det in dets:
                x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
                cls = det["class"]
                tid = det["track_id"]
                conf = det["conf"]
                color = class_colors.get(cls, default_color)
                thickness = 2 if not det["interpolated"] else 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                label = f"#{tid} {cls}"
                if conf > 0:
                    label += f" {conf:.2f}"
                if det["interpolated"]:
                    label += " [interp]"

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                frame_annotations.append(
                    {
                        "track_id": tid,
                        "class": cls,
                        "bbox": det["bbox"],
                        "conf": conf,
                        "interpolated": det["interpolated"],
                    }
                )

            filename = f"{session_id}_frame_{frame_num:06d}.jpg"
            cv2.imwrite(
                str(output_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            annotations.append(
                {
                    "frame": frame_num,
                    "file": filename,
                    "detections": frame_annotations,
                }
            )
            exported += 1

            # Update export job progress
            if session_id in self.export_jobs:
                self.export_jobs[session_id]["progress"] = round(
                    exported / total_to_export * 100
                )

        cap.release()

        # Write annotations manifest
        ann_file = self.exports_dir / session_id / "annotations.json"
        ann_data = {
            "video": video_path.name,
            "session_id": session_id,
            "total_frames_exported": exported,
            "total_detections": sum(len(a["detections"]) for a in annotations),
            "frames": annotations,
        }
        _atomic_write_json(ann_file, ann_data)

        return exported

    def export_coco(self, session_id: str, roi_id: str | None = None) -> Path:
        """Export session data in COCO format. Returns the path to the output file.

        ``roi_id`` (PRD F3-2): when set, only tracks that pass through the
        named/id'd ROI are exported.
        """
        data = self.load_session_data(session_id)
        if data is None:
            raise FileNotFoundError(f"No data for session: {session_id}")

        video_path = self.get_video_path(session_id)
        video_name = video_path.name if video_path else session_id

        tracks = data.get("tracks", [])
        if roi_id:
            from cctv_yolo.exports import filter_tracks_by_roi
            tracks = filter_tracks_by_roi(tracks, data.get("rois", []), roi_id)

        # Gather all unique classes
        categories = []
        cat_name_to_id = {}
        for track in tracks:
            cls = track.get("class", "vehicle")
            if cls not in cat_name_to_id:
                cat_id = len(cat_name_to_id) + 1
                cat_name_to_id[cls] = cat_id
                categories.append(
                    {"id": cat_id, "name": cls, "supercategory": "vehicle"}
                )

        images = []
        coco_annotations = []
        ann_id = 1
        seen_frames = set()

        # Get video dimensions
        width, height = 0, 0
        if video_path and video_path.exists():
            info = self.get_video_info(video_path)
            width = info["width"]
            height = info["height"]

        for track in tracks:
            cls = track.get("class", "vehicle")
            cat_id = cat_name_to_id.get(cls, 1)
            tid = track.get("track_id", 0)

            for fd in track.get("frames", []):
                fn = fd["frame"]

                if fn not in seen_frames:
                    seen_frames.add(fn)
                    images.append(
                        {
                            "id": fn,
                            "file_name": f"{video_name}_frame_{fn:06d}.jpg",
                            "width": width,
                            "height": height,
                        }
                    )

                x1, y1, x2, y2 = fd["bbox"]
                bw = x2 - x1
                bh = y2 - y1

                coco_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": fn,
                        "category_id": cat_id,
                        "bbox": [x1, y1, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                        "attributes": {
                            "track_id": tid,
                            "confidence": fd.get("conf", 0),
                            "interpolated": fd.get("interpolated", False),
                        },
                    }
                )
                ann_id += 1

        coco_data = {
            "info": {
                "description": f"CCTV-YOLO export for {session_id}",
                "date_created": datetime.now().isoformat(),
                "version": "1.0",
            },
            "images": sorted(images, key=lambda x: x["id"]),
            "annotations": coco_annotations,
            "categories": categories,
        }

        output_dir = self.exports_dir / session_id / "coco"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "coco_annotations.json"
        _atomic_write_json(output_file, coco_data)

        return output_file

    # ------------------------------------------------------------------
    # Processing job helpers
    # ------------------------------------------------------------------

    def set_processing_status(self, session_id: str, status: str, **kwargs):
        """Update the processing job status for a session."""
        if session_id not in self.processing_jobs:
            self.processing_jobs[session_id] = {}
        self.processing_jobs[session_id]["status"] = status
        self.processing_jobs[session_id].update(kwargs)

    def get_processing_status(self, session_id: str) -> dict:
        """Get current processing status for a session."""
        return self.processing_jobs.get(session_id, {"status": "unknown"})

    def set_export_status(self, session_id: str, status: str, **kwargs):
        """Update the export job status for a session."""
        if session_id not in self.export_jobs:
            self.export_jobs[session_id] = {}
        self.export_jobs[session_id]["status"] = status
        self.export_jobs[session_id].update(kwargs)

    def get_export_status(self, session_id: str) -> dict:
        """Get current export status for a session."""
        return self.export_jobs.get(session_id, {"status": "unknown"})

    def clear_processing_job(self, session_id: str):
        """Remove a processing job entry."""
        self.processing_jobs.pop(session_id, None)

    def clear_export_job(self, session_id: str):
        """Remove an export job entry."""
        self.export_jobs.pop(session_id, None)

    # ------------------------------------------------------------------
    # Processing ROI persistence
    # ------------------------------------------------------------------

    def get_processing_roi(self, session_id: str) -> dict | None:
        """Load a saved processing ROI for a session."""
        roi_file = self.config_dir / "processing_rois.json"
        if not roi_file.exists():
            return None
        try:
            with open(roi_file, "r") as f:
                data = json.load(f)
            return data.get(session_id)
        except (json.JSONDecodeError, OSError):
            return None

    def set_processing_roi(self, session_id: str, roi: dict | None):
        """Save (or clear) a processing ROI for a session."""
        roi_file = self.config_dir / "processing_rois.json"
        data = {}
        if roi_file.exists():
            try:
                with open(roi_file, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        if roi is None:
            data.pop(session_id, None)
        else:
            data[session_id] = roi
        _atomic_write_json(roi_file, data)

    # ------------------------------------------------------------------
    # Global Processing ROI
    # ------------------------------------------------------------------

    def get_global_processing_roi(self) -> dict | None:
        """Load the global processing ROI."""
        roi_file = self.config_dir / "global_roi.json"
        if not roi_file.exists():
            return None
        try:
            with open(roi_file, "r") as f:
                data = json.load(f)
            return data if data else None
        except (json.JSONDecodeError, OSError):
            return None

    def set_global_processing_roi(self, roi: dict | None):
        """Save (or clear) the global processing ROI."""
        roi_file = self.config_dir / "global_roi.json"
        if roi is None:
            roi_file.unlink(missing_ok=True)
        else:
            _atomic_write_json(roi_file, roi)

    # ------------------------------------------------------------------
    # Session Groups (PRD Part M)
    #
    # Groups are named collections of sessions ("Snow weather", "Night-time"
    # etc.). They're used across Performance, Analytics, Insights, Training
    # to scope multi-session views. Single source of truth here.
    # ------------------------------------------------------------------

    def _groups_file(self) -> Path:
        return self.config_dir / "session_groups.json"

    def _load_groups_raw(self) -> dict:
        f = self._groups_file()
        if not f.exists():
            return {"_version": 1, "groups": []}
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            if "groups" not in data:
                data["groups"] = []
            return data
        except (json.JSONDecodeError, OSError):
            logger.warning("session_groups.json unreadable, returning empty")
            return {"_version": 1, "groups": []}

    def list_groups(self) -> list[dict]:
        """All groups, sorted by name."""
        data = self._load_groups_raw()
        groups = list(data.get("groups", []))
        groups.sort(key=lambda g: g.get("name", "").lower())
        return groups

    def get_group(self, group_id: str) -> dict | None:
        for g in self._load_groups_raw().get("groups", []):
            if g.get("id") == group_id:
                return g
        return None

    def _save_groups(self, data: dict) -> None:
        _atomic_write_json(self._groups_file(), data)
        self.groups_changed.emit()

    def create_group(self, name: str, color: str | None = None,
                     description: str = "") -> str:
        """Create a new group. Returns its id.

        If the name already exists, " (2)", " (3)" etc. is appended so
        combo boxes can disambiguate visually. The id is slug-ified from
        the (possibly-suffixed) name; on slug collision a short random
        suffix is added as a last resort.
        """
        from cctv_yolo.theme import ROI_COLOR_ROTATION
        data = self._load_groups_raw()

        # Auto-disambiguate display name on collision: "Snow" + "Snow (2)" + ...
        existing_names = {g.get("name", "").strip().lower()
                          for g in data["groups"]}
        if name.strip().lower() in existing_names:
            n = 2
            while f"{name} ({n})".lower() in existing_names:
                n += 1
            name = f"{name} ({n})"

        # Slug-ify (possibly-suffixed) name for the id.
        import uuid
        base = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.lower()).strip("_") or "group"
        existing_ids = {g["id"] for g in data["groups"]}
        gid = base
        if gid in existing_ids:
            gid = f"{base}_{uuid.uuid4().hex[:6]}"
        if not color:
            # Cycle through palette for visual distinction
            color = ROI_COLOR_ROTATION[len(data["groups"]) % len(ROI_COLOR_ROTATION)]
        now = datetime.now().isoformat()
        data["groups"].append({
            "id": gid,
            "name": name,
            "color": color,
            "description": description,
            "session_ids": [],
            "created_at": now,
            "updated_at": now,
        })
        self._save_groups(data)
        return gid

    def delete_group(self, group_id: str) -> bool:
        data = self._load_groups_raw()
        before = len(data["groups"])
        data["groups"] = [g for g in data["groups"] if g.get("id") != group_id]
        if len(data["groups"]) == before:
            return False
        self._save_groups(data)
        return True

    def rename_group(self, group_id: str, new_name: str) -> bool:
        data = self._load_groups_raw()
        for g in data["groups"]:
            if g.get("id") == group_id:
                g["name"] = new_name
                g["updated_at"] = datetime.now().isoformat()
                self._save_groups(data)
                return True
        return False

    def recolor_group(self, group_id: str, color: str) -> bool:
        data = self._load_groups_raw()
        for g in data["groups"]:
            if g.get("id") == group_id:
                g["color"] = color
                g["updated_at"] = datetime.now().isoformat()
                self._save_groups(data)
                return True
        return False

    def add_to_group(self, group_id: str, session_ids: list[str]) -> int:
        """Add session_ids to a group, skipping duplicates. Returns count added."""
        data = self._load_groups_raw()
        added = 0
        for g in data["groups"]:
            if g.get("id") == group_id:
                existing = set(g.get("session_ids", []))
                for sid in session_ids:
                    if sid not in existing:
                        existing.add(sid)
                        added += 1
                g["session_ids"] = sorted(existing)
                g["updated_at"] = datetime.now().isoformat()
                self._save_groups(data)
                return added
        return 0

    def remove_from_group(self, group_id: str, session_ids: list[str]) -> int:
        """Remove session_ids from a group. Returns count removed."""
        data = self._load_groups_raw()
        for g in data["groups"]:
            if g.get("id") == group_id:
                to_remove = set(session_ids)
                before = len(g.get("session_ids", []))
                g["session_ids"] = [s for s in g.get("session_ids", []) if s not in to_remove]
                removed = before - len(g["session_ids"])
                if removed:
                    g["updated_at"] = datetime.now().isoformat()
                    self._save_groups(data)
                return removed
        return 0

    def get_sessions_in_group(self, group_id: str) -> list[dict]:
        """Return session dicts for every session_id in the group that still exists."""
        g = self.get_group(group_id)
        if not g:
            return []
        all_sessions = {s["id"]: s for s in self.get_sessions()}
        return [all_sessions[sid] for sid in g.get("session_ids", []) if sid in all_sessions]

    # ------------------------------------------------------------------
    # Training history — drives "Build from unused corrections" (PRD J2)
    # ------------------------------------------------------------------

    def _training_history_file(self) -> Path:
        return self.config_dir / "training_history.json"

    def _load_training_history(self) -> dict:
        f = self._training_history_file()
        if not f.exists():
            return {"_version": 1, "history": []}
        try:
            with open(f, "r") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {"_version": 1, "history": []}

    def list_training_history(self) -> list[dict]:
        return list(self._load_training_history().get("history", []))

    def get_last_training_build(self) -> dict | None:
        h = self.list_training_history()
        return h[-1] if h else None

    def record_training_build(self, build_id: str, trained_model: str | None = None) -> None:
        """Snapshot the current corrections+tracks mtimes for every session
        so the next 'unused corrections' build can diff against them.
        PRD J2.
        """
        snapshot = {}
        for f in self.corrections_dir.glob("*.json"):
            sid = f.stem
            try:
                cmt = f.stat().st_mtime
            except OSError:
                cmt = 0
            tf = self.tracks_dir / f.name
            try:
                tmt = tf.stat().st_mtime if tf.exists() else 0
            except OSError:
                tmt = 0
            snapshot[sid] = {"corrections_mtime": cmt, "tracks_mtime": tmt}

        data = self._load_training_history()
        data["history"].append({
            "build_id": build_id,
            "trained_model": trained_model,
            "built_at": datetime.now().isoformat(),
            "session_files_at_build": snapshot,
        })
        _atomic_write_json(self._training_history_file(), data)

    def list_unused_corrections(self) -> list[dict]:
        """Return session dicts for sessions whose corrections (or tracks) have
        been touched since the last training build, or that didn't exist at
        the last build. PRD J2 — drives the 'Build from unused corrections'
        button.
        """
        last = self.get_last_training_build()
        snap = last.get("session_files_at_build", {}) if last else {}
        all_sessions = {s["id"]: s for s in self.get_sessions()}

        result = []
        for sid, s in all_sessions.items():
            if not s.get("has_corrections"):
                continue
            cf = self.corrections_dir / f"{sid}.json"
            tf = self.tracks_dir / f"{sid}.json"
            try:
                cmt = cf.stat().st_mtime if cf.exists() else 0
                tmt = tf.stat().st_mtime if tf.exists() else 0
            except OSError:
                continue
            ref = snap.get(sid)
            if ref is None:
                # New since last build
                result.append({**s, "_reason": "new"})
            elif cmt > ref.get("corrections_mtime", 0):
                result.append({**s, "_reason": "corrected"})
            elif tmt > ref.get("tracks_mtime", 0):
                # Re-processed: tracks newer than at last build
                result.append({**s, "_reason": "reprocessed"})
        return result

    def get_group_stats(self, group_id: str) -> dict:
        """Aggregate stats across every session in a group.

        Returns dict with total_tracks, class_counts, mean_conf (track-weighted),
        n_sessions, n_corrected.
        """
        sessions = self.get_sessions_in_group(group_id)
        if not sessions:
            return {"total_tracks": 0, "class_counts": {}, "mean_conf": 0.0,
                    "n_sessions": 0, "n_corrected": 0}
        total_tracks = 0
        class_counts: dict = defaultdict(int)
        conf_sum = 0.0
        n_corrected = 0
        for s in sessions:
            if s.get("has_corrections"):
                n_corrected += 1
                data = self.load_corrections(s["id"])
            else:
                data = self.load_tracks(s["id"])
            if not data:
                continue
            tracks = data.get("tracks", [])
            for t in tracks:
                cls = t.get("class", "unknown")
                class_counts[cls] += 1
                total_tracks += 1
                conf_sum += t.get("avg_confidence", 0.0)
        mean_conf = (conf_sum / total_tracks) if total_tracks else 0.0
        return {
            "total_tracks": total_tracks,
            "class_counts": dict(class_counts),
            "mean_conf": round(mean_conf, 3),
            "n_sessions": len(sessions),
            "n_corrected": n_corrected,
        }

    # ----- Batch (PRD E) -----
    def build_session_id_for_batch(self, video_path: Path, source_folder: Path) -> str:
        """Build a globally-unique session_id for a video in any source folder.

        Path-aware so nested files keep their structure in the ID, with an
        8-char sha256 of the absolute path appended so two files with the
        same relative name (in different source folders) never collide.
        """
        import hashlib
        video_path = Path(video_path)
        source_folder = Path(source_folder)
        try:
            rel = video_path.relative_to(source_folder)
            parts = list(rel.parts[:-1]) + [rel.stem]
        except ValueError:
            # video isn't inside source_folder — fall back to its stem only
            parts = [video_path.stem]
        safe = "--".join(p.replace(" ", "_") for p in parts)
        safe = re.sub(r"[^a-zA-Z0-9_\-.]", "", safe) or "video"
        abs_hash = hashlib.sha256(str(video_path.resolve()).encode()).hexdigest()[:8]
        return f"{safe}--{abs_hash}"

    def _batch_session_map_file(self) -> Path:
        return self.config_dir / "batch_session_map.json"

    def _load_batch_session_map(self) -> dict:
        f = self._batch_session_map_file()
        if not f.exists():
            return {}
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_batch_session_map(self, data: dict) -> None:
        _atomic_write_json(self._batch_session_map_file(), data)

    def register_batch_session(self, session_id: str, video_path: Path) -> None:
        """Record session_id -> absolute video path so get_video_path() can
        resolve batch-imported videos that live outside ``videos_dir``."""
        data = self._load_batch_session_map()
        data[session_id] = str(Path(video_path).resolve())
        self._save_batch_session_map(data)

    def unregister_batch_session(self, session_id: str) -> None:
        data = self._load_batch_session_map()
        if session_id in data:
            data.pop(session_id, None)
            self._save_batch_session_map(data)

    def _lookup_batch_session(self, session_id: str) -> Path | None:
        data = self._load_batch_session_map()
        raw = data.get(session_id)
        if not raw:
            return None
        p = Path(raw)
        return p if p.exists() else None

    # ----- Batch (PRD E) -----
    def _batch_registry_file(self) -> Path:
        return self.config_dir / "batch_registry.json"

    def load_batch_registry(self) -> dict:
        """Per-folder UI state ({folders: {path: {...}}, active_folder: path})."""
        f = self._batch_registry_file()
        if not f.exists():
            return {"folders": {}, "active_folder": None}
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return {"folders": {}, "active_folder": None}
            data.setdefault("folders", {})
            data.setdefault("active_folder", None)
            return data
        except (json.JSONDecodeError, OSError):
            return {"folders": {}, "active_folder": None}

    def save_batch_registry(self, data: dict) -> None:
        _atomic_write_json(self._batch_registry_file(), data)

    # ----- Correction (PRD F) -----

    def get_rois(self, session_id: str) -> list:
        """Return the ROIs stored in the session's corrections JSON (or tracks
        JSON if no corrections exist). Empty list if neither file exists.
        PRD F3-1.
        """
        data = self.load_session_data(session_id)
        if not data:
            return []
        return list(data.get("rois", []) or [])

    def count_tracks_in_roi(self, session_id: str, roi_id: str) -> int:
        """Count tracks whose bbox center falls in the named/id'd ROI on at
        least one frame. PRD F3-1.
        """
        data = self.load_session_data(session_id)
        if not data:
            return 0
        from cctv_yolo.exports import filter_tracks_by_roi
        return len(filter_tracks_by_roi(data.get("tracks", []), data.get("rois", []), roi_id))
