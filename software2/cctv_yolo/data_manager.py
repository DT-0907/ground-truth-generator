"""
Data manager — all file I/O for tracks, corrections, videos, exports.
Replaces Flask routes with direct Python calls.
"""
import json
import os
import re
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class DataManager:
    """Centralized data access — no HTTP, direct file operations."""

    def __init__(self):
        # Use "CCTV-YOLO-Data" (not "CCTV-YOLO") so the data folder can't
        # collide with a clone of this project on macOS's case-insensitive
        # filesystem (~/Documents/CCTV-YOLO/ and ~/Documents/cctv-yolo/
        # resolve to the same directory — which silently merged the app's
        # data dir with the source repo and surfaced stale files).
        self._data_root = Path.home() / "Documents" / "CCTV-YOLO-Data"
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
        with open(config_file, "w") as f:
            json.dump(data, f, indent=2)

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
        with open(config_file, "w") as f:
            json.dump(data, f, indent=2)

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
        """Save corrected track data."""
        correction_file = self.corrections_dir / f"{session_id}.json"
        with open(correction_file, "w") as f:
            json.dump(data, f, indent=2)

    def save_tracks(self, session_id: str, data: dict):
        """Save raw track data (used by processor after detection)."""
        track_file = self.tracks_dir / f"{session_id}.json"
        with open(track_file, "w") as f:
            json.dump(data, f, indent=2)

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
        with open(ann_file, "w") as f:
            json.dump(ann_data, f, indent=2)

        return exported

    def export_coco(self, session_id: str) -> Path:
        """Export session data in COCO format. Returns the path to the output file."""
        data = self.load_session_data(session_id)
        if data is None:
            raise FileNotFoundError(f"No data for session: {session_id}")

        video_path = self.get_video_path(session_id)
        video_name = video_path.name if video_path else session_id

        tracks = data.get("tracks", [])

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

        output_dir = self.exports_dir / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "coco_annotations.json"
        with open(output_file, "w") as f:
            json.dump(coco_data, f, indent=2)

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
        with open(roi_file, "w") as f:
            json.dump(data, f, indent=2)

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
            with open(roi_file, "w") as f:
                json.dump(roi, f, indent=2)
