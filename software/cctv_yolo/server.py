"""
Flask web application for video track correction — standalone desktop version.

Data is stored in ~/Documents/CCTV-YOLO/ so it persists across app updates.
"""

import csv
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import base64
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import cv2


# ---------------------------------------------------------------------------
# Path helpers — work both in dev and inside a PyInstaller bundle
# ---------------------------------------------------------------------------

def _resource_path(relative):
    """Absolute path to a bundled resource (templates, etc.)."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "cctv_yolo" / relative
    return Path(__file__).parent / relative


# ---------------------------------------------------------------------------
# Data directories — live in the user's Documents folder
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(os.environ.get("CCTV_YOLO_DATA_DIR", ""))
if not _DATA_ROOT or not _DATA_ROOT.is_absolute():
    _DATA_ROOT = Path.home() / "Documents" / "CCTV-YOLO"

DATA_DIR = _DATA_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
TRACKS_DIR = DATA_DIR / "tracks"
CORRECTIONS_DIR = DATA_DIR / "corrections"
EXPORTS_DIR = DATA_DIR / "exports"
MODELS_DIR = _DATA_ROOT / "models"

for _d in [VIDEOS_DIR, TRACKS_DIR, CORRECTIONS_DIR, EXPORTS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

NAS_CONFIG_FILE = _DATA_ROOT / "config" / "nas.json"

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    template_folder=str(_resource_path("templates")),
)
CORS(app)

# ---------------------------------------------------------------------------
# NAS mode state
# ---------------------------------------------------------------------------

_active_mode = "local"
_nas_mount_point = None
_session_map = {}  # session_id -> absolute video Path

_local_videos = VIDEOS_DIR
_local_tracks = TRACKS_DIR
_local_corrections = CORRECTIONS_DIR
_local_exports = EXPORTS_DIR


def switch_to_nas(mount_point):
    global VIDEOS_DIR, TRACKS_DIR, CORRECTIONS_DIR, EXPORTS_DIR
    global _active_mode, _nas_mount_point
    VIDEOS_DIR = mount_point
    nas_proc = mount_point / "_cctv_processing"
    TRACKS_DIR = nas_proc / "tracks"
    CORRECTIONS_DIR = nas_proc / "corrections"
    EXPORTS_DIR = nas_proc / "exports"
    for d in [TRACKS_DIR, CORRECTIONS_DIR, EXPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    _active_mode = "nas"
    _nas_mount_point = mount_point


def switch_to_local():
    global VIDEOS_DIR, TRACKS_DIR, CORRECTIONS_DIR, EXPORTS_DIR
    global _active_mode, _nas_mount_point, _session_map
    VIDEOS_DIR = _local_videos
    TRACKS_DIR = _local_tracks
    CORRECTIONS_DIR = _local_corrections
    EXPORTS_DIR = _local_exports
    _active_mode = "local"
    _nas_mount_point = None
    _session_map = {}


# ---------------------------------------------------------------------------
# Session-ID helpers
# ---------------------------------------------------------------------------

def build_session_id(video_path, root):
    relative = video_path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    sanitized = "--".join(p.replace(" ", "_") for p in parts)
    sanitized = re.sub(r"[^a-zA-Z0-9_\-.]", "", sanitized)
    return sanitized


def get_video_path(session_id):
    if session_id in _session_map:
        return _session_map[session_id]
    if _active_mode == "nas" and not _session_map:
        get_videos()
        if session_id in _session_map:
            return _session_map[session_id]
    for ext in [".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV"]:
        p = VIDEOS_DIR / (session_id + ext)
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Background job state
# ---------------------------------------------------------------------------

processing_jobs = {}
export_jobs = {}
processing_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def get_sessions():
    sessions = []
    track_files = list(TRACKS_DIR.glob("*.json"))
    for track_file in sorted(track_files):
        session_id = track_file.stem
        with open(track_file, "r") as f:
            data = json.load(f)
        video_path = get_video_path(session_id)
        video_name = data.get("video_name", "")
        correction_file = CORRECTIONS_DIR / f"{session_id}.json"
        sessions.append({
            "id": session_id,
            "video_name": video_path.name if video_path else video_name,
            "video_exists": video_path is not None,
            "track_count": len(data.get("tracks", [])),
            "needs_review": data.get("stats", {}).get("needs_review", 0),
            "has_corrections": correction_file.exists(),
            "processed_at": data.get("processed_at", "Unknown"),
        })
    return sessions


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    sessions = get_sessions()
    return render_template("index.html", sessions=sessions)


@app.route("/review/<session_id>")
def review(session_id):
    track_file = TRACKS_DIR / f"{session_id}.json"
    if not track_file.exists():
        return "Session not found", 404
    correction_file = CORRECTIONS_DIR / f"{session_id}.json"
    if correction_file.exists():
        with open(correction_file, "r") as f:
            data = json.load(f)
    else:
        with open(track_file, "r") as f:
            data = json.load(f)
    if "rois" not in data:
        data["rois"] = []
    return render_template("review.html", session_id=session_id, data=data)


# ---------------------------------------------------------------------------
# API — sessions & tracks
# ---------------------------------------------------------------------------

@app.route("/api/sessions")
def api_sessions():
    return jsonify(get_sessions())


@app.route("/api/session/<session_id>/tracks")
def api_get_tracks(session_id):
    correction_file = CORRECTIONS_DIR / f"{session_id}.json"
    track_file = TRACKS_DIR / f"{session_id}.json"
    if correction_file.exists():
        file_to_load = correction_file
    elif track_file.exists():
        file_to_load = track_file
    else:
        return jsonify({"error": "Session not found"}), 404
    with open(file_to_load, "r") as f:
        data = json.load(f)
    data["source"] = "corrections" if correction_file.exists() else "tracks"
    return jsonify(data)


@app.route("/api/session/<session_id>/tracks", methods=["POST"])
def api_save_tracks(session_id):
    try:
        data = request.json
        correction_file = CORRECTIONS_DIR / f"{session_id}.json"
        with open(correction_file, "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"success": True, "message": "Saved"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<session_id>/delete_track", methods=["POST"])
def api_delete_track(session_id):
    try:
        track_id = request.json.get("track_id")
        if track_id is None:
            return jsonify({"error": "track_id required"}), 400
        correction_file = CORRECTIONS_DIR / f"{session_id}.json"
        track_file = TRACKS_DIR / f"{session_id}.json"
        source_file = correction_file if correction_file.exists() else track_file
        with open(source_file, "r") as f:
            data = json.load(f)
        original_count = len(data["tracks"])
        data["tracks"] = [t for t in data["tracks"] if t["track_id"] != track_id]
        if len(data["tracks"]) == original_count:
            return jsonify({"error": f"Track {track_id} not found"}), 404
        with open(correction_file, "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"success": True, "remaining": len(data["tracks"])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<session_id>/change_class", methods=["POST"])
def api_change_class(session_id):
    try:
        track_id = request.json.get("track_id")
        new_class = request.json.get("new_class")
        if track_id is None or not new_class:
            return jsonify({"error": "track_id and new_class required"}), 400
        correction_file = CORRECTIONS_DIR / f"{session_id}.json"
        track_file = TRACKS_DIR / f"{session_id}.json"
        source_file = correction_file if correction_file.exists() else track_file
        with open(source_file, "r") as f:
            data = json.load(f)
        found = False
        for track in data["tracks"]:
            if track["track_id"] == track_id:
                track["class"] = new_class
                found = True
                break
        if not found:
            return jsonify({"error": f"Track {track_id} not found"}), 404
        with open(correction_file, "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<session_id>/merge_tracks", methods=["POST"])
def api_merge_tracks(session_id):
    try:
        track_id = request.json.get("track_id")
        target_id = request.json.get("target_id")
        if track_id is None or target_id is None:
            return jsonify({"error": "track_id and target_id required"}), 400
        if track_id == target_id:
            return jsonify({"error": "Cannot merge track with itself"}), 400
        correction_file = CORRECTIONS_DIR / f"{session_id}.json"
        track_file = TRACKS_DIR / f"{session_id}.json"
        source_file = correction_file if correction_file.exists() else track_file
        with open(source_file, "r") as f:
            data = json.load(f)
        source_track = None
        target_track = None
        for track in data["tracks"]:
            if track["track_id"] == track_id:
                source_track = track
            elif track["track_id"] == target_id:
                target_track = track
        if not source_track:
            return jsonify({"error": f"Source track {track_id} not found"}), 404
        if not target_track:
            return jsonify({"error": f"Target track {target_id} not found"}), 404
        target_track["frames"].extend(source_track["frames"])
        seen = {}
        for f in target_track["frames"]:
            fn = f["frame"]
            if fn not in seen or f.get("conf", 0) > seen[fn].get("conf", 0):
                seen[fn] = f
        target_track["frames"] = sorted(seen.values(), key=lambda f: f["frame"])
        all_sorted = sorted(target_track["frames"], key=lambda f: f["frame"])
        interpolated = []
        for i in range(len(all_sorted) - 1):
            curr = all_sorted[i]
            nxt = all_sorted[i + 1]
            gap = nxt["frame"] - curr["frame"]
            if gap > 1:
                for g in range(1, gap):
                    t = g / gap
                    bbox = [
                        round(curr["bbox"][j] + (nxt["bbox"][j] - curr["bbox"][j]) * t, 1)
                        for j in range(4)
                    ]
                    interpolated.append({
                        "frame": curr["frame"] + g,
                        "bbox": bbox,
                        "conf": 0,
                        "interpolated": True,
                    })
        if interpolated:
            target_track["frames"].extend(interpolated)
            target_track["frames"].sort(key=lambda f: f["frame"])
        if target_track["frames"]:
            target_track["start_frame"] = target_track["frames"][0]["frame"]
            target_track["end_frame"] = target_track["frames"][-1]["frame"]
        data["tracks"] = [t for t in data["tracks"] if t["track_id"] != track_id]
        with open(correction_file, "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<session_id>/split_track", methods=["POST"])
def api_split_track(session_id):
    try:
        track_id = request.json.get("track_id")
        frame = request.json.get("frame")
        if track_id is None or frame is None:
            return jsonify({"error": "track_id and frame required"}), 400
        correction_file = CORRECTIONS_DIR / f"{session_id}.json"
        track_file = TRACKS_DIR / f"{session_id}.json"
        source_file = correction_file if correction_file.exists() else track_file
        with open(source_file, "r") as f:
            data = json.load(f)
        track = None
        for t in data["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        if not track:
            return jsonify({"error": f"Track {track_id} not found"}), 404
        before = [f for f in track["frames"] if f["frame"] < frame]
        after = [f for f in track["frames"] if f["frame"] >= frame]
        if not before or not after:
            return jsonify({"error": "Cannot split: no frames on one side"}), 400
        track["frames"] = before
        track["end_frame"] = before[-1]["frame"]
        new_id = max(t["track_id"] for t in data["tracks"]) + 1
        new_track = {
            "track_id": new_id,
            "class": track["class"],
            "class_id": track.get("class_id", 0),
            "frames": after,
            "start_frame": after[0]["frame"],
            "end_frame": after[-1]["frame"],
            "needs_review": True,
            "avg_confidence": track.get("avg_confidence", 0.5),
        }
        data["tracks"].append(new_track)
        with open(correction_file, "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"success": True, "new_track_id": new_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Routes — video serving
# ---------------------------------------------------------------------------

@app.route("/video/<session_id>")
def serve_video(session_id):
    video_path = get_video_path(session_id)
    if video_path and video_path.exists():
        return send_from_directory(video_path.parent, video_path.name)
    track_file = TRACKS_DIR / f"{session_id}.json"
    if track_file.exists():
        with open(track_file, "r") as f:
            data = json.load(f)
        video_name = data.get("video_name", "")
        if video_name and (VIDEOS_DIR / video_name).exists():
            return send_from_directory(VIDEOS_DIR, video_name)
    return "Video not found", 404


@app.route("/frame/<session_id>/<int:frame_num>")
def get_frame(session_id, frame_num):
    video_path = get_video_path(session_id)
    if not video_path:
        track_file = TRACKS_DIR / f"{session_id}.json"
        if track_file.exists():
            with open(track_file, "r") as f:
                data = json.load(f)
            video_name = data.get("video_name", "")
            candidate = VIDEOS_DIR / video_name
            if candidate.exists():
                video_path = candidate
    if not video_path:
        return "Video not found", 404
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Frame not found", 404
    _, buffer = cv2.imencode(".jpg", frame)
    return Response(buffer.tobytes(), mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# API — video listing & discovery
# ---------------------------------------------------------------------------

def get_videos():
    global _session_map
    videos = []
    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
    new_map = {}

    if _active_mode == "nas":
        candidates = sorted(VIDEOS_DIR.rglob("*"))
    else:
        candidates = sorted(VIDEOS_DIR.iterdir()) if VIDEOS_DIR.exists() else []

    for f in candidates:
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXTS:
            continue

        if _active_mode == "nas":
            session_id = build_session_id(f, VIDEOS_DIR)
            display_name = str(f.relative_to(VIDEOS_DIR))
            rel = f.relative_to(VIDEOS_DIR)
            folder = str(rel.parent) if len(rel.parts) > 1 else ""
        else:
            session_id = f.stem
            display_name = f.name
            folder = ""

        new_map[session_id] = f

        track_file = TRACKS_DIR / f"{session_id}.json"
        correction_file = CORRECTIONS_DIR / f"{session_id}.json"

        size_mb = round(f.stat().st_size / (1024 * 1024), 1)

        cap = cv2.VideoCapture(str(f))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = round(total_frames / fps, 1) if fps > 0 else 0
        cap.release()

        status = "unprocessed"
        track_count = 0
        with processing_lock:
            if session_id in processing_jobs:
                job = processing_jobs[session_id]
                status = job["status"]
        if track_file.exists() and status != "processing":
            status = "processed"
            try:
                with open(track_file, "r") as tf:
                    track_data = json.load(tf)
                track_count = len(track_data.get("tracks", []))
            except Exception:
                pass

        has_corrections = correction_file.exists()

        export_dir = EXPORTS_DIR / session_id
        export_status = "none"
        export_count = 0
        with processing_lock:
            if session_id in export_jobs:
                ej = export_jobs[session_id]
                export_status = ej["status"]
        if export_dir.exists() and export_status != "exporting":
            labeled_dir = export_dir / "labeled"
            if labeled_dir.exists():
                export_count = len(list(labeled_dir.glob("*.jpg")))
                if export_count > 0:
                    export_status = "exported"

        videos.append({
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
        })

    _session_map = new_map
    return videos


@app.route("/api/videos")
def api_videos():
    return jsonify(get_videos())


@app.route("/api/videos/thumbnail/<session_id>")
def api_video_thumbnail(session_id):
    video_path = get_video_path(session_id)
    if not video_path or not video_path.exists():
        return "Video not found", 404
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(total // 10, 30))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Could not read frame", 500
    h, w = frame.shape[:2]
    thumb_w = 320
    thumb_h = int(h * thumb_w / w)
    thumb = cv2.resize(frame, (thumb_w, thumb_h))
    _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(buffer.tobytes(), mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# API — processing
# ---------------------------------------------------------------------------

@app.route("/api/videos/process", methods=["POST"])
def api_process_video():
    session_id = request.json.get("session_id") or request.json.get("video_name", "")
    if not request.json.get("session_id") and session_id:
        session_id_lookup = Path(session_id).stem
    else:
        session_id_lookup = session_id

    video_path = get_video_path(session_id_lookup)
    if not video_path or not video_path.exists():
        return jsonify({"error": f"Video not found: {session_id}"}), 404

    model = request.json.get("model", "yolov8m.pt")
    conf = request.json.get("conf", 0.25)
    processing_roi = request.json.get("processing_roi", None)

    with processing_lock:
        if session_id_lookup in processing_jobs and processing_jobs[session_id_lookup]["status"] == "processing":
            return jsonify({"error": "Already processing"}), 409
        processing_jobs[session_id_lookup] = {
            "status": "processing",
            "progress": 0,
            "error": None,
            "started_at": time.time(),
        }

    def run_processing():
        def _progress(pct):
            with processing_lock:
                if session_id_lookup in processing_jobs:
                    processing_jobs[session_id_lookup]["progress"] = pct

        try:
            from cctv_yolo.processor import process_video
            process_video(
                str(video_path),
                str(TRACKS_DIR),
                model,
                conf,
                session_id=session_id_lookup,
                progress_callback=_progress,
                models_dir=str(MODELS_DIR),
                processing_roi=processing_roi,
            )
            with processing_lock:
                processing_jobs[session_id_lookup]["status"] = "done"
                processing_jobs[session_id_lookup]["progress"] = 100
        except Exception as e:
            with processing_lock:
                processing_jobs[session_id_lookup]["status"] = "error"
                processing_jobs[session_id_lookup]["error"] = str(e)

    threading.Thread(target=run_processing, daemon=True).start()
    return jsonify({"success": True, "message": f"Processing started: {session_id_lookup}"})


@app.route("/api/videos/status/<session_id>")
def api_processing_status(session_id):
    with processing_lock:
        if session_id in processing_jobs:
            return jsonify(processing_jobs[session_id])
    track_file = TRACKS_DIR / f"{session_id}.json"
    if track_file.exists():
        return jsonify({"status": "done", "progress": 100, "error": None})
    return jsonify({"status": "unprocessed", "progress": 0, "error": None})


# ---------------------------------------------------------------------------
# API — export
# ---------------------------------------------------------------------------

def export_labeled_images(session_id, sample_rate=1):
    video_path = get_video_path(session_id)
    if not video_path or not video_path.exists():
        raise FileNotFoundError(f"Video not found for session: {session_id}")

    correction_file = CORRECTIONS_DIR / f"{session_id}.json"
    track_file = TRACKS_DIR / f"{session_id}.json"
    if correction_file.exists():
        with open(correction_file, "r") as f:
            data = json.load(f)
    elif track_file.exists():
        with open(track_file, "r") as f:
            data = json.load(f)
    else:
        raise FileNotFoundError(f"No tracks found for {session_id}")

    tracks = data.get("tracks", [])

    # Filter tracks by active ROIs if set
    active_roi_ids = data.get("active_roi_ids", [])
    rois = data.get("rois", [])
    if active_roi_ids and rois:
        active_rois = [r for r in rois if r.get("id") in active_roi_ids]
        if active_rois:
            filtered = []
            for track in tracks:
                for roi in active_rois:
                    pts = roi.get("points", [])
                    in_roi = False
                    for fd in track.get("frames", []):
                        bbox = fd.get("bbox", [0, 0, 0, 0])
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        if roi.get("type") == "rect" and len(pts) >= 2:
                            x1 = min(pts[0]["x"], pts[1]["x"])
                            y1 = min(pts[0]["y"], pts[1]["y"])
                            x2 = max(pts[0]["x"], pts[1]["x"])
                            y2 = max(pts[0]["y"], pts[1]["y"])
                            if x1 <= cx <= x2 and y1 <= cy <= y2:
                                in_roi = True
                                break
                        elif roi.get("type") == "polygon" and len(pts) >= 3:
                            inside = False
                            j = len(pts) - 1
                            for i in range(len(pts)):
                                xi, yi = pts[i]["x"], pts[i]["y"]
                                xj, yj = pts[j]["x"], pts[j]["y"]
                                if ((yi > cy) != (yj > cy)) and (cx < (xj - xi) * (cy - yi) / (yj - yi) + xi):
                                    inside = not inside
                                j = i
                            if inside:
                                in_roi = True
                                break
                    if in_roi:
                        filtered.append(track)
                        break
            tracks = filtered

    frame_detections = {}
    for track in tracks:
        cls = track.get("class", "vehicle")
        tid = track.get("track_id", 0)
        for fd in track.get("frames", []):
            fn = fd["frame"]
            if fn not in frame_detections:
                frame_detections[fn] = []
            frame_detections[fn].append({
                "bbox": fd["bbox"],
                "class": cls,
                "track_id": tid,
                "conf": fd.get("conf", 0),
                "interpolated": fd.get("interpolated", False),
            })

    if not frame_detections:
        raise ValueError("No detections to export")

    output_dir = EXPORTS_DIR / session_id / "labeled"
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

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            frame_annotations.append({
                "track_id": tid,
                "class": cls,
                "bbox": det["bbox"],
                "conf": conf,
                "interpolated": det["interpolated"],
            })

        filename = f"{session_id}_frame_{frame_num:06d}.jpg"
        cv2.imwrite(str(output_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        annotations.append({
            "frame": frame_num,
            "file": filename,
            "detections": frame_annotations,
        })

        exported += 1
        with processing_lock:
            if session_id in export_jobs:
                export_jobs[session_id]["progress"] = round(exported / total_to_export * 100)

    cap.release()

    ann_file = EXPORTS_DIR / session_id / "annotations.json"
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


@app.route("/api/videos/export", methods=["POST"])
def api_export_video():
    session_id = request.json.get("session_id") or request.json.get("video_name", "")
    if not request.json.get("session_id") and session_id:
        session_id = Path(session_id).stem

    video_path = get_video_path(session_id)
    if not video_path or not video_path.exists():
        return jsonify({"error": f"Video not found: {session_id}"}), 404

    track_file = TRACKS_DIR / f"{session_id}.json"
    correction_file = CORRECTIONS_DIR / f"{session_id}.json"
    if not track_file.exists() and not correction_file.exists():
        return jsonify({"error": "No tracks available — process the video first"}), 400

    sample_rate = request.json.get("sample_rate", 1)

    with processing_lock:
        if session_id in export_jobs and export_jobs[session_id]["status"] == "exporting":
            return jsonify({"error": "Already exporting"}), 409
        export_jobs[session_id] = {
            "status": "exporting",
            "progress": 0,
            "error": None,
            "started_at": time.time(),
            "output_dir": str(EXPORTS_DIR / session_id),
        }

    def run_export():
        try:
            count = export_labeled_images(session_id, sample_rate)
            with processing_lock:
                export_jobs[session_id]["status"] = "done"
                export_jobs[session_id]["progress"] = 100
                export_jobs[session_id]["count"] = count
        except Exception as e:
            with processing_lock:
                export_jobs[session_id]["status"] = "error"
                export_jobs[session_id]["error"] = str(e)

    threading.Thread(target=run_export, daemon=True).start()
    return jsonify({"success": True, "message": f"Export started: {session_id}"})


@app.route("/api/videos/export-status/<session_id>")
def api_export_status(session_id):
    with processing_lock:
        if session_id in export_jobs:
            return jsonify(export_jobs[session_id])
    labeled_dir = EXPORTS_DIR / session_id / "labeled"
    if labeled_dir.exists():
        count = len(list(labeled_dir.glob("*.jpg")))
        if count > 0:
            return jsonify({"status": "done", "progress": 100, "error": None, "count": count})
    return jsonify({"status": "none", "progress": 0, "error": None})


# ---------------------------------------------------------------------------
# API — models
# ---------------------------------------------------------------------------

@app.route("/api/models")
def api_models():
    """List available .pt model files."""
    local_models = sorted([f.name for f in MODELS_DIR.glob("*.pt")])
    builtins = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    all_models = list(dict.fromkeys(builtins + local_models))  # dedup preserving order
    return jsonify({"models": all_models})


# ---------------------------------------------------------------------------
# API — performance
# ---------------------------------------------------------------------------

def _point_in_polygon_js(px, py, polygon):
    """Ray-casting point-in-polygon test (JS-style point dicts)."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]["x"], polygon[i]["y"]
        xj, yj = polygon[j]["x"], polygon[j]["y"]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _track_in_roi(track, roi):
    """Check if any frame of a track has its bbox center inside the ROI."""
    for fd in track.get("frames", []):
        bbox = fd.get("bbox", [0, 0, 0, 0])
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        if roi.get("type") == "rect":
            pts = roi["points"]
            x1, y1 = pts[0]["x"], pts[0]["y"]
            x2, y2 = pts[1]["x"], pts[1]["y"]
            if min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2):
                return True
        else:  # polygon
            if _point_in_polygon_js(cx, cy, roi["points"]):
                return True
    return False


def _load_session_data(session_id):
    """Load corrections (preferred) or tracks for a session."""
    correction_file = CORRECTIONS_DIR / f"{session_id}.json"
    track_file = TRACKS_DIR / f"{session_id}.json"
    if correction_file.exists():
        with open(correction_file, "r") as f:
            return json.load(f)
    elif track_file.exists():
        with open(track_file, "r") as f:
            return json.load(f)
    return None


@app.route("/api/performance/<session_id>")
def api_performance(session_id):
    """Get performance stats for a session."""
    data = _load_session_data(session_id)
    if not data:
        return jsonify({"error": "Session not found"}), 404

    tracks = data.get("tracks", [])
    rois = data.get("rois", [])

    # Count by vehicle type
    by_type = {}
    for t in tracks:
        cls = t.get("class", "unknown")
        by_type[cls] = by_type.get(cls, 0) + 1

    # Count by ROI (if ROIs defined)
    roi_counts = []
    for roi in rois:
        count = 0
        by_class = {}
        for t in tracks:
            if _track_in_roi(t, roi):
                count += 1
                cls = t.get("class", "unknown")
                by_class[cls] = by_class.get(cls, 0) + 1
        roi_counts.append({
            "name": roi.get("name", ""),
            "count": count,
            "by_class": by_class,
        })

    return jsonify({
        "total": len(tracks),
        "by_type": by_type,
        "roi_counts": roi_counts,
        "model": data.get("model", ""),
        "conf_threshold": data.get("conf_threshold", 0),
        "processed_at": data.get("processed_at", ""),
    })


@app.route("/api/performance/<session_id>/csv")
def api_performance_csv(session_id):
    """Export performance stats as CSV."""
    data = _load_session_data(session_id)
    if not data:
        return jsonify({"error": "Session not found"}), 404

    tracks = data.get("tracks", [])
    rois = data.get("rois", [])

    output = io.StringIO()
    writer = csv.writer(output)

    # Summary section
    writer.writerow(["Session", session_id])
    writer.writerow(["Total Vehicles", len(tracks)])
    writer.writerow(["Model", data.get("model", "")])
    writer.writerow(["Confidence Threshold", data.get("conf_threshold", "")])
    writer.writerow(["Processed At", data.get("processed_at", "")])
    writer.writerow([])

    # By type
    writer.writerow(["Vehicle Type", "Count"])
    by_type = {}
    for t in tracks:
        cls = t.get("class", "unknown")
        by_type[cls] = by_type.get(cls, 0) + 1
    for cls, count in sorted(by_type.items()):
        writer.writerow([cls, count])
    writer.writerow([])

    # By ROI
    if rois:
        writer.writerow(["ROI Name", "Total Count"])
        for roi in rois:
            count = sum(1 for t in tracks if _track_in_roi(t, roi))
            writer.writerow([roi.get("name", ""), count])
        writer.writerow([])

    # Track detail
    writer.writerow(["Track ID", "Class", "Start Frame", "End Frame", "Frames", "Avg Confidence"])
    for t in tracks:
        writer.writerow([
            t.get("track_id", ""),
            t.get("class", ""),
            t.get("start_frame", ""),
            t.get("end_frame", ""),
            len(t.get("frames", [])),
            t.get("avg_confidence", ""),
        ])

    csv_content = output.getvalue()
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={session_id}_performance.csv"},
    )


# ---------------------------------------------------------------------------
# API — NAS mount / config
# ---------------------------------------------------------------------------

def load_nas_config():
    if NAS_CONFIG_FILE.exists():
        with open(NAS_CONFIG_FILE) as f:
            data = json.load(f)
        data["password"] = base64.b64decode(data["password"]).decode()
        return data
    return None


def save_nas_config(config):
    safe = dict(config)
    safe["password"] = base64.b64encode(config["password"].encode()).decode()
    NAS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NAS_CONFIG_FILE, "w") as f:
        json.dump(safe, f, indent=2)


def mount_nas(config):
    mount_point = Path(config.get("mount_point") or "/tmp/cctv_nas_mount")
    mount_point.mkdir(parents=True, exist_ok=True)
    if os.path.ismount(str(mount_point)):
        return True, "Already mounted", mount_point
    if sys.platform == "darwin":
        url = f"//{config['username']}:{config['password']}@{config['ip']}/{config['share']}"
        cmd = ["mount_smbfs", url, str(mount_point)]
    elif sys.platform.startswith("linux"):
        cmd = [
            "mount", "-t", "cifs",
            f"//{config['ip']}/{config['share']}",
            str(mount_point),
            "-o", f"username={config['username']},password={config['password']}",
        ]
    elif sys.platform == "win32":
        # On Windows use net use
        drive = config.get("mount_point") or "Z:"
        cmd = [
            "net", "use", drive,
            f"\\\\{config['ip']}\\{config['share']}",
            f"/user:{config['username']}", config["password"],
        ]
        mount_point = Path(drive + "\\")
    else:
        return False, f"Unsupported platform: {sys.platform}", None
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True, "Connected", mount_point
        return False, result.stderr.strip() or "Mount failed", None
    except subprocess.TimeoutExpired:
        return False, "Mount timed out", None
    except Exception as e:
        return False, str(e), None


def unmount_nas(mount_point):
    if not mount_point:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["net", "use", str(mount_point), "/delete", "/y"],
                           capture_output=True, timeout=10)
        else:
            subprocess.run(["umount", str(mount_point)], capture_output=True, timeout=10)
    except Exception:
        pass


@app.route("/api/nas/status")
def api_nas_status():
    result = {
        "mode": _active_mode,
        "connected": _active_mode == "nas",
        "mount_point": str(_nas_mount_point) if _nas_mount_point else None,
    }
    if _active_mode == "nas":
        config = load_nas_config()
        if config:
            result["ip"] = config.get("ip", "")
            result["share"] = config.get("share", "")
    return jsonify(result)


@app.route("/api/nas/config")
def api_nas_config():
    config = load_nas_config()
    if config:
        safe = dict(config)
        safe.pop("password", None)
        return jsonify(safe)
    return jsonify({})


@app.route("/api/nas/connect", methods=["POST"])
def api_nas_connect():
    data = request.json
    for field in ["ip", "share", "username", "password"]:
        if not data.get(field):
            return jsonify({"error": f"{field} is required"}), 400
    config = {
        "ip": data["ip"],
        "share": data["share"],
        "username": data["username"],
        "password": data["password"],
        "mount_point": data.get("mount_point", "/tmp/cctv_nas_mount"),
    }
    save_nas_config(config)
    success, message, mount_point = mount_nas(config)
    if not success:
        return jsonify({"success": False, "message": message}), 500
    switch_to_nas(mount_point)
    videos = get_videos()
    return jsonify({
        "success": True,
        "message": f"Connected! Found {len(videos)} videos",
        "video_count": len(videos),
        "mount_point": str(mount_point),
    })


@app.route("/api/nas/disconnect", methods=["POST"])
def api_nas_disconnect():
    mount = _nas_mount_point
    switch_to_local()
    unmount_nas(mount)
    return jsonify({"success": True, "message": "Disconnected"})


# ---------------------------------------------------------------------------
# API — desktop helpers (open folders, app info)
# ---------------------------------------------------------------------------

@app.route("/api/app-info")
def api_app_info():
    """Return data directory path and platform info."""
    return jsonify({
        "data_dir": str(_DATA_ROOT),
        "videos_dir": str(VIDEOS_DIR),
        "exports_dir": str(EXPORTS_DIR),
        "platform": sys.platform,
        "mode": _active_mode,
    })


@app.route("/api/open-folder", methods=["POST"])
def api_open_folder():
    """Open a data folder in the system file manager."""
    folder_type = request.json.get("folder")
    folder_map = {
        "data": _DATA_ROOT,
        "videos": VIDEOS_DIR,
        "tracks": TRACKS_DIR,
        "corrections": CORRECTIONS_DIR,
        "exports": EXPORTS_DIR,
    }
    path = folder_map.get(folder_type)
    if not path:
        return jsonify({"error": "Unknown folder type"}), 400

    path.mkdir(parents=True, exist_ok=True)

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"success": True, "path": str(path)})


# ---------------------------------------------------------------------------
# Auto-reconnect NAS on startup
# ---------------------------------------------------------------------------

def _check_nas_on_startup():
    config = load_nas_config()
    if config:
        mount_point = Path(config.get("mount_point") or "/tmp/cctv_nas_mount")
        if mount_point.exists() and os.path.ismount(str(mount_point)):
            switch_to_nas(mount_point)
            get_videos()
            print(f"  NAS auto-reconnected: {mount_point}")

_check_nas_on_startup()
