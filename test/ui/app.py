#!/usr/bin/env python3
"""
Flask web application for video track correction UI.

Usage:
    python ui/app.py
    Then open: http://localhost:5000
"""

import json
import os
import sys
import threading
import io
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import cv2

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
CORS(app)

# Base directories (relative to project root)
BASE_DIR = Path(__file__).parent.parent
VIDEOS_DIR = BASE_DIR / 'data' / 'videos'
TRACKS_DIR = BASE_DIR / 'data' / 'tracks'
ANNOTATIONS_DIR = BASE_DIR / 'data' / 'annotations'
MODELS_DIR = BASE_DIR / 'models'

# Ensure directories exist
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
TRACKS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Processing state: session_id -> {status, progress, error}
_processing_state = {}
_processing_lock = threading.Lock()


def find_sessions():
    """Find all video sessions (videos with corresponding track files)."""
    sessions = []

    if not VIDEOS_DIR.exists():
        return sessions

    # Get all video files
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(VIDEOS_DIR.glob(f'*{ext}'))
        video_files.extend(VIDEOS_DIR.glob(f'*{ext.upper()}'))

    # Find matching track files
    if TRACKS_DIR.exists():
        track_files = list(TRACKS_DIR.glob('*.json'))
        track_dict = {f.stem: f for f in track_files}
    else:
        track_dict = {}

    for video_file in sorted(video_files):
        video_name = video_file.stem
        track_file = track_dict.get(video_name)

        # Check if annotation exists
        annotation_file = None
        if ANNOTATIONS_DIR.exists():
            ann_file = ANNOTATIONS_DIR / f"{video_name}.json"
            if ann_file.exists():
                annotation_file = str(ann_file.relative_to(BASE_DIR))

        # Check processing state
        proc_status = 'idle'
        proc_progress = 0
        with _processing_lock:
            if video_name in _processing_state:
                proc_status = _processing_state[video_name].get('status', 'idle')
                proc_progress = _processing_state[video_name].get('progress', 0)

        # Count tracks if available
        track_count = 0
        needs_review = 0
        if track_file:
            try:
                with open(track_file, 'r') as f:
                    td = json.load(f)
                track_count = len(td.get('tracks', []))
                needs_review = sum(1 for t in td.get('tracks', []) if t.get('needs_review'))
            except Exception:
                pass

        sessions.append({
            'id': video_name,
            'video_name': video_file.name,
            'video_path': str(video_file.relative_to(BASE_DIR)),
            'has_tracks': track_file is not None,
            'track_path': str(track_file.relative_to(BASE_DIR)) if track_file else None,
            'has_annotations': annotation_file is not None,
            'annotation_path': annotation_file,
            'processing_status': proc_status,
            'processing_progress': proc_progress,
            'track_count': track_count,
            'needs_review': needs_review
        })

    return sessions


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Main page - 3-tab layout."""
    sessions = find_sessions()
    return render_template('index.html', sessions=sessions)


@app.route('/review/<session_id>')
def review(session_id):
    """Review page for a specific session."""
    sessions = find_sessions()
    session = next((s for s in sessions if s['id'] == session_id), None)

    if not session:
        return "Session not found", 404

    return render_template('review.html', session=session)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route('/api/sessions')
def api_sessions():
    """API endpoint to get all sessions."""
    return jsonify(find_sessions())


@app.route('/api/models')
def api_models():
    """List available YOLO models (builtins + local .pt files)."""
    local_models = sorted([f.name for f in MODELS_DIR.glob('*.pt')]) if MODELS_DIR.exists() else []
    builtins = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    # Deduplicate while preserving order
    combined = list(dict.fromkeys(builtins + local_models))
    return jsonify({'models': combined})


@app.route('/api/session/<session_id>/tracks')
def api_get_tracks(session_id):
    """Get tracks for a session. Loads annotations first, falls back to tracks."""
    # Check annotations first (user corrections)
    ann_file = ANNOTATIONS_DIR / f"{session_id}.json"
    if ann_file.exists():
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        except Exception:
            pass

    # Fall back to original tracks
    track_file = TRACKS_DIR / f"{session_id}.json"
    if not track_file.exists():
        return jsonify({'error': 'Track file not found'}), 404

    try:
        with open(track_file, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/tracks', methods=['POST'])
def api_save_tracks(session_id):
    """Save corrected tracks (including individual bbox edits)."""
    try:
        data = request.json

        # Save to annotations directory
        annotation_file = ANNOTATIONS_DIR / f"{session_id}.json"

        with open(annotation_file, 'w') as f:
            json.dump(data, f, indent=2)

        return jsonify({'success': True, 'message': 'Annotations saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/bbox', methods=['POST'])
def api_update_bbox(session_id):
    """Update a single bbox for a specific track/frame."""
    try:
        data = request.json
        track_id = data.get('track_id')
        frame_num = data.get('frame')
        bbox = data.get('bbox')

        if track_id is None or frame_num is None or bbox is None:
            return jsonify({'error': 'track_id, frame, and bbox required'}), 400

        # Load current data (annotations first, then tracks)
        ann_file = ANNOTATIONS_DIR / f"{session_id}.json"
        track_file = TRACKS_DIR / f"{session_id}.json"

        if ann_file.exists():
            src = ann_file
        elif track_file.exists():
            src = track_file
        else:
            return jsonify({'error': 'No track data found'}), 404

        with open(src, 'r') as f:
            tracks_data = json.load(f)

        # Find and update the specific bbox
        found = False
        for track in tracks_data.get('tracks', []):
            if track.get('track_id') == track_id:
                for fr in track.get('frames', []):
                    if fr.get('frame') == frame_num:
                        fr['bbox'] = bbox
                        found = True
                        break
                break

        if not found:
            return jsonify({'error': 'Track/frame not found'}), 404

        # Save to annotations
        with open(ann_file, 'w') as f:
            json.dump(tracks_data, f, indent=2)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/action', methods=['POST'])
def api_track_action(session_id):
    """Perform track actions: delete, merge, split, change_class."""
    try:
        if not request.json:
            return jsonify({'error': 'JSON data required'}), 400

        action = request.json.get('action')
        track_id = request.json.get('track_id')
        data = request.json.get('data', {})

        if not action:
            return jsonify({'error': 'Action required'}), 400
        if track_id is None:
            return jsonify({'error': 'Track ID required'}), 400

        # Load current tracks (annotations first)
        ann_file = ANNOTATIONS_DIR / f"{session_id}.json"
        track_file = TRACKS_DIR / f"{session_id}.json"

        if ann_file.exists():
            src = ann_file
        elif track_file.exists():
            src = track_file
        else:
            return jsonify({'error': 'Track file not found'}), 404

        with open(src, 'r') as f:
            tracks_data = json.load(f)

        tracks = tracks_data.get('tracks', [])

        if action == 'delete':
            original_count = len(tracks)
            tracks = [t for t in tracks if t.get('track_id') != track_id]
            if len(tracks) == original_count:
                return jsonify({'error': f'Track {track_id} not found'}), 404

        elif action == 'change_class':
            new_class = data.get('new_class')
            if not new_class:
                return jsonify({'error': 'New class name required'}), 400

            found = False
            for track in tracks:
                if track.get('track_id') == track_id:
                    track['class'] = new_class
                    found = True
                    break

            if not found:
                return jsonify({'error': f'Track {track_id} not found'}), 404

        elif action == 'merge':
            target_id = data.get('target_track_id')
            if target_id is None:
                return jsonify({'error': 'Target track ID required for merge'}), 400

            if track_id == target_id:
                return jsonify({'error': 'Cannot merge track with itself'}), 400

            track_to_merge = None
            target_track = None

            for track in tracks:
                if track.get('track_id') == track_id:
                    track_to_merge = track
                elif track.get('track_id') == target_id:
                    target_track = track

            if not track_to_merge:
                return jsonify({'error': f'Track {track_id} not found'}), 404
            if not target_track:
                return jsonify({'error': f'Target track {target_id} not found'}), 404

            # Merge frames, deduplicate by frame number (keep higher confidence)
            existing = {f.get('frame'): f for f in target_track.get('frames', [])}
            for fr in track_to_merge.get('frames', []):
                fn = fr.get('frame')
                if fn in existing:
                    if fr.get('conf', 0) > existing[fn].get('conf', 0):
                        existing[fn] = fr
                else:
                    existing[fn] = fr

            target_track['frames'] = sorted(existing.values(), key=lambda x: x.get('frame', 0))
            if target_track['frames']:
                target_track['start_frame'] = min(f.get('frame', 0) for f in target_track['frames'])
                target_track['end_frame'] = max(f.get('frame', 0) for f in target_track['frames'])

            tracks = [t for t in tracks if t.get('track_id') != track_id]

        elif action == 'split':
            split_frame = data.get('frame')
            if split_frame is None:
                return jsonify({'error': 'Frame number required for split'}), 400

            for track in tracks:
                if track.get('track_id') == track_id:
                    frames = track.get('frames', [])
                    if not frames:
                        return jsonify({'error': 'Track has no frames to split'}), 400

                    before = [f for f in frames if f.get('frame', 0) < split_frame]
                    after = [f for f in frames if f.get('frame', 0) >= split_frame]

                    if not before or not after:
                        return jsonify({'error': 'Cannot split: no frames on one side'}), 400

                    track['frames'] = before
                    track['end_frame'] = max(f.get('frame', 0) for f in before)

                    new_track_id = max([t.get('track_id', 0) for t in tracks], default=0) + 1
                    new_track = track.copy()
                    new_track['track_id'] = new_track_id
                    new_track['frames'] = after
                    new_track['start_frame'] = min(f.get('frame', 0) for f in after)
                    new_track['end_frame'] = max(f.get('frame', 0) for f in after)
                    tracks.append(new_track)
                    break

        tracks_data['tracks'] = tracks

        # Save to annotations dir (non-destructive)
        with open(ann_file, 'w') as f:
            json.dump(tracks_data, f, indent=2)

        return jsonify({'success': True, 'tracks': tracks_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Frame serving (for canvas-based playback)
# ---------------------------------------------------------------------------

@app.route('/frame/<session_id>/<int:frame_num>')
def get_frame(session_id, frame_num):
    """Extract and return a single frame as JPEG."""
    # Find video file
    video_file = None
    for ext in ['.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV']:
        candidate = VIDEOS_DIR / f"{session_id}{ext}"
        if candidate.exists():
            video_file = candidate
            break

    if not video_file:
        return "Video not found", 404

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return "Cannot open video", 500

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Frame not available", 404

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/video-info/<session_id>')
def api_video_info(session_id):
    """Return video metadata (fps, total frames, resolution)."""
    video_file = None
    for ext in ['.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV']:
        candidate = VIDEOS_DIR / f"{session_id}{ext}"
        if candidate.exists():
            video_file = candidate
            break

    if not video_file:
        return jsonify({'error': 'Video not found'}), 404

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open video'}), 500

    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    cap.release()
    return jsonify(info)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

@app.route('/api/process/<session_id>', methods=['POST'])
def api_process(session_id):
    """Start background processing for a session."""
    with _processing_lock:
        if session_id in _processing_state and _processing_state[session_id].get('status') == 'processing':
            return jsonify({'error': 'Already processing'}), 409

    # Find video
    video_file = None
    for ext in ['.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV']:
        candidate = VIDEOS_DIR / f"{session_id}{ext}"
        if candidate.exists():
            video_file = candidate
            break

    if not video_file:
        return jsonify({'error': 'Video not found'}), 404

    body = request.json or {}
    model_name = body.get('model', 'yolov8m.pt')
    conf_threshold = body.get('confidence', 0.25)
    processing_roi = body.get('roi', None)

    with _processing_lock:
        _processing_state[session_id] = {'status': 'processing', 'progress': 0, 'error': None}

    def run():
        try:
            # Add src to path so processor can be imported
            src_dir = str(BASE_DIR / 'src')
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            from processor import process_video

            def on_progress(pct):
                with _processing_lock:
                    _processing_state[session_id]['progress'] = pct

            process_video(
                video_path=str(video_file),
                output_dir=str(TRACKS_DIR),
                model_name=model_name,
                conf_threshold=conf_threshold,
                session_id=session_id,
                progress_callback=on_progress,
                models_dir=str(MODELS_DIR),
                processing_roi=processing_roi
            )
            with _processing_lock:
                _processing_state[session_id] = {'status': 'done', 'progress': 100, 'error': None}
        except Exception as e:
            with _processing_lock:
                _processing_state[session_id] = {'status': 'error', 'progress': 0, 'error': str(e)}

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return jsonify({'success': True, 'message': 'Processing started'})


@app.route('/api/process/status/<session_id>')
def api_process_status(session_id):
    """Return processing status for a session."""
    with _processing_lock:
        state = _processing_state.get(session_id, {'status': 'idle', 'progress': 0, 'error': None})
    return jsonify(state)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

@app.route('/api/performance/<session_id>')
def api_performance(session_id):
    """Return performance/counting data for a session."""
    # Load tracks (annotations first)
    ann_file = ANNOTATIONS_DIR / f"{session_id}.json"
    track_file = TRACKS_DIR / f"{session_id}.json"

    src = None
    if ann_file.exists():
        src = ann_file
    elif track_file.exists():
        src = track_file

    if not src:
        return jsonify({'error': 'No track data found'}), 404

    with open(src, 'r') as f:
        data = json.load(f)

    tracks = data.get('tracks', [])

    # Count by class
    by_class = {}
    for t in tracks:
        cls = t.get('class', 'unknown')
        by_class[cls] = by_class.get(cls, 0) + 1

    total = len(tracks)
    needs_review = sum(1 for t in tracks if t.get('needs_review'))

    # ROI counts (if processing_roi was used or if rois are stored)
    rois = data.get('rois', [])
    roi_counts = []
    for roi in rois:
        roi_name = roi.get('name', roi.get('id', 'ROI'))
        count = 0
        for t in tracks:
            for fr in t.get('frames', []):
                bbox = fr.get('bbox')
                if bbox and _bbox_in_roi(bbox, roi):
                    count += 1
                    break  # count track once per ROI
        roi_counts.append({'name': roi_name, 'count': count})

    return jsonify({
        'session_id': session_id,
        'total_tracks': total,
        'needs_review': needs_review,
        'by_class': by_class,
        'roi_counts': roi_counts,
        'fps': data.get('fps', 30),
        'total_frames': data.get('total_frames', 0),
        'model': data.get('model', 'unknown'),
        'processed_at': data.get('processed_at', '')
    })


def _bbox_in_roi(bbox, roi):
    """Check if bbox center is inside ROI."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    roi_type = roi.get('type', 'rect')
    pts = roi.get('points', [])
    if not pts:
        return False
    if roi_type == 'rect' and len(pts) >= 2:
        x1, y1 = pts[0].get('x', 0), pts[0].get('y', 0)
        x2, y2 = pts[1].get('x', 0), pts[1].get('y', 0)
        return min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2)
    else:
        poly = [(p.get('x', 0), p.get('y', 0)) for p in pts]
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > cy) != (yj > cy)) and (cx < (xj - xi) * (cy - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside


# ---------------------------------------------------------------------------
# Static data serving
# ---------------------------------------------------------------------------

@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve files from data directory."""
    if filename.startswith('data/'):
        filename = filename[5:]
    return send_from_directory(BASE_DIR / 'data', filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Correction UI Server")
    print("=" * 60)
    print(f"Videos directory: {VIDEOS_DIR}")
    print(f"Tracks directory: {TRACKS_DIR}")
    print(f"Annotations directory: {ANNOTATIONS_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print()
    print("Starting server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
