#!/usr/bin/env python3
"""
Flask web application for video track correction UI.

Usage:
    python ui/app.py
    Then open: http://localhost:5000
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Base directories (relative to project root)
BASE_DIR = Path(__file__).parent.parent
VIDEOS_DIR = BASE_DIR / 'data' / 'videos'
TRACKS_DIR = BASE_DIR / 'data' / 'tracks'
ANNOTATIONS_DIR = BASE_DIR / 'data' / 'annotations'

# Ensure directories exist
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


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
        
        sessions.append({
            'id': video_name,
            'video_name': video_file.name,
            'video_path': str(video_file.relative_to(BASE_DIR)),
            'has_tracks': track_file is not None,
            'track_path': str(track_file.relative_to(BASE_DIR)) if track_file else None,
            'has_annotations': annotation_file is not None,
            'annotation_path': annotation_file
        })
    
    return sessions


@app.route('/')
def index():
    """Main page - list all sessions."""
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


@app.route('/api/sessions')
def api_sessions():
    """API endpoint to get all sessions."""
    return jsonify(find_sessions())


@app.route('/api/session/<session_id>/tracks')
def api_get_tracks(session_id):
    """Get tracks for a session."""
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
    """Save corrected tracks."""
    try:
        data = request.json
        
        # Save to annotations directory
        annotation_file = ANNOTATIONS_DIR / f"{session_id}.json"
        
        with open(annotation_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify({'success': True, 'message': 'Annotations saved'})
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
        
        # Load current tracks
        track_file = TRACKS_DIR / f"{session_id}.json"
        if not track_file.exists():
            return jsonify({'error': 'Track file not found'}), 404
        
        with open(track_file, 'r') as f:
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
            # Merge track_id into target_track_id
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
            
            # Merge frames
            target_frames = target_track.get('frames', [])
            merge_frames = track_to_merge.get('frames', [])
            target_frames.extend(merge_frames)
            target_frames.sort(key=lambda x: x.get('frame', 0))
            
            target_track['frames'] = target_frames
            # Update start/end frame
            if target_frames:
                target_track['start_frame'] = min(f.get('frame', 0) for f in target_frames)
                target_track['end_frame'] = max(f.get('frame', 0) for f in target_frames)
            
            # Remove merged track
            tracks = [t for t in tracks if t.get('track_id') != track_id]
        
        elif action == 'split':
            # Split track at frame
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
                        return jsonify({'error': 'Cannot split: no frames before or after split point'}), 400
                    
                    # Update original track
                    track['frames'] = before
                    track['end_frame'] = max(f.get('frame', 0) for f in before) if before else track.get('end_frame', 0)
                    
                    # Create new track
                    new_track_id = max([t.get('track_id', 0) for t in tracks], default=0) + 1
                    new_track = track.copy()
                    new_track['track_id'] = new_track_id
                    new_track['frames'] = after
                    new_track['start_frame'] = min(f.get('frame', 0) for f in after) if after else split_frame
                    new_track['end_frame'] = max(f.get('frame', 0) for f in after) if after else track.get('end_frame', 0)
                    tracks.append(new_track)
                    break
        
        # Update tracks data
        tracks_data['tracks'] = tracks
        
        # Save back to tracks file (temporary, until user saves annotations)
        with open(track_file, 'w') as f:
            json.dump(tracks_data, f, indent=2)
        
        return jsonify({'success': True, 'tracks': tracks_data})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve files from data directory."""
    # Handle paths that may or may not include 'data/' prefix
    if filename.startswith('data/'):
        filename = filename[5:]  # Remove 'data/' prefix
    return send_from_directory(BASE_DIR / 'data', filename)


if __name__ == '__main__':
    print("=" * 60)
    print("Correction UI Server")
    print("=" * 60)
    print(f"Videos directory: {VIDEOS_DIR}")
    print(f"Tracks directory: {TRACKS_DIR}")
    print(f"Annotations directory: {ANNOTATIONS_DIR}")
    print()
    print("Starting server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

