"""
Video processor: Detection + Tracking pipeline using Ultralytics.
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


def _get_device():
    """Detect the best available device: CUDA GPU, Apple MPS, or CPU."""
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {name} ({count} device(s))")
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple MPS (Metal) detected")
        return "mps"
    print("No GPU detected, using CPU")
    return "cpu"


# COCO vehicle class IDs
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    1: 'bicycle'
}


def _point_in_polygon(px, py, polygon):
    """Ray-casting point-in-polygon test."""
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


def _bbox_center_in_roi(bbox, roi):
    """Check if bbox center is inside ROI polygon/rect."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    if roi.get('type') == 'rect':
        pts = roi['points']
        x1, y1 = pts[0]['x'], pts[0]['y']
        x2, y2 = pts[1]['x'], pts[1]['y']
        return min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2)
    else:  # polygon
        poly = [(p['x'], p['y']) for p in roi['points']]
        return _point_in_polygon(cx, cy, poly)


def process_video(video_path: str, output_dir: str = "data/tracks",
                  model_name: str = "yolov8m.pt", conf_threshold: float = 0.25,
                  feedback_file: str = None, session_id: str = None,
                  progress_callback=None, models_dir: str = None,
                  processing_roi: dict = None) -> dict:
    """
    Process a video file: detect vehicles and track them across frames.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save track results
        model_name: YOLO model to use
        conf_threshold: Minimum confidence threshold
        feedback_file: Optional path to feedback file for confidence adjustment
        session_id: Optional session_id for output filename (defaults to video stem)
        progress_callback: Optional callable(percent: int) for progress updates
        models_dir: Optional path to local models directory. If not given,
                    defaults to ~/Documents/CCTV-YOLO/models/
        processing_roi: Optional ROI dict to filter detections. Only detections
                       whose bbox center falls inside the ROI are kept.
                       Format: {"type": "rect"|"polygon", "points": [...]}

    Returns:
        dict with tracks data
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not video_path.is_file():
        raise ValueError(f"Video path is not a file: {video_path}")

    # Load feedback for confidence adjustment if available
    confidence_adjustments = {}
    if feedback_file and Path(feedback_file).exists():
        with open(feedback_file, 'r') as f:
            feedback = json.load(f)
            confidence_adjustments = feedback.get('confidence_adjustments', {})
        print(f"Loaded confidence adjustments from feedback")

    # Load YOLO model — check local models dir first to avoid downloading
    if models_dir:
        _models_dir = Path(models_dir)
    else:
        _models_dir = Path.home() / "Documents" / "CCTV-YOLO" / "models"
    _models_dir.mkdir(parents=True, exist_ok=True)

    local_model = _models_dir / model_name
    try:
        if local_model.exists():
            print(f"Loading model from local: {local_model}")
            model = YOLO(str(local_model))
        else:
            print(f"Loading model: {model_name} (will download if not cached)")
            model = YOLO(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load YOLO model '{model_name}'. "
            f"Checked local path: {local_model}\n"
            f"If you have no internet connection, place the model file in:\n"
            f"  {_models_dir}\n\n"
            f"Original error: {e}"
        ) from e

    # Select best available device (GPU > MPS > CPU)
    device = _get_device()
    model.to(device)
    print(f"Model loaded on device: {device}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")

    cap.release()

    # Run tracking using Ultralytics built-in tracker
    print("Running detection + tracking...")
    results = model.track(
        source=str(video_path),
        conf=conf_threshold,
        classes=list(VEHICLE_CLASSES.keys()),
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False
    )

    # Collect tracks
    tracks_dict = {}  # track_id -> track data

    last_pct = 0
    for frame_idx, result in enumerate(results):
        # Report progress
        if progress_callback and total_frames > 0:
            pct = int((frame_idx + 1) / total_frames * 100)
            if pct != last_pct:
                progress_callback(pct)
                last_pct = pct

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes

        # Check if tracking IDs are available
        if boxes.id is None:
            continue

        for i in range(len(boxes)):
            track_id = int(boxes.id[i].item())
            bbox = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].item())
            class_id = int(boxes.cls[i].item())
            class_name = VEHICLE_CLASSES.get(class_id, 'unknown')

            # Filter by processing ROI if defined
            if processing_roi and not _bbox_center_in_roi(bbox, processing_roi):
                continue

            # Apply confidence adjustment from feedback
            adjustment_key = f"{class_name}"
            if adjustment_key in confidence_adjustments:
                adj = confidence_adjustments[adjustment_key]
                if conf < adj.get('flag_threshold', 1.0):
                    pass

            if track_id not in tracks_dict:
                tracks_dict[track_id] = {
                    'track_id': track_id,
                    'class': class_name,
                    'class_id': class_id,
                    'frames': [],
                    'start_frame': frame_idx,
                    'end_frame': frame_idx,
                    'needs_review': False,
                    'avg_confidence': 0.0
                }

            tracks_dict[track_id]['frames'].append({
                'frame': frame_idx,
                'bbox': [round(x, 1) for x in bbox],
                'conf': round(conf, 3)
            })
            tracks_dict[track_id]['end_frame'] = frame_idx

    # Post-process tracks
    tracks = list(tracks_dict.values())

    for track in tracks:
        confs = [f['conf'] for f in track['frames']]
        track['avg_confidence'] = round(sum(confs) / len(confs), 3) if confs else 0

        duration = track['end_frame'] - track['start_frame']
        if duration < 5 or track['avg_confidence'] < 0.4:
            track['needs_review'] = True

    tracks.sort(key=lambda t: t['start_frame'])

    output_data = {
        'video_path': str(video_path),
        'video_name': video_path.name,
        'fps': fps,
        'total_frames': total_frames,
        'resolution': f"{width}x{height}",
        'processed_at': datetime.now().isoformat(),
        'model': model_name,
        'conf_threshold': conf_threshold,
        'processing_roi': processing_roi,
        'tracks': tracks,
        'stats': {
            'total_tracks': len(tracks),
            'needs_review': sum(1 for t in tracks if t['needs_review']),
            'by_class': {}
        }
    }

    for track in tracks:
        cls = track['class']
        output_data['stats']['by_class'][cls] = output_data['stats']['by_class'].get(cls, 0) + 1

    output_name = session_id if session_id else video_path.stem
    output_file = output_dir / f"{output_name}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Total tracks: {len(tracks)}")
    print(f"Needs review: {output_data['stats']['needs_review']}")
    print(f"By class: {output_data['stats']['by_class']}")

    return output_data
