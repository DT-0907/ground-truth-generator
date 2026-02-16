"""
Video processor: Detection + Tracking pipeline using Ultralytics.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO


# COCO vehicle class IDs
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    1: 'bicycle'
}


def process_video(video_path: str, output_dir: str = "data/tracks",
                  model_name: str = "yolov8m.pt", conf_threshold: float = 0.25,
                  feedback_file: str = None, session_id: str = None) -> dict:
    """
    Process a video file: detect vehicles and track them across frames.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save track results
        model_name: YOLO model to use
        conf_threshold: Minimum confidence threshold
        feedback_file: Optional path to feedback file for confidence adjustment
        session_id: Optional session_id for output filename (defaults to video stem)

    Returns:
        dict with tracks data
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load feedback for confidence adjustment if available
    confidence_adjustments = {}
    if feedback_file and Path(feedback_file).exists():
        with open(feedback_file, 'r') as f:
            feedback = json.load(f)
            confidence_adjustments = feedback.get('confidence_adjustments', {})
        print(f"Loaded confidence adjustments from feedback")

    # Load YOLO model
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

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

    for frame_idx, result in enumerate(tqdm(results, total=total_frames, desc="Processing")):
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
        'resolution': [width, height],
        'processed_at': datetime.now().isoformat(),
        'model': model_name,
        'conf_threshold': conf_threshold,
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
