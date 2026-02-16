"""
Feedback system: Track corrections and export for fine-tuning.

Two outputs:
1. Confidence calibration file - adjusts thresholds based on correction patterns
2. COCO format export - for fine-tuning YOLO on corrected data
"""

import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def analyze_corrections(tracks_dir: str, corrections_dir: str, output_file: str = "config/feedback.json"):
    tracks_dir = Path(tracks_dir)
    corrections_dir = Path(corrections_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_sessions': 0,
        'total_original_tracks': 0,
        'total_corrected_tracks': 0,
        'deleted_tracks': 0,
        'class_changes': defaultdict(lambda: defaultdict(int)),
        'false_positive_by_class': defaultdict(int),
        'false_positive_by_confidence': defaultdict(int),
        'merged_tracks': 0,
        'split_tracks': 0
    }

    if not corrections_dir.exists():
        print("No corrections directory found")
        return

    correction_files = list(corrections_dir.glob("*.json"))
    if not correction_files:
        print("No correction files found")
        return

    for corr_file in correction_files:
        orig_file = tracks_dir / corr_file.name
        if not orig_file.exists():
            continue

        stats['total_sessions'] += 1

        with open(orig_file, 'r') as f:
            original = json.load(f)
        with open(corr_file, 'r') as f:
            corrected = json.load(f)

        orig_tracks = {t['track_id']: t for t in original.get('tracks', [])}
        corr_tracks = {t['track_id']: t for t in corrected.get('tracks', [])}

        stats['total_original_tracks'] += len(orig_tracks)
        stats['total_corrected_tracks'] += len(corr_tracks)

        for track_id, track in orig_tracks.items():
            if track_id not in corr_tracks:
                stats['deleted_tracks'] += 1
                stats['false_positive_by_class'][track['class']] += 1
                avg_conf = track.get('avg_confidence', 0.5)
                bucket = int(avg_conf * 10) / 10
                stats['false_positive_by_confidence'][f"{bucket:.1f}"] += 1

        for track_id, corr_track in corr_tracks.items():
            if track_id in orig_tracks:
                orig_track = orig_tracks[track_id]
                if orig_track['class'] != corr_track['class']:
                    stats['class_changes'][orig_track['class']][corr_track['class']] += 1

    confidence_adjustments = {}
    for cls, fp_count in stats['false_positive_by_class'].items():
        total_class = sum(1 for t in orig_tracks.values() if t.get('class') == cls)
        if total_class > 0:
            fp_rate = fp_count / total_class
            if fp_rate > 0.2:
                confidence_adjustments[cls] = {
                    'flag_threshold': 0.6,
                    'false_positive_rate': round(fp_rate, 3)
                }

    output = {
        'generated_at': datetime.now().isoformat(),
        'sessions_analyzed': stats['total_sessions'],
        'stats': {
            'total_original_tracks': stats['total_original_tracks'],
            'total_corrected_tracks': stats['total_corrected_tracks'],
            'deleted_tracks': stats['deleted_tracks'],
            'deletion_rate': round(stats['deleted_tracks'] / max(stats['total_original_tracks'], 1), 3),
            'false_positive_by_class': dict(stats['false_positive_by_class']),
            'false_positive_by_confidence': dict(stats['false_positive_by_confidence']),
            'class_changes': {k: dict(v) for k, v in stats['class_changes'].items()}
        },
        'confidence_adjustments': confidence_adjustments
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Feedback saved to: {output_file}")
    print(f"Sessions analyzed: {stats['total_sessions']}")
    print(f"Deletion rate: {output['stats']['deletion_rate']:.1%}")

    return output


def export_coco(tracks_dir: str, corrections_dir: str, video_dir: str,
                output_dir: str = "data/exports", sample_rate: int = 10):
    tracks_dir = Path(tracks_dir)
    corrections_dir = Path(corrections_dir)
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": 1, "name": "bicycle"},
        {"id": 2, "name": "car"},
        {"id": 3, "name": "motorcycle"},
        {"id": 5, "name": "bus"},
        {"id": 7, "name": "truck"}
    ]
    class_to_id = {c['name']: c['id'] for c in categories}

    coco_data = {
        "info": {
            "description": "Vehicle detection dataset from corrected tracks",
            "date_created": datetime.now().isoformat(),
            "version": "1.0"
        },
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    image_id = 0
    annotation_id = 0

    correction_files = list(corrections_dir.glob("*.json")) if corrections_dir.exists() else []
    if not correction_files:
        print("No correction files found.")
        return None

    for corr_file in correction_files:
        with open(corr_file, 'r') as f:
            data = json.load(f)

        video_name = data.get('video_name', corr_file.stem + '.mp4')
        video_path = video_dir / video_name

        if not video_path.exists():
            for ext in ['.mp4', '.mov', '.avi', '.mkv']:
                candidate = video_dir / (corr_file.stem + ext)
                if candidate.exists():
                    video_path = candidate
                    break

        if not video_path.exists():
            print(f"Video not found for {corr_file.name}, skipping")
            continue

        print(f"Processing: {video_path.name}")

        frame_detections = defaultdict(list)
        for track in data.get('tracks', []):
            class_name = track['class']
            if class_name not in class_to_id:
                continue
            for frame_data in track.get('frames', []):
                frame_num = frame_data['frame']
                if frame_num % sample_rate != 0:
                    continue
                frame_detections[frame_num].append({
                    'bbox': frame_data['bbox'],
                    'class_id': class_to_id[class_name]
                })

        if not frame_detections:
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for frame_num in sorted(frame_detections.keys()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            image_filename = f"{corr_file.stem}_frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(images_dir / image_filename), frame)

            coco_data['images'].append({
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height
            })

            for det in frame_detections[frame_num]:
                x1, y1, x2, y2 = det['bbox']
                w = x2 - x1
                h = y2 - y1
                coco_data['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": det['class_id'],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1

            image_id += 1

        cap.release()

    coco_file = output_dir / "annotations.json"
    with open(coco_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\nCOCO dataset exported to: {output_dir}")
    print(f"Images: {len(coco_data['images'])}")
    print(f"Annotations: {len(coco_data['annotations'])}")

    return coco_data
