"""
Video processor: Detection + Tracking pipeline using Ultralytics.
"""

import json
import logging
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ProcessingError(RuntimeError):
    """User-friendly processing error. ``str(err)`` is safe to show in the UI."""
    pass


class BatchCancelled(Exception):
    """Raised by process_video when a should_cancel() callback returns True.

    Caught by the batch scheduler so partial outputs can be cleaned up and
    the item can be re-queued (or removed) cleanly.
    """
    pass


def _get_device():
    """Detect the best available device: CUDA GPU, Apple MPS, or CPU.

    Cross-platform behavior:
      - Windows w/ NVIDIA: CUDA (fastest)        — requires CUDA-enabled torch
                                                    (see build_windows.bat)
      - Apple Silicon Mac: MPS (Metal)           — built into PyTorch
      - Intel Mac:         CPU
      - Linux w/ NVIDIA:   CUDA
      - Anywhere else:     CPU
    """
    import logging
    _log = logging.getLogger(__name__)
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        msg = f"CUDA GPU detected: {name} ({count} device(s))"
        print(msg)
        _log.info(msg)
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        msg = "Apple MPS (Metal) detected"
        print(msg)
        _log.info(msg)
        return "mps"
    msg = "No GPU detected, using CPU"
    print(msg)
    _log.info(msg)
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
                  processing_roi: dict = None,
                  should_cancel=None,
                  progress_detail_callback=None,
                  sample_rate: int = 1) -> dict:
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
                    defaults to <data_root>/models/ (resolved via
                    cctv_yolo.paths.get_data_root() — same folder as the
                    app / repo so the install is portable).
        processing_roi: Optional ROI dict to filter detections. Only detections
                       whose bbox center falls inside the ROI are kept.
                       Format: {"type": "rect"|"polygon", "points": [...]}
        should_cancel: Optional callable returning True if the caller wants
                       processing to abort. Checked once per frame. When it
                       returns True, ``BatchCancelled`` is raised so the
                       scheduler can clean up partial output.
        progress_detail_callback: Optional callable(percent, fps, eta_seconds).
                       Called once per frame with throughput and ETA.
        sample_rate: Process every Nth frame (1 = all frames). Frames that are
                     skipped do not generate detections but the frame index is
                     preserved so downstream tools can still interpolate.

    Returns:
        dict with tracks data
    """
    sample_rate = max(1, int(sample_rate))
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
        with open(feedback_file, 'r', encoding="utf-8") as f:
            feedback = json.load(f)
            confidence_adjustments = feedback.get('confidence_adjustments', {})
        print(f"Loaded confidence adjustments from feedback")

    # Load YOLO model — check local models dir first to avoid downloading
    if models_dir:
        _models_dir = Path(models_dir)
    else:
        from cctv_yolo.paths import get_models_dir
        _models_dir = get_models_dir()
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
        logger.exception("Model load failed: %s", model_name)
        raise ProcessingError(
            "Model file is invalid or corrupt. Try re-importing."
        ) from e

    # Select best available device (CUDA > MPS > CPU). GPU is always
    # used if present — same logic on every platform.
    device = _get_device()
    model.to(device)
    import logging
    logging.getLogger(__name__).info(
        "Model loaded on device: %s (%s)", device,
        "CUDA NVIDIA GPU" if device.startswith("cuda")
        else "Apple Metal (MPS)" if device == "mps"
        else "CPU fallback"
    )
    print(f"Model loaded on device: {device}")

    # Open video
    try:
        cap = cv2.VideoCapture(str(video_path))
    except cv2.error as e:
        logger.exception("OpenCV error opening %s", video_path)
        raise ProcessingError(
            "Couldn't read this video. Try re-encoding to H.264 MP4."
        ) from e
    if not cap.isOpened():
        logger.error("cv2.VideoCapture.isOpened() returned False for %s", video_path)
        raise ProcessingError(
            "Couldn't read this video. Try re-encoding to H.264 MP4."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")

    cap.release()

    # Run tracking using Ultralytics built-in tracker.
    # Pass device= explicitly so Ultralytics doesn't fall back to CPU even
    # though model.to(device) was already called. (Some Ultralytics versions
    # ignore the model's current device on track() unless told.)
    print("Running detection + tracking...")
    try:
        results = model.track(
            source=str(video_path),
            conf=conf_threshold,
            classes=list(VEHICLE_CLASSES.keys()),
            tracker="bytetrack.yaml",
            stream=True,
            device=device,
            verbose=False,
            vid_stride=sample_rate,
        )
    except Exception as e:
        msg = str(e).lower()
        if "out of memory" in msg or "oom" in msg:
            logger.exception("OOM starting tracker")
            raise ProcessingError(
                "Out of memory. Try a smaller model (e.g. yolov8n) or "
                "reduce video resolution."
            ) from e
        logger.exception("Tracker start failed")
        raise

    # Collect tracks
    tracks_dict = {}  # track_id -> track data

    last_pct = 0
    start_time = time.time()
    processed_frames = 0
    try:
        for stream_idx, result in enumerate(results):
            # vid_stride means each yielded frame corresponds to
            # stream_idx * sample_rate in the source video.
            frame_idx = stream_idx * sample_rate
            processed_frames = stream_idx + 1
            # Cooperative cancel — checked once per frame before any heavy work.
            if should_cancel is not None and should_cancel():
                raise BatchCancelled(f"Cancelled at frame {frame_idx}/{total_frames}")

            # Report progress + detail
            if total_frames > 0:
                pct = int((frame_idx + 1) / total_frames * 100)
                if pct != last_pct:
                    if progress_callback:
                        progress_callback(pct)
                    if progress_detail_callback:
                        elapsed = max(time.time() - start_time, 1e-6)
                        cur_fps = processed_frames / elapsed
                        remaining_src = max(total_frames - frame_idx - 1, 0)
                        remaining_processed = remaining_src / max(sample_rate, 1)
                        eta = remaining_processed / cur_fps if cur_fps > 0 else 0.0
                        try:
                            progress_detail_callback(pct, cur_fps, eta)
                        except Exception:
                            logger.exception("progress_detail_callback failed")
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
    except BatchCancelled:
        raise
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        logger.exception("CUDA OOM during inference")
        raise ProcessingError(
            "Out of memory. Try a smaller model (e.g. yolov8n) or "
            "reduce video resolution."
        ) from e
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "oom" in msg:
            logger.exception("Runtime OOM during inference")
            raise ProcessingError(
                "Out of memory. Try a smaller model (e.g. yolov8n) or "
                "reduce video resolution."
            ) from e
        if "cannot open video" in msg or "could not open" in msg:
            logger.exception("Video read error during inference")
            raise ProcessingError(
                "Couldn't read this video. Try re-encoding to H.264 MP4."
            ) from e
        logger.exception("Unhandled runtime error during inference")
        raise
    except cv2.error as e:
        logger.exception("OpenCV error during inference")
        raise ProcessingError(
            "Couldn't read this video. Try re-encoding to H.264 MP4."
        ) from e

    # Post-process tracks
    tracks = list(tracks_dict.values())

    for track in tracks:
        confs = [f['conf'] for f in track['frames']]
        track['avg_confidence'] = round(sum(confs) / len(confs), 3) if confs else 0

        duration = track['end_frame'] - track['start_frame']
        if duration < 5 or track['avg_confidence'] < 0.4:
            track['needs_review'] = True

    tracks.sort(key=lambda t: t['start_frame'])

    processing_time = round(time.time() - start_time, 2)
    output_data = {
        'video_path': str(video_path),
        'video_name': video_path.name,
        'fps': fps,
        'total_frames': total_frames,
        'resolution': f"{width}x{height}",
        'processed_at': datetime.now().isoformat(),
        'model': model_name,
        'conf_threshold': conf_threshold,
        'sample_rate': sample_rate,
        'processing_time': processing_time,
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
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Total tracks: {len(tracks)}")
    print(f"Needs review: {output_data['stats']['needs_review']}")
    print(f"By class: {output_data['stats']['by_class']}")

    return output_data
