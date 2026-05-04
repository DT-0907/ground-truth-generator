"""
Auto pixels-per-meter calibration.

Two strategies, layered:

1. **Track-velocity heuristic** (fast, no scene assumptions):
   For tracks long enough to have a stable velocity, assume the median
   inter-frame displacement of car tracks is roughly the average car
   speed (~13.4 m/s for 30 mph). Solve for ppm such that the median
   pixel speed maps to that.

2. **Vanishing-point geometry** (more accurate when the road is
   visible): Hough-line detection -> RANSAC vanishing point -> use the
   median bounding-box height of cars as a reference for "1.5 m
   sedan height" and convert to ppm.

Both produce a number; we report both so the reviewer can pick.
"""
from __future__ import annotations
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# Defaults tuned to "average North American urban scene"
DEFAULT_CAR_SPEED_MPS = 13.4   # ~30 mph
DEFAULT_CAR_HEIGHT_M = 1.5


def _bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def calibrate_from_tracks(
    track_data: dict,
    assumed_speed_mps: float = DEFAULT_CAR_SPEED_MPS,
    min_track_frames: int = 12,
) -> Optional[dict]:
    """Estimate pixels-per-meter from track velocities.

    Picks tracks tagged as ``car`` with at least *min_track_frames*
    frames, computes the median pixel speed (px/sec), and divides by
    *assumed_speed_mps*.
    """
    fps = float(track_data.get("fps", 30.0)) or 30.0
    px_per_sec_samples: list[float] = []

    for tr in track_data.get("tracks", []):
        if tr.get("class") != "car":
            continue
        frames = sorted(tr.get("frames", []), key=lambda f: f["frame"])
        if len(frames) < min_track_frames:
            continue
        speeds = []
        for prev, cur in zip(frames, frames[1:]):
            df = cur["frame"] - prev["frame"]
            if df <= 0:
                continue
            pcx, pcy = _bbox_center(prev["bbox"])
            ccx, ccy = _bbox_center(cur["bbox"])
            speeds.append(math.hypot(ccx - pcx, ccy - pcy) / (df / fps))
        if not speeds:
            continue
        speeds.sort()
        px_per_sec_samples.append(speeds[len(speeds) // 2])  # median

    if not px_per_sec_samples:
        return None

    px_speed = statistics.median(px_per_sec_samples)
    if assumed_speed_mps <= 0:
        return None
    ppm = px_speed / assumed_speed_mps
    return {
        "method": "tracks",
        "pixels_per_meter": round(ppm, 2),
        "n_tracks": len(px_per_sec_samples),
        "median_px_per_sec": round(px_speed, 2),
        "assumed_speed_mps": assumed_speed_mps,
    }


def _vanishing_point(lines: np.ndarray, image_shape) -> Optional[tuple[float, float]]:
    """Naive RANSAC: pick pairs of lines, intersect, vote."""
    if lines is None or len(lines) < 4:
        return None
    h, w = image_shape[:2]
    rng = np.random.default_rng(42)

    best = None
    best_count = 0
    n = min(60, len(lines))
    indices = rng.choice(len(lines), size=n, replace=False)
    sample = lines[indices].reshape(-1, 4)

    intersections = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            x1, y1, x2, y2 = sample[i]
            x3, y3, x4, y4 = sample[j]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            # Keep only intersections roughly within the image
            if -w <= px <= 2 * w and -h <= py <= 2 * h:
                intersections.append((px, py))

    if not intersections:
        return None

    pts = np.asarray(intersections)
    # Vote with a tolerance grid
    grid = 25
    counts: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for x, y in pts:
        key = (int(x // grid), int(y // grid))
        counts[key].append((x, y))
    best_key = max(counts.items(), key=lambda kv: len(kv[1]))
    cell_pts = best_key[1]
    if len(cell_pts) < 3:
        return None
    cx = float(np.mean([p[0] for p in cell_pts]))
    cy = float(np.mean([p[1] for p in cell_pts]))
    return (cx, cy)


def calibrate_from_scene(
    video_path: Path,
    track_data: Optional[dict] = None,
    car_height_m: float = DEFAULT_CAR_HEIGHT_M,
) -> Optional[dict]:
    """Estimate ppm from scene geometry: detect lane/road lines via
    Hough on a sampled frame, find the vanishing point, then use the
    median bbox height of cars as a 1.5 m reference."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, total // 3))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=int(min(w, h) * 0.15), maxLineGap=20,
    )
    vp = _vanishing_point(lines, frame.shape) if lines is not None else None

    if track_data is None:
        return {"method": "scene", "vanishing_point": vp,
                "pixels_per_meter": None,
                "note": "Need track_data to estimate ppm."}

    car_heights = []
    for tr in track_data.get("tracks", []):
        if tr.get("class") != "car":
            continue
        for fd in tr.get("frames", []):
            if fd.get("interpolated"):
                continue
            x1, y1, x2, y2 = fd["bbox"]
            car_heights.append(y2 - y1)
    if not car_heights:
        return {"method": "scene", "vanishing_point": vp,
                "pixels_per_meter": None,
                "note": "No car bboxes found."}
    median_h_px = statistics.median(car_heights)
    ppm = median_h_px / car_height_m

    return {
        "method": "scene",
        "vanishing_point": vp,
        "pixels_per_meter": round(ppm, 2),
        "median_car_height_px": round(median_h_px, 1),
        "assumed_car_height_m": car_height_m,
    }


def auto_calibrate(
    video_path: Path,
    track_data: dict,
) -> dict:
    """Run both heuristics and return both results — UI lets the
    reviewer pick the one that matches their scene."""
    by_track = calibrate_from_tracks(track_data)
    by_scene = calibrate_from_scene(video_path, track_data)
    return {"by_track": by_track, "by_scene": by_scene}
