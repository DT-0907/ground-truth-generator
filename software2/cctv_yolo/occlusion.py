"""
Occlusion / track-gap detection — find pairs of tracks that are likely
the same vehicle separated by a temporary occlusion (sign, pillar,
overlap with another vehicle).

The reviewer gets a ranked list of "merge candidate" pairs with a
confidence score, predicted gap location, and a one-click action that
flows through the existing merge code path so gap interpolation runs.

Each interpolated frame produced by an occlusion-merge gets the
``"occluded": True`` attribute so it shows up distinctly in the
timeline minimap and the canvas.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable


@dataclass
class GapCandidate:
    """A suggested merge between two tracks across an occlusion."""

    track_a: int            # the earlier track ("dies" first)
    track_b: int            # the later track ("appears" after)
    gap_frames: int
    spatial_distance_px: float
    velocity_distance_px: float  # how far off the predicted continuation is
    score: float            # 0 (not similar) ... 1 (very likely same vehicle)
    same_class: bool
    a_end: tuple[int, float, float]  # frame, cx, cy
    b_start: tuple[int, float, float]


def _bbox_center(bbox) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_size(bbox) -> tuple[float, float]:
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aarea = (ax2 - ax1) * (ay2 - ay1)
    barea = (bx2 - bx1) * (by2 - by1)
    union = aarea + barea - inter
    return inter / union if union > 0 else 0.0


def _track_endpoint_velocity(track: dict, n: int = 5) -> tuple[float, float]:
    """Average per-frame velocity over the last *n* frames of a track.
    Returns (vx, vy) in pixels per frame. Returns (0, 0) for very
    short tracks.
    """
    frames = sorted(track.get("frames", []), key=lambda f: f["frame"])
    if len(frames) < 2:
        return 0.0, 0.0
    tail = frames[-min(n, len(frames)):]
    vx, vy = [], []
    for prev, cur in zip(tail, tail[1:]):
        df = cur["frame"] - prev["frame"]
        if df <= 0:
            continue
        pcx, pcy = _bbox_center(prev["bbox"])
        ccx, ccy = _bbox_center(cur["bbox"])
        vx.append((ccx - pcx) / df)
        vy.append((ccy - pcy) / df)
    if not vx:
        return 0.0, 0.0
    return sum(vx) / len(vx), sum(vy) / len(vy)


def _track_start_velocity(track: dict, n: int = 5) -> tuple[float, float]:
    frames = sorted(track.get("frames", []), key=lambda f: f["frame"])
    if len(frames) < 2:
        return 0.0, 0.0
    head = frames[: min(n, len(frames))]
    vx, vy = [], []
    for prev, cur in zip(head, head[1:]):
        df = cur["frame"] - prev["frame"]
        if df <= 0:
            continue
        pcx, pcy = _bbox_center(prev["bbox"])
        ccx, ccy = _bbox_center(cur["bbox"])
        vx.append((ccx - pcx) / df)
        vy.append((ccy - pcy) / df)
    if not vx:
        return 0.0, 0.0
    return sum(vx) / len(vx), sum(vy) / len(vy)


def find_gap_candidates(
    track_data: dict,
    max_gap_frames: int = 90,
    max_predicted_offset: float = 120.0,
    min_score: float = 0.35,
    require_same_class: bool = True,
) -> list[GapCandidate]:
    """Score every (A.end → B.start) pair within *max_gap_frames* and
    return those that look like the same vehicle.

    Score combines:
      - velocity continuation (predict where A would have been at B.start
        and measure distance to B's actual position)
      - bbox size similarity (vehicles don't change size much)
      - same class (unless ``require_same_class`` is False)
      - small temporal gap (closer is better)
    """
    tracks = track_data.get("tracks", []) or []
    if len(tracks) < 2:
        return []

    # Pre-compute per-track endpoints
    info = []
    for t in tracks:
        frames = sorted(t.get("frames", []), key=lambda f: f["frame"])
        if not frames:
            continue
        first, last = frames[0], frames[-1]
        info.append({
            "id": t.get("track_id"),
            "class": t.get("class"),
            "first_frame": first["frame"],
            "last_frame": last["frame"],
            "first_bbox": first["bbox"],
            "last_bbox": last["bbox"],
            "first_center": _bbox_center(first["bbox"]),
            "last_center": _bbox_center(last["bbox"]),
            "first_size": _bbox_size(first["bbox"]),
            "last_size": _bbox_size(last["bbox"]),
            "vel_end": _track_endpoint_velocity(t),
            "vel_start": _track_start_velocity(t),
            "n_frames": len(frames),
        })

    candidates: list[GapCandidate] = []
    seen_pairs: set[tuple[int, int]] = set()

    for a in info:
        for b in info:
            if a["id"] == b["id"]:
                continue
            gap = b["first_frame"] - a["last_frame"]
            if gap <= 0 or gap > max_gap_frames:
                continue

            if require_same_class and a["class"] != b["class"]:
                continue

            # Predict where A would be at B's first frame
            vx, vy = a["vel_end"]
            ax, ay = a["last_center"]
            pred_x = ax + vx * gap
            pred_y = ay + vy * gap
            bx, by = b["first_center"]
            offset = math.hypot(pred_x - bx, pred_y - by)
            if offset > max_predicted_offset:
                continue

            # Spatial dist (raw — even without velocity)
            spatial = math.hypot(ax - bx, ay - by)

            # Bbox size similarity
            aw, ah = a["last_size"]
            bw, bh = b["first_size"]
            if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
                size_sim = 0.0
            else:
                size_sim = min(aw, bw) / max(aw, bw) * min(ah, bh) / max(ah, bh)

            # Velocity continuity bonus — does B keep moving in A's direction?
            bvx, bvy = b["vel_start"]
            vmag_a = math.hypot(vx, vy) + 1e-6
            vmag_b = math.hypot(bvx, bvy) + 1e-6
            cos_sim = (vx * bvx + vy * bvy) / (vmag_a * vmag_b)
            cos_sim = max(0.0, cos_sim)  # 0..1

            # Composite score, all sub-scores in [0, 1]
            offset_score = max(0.0, 1.0 - offset / max_predicted_offset)
            time_score = max(0.0, 1.0 - gap / max_gap_frames)
            class_score = 1.0 if a["class"] == b["class"] else 0.4
            score = (
                0.45 * offset_score
                + 0.20 * time_score
                + 0.20 * size_sim
                + 0.10 * cos_sim
                + 0.05 * class_score
            )

            if score < min_score:
                continue

            # Deduplicate symmetric pairs (we only want A→B forward in time)
            key = (a["id"], b["id"])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            candidates.append(GapCandidate(
                track_a=a["id"],
                track_b=b["id"],
                gap_frames=gap,
                spatial_distance_px=spatial,
                velocity_distance_px=offset,
                score=round(score, 3),
                same_class=a["class"] == b["class"],
                a_end=(a["last_frame"], ax, ay),
                b_start=(b["first_frame"], bx, by),
            ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def mark_track_uncertain_segments(
    track: dict,
    min_visible_run: int = 3,
    max_gap_within: int = 5,
) -> int:
    """Annotate frames within a single track that look like internal
    occlusions: a frame whose neighbors-within-3-frames are missing in
    the per-frame index even though the track is alive.

    Adds ``"occluded": True`` to interpolated frames inside such gaps
    and returns the count of frames flagged.
    """
    frames = sorted(track.get("frames", []), key=lambda f: f["frame"])
    if len(frames) < 2:
        return 0
    flagged = 0
    out: list[dict] = []
    for i, fd in enumerate(frames):
        out.append(fd)
        if i + 1 >= len(frames):
            continue
        next_fd = frames[i + 1]
        gap = next_fd["frame"] - fd["frame"]
        if 1 < gap <= max_gap_within:
            # Linearly interpolate and mark each as occluded
            x1a, y1a, x2a, y2a = fd["bbox"]
            x1b, y1b, x2b, y2b = next_fd["bbox"]
            for k in range(1, gap):
                t = k / gap
                interp = [
                    x1a + (x1b - x1a) * t,
                    y1a + (y1b - y1a) * t,
                    x2a + (x2b - x2a) * t,
                    y2a + (y2b - y2a) * t,
                ]
                out.append({
                    "frame": fd["frame"] + k,
                    "bbox": [round(c, 1) for c in interp],
                    "conf": 0.0,
                    "interpolated": True,
                    "occluded": True,
                })
                flagged += 1
    track["frames"] = sorted(out, key=lambda f: f["frame"])
    return flagged
