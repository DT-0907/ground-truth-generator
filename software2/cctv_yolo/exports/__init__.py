"""Export writers for the Correction tab.

Per-format modules (CVAT, MOT, CSV, Review Pack) all share one helper:
``filter_tracks_by_roi(tracks, rois, roi_id)`` — keeps only tracks whose
bbox center falls in the named ROI on at least one frame. Mirrors
processor._bbox_center_in_roi so the ROI semantics stay consistent
between processing and export.
"""
from __future__ import annotations

from typing import Iterable


def _point_in_polygon(px: float, py: float, polygon: list) -> bool:
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


def _bbox_center_in_roi(bbox, roi) -> bool:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    if roi.get("type") == "rect":
        pts = roi["points"]
        x1, y1 = pts[0]["x"], pts[0]["y"]
        x2, y2 = pts[1]["x"], pts[1]["y"]
        return min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2)
    poly = roi.get("points", [])
    return _point_in_polygon(cx, cy, poly)


def find_roi(rois: list, roi_id: str | None):
    """Locate an ROI by id or name. Returns None if not found / no filter."""
    if not roi_id:
        return None
    for r in rois or []:
        if r.get("id") == roi_id or r.get("name") == roi_id:
            return r
    return None


def filter_tracks_by_roi(tracks: list, rois: list, roi_id: str | None) -> list:
    """Return tracks with at least one frame whose bbox center is in the named ROI.

    If ``roi_id`` is None or the ROI isn't found, the input list is returned
    unchanged (no filter).
    """
    roi = find_roi(rois, roi_id)
    if roi is None:
        return list(tracks)
    out = []
    for tr in tracks:
        for fd in tr.get("frames", []):
            if _bbox_center_in_roi(fd["bbox"], roi):
                out.append(tr)
                break
    return out
