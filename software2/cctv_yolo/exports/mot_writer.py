"""MOT Challenge format writer.

One CSV row per detection:
``<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,-1,-1,-1``

Frames are 1-indexed (MOT convention); we add 1 to the 0-indexed frame
numbers stored in the tracks JSON.
"""
from __future__ import annotations

from pathlib import Path

from . import filter_tracks_by_roi


def write_mot_txt(
    output_path: Path,
    track_data: dict,
    roi_id: str | None = None,
) -> Path:
    """Write MOT Challenge gt.txt to ``output_path`` and return it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = filter_tracks_by_roi(
        track_data.get("tracks", []),
        track_data.get("rois", []),
        roi_id,
    )

    rows: list[str] = []
    for tr in tracks:
        tid = tr.get("track_id", 0)
        for fd in sorted(tr.get("frames", []), key=lambda f: f["frame"]):
            x1, y1, x2, y2 = fd["bbox"]
            bb_left = x1
            bb_top = y1
            bb_w = x2 - x1
            bb_h = y2 - y1
            conf = fd.get("conf", 1.0) or 1.0
            # MOT frames are 1-indexed; tracks JSON uses 0-indexed frames.
            rows.append(
                f"{fd['frame'] + 1},{tid},{bb_left:.2f},{bb_top:.2f},"
                f"{bb_w:.2f},{bb_h:.2f},{conf:.4f},-1,-1,-1"
            )

    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return output_path
