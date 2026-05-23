"""CVAT for-video XML 1.1 writer.

Emits an XML doc CVAT can import directly. One <track> per track_id with
a <box> per frame (interpolated frames marked keyframe="0", originals
"1"). Bbox coords are video pixels.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from . import filter_tracks_by_roi


def write_cvat_xml(
    output_path: Path,
    track_data: dict,
    video_name: str = "",
    width: int = 0,
    height: int = 0,
    roi_id: str | None = None,
) -> Path:
    """Write CVAT 1.1 XML to ``output_path`` and return it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracks = filter_tracks_by_roi(
        track_data.get("tracks", []),
        track_data.get("rois", []),
        roi_id,
    )

    labels = sorted({t.get("class", "vehicle") for t in tracks})

    parts: list[str] = []
    parts.append('<?xml version="1.0" encoding="utf-8"?>')
    parts.append("<annotations>")
    parts.append("  <version>1.1</version>")
    parts.append("  <meta>")
    parts.append("    <task>")
    parts.append(f"      <name>{escape(video_name or output_path.stem)}</name>")
    parts.append(f"      <size>{int(track_data.get('total_frames', 0))}</size>")
    parts.append("      <mode>interpolation</mode>")
    parts.append("      <overlap>0</overlap>")
    parts.append(f"      <created>{datetime.now().isoformat()}</created>")
    parts.append("      <labels>")
    for label in labels:
        parts.append("        <label>")
        parts.append(f"          <name>{escape(label)}</name>")
        parts.append("          <attributes></attributes>")
        parts.append("        </label>")
    parts.append("      </labels>")
    parts.append("    </task>")
    parts.append(
        f"      <original_size><width>{int(width)}</width>"
        f"<height>{int(height)}</height></original_size>"
    )
    parts.append("  </meta>")

    for idx, tr in enumerate(tracks):
        label = escape(tr.get("class", "vehicle"))
        tid = tr.get("track_id", idx)
        parts.append(f'  <track id="{idx}" label="{label}" source="manual">')
        frames = sorted(tr.get("frames", []), key=lambda f: f["frame"])
        last_frame = frames[-1]["frame"] if frames else 0
        for fd in frames:
            x1, y1, x2, y2 = fd["bbox"]
            keyframe = 0 if fd.get("interpolated") else 1
            occluded = 1 if fd.get("occluded") else 0
            outside = 0
            parts.append(
                f'    <box frame="{fd["frame"]}" outside="{outside}" '
                f'occluded="{occluded}" keyframe="{keyframe}" '
                f'xtl="{x1:.2f}" ytl="{y1:.2f}" '
                f'xbr="{x2:.2f}" ybr="{y2:.2f}" '
                f'z_order="0"></box>'
            )
        # Sentinel "outside" frame after the last keyframe — required by CVAT
        # so the track ends cleanly.
        if frames:
            x1, y1, x2, y2 = frames[-1]["bbox"]
            parts.append(
                f'    <box frame="{last_frame + 1}" outside="1" '
                f'occluded="0" keyframe="1" '
                f'xtl="{x1:.2f}" ytl="{y1:.2f}" '
                f'xbr="{x2:.2f}" ybr="{y2:.2f}" '
                f'z_order="0"></box>'
            )
        parts.append("  </track>")

    parts.append("</annotations>")

    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
