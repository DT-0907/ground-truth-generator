"""Review Pack zip — bundles annotated MP4 + per-track CSV + summary PDF
+ README for handoff to a reviewer.

PDF generation is optional; if reportlab isn't installed, the pack still
ships the MP4, CSV, and README with a note about the missing PDF.
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from cctv_yolo.annotated_export import annotate_video

from .csv_writer import write_per_track_csv


def _build_summary_pdf(pdf_path: Path, track_data: dict, session_id: str,
                       video_name: str) -> bool:
    """Render a one-page summary PDF. Returns False if reportlab missing."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        return False

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    y = height - 60

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, f"CCTV-YOLO Review Pack — {session_id}")
    y -= 28
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Video: {video_name or '(unknown)'}")
    y -= 14
    c.drawString(50, y, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    y -= 20

    tracks = track_data.get("tracks", [])
    rois = track_data.get("rois", [])
    class_counts = Counter(t.get("class", "unknown") for t in tracks)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Summary")
    y -= 18
    c.setFont("Helvetica", 11)
    for line in [
        f"Total tracks: {len(tracks)}",
        f"Total ROIs: {len(rois)}",
        f"Total frames: {track_data.get('total_frames', 0)}",
        f"Resolution: {track_data.get('resolution', '')}",
        f"FPS: {track_data.get('fps', 0)}",
    ]:
        c.drawString(60, y, line)
        y -= 14

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Tracks by class")
    y -= 16
    c.setFont("Helvetica", 11)
    for cls, n in class_counts.most_common():
        c.drawString(60, y, f"{cls}: {n}")
        y -= 14

    if rois:
        y -= 6
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "ROIs")
        y -= 16
        c.setFont("Helvetica", 11)
        for r in rois:
            c.drawString(60, y, f"- {r.get('name', '?')} ({r.get('type', '?')})")
            y -= 14
            if y < 60:
                c.showPage()
                y = height - 60

    c.showPage()
    c.save()
    return True


def build_review_pack(
    data_manager,
    session_id: str,
    output_zip: Path | None = None,
    roi_id: str | None = None,
    progress_callback=None,
) -> Path:
    """Bundle annotated video, CSV, PDF, README into a zip.

    Returns the path to the output zip.
    """
    track_data = data_manager.load_session_data(session_id) or {"tracks": [], "rois": []}
    video_path = data_manager.get_video_path(session_id)
    if video_path is None or not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found for session: {session_id}")

    out_dir = data_manager.exports_dir / session_id / "review_pack"
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_zip is None:
        output_zip = out_dir.parent / f"{session_id}_review_pack.zip"
    output_zip = Path(output_zip)

    # Working files
    work_dir = out_dir / "_build"
    work_dir.mkdir(parents=True, exist_ok=True)

    annotated_mp4 = work_dir / f"{session_id}_annotated.mp4"
    csv_path = work_dir / f"{session_id}_per_track.csv"
    pdf_path = work_dir / f"{session_id}_summary.pdf"
    readme_path = work_dir / "README.md"

    if progress_callback:
        progress_callback(5)

    annotate_video(
        video_path=Path(video_path),
        track_data=track_data,
        output_path=annotated_mp4,
        progress_callback=lambda p: progress_callback(5 + int(p * 0.75)) if progress_callback else None,
    )

    if progress_callback:
        progress_callback(82)

    write_per_track_csv(csv_path, track_data, roi_id=roi_id)
    pdf_ok = _build_summary_pdf(
        pdf_path, track_data, session_id, track_data.get("video_name", ""),
    )

    if progress_callback:
        progress_callback(90)

    readme_lines = [
        f"# Review Pack — {session_id}",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Contents",
        f"- `{annotated_mp4.name}` — annotated MP4 (bboxes, labels, ROIs, HUD).",
        f"- `{csv_path.name}` — per-track CSV (id, class, frame range, conf).",
    ]
    if pdf_ok:
        readme_lines.append(f"- `{pdf_path.name}` — one-page PDF summary.")
    else:
        readme_lines.append(
            "- _PDF summary skipped:_ `reportlab` is not installed. "
            "Run `pip install reportlab` and rebuild the pack to include it."
        )
    if roi_id:
        readme_lines.append(f"\nROI filter applied: **{roi_id}**")
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    # Zip everything
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(annotated_mp4, annotated_mp4.name)
        zf.write(csv_path, csv_path.name)
        if pdf_ok:
            zf.write(pdf_path, pdf_path.name)
        zf.write(readme_path, readme_path.name)

    # Clean up working dir
    try:
        shutil.rmtree(work_dir)
    except OSError:
        pass

    if progress_callback:
        progress_callback(100)

    return output_zip
