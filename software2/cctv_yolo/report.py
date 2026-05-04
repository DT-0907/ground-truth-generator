"""
Self-contained HTML session report.

Includes:
- Header (video name, processed-at, resolution, FPS)
- Summary stats (track count, by-class counts, ROI tallies)
- Embedded heatmap PNG (base64) if present
- Embedded annotated MP4 link (file:// to keep file size sane)
- OD matrix table
- Time-series chart (CSV link)
- Speed table (top-10 fastest)
- Anomalies (if any baseline available)

No template engine — plain f-strings keep it dependency-free.
"""
from __future__ import annotations
import base64
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from cctv_yolo import analytics


def _b64_image(path: Path) -> str:
    if not path or not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CCTV-YOLO report — {session_id}</title>
<style>
  body {{
    margin: 0; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    background: #0e1424; color: #eee;
  }}
  header {{ padding: 22px 32px; background: #16213e; border-bottom: 2px solid #4ecca3; }}
  h1 {{ margin: 0; color: #4ecca3; font-size: 22px; }}
  h2 {{ color: #4ecca3; margin-top: 32px; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 28px 32px; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
  .card {{
    background: #16213e; padding: 14px; border-radius: 8px;
    border-top: 2px solid #4ecca3;
  }}
  .num {{ color: #4ecca3; font-size: 28px; font-weight: 700; }}
  .lab {{ font-size: 12px; opacity: 0.7; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ padding: 8px; border: 1px solid #2d3a5a; text-align: left; }}
  th {{ background: #16213e; color: #4ecca3; font-size: 12px; }}
  td.num {{ text-align: right; font-size: 14px; color: #eee; font-weight: 400; }}
  img {{ max-width: 100%; border: 1px solid #2d3a5a; border-radius: 6px; }}
  video {{ max-width: 100%; border-radius: 6px; }}
  .muted {{ color: #888; font-size: 13px; }}
  details {{ margin: 12px 0; }}
  summary {{ cursor: pointer; color: #4ecca3; font-weight: 600; }}
</style>
</head>
<body>
<header>
  <h1>{session_id}</h1>
  <div class="muted">{video_name} · {resolution} · {fps:.1f} fps · processed {processed_at}</div>
</header>
<main>
  <h2>Summary</h2>
  <div class="stats">
    <div class="card"><div class="num">{n_tracks}</div><div class="lab">Tracks</div></div>
    <div class="card"><div class="num">{n_review}</div><div class="lab">Needs review</div></div>
    <div class="card"><div class="num">{n_corrected}</div><div class="lab">Has corrections</div></div>
    <div class="card"><div class="num">{n_rois}</div><div class="lab">ROIs</div></div>
  </div>

  <h2>By class</h2>
  <table>
    <tr><th>Class</th><th>Count</th></tr>
    {class_rows}
  </table>

  {heatmap_block}

  {od_block}

  {speeds_block}

  {anomalies_block}

  {video_block}

  <h2>Files</h2>
  <ul>
    {file_links}
  </ul>

  <p class="muted">Generated {gen_at} by CCTV-YOLO.</p>
</main>
</body>
</html>
"""


def render_html_report(
    data_manager,
    session_id: str,
    output_path: Optional[Path] = None,
    embed_video: bool = False,
) -> Path:
    track_data = data_manager.load_session_data(session_id)
    if track_data is None:
        raise FileNotFoundError(f"No data for session {session_id}")

    sess_dir = data_manager.exports_dir / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = sess_dir / "report.html"
    output_path = Path(output_path)

    video_path = data_manager.get_video_path(session_id)
    info = data_manager.get_video_info(video_path) if video_path else {
        "fps": 0, "resolution": "n/a", "total_frames": 0,
    }

    tracks = track_data.get("tracks", [])
    rois = track_data.get("rois", [])
    n_review = sum(1 for t in tracks if t.get("needs_review"))
    n_corrected = 1 if data_manager.has_corrections(session_id) else 0

    cls_counter = Counter(t.get("class", "unknown") for t in tracks)
    class_rows = "\n".join(
        f"<tr><td>{cls}</td><td class='num'>{n}</td></tr>"
        for cls, n in cls_counter.most_common()
    ) or "<tr><td colspan='2' class='muted'>(none)</td></tr>"

    # Heatmap — render if missing
    hm = sess_dir / f"{session_id}_heatmap.png"
    heatmap_block = ""
    try:
        if not hm.exists() and video_path:
            analytics.render_heatmap(video_path, track_data, hm)
        if hm.exists():
            heatmap_block = (
                "<h2>Path-density heatmap</h2>\n"
                f"<img src='data:image/png;base64,{_b64_image(hm)}'>"
            )
    except Exception as e:
        heatmap_block = f"<h2>Heatmap</h2><p class='muted'>error: {e}</p>"

    # OD matrix
    od_block = ""
    if rois:
        od = analytics.origin_destination_matrix(track_data)
        rows = ["<tr><th>O / D</th>" +
                "".join(f"<th>{d}</th>" for d in od["rois"]) + "</tr>"]
        for o in od["rois"]:
            row = f"<tr><td>{o}</td>" + "".join(
                f"<td class='num'>{od['matrix'].get(o, {}).get(d, 0)}</td>"
                for d in od["rois"]
            ) + "</tr>"
            rows.append(row)
        od_block = ("<h2>Origin → destination</h2>"
                    "<table>" + "\n".join(rows) + "</table>")

    # Top speeds (using 20 px/m default if not provided — purely for display)
    try:
        speeds = analytics.estimate_speeds(track_data, pixels_per_meter=20.0)
        speeds.sort(key=lambda s: -s["peak_speed_mph"])
        top = speeds[:10]
        if top:
            rows = "\n".join(
                f"<tr><td>#{s['track_id']}</td><td>{s['class']}</td>"
                f"<td class='num'>{s['avg_speed_mph']}</td>"
                f"<td class='num'>{s['peak_speed_mph']}</td></tr>"
                for s in top
            )
            speeds_block = (
                "<h2>Top-10 speeds (assumes ppm=20)</h2>"
                "<table><tr><th>Track</th><th>Class</th>"
                "<th>Avg mph</th><th>Peak mph</th></tr>"
                + rows + "</table>"
                "<p class='muted'>Re-run with the correct pixels_per_meter "
                "for accurate numbers.</p>"
            )
        else:
            speeds_block = ""
    except Exception:
        speeds_block = ""

    # Anomalies (best-effort — silent on failure since it requires baselines)
    anomalies_block = ""
    try:
        from cctv_yolo.anomaly import detect_anomalies
        anomalies = detect_anomalies(data_manager, session_id, z_threshold=2.0)
        if anomalies:
            rows = "\n".join(
                f"<tr><td>{a.metric}</td><td>{a.roi}</td><td>{a.hour}</td>"
                f"<td class='num'>{a.value}</td><td class='num'>{a.baseline_mean}</td>"
                f"<td class='num'>{a.z_score}</td></tr>"
                for a in anomalies[:20]
            )
            anomalies_block = (
                "<h2>Anomalies (z >= 2)</h2>"
                "<table><tr><th>Metric</th><th>ROI</th><th>Hour</th>"
                "<th>Value</th><th>Baseline</th><th>Z</th></tr>"
                + rows + "</table>"
            )
    except Exception:
        pass

    # Annotated video (link, not embedded by default — file size)
    video_block = ""
    annotated = sess_dir / f"{session_id}_annotated.mp4"
    if annotated.exists():
        if embed_video:
            b64 = base64.b64encode(annotated.read_bytes()).decode("ascii")
            video_block = (
                "<h2>Annotated video</h2>"
                f"<video controls src='data:video/mp4;base64,{b64}'></video>"
            )
        else:
            video_block = (
                "<h2>Annotated video</h2>"
                f"<video controls src='{annotated.name}'></video>"
                "<p class='muted'>(file linked relative to this report)</p>"
            )

    # File listing
    files = sorted(sess_dir.iterdir())
    file_links = "\n".join(
        f"<li><a href='{p.name}'>{p.name}</a></li>"
        for p in files if p.is_file()
    ) or "<li class='muted'>(no files yet)</li>"

    html = _TEMPLATE.format(
        session_id=session_id,
        video_name=info.get("resolution", "n/a") if not video_path else video_path.name,
        resolution=info.get("resolution", "n/a"),
        fps=info.get("fps", 0),
        processed_at=track_data.get("processed_at", "?"),
        n_tracks=len(tracks),
        n_review=n_review,
        n_corrected=n_corrected,
        n_rois=len(rois),
        class_rows=class_rows,
        heatmap_block=heatmap_block,
        od_block=od_block,
        speeds_block=speeds_block,
        anomalies_block=anomalies_block,
        video_block=video_block,
        file_links=file_links,
        gen_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path
