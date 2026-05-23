"""
Anomaly detection from learned traffic patterns.

Builds a per-ROI / per-hour-of-day baseline from every available
session: count, class-mix, mean speed (if available). Flag any session
whose values fall outside ``mean +/- N*sigma``.

Time-of-day is inferred from the session's ``processed_at`` timestamp
and the offset of each event within the video — for a 60-minute clip
processed at 14:00, an event 30 minutes in is bucketed at hour 14.

When a session lacks ROIs, falls back to whole-frame statistics.
"""
from __future__ import annotations
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from cctv_yolo.analytics import bbox_in_roi


@dataclass
class Anomaly:
    session_id: str
    hour: int
    roi: str
    metric: str          # "count", "class_share:car", "mean_speed", ...
    value: float
    baseline_mean: float
    baseline_std: float
    z_score: float


def _hour_of_event(processed_at_iso: str, event_seconds_in: float) -> int:
    """Return hour-of-day [0..23] of an event."""
    try:
        base = datetime.fromisoformat(processed_at_iso)
    except (ValueError, TypeError):
        base = datetime.now()
    sec = (base.hour * 3600 + base.minute * 60 + base.second + event_seconds_in)
    return int(sec / 3600) % 24


def _aggregate_session(track_data: dict) -> dict:
    """Aggregate per-ROI per-hour stats for a single session."""
    fps = float(track_data.get("fps", 30.0)) or 30.0
    processed_at = track_data.get("processed_at") or datetime.now().isoformat()
    rois = track_data.get("rois", []) or []
    roi_pairs = list(zip(
        [r.get("name") or f"ROI {i+1}" for i, r in enumerate(rois)],
        rois,
    ))

    # bucket: {(hour, roi_name): {"count": n, "class": {cls: n}, ...}}
    buckets = defaultdict(lambda: {"count": 0, "class": defaultdict(int)})
    no_roi_label = "__all__"

    for tr in track_data.get("tracks", []):
        cls = tr.get("class", "vehicle")
        frames = sorted(tr.get("frames", []), key=lambda f: f["frame"])
        if not frames:
            continue
        f0 = frames[0]["frame"]
        sec0 = f0 / fps
        hour = _hour_of_event(processed_at, sec0)

        # Whole-session bucket
        b = buckets[(hour, no_roi_label)]
        b["count"] += 1
        b["class"][cls] += 1

        # Per-ROI: count once per (track, roi) if track ever enters
        for roi_name, roi in roi_pairs:
            if any(bbox_in_roi(f["bbox"], roi) for f in frames):
                b2 = buckets[(hour, roi_name)]
                b2["count"] += 1
                b2["class"][cls] += 1

    return buckets


def build_baseline(data_manager, exclude_session: Optional[str] = None) -> dict:
    """Aggregate stats across every session into a baseline distribution.

    Returns a dict keyed by ``(hour, roi)`` of arrays of per-session
    metric values, ready for outlier detection.
    """
    distrib: dict = defaultdict(lambda: {
        "count": [],
        "class_share": defaultdict(list),
        "sessions": [],
    })

    for s in data_manager.get_sessions():
        sid = s["id"]
        if sid == exclude_session:
            continue
        data = data_manager.load_session_data(sid)
        if not data:
            continue
        agg = _aggregate_session(data)
        for (hour, roi), stats in agg.items():
            d = distrib[(hour, roi)]
            d["count"].append(stats["count"])
            total = max(1, stats["count"])
            for cls, n in stats["class"].items():
                d["class_share"][cls].append(n / total)
            d["sessions"].append(sid)
    return distrib


def _zscore(x: float, mean: float, std: float) -> float:
    if std <= 1e-9:
        return 0.0
    return (x - mean) / std


def detect_anomalies(
    data_manager,
    target_session: str,
    z_threshold: float = 2.0,
) -> list[Anomaly]:
    """Compare *target_session* against the baseline of all OTHER
    sessions. Returns a list of metrics that exceed the z-threshold.
    """
    target = data_manager.load_session_data(target_session)
    if not target:
        return []
    baseline = build_baseline(data_manager, exclude_session=target_session)
    target_agg = _aggregate_session(target)

    anomalies: list[Anomaly] = []
    for (hour, roi), stats in target_agg.items():
        b = baseline.get((hour, roi))
        if not b or len(b["count"]) < 2:
            continue
        mean = statistics.mean(b["count"])
        std = statistics.pstdev(b["count"])
        z = _zscore(stats["count"], mean, std)
        if abs(z) >= z_threshold:
            anomalies.append(Anomaly(
                session_id=target_session,
                hour=hour, roi=roi,
                metric="count",
                value=stats["count"],
                baseline_mean=round(mean, 2),
                baseline_std=round(std, 2),
                z_score=round(z, 2),
            ))

        # Class-share anomalies
        total = max(1, stats["count"])
        for cls, n in stats["class"].items():
            share = n / total
            shares = b["class_share"].get(cls, [])
            if len(shares) < 2:
                continue
            cmean = statistics.mean(shares)
            cstd = statistics.pstdev(shares)
            cz = _zscore(share, cmean, cstd)
            if abs(cz) >= z_threshold:
                anomalies.append(Anomaly(
                    session_id=target_session,
                    hour=hour, roi=roi,
                    metric=f"class_share:{cls}",
                    value=round(share, 3),
                    baseline_mean=round(cmean, 3),
                    baseline_std=round(cstd, 3),
                    z_score=round(cz, 2),
                ))

    anomalies.sort(key=lambda a: -abs(a.z_score))
    return anomalies


def detect_anomalies_batch(
    data_manager,
    session_ids: Iterable[str],
    z_threshold: float = 2.0,
) -> list[Anomaly]:
    """Run :func:`detect_anomalies` for each session in *session_ids* and
    return a flattened, z-sorted list. The ``session_id`` attribute on each
    Anomaly identifies the originating clip — used as a column in the
    Group / Dataset / Multi insights sub-tabs (PRD I2).
    """
    out: list[Anomaly] = []
    sids = list(session_ids)
    sid_set = set(sids)
    for sid in sids:
        target = data_manager.load_session_data(sid)
        if not target:
            continue
        # Build a baseline excluding all sessions in the requested set
        # so per-session anomalies don't get suppressed by their own data.
        baseline: dict = defaultdict(lambda: {
            "count": [],
            "class_share": defaultdict(list),
            "sessions": [],
        })
        for s in data_manager.get_sessions():
            other = s["id"]
            if other in sid_set:
                continue
            data = data_manager.load_session_data(other)
            if not data:
                continue
            agg = _aggregate_session(data)
            for (hour, roi), stats in agg.items():
                d = baseline[(hour, roi)]
                d["count"].append(stats["count"])
                total = max(1, stats["count"])
                for cls, n in stats["class"].items():
                    d["class_share"][cls].append(n / total)
                d["sessions"].append(other)

        target_agg = _aggregate_session(target)
        for (hour, roi), stats in target_agg.items():
            b = baseline.get((hour, roi))
            if not b or len(b["count"]) < 2:
                continue
            mean = statistics.mean(b["count"])
            std = statistics.pstdev(b["count"])
            z = _zscore(stats["count"], mean, std)
            if abs(z) >= z_threshold:
                out.append(Anomaly(
                    session_id=sid, hour=hour, roi=roi, metric="count",
                    value=stats["count"],
                    baseline_mean=round(mean, 2),
                    baseline_std=round(std, 2),
                    z_score=round(z, 2),
                ))
            total = max(1, stats["count"])
            for cls, n in stats["class"].items():
                share = n / total
                shares = b["class_share"].get(cls, [])
                if len(shares) < 2:
                    continue
                cmean = statistics.mean(shares)
                cstd = statistics.pstdev(shares)
                cz = _zscore(share, cmean, cstd)
                if abs(cz) >= z_threshold:
                    out.append(Anomaly(
                        session_id=sid, hour=hour, roi=roi,
                        metric=f"class_share:{cls}",
                        value=round(share, 3),
                        baseline_mean=round(cmean, 3),
                        baseline_std=round(cstd, 3),
                        z_score=round(cz, 2),
                    ))
    out.sort(key=lambda a: -abs(a.z_score))
    return out


def save_anomalies(anomalies: list[Anomaly], out_dir: Path,
                   ts: Optional[str] = None) -> Path:
    """Write a list of Anomaly objects to ``out_dir/anomalies_<ts>.json``.

    PRD I3.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"anomalies_{ts}.json"
    payload = {
        "saved_at": datetime.now().isoformat(),
        "count": len(anomalies),
        "anomalies": [asdict(a) for a in anomalies],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def load_anomalies(path: Path) -> list[Anomaly]:
    """Load anomalies previously saved by :func:`save_anomalies`."""
    with open(Path(path), "r") as f:
        data = json.load(f)
    rows = data.get("anomalies", [])
    return [Anomaly(**row) for row in rows]


def list_anomaly_history(folder: Path) -> list[Path]:
    """Return all anomaly snapshots under ``folder``, newest first."""
    p = Path(folder)
    if not p.exists():
        return []
    return sorted(p.glob("anomalies_*.json"), reverse=True)


def baseline_summary(distrib: dict) -> list[dict]:
    """Tabular view of the baseline for the UI."""
    rows = []
    for (hour, roi), data in sorted(distrib.items()):
        if len(data["count"]) < 2:
            continue
        rows.append({
            "hour": hour,
            "roi": roi,
            "n_sessions": len(data["count"]),
            "count_mean": round(statistics.mean(data["count"]), 2),
            "count_std": round(statistics.pstdev(data["count"]), 2),
        })
    return rows
