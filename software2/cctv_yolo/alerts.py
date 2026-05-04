"""
Simple rule engine for live-stream alerts.

Tracks per-track state (entry frame, last position, last seen time)
and emits alerts when rules trigger.
"""
from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackState:
    track_id: int
    cls: str
    first_seen_t: float
    last_seen_t: float
    entry_pos: Optional[tuple[float, float]] = None
    exit_pos: Optional[tuple[float, float]] = None
    samples: int = 0
    centers: list[tuple[float, float, float]] = field(default_factory=list)  # (t, cx, cy)


@dataclass
class Alert:
    track_id: int
    rule: str
    message: str
    timestamp: float


class AlertEngine:
    """Stateful rule engine. Feed detections each frame; pull pending alerts."""

    def __init__(
        self,
        loiter_seconds: float = 30.0,
        speed_pixels_per_sec_max: float = 800.0,
        wrong_way_dx_threshold: float = -10.0,
    ):
        """
        Parameters
        ----------
        loiter_seconds:
            Track present for >= this many seconds → loitering alert.
        speed_pixels_per_sec_max:
            Approximate hard cap on pixel velocity. Calibrate per camera.
        wrong_way_dx_threshold:
            If the typical traffic flow is left-to-right (+x), a sustained
            negative dx triggers a wrong-way alert. Provide negative number
            for a flow you expect to *not* see.
        """
        self.loiter_seconds = loiter_seconds
        self.speed_max = speed_pixels_per_sec_max
        self.wrong_way_dx_threshold = wrong_way_dx_threshold
        self.tracks: dict[int, TrackState] = {}
        self.pending: list[Alert] = []
        self.fired_rules: dict[int, set[str]] = defaultdict(set)

    def update(self, detections: list[dict], now: Optional[float] = None) -> list[Alert]:
        """Feed current-frame detections.

        Each detection: {"track_id", "class", "bbox": [x1,y1,x2,y2]}
        Returns the list of new alerts since last call.
        """
        if now is None:
            now = time.time()

        new_alerts: list[Alert] = []

        for d in detections:
            tid = d.get("track_id")
            if tid is None:
                continue
            cls = d.get("class", "vehicle")
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            st = self.tracks.get(tid)
            if st is None:
                st = TrackState(
                    track_id=tid, cls=cls,
                    first_seen_t=now, last_seen_t=now,
                    entry_pos=(cx, cy),
                )
                self.tracks[tid] = st
            else:
                # Speed check
                if st.centers:
                    pt, pcx, pcy = st.centers[-1]
                    dt = now - pt
                    if dt > 0:
                        v = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5 / dt
                        if v > self.speed_max and "speed" not in self.fired_rules[tid]:
                            new_alerts.append(Alert(
                                tid, "speed",
                                f"Track #{tid} ({cls}) exceeded speed cap "
                                f"({v:.0f} px/s)",
                                now,
                            ))
                            self.fired_rules[tid].add("speed")
                st.last_seen_t = now
                st.exit_pos = (cx, cy)

            st.centers.append((now, cx, cy))
            if len(st.centers) > 60:
                st.centers = st.centers[-60:]
            st.samples += 1

            # Loiter
            present = now - st.first_seen_t
            if present >= self.loiter_seconds and "loiter" not in self.fired_rules[tid]:
                new_alerts.append(Alert(
                    tid, "loiter",
                    f"Track #{tid} ({cls}) loitering for {present:.0f}s",
                    now,
                ))
                self.fired_rules[tid].add("loiter")

            # Wrong-way (very simple: smoothed dx vs threshold)
            if len(st.centers) >= 6:
                dx_samples = [
                    st.centers[i][1] - st.centers[i - 1][1]
                    for i in range(-5, 0)
                ]
                avg_dx = sum(dx_samples) / len(dx_samples)
                if (
                    self.wrong_way_dx_threshold < 0
                    and avg_dx <= self.wrong_way_dx_threshold
                ) or (
                    self.wrong_way_dx_threshold > 0
                    and avg_dx >= self.wrong_way_dx_threshold
                ):
                    if "wrong_way" not in self.fired_rules[tid]:
                        new_alerts.append(Alert(
                            tid, "wrong_way",
                            f"Track #{tid} ({cls}) appears to travel wrong way "
                            f"(avg dx={avg_dx:.1f})",
                            now,
                        ))
                        self.fired_rules[tid].add("wrong_way")

        # Garbage-collect tracks not seen in >30s
        stale_cutoff = now - 30.0
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].last_seen_t < stale_cutoff:
                self.tracks.pop(tid, None)
                self.fired_rules.pop(tid, None)

        self.pending.extend(new_alerts)
        return new_alerts

    def all_alerts(self) -> list[Alert]:
        return list(self.pending)
