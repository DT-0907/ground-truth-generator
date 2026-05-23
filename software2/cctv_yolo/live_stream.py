"""
Live video source — RTSP / webcam / file with YOLO detection + tracking
in a Qt-friendly worker.

Emits annotated frames as QImage so they can be displayed without
synchronization headaches. Supports:

- A/B model compare (two models run on the same frames; model B can be
  throttled to every Nth frame)
- Per-ROI live counts + entry/exit/dwell events
- Recording (mp4v) — start/stop, auto on ROI event
- 10s connect timeout + exponential-backoff reconnect for RTSP
- FPS / inference-ms stats

Public Start/Stop API and the original ``frame_ready`` / ``alert`` /
``failed`` / ``stopped`` signals are preserved.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from cctv_yolo.alerts import AlertEngine


VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}

# Bounding-box colors — Correction's spec: PURPLE default, PINK for selected.
# Live has no "selected" track, so all boxes use PURPLE. The values below are
# in BGR for cv2.
BBOX_BGR        = (152, 37, 152)[::-1]    # PURPLE #982598 -> BGR
BBOX_SEL_BGR    = (228, 145, 201)[::-1]   # PINK
TEXT_BGR        = (241, 233, 233)[::-1]   # OFFWHITE
PANEL_BGR       = (30, 32, 80)[::-1]      # PANEL #1E2050
RED_BGR         = (255, 107, 122)[::-1]   # ERROR #FF6B7A


def _device():
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    bytes_per_line = 3 * w
    # .copy() — detach from the underlying numpy buffer so Qt can keep it alive
    return QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)


def source_hash(source: str) -> str:
    """Stable short hash used to key per-source on-disk artifacts."""
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """Ray-cast point-in-polygon. ``polygon`` is a list of (x, y)."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def _roi_contains(roi: dict, cx: float, cy: float) -> bool:
    if roi["type"] == "rect":
        p1, p2 = roi["points"][0], roi["points"][1]
        x1 = min(p1["x"], p2["x"])
        y1 = min(p1["y"], p2["y"])
        x2 = max(p1["x"], p2["x"])
        y2 = max(p1["y"], p2["y"])
        return x1 <= cx <= x2 and y1 <= cy <= y2
    pts = [(p["x"], p["y"]) for p in roi["points"]]
    return _point_in_polygon(cx, cy, pts)


def _draw_box_xyxy(frame: np.ndarray, bbox, color, label: str = ""):
    x1, y1, x2, y2 = [int(c) for c in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(
            frame, label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )


def _draw_rois(frame: np.ndarray, rois: list, counts: dict):
    """Overlay ROI shapes + live count badges."""
    for i, roi in enumerate(rois):
        name = roi.get("name", f"ROI {i+1}")
        count = counts.get(name, 0)
        if roi["type"] == "rect":
            p1, p2 = roi["points"][0], roi["points"][1]
            x1 = int(min(p1["x"], p2["x"]))
            y1 = int(min(p1["y"], p2["y"]))
            x2 = int(max(p1["x"], p2["x"]))
            y2 = int(max(p1["y"], p2["y"]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_SEL_BGR, 2)
            label = f"{name}  {count}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(
                frame, (x1, max(0, y1 - th - 8)),
                (x1 + tw + 8, y1), PANEL_BGR, -1,
            )
            cv2.putText(
                frame, label, (x1 + 4, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_BGR, 1,
            )
        elif roi["type"] == "polygon" and len(roi["points"]) >= 3:
            pts = np.array(
                [[int(p["x"]), int(p["y"])] for p in roi["points"]],
                dtype=np.int32,
            )
            cv2.polylines(frame, [pts], True, BBOX_SEL_BGR, 2)
            x0, y0 = pts[0]
            label = f"{name}  {count}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(
                frame, (x0, max(0, y0 - th - 8)),
                (x0 + tw + 8, y0), PANEL_BGR, -1,
            )
            cv2.putText(
                frame, label, (x0 + 4, max(0, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_BGR, 1,
            )


def _draw_fps_overlay(frame: np.ndarray, fps: float, infer_ms: float, label: str = ""):
    """Bottom-left FPS/inference badge."""
    h, w = frame.shape[:2]
    text = f"{fps:5.1f} fps  ·  {infer_ms:5.1f}ms infer"
    if label:
        text = f"{label}  ·  " + text
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x = 10
    y = h - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, y - th - 6), (x + tw + 6, y + 4), PANEL_BGR, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_BGR, 1)


def _draw_rec_indicator(frame: np.ndarray):
    """Top-right recording dot."""
    h, w = frame.shape[:2]
    cv2.circle(frame, (w - 22, 22), 8, RED_BGR, -1)
    cv2.putText(
        frame, "REC", (w - 70, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED_BGR, 2,
    )


def _detections_from_results(results, vehicle_classes: dict) -> list[dict]:
    detections = []
    if not results or results[0].boxes is None:
        return detections
    boxes = results[0].boxes
    for i in range(len(boxes)):
        bbox = boxes.xyxy[i].cpu().numpy().tolist()
        cid = int(boxes.cls[i].item())
        cname = vehicle_classes.get(cid, "unknown")
        conf_v = float(boxes.conf[i].item())
        tid = int(boxes.id[i].item()) if boxes.id is not None else -1
        detections.append({
            "track_id": tid,
            "class": cname,
            "conf": conf_v,
            "bbox": bbox,
        })
    return detections


class _RoiTracker:
    """Tracks per-ROI track_id occupancy across frames to emit
    entry / exit / dwell events."""

    def __init__(self, rois: list, dwell_seconds: float = 5.0):
        self.rois = rois or []
        self.dwell_seconds = dwell_seconds
        # roi_name -> set of track_ids currently inside
        self._prev: dict[str, set] = {}
        # (roi_name, tid) -> entered_at
        self._entered_at: dict[tuple, float] = {}
        # Tracks that already fired a dwell event so we don't spam
        self._fired_dwell: set[tuple] = set()

    def update(self, detections: list[dict], now: float) -> tuple[dict, list[dict]]:
        """Return (counts_per_roi, events). Events are dicts with
        keys: rule, roi, track_id, class, timestamp."""
        counts: dict[str, int] = {}
        events: list[dict] = []
        current: dict[str, set] = {}

        for idx, roi in enumerate(self.rois):
            name = roi.get("name") or f"ROI {idx+1}"
            inside_ids: set = set()
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if _roi_contains(roi, cx, cy):
                    inside_ids.add(d["track_id"])
            current[name] = inside_ids
            counts[name] = len(inside_ids)

            prev = self._prev.get(name, set())
            entered = inside_ids - prev
            exited = prev - inside_ids
            for tid in entered:
                self._entered_at[(name, tid)] = now
                self._fired_dwell.discard((name, tid))
                events.append({
                    "rule": "roi_enter",
                    "roi": name,
                    "track_id": tid,
                    "timestamp": now,
                    "message": f"track #{tid} entered {name}",
                })
            for tid in exited:
                self._entered_at.pop((name, tid), None)
                self._fired_dwell.discard((name, tid))
                events.append({
                    "rule": "roi_exit",
                    "roi": name,
                    "track_id": tid,
                    "timestamp": now,
                    "message": f"track #{tid} left {name}",
                })
            for tid in inside_ids:
                key = (name, tid)
                t0 = self._entered_at.get(key)
                if t0 and (now - t0) >= self.dwell_seconds and key not in self._fired_dwell:
                    self._fired_dwell.add(key)
                    events.append({
                        "rule": "roi_dwell",
                        "roi": name,
                        "track_id": tid,
                        "timestamp": now,
                        "message": (
                            f"track #{tid} dwelled in {name} for "
                            f"{self.dwell_seconds:.0f}s"
                        ),
                    })

        self._prev = current
        return counts, events


class LiveStreamWorker(QThread):
    """Pull frames from any cv2.VideoCapture source, run detection,
    emit annotated frames + alerts.

    A/B compare mode emits both ``frame_ready`` (model A) and
    ``frame_ready_b`` (model B). In single-model mode only model A runs.
    """

    # Model-A signals (kept exactly compatible with prior consumers)
    frame_ready = Signal(QImage, dict)          # QImage, stats
    alert       = Signal(dict)                  # serialized Alert / ROI event
    failed      = Signal(str)
    stopped     = Signal()

    # A/B + lifecycle signals
    frame_ready_b      = Signal(QImage, dict)   # only fires when A/B is on
    reconnecting       = Signal(str)            # message banner
    reconnected        = Signal()
    recording_started  = Signal(str)            # path
    recording_stopped  = Signal(str)            # path

    def __init__(
        self,
        source: str,
        model_path: str,
        models_dir: Path,
        conf: float = 0.3,
        loiter_seconds: float = 30.0,
        wrong_way_dx_threshold: float = -10.0,
        max_fps: int = 15,
        rois: Optional[list] = None,
        roi_dwell_seconds: float = 5.0,
        # A/B compare
        model_b_path: Optional[str] = None,
        b_every_n: int = 1,
        # Recording
        record_dir: Optional[Path] = None,
        record_on_event: bool = False,
        event_clip_seconds: float = 10.0,
        # RTSP behavior
        connect_timeout_seconds: float = 10.0,
        reconnect: bool = True,
        max_reconnect_attempts: int = 0,   # 0 = infinite
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.model_path = model_path
        self.model_b_path = model_b_path
        self.models_dir = Path(models_dir)
        self.conf = conf
        self.loiter_seconds = loiter_seconds
        self.wrong_way_dx_threshold = wrong_way_dx_threshold
        self.max_fps = max(1, int(max_fps))
        self.rois = list(rois or [])
        self.roi_dwell_seconds = roi_dwell_seconds

        self.b_every_n = max(1, int(b_every_n))

        self.record_dir = Path(record_dir) if record_dir else None
        self.record_on_event = record_on_event
        self.event_clip_seconds = event_clip_seconds

        self.connect_timeout_seconds = connect_timeout_seconds
        self.reconnect = reconnect
        self.max_reconnect_attempts = max_reconnect_attempts

        # Runtime state
        self._stop = False
        self._lock = threading.Lock()
        self._record_request: Optional[bool] = None   # set by external calls
        self._writer_a: Optional[cv2.VideoWriter] = None
        self._writer_b: Optional[cv2.VideoWriter] = None
        self._writer_paths: tuple[Optional[Path], Optional[Path]] = (None, None)
        self._event_record_until: float = 0.0
        self._snapshot_request: bool = False
        self._snapshot_dir: Optional[Path] = None
        self._last_frame_a: Optional[np.ndarray] = None
        self._last_frame_b: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public control surface (thread-safe — only flips flags)
    # ------------------------------------------------------------------

    def stop(self):
        self._stop = True

    def start_recording(self):
        with self._lock:
            self._record_request = True

    def stop_recording(self):
        with self._lock:
            self._record_request = False

    def request_snapshot(self, snapshot_dir: Path):
        with self._lock:
            self._snapshot_request = True
            self._snapshot_dir = Path(snapshot_dir)

    def update_rois(self, rois: list):
        with self._lock:
            self.rois = list(rois or [])

    # ------------------------------------------------------------------
    # Capture helpers
    # ------------------------------------------------------------------

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap_source = int(self.source) if self.source.isdigit() else self.source
        # FFMPEG is best for RTSP — fall back automatically otherwise.
        try:
            cap = cv2.VideoCapture(cap_source, cv2.CAP_FFMPEG) \
                if not isinstance(cap_source, int) else cv2.VideoCapture(cap_source)
        except Exception:
            cap = cv2.VideoCapture(cap_source)

        deadline = time.time() + self.connect_timeout_seconds
        while time.time() < deadline:
            if cap.isOpened():
                # Try one frame read to confirm RTSP handshake actually flows
                ret, _ = cap.read()
                if ret:
                    # Reopen so we don't lose this first frame
                    cap.release()
                    cap = cv2.VideoCapture(cap_source, cv2.CAP_FFMPEG) \
                        if not isinstance(cap_source, int) else cv2.VideoCapture(cap_source)
                    return cap
            time.sleep(0.2)
        try:
            cap.release()
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def _ensure_writers(self, frame_a: np.ndarray, frame_b: Optional[np.ndarray]):
        if self._writer_a is not None:
            return
        if not self.record_dir:
            return
        out_dir = self.record_dir / source_hash(self.source)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path_a = out_dir / f"{ts}.mp4"
        h, w = frame_a.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = float(self.max_fps)
        self._writer_a = cv2.VideoWriter(str(path_a), fourcc, fps, (w, h))
        path_b = None
        if frame_b is not None:
            path_b = out_dir / f"{ts}_b.mp4"
            hb, wb = frame_b.shape[:2]
            self._writer_b = cv2.VideoWriter(str(path_b), fourcc, fps, (wb, hb))
        self._writer_paths = (path_a, path_b)
        self.recording_started.emit(str(path_a))

    def _close_writers(self):
        path_a, path_b = self._writer_paths
        if self._writer_a is not None:
            self._writer_a.release()
            self._writer_a = None
        if self._writer_b is not None:
            self._writer_b.release()
            self._writer_b = None
        if path_a is not None:
            self.recording_stopped.emit(str(path_a))
        self._writer_paths = (None, None)

    # ------------------------------------------------------------------
    # Snapshot helper (A/B compare delta)
    # ------------------------------------------------------------------

    def _write_snapshot(self, frame_a: np.ndarray, frame_b: Optional[np.ndarray]):
        if not self._snapshot_dir:
            self._snapshot_request = False
            return
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self._snapshot_dir / ts
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "a.png"), frame_a)
        if frame_b is not None:
            # Match shapes (B might be a different model on the same source —
            # frames are the same size though).
            if frame_b.shape != frame_a.shape:
                frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))
            cv2.imwrite(str(out / "b.png"), frame_b)
            diff = cv2.absdiff(frame_a, frame_b)
            cv2.imwrite(str(out / "diff.png"), diff)
        self._snapshot_request = False
        self._snapshot_dir = None

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self):
        try:
            from ultralytics import YOLO

            local_a = self.models_dir / self.model_path
            model_a = YOLO(str(local_a) if local_a.exists() else self.model_path)
            model_a.to(_device())

            model_b = None
            if self.model_b_path:
                local_b = self.models_dir / self.model_b_path
                model_b = YOLO(str(local_b) if local_b.exists() else self.model_b_path)
                model_b.to(_device())

            cap = self._open_capture()
            if cap is None:
                self.failed.emit(
                    f"Couldn't connect to {self.source}. "
                    f"Check the URL and network."
                )
                return

            engine = AlertEngine(
                loiter_seconds=self.loiter_seconds,
                wrong_way_dx_threshold=self.wrong_way_dx_threshold,
            )
            roi_tracker = _RoiTracker(self.rois, self.roi_dwell_seconds)

            min_dt = 1.0 / self.max_fps
            last_t = 0.0
            frame_count = 0
            attempts = 0
            fps_ema = 0.0
            infer_ema_a = 0.0
            infer_ema_b = 0.0
            EMA = 0.2
            last_emit_t = time.time()
            last_b_frame: Optional[np.ndarray] = None
            last_b_stats: Optional[dict] = None

            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    # Lost stream — try to reconnect.
                    if not self.reconnect:
                        self.failed.emit("Stream ended or read error")
                        break
                    self.reconnecting.emit(
                        f"Reconnecting to {self.source}… (attempt {attempts + 1})"
                    )
                    try:
                        cap.release()
                    except Exception:
                        pass
                    backoff = min(30.0, 1.5 * (2 ** min(attempts, 5)))
                    deadline = time.time() + backoff
                    while time.time() < deadline:
                        if self._stop:
                            break
                        time.sleep(0.2)
                    if self._stop:
                        break
                    cap = self._open_capture()
                    attempts += 1
                    if cap is None:
                        if self.max_reconnect_attempts and attempts >= self.max_reconnect_attempts:
                            self.failed.emit(
                                f"Lost connection to {self.source} after "
                                f"{attempts} attempts."
                            )
                            return
                        continue
                    self.reconnected.emit()
                    attempts = 0
                    continue

                now = time.time()
                if now - last_t < min_dt:
                    continue
                inst_fps = 1.0 / max(1e-3, now - last_emit_t)
                fps_ema = inst_fps if fps_ema == 0 else (EMA * inst_fps + (1 - EMA) * fps_ema)
                last_emit_t = now
                last_t = now
                frame_count += 1

                # ---- Model A inference ----
                t0 = time.time()
                results_a = model_a.track(
                    source=frame,
                    conf=self.conf,
                    classes=list(VEHICLE_CLASSES.keys()),
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
                infer_ms_a = (time.time() - t0) * 1000.0
                infer_ema_a = infer_ms_a if infer_ema_a == 0 else (
                    EMA * infer_ms_a + (1 - EMA) * infer_ema_a
                )
                detections_a = _detections_from_results(results_a, VEHICLE_CLASSES)

                # ---- Model B inference (every Nth frame) ----
                detections_b: list[dict] = []
                run_b = bool(model_b) and (frame_count % self.b_every_n == 0)
                if run_b:
                    tb0 = time.time()
                    results_b = model_b.track(
                        source=frame,
                        conf=self.conf,
                        classes=list(VEHICLE_CLASSES.keys()),
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False,
                    )
                    infer_ms_b = (time.time() - tb0) * 1000.0
                    infer_ema_b = infer_ms_b if infer_ema_b == 0 else (
                        EMA * infer_ms_b + (1 - EMA) * infer_ema_b
                    )
                    detections_b = _detections_from_results(results_b, VEHICLE_CLASSES)

                # ---- ROI counts + events (model A only — A is the authoritative one) ----
                with self._lock:
                    roi_tracker.rois = self.rois
                roi_counts, roi_events = roi_tracker.update(detections_a, now)

                # ---- Annotate frame A ----
                frame_a = frame.copy()
                for d in detections_a:
                    _draw_box_xyxy(
                        frame_a, d["bbox"], BBOX_BGR,
                        label=f"#{d['track_id']} {d['class']} {d['conf']:.2f}",
                    )
                _draw_rois(frame_a, self.rois, roi_counts)

                # ---- Annotate frame B ----
                frame_b: Optional[np.ndarray] = None
                if model_b is not None:
                    if run_b:
                        frame_b = frame.copy()
                        for d in detections_b:
                            _draw_box_xyxy(
                                frame_b, d["bbox"], BBOX_BGR,
                                label=f"#{d['track_id']} {d['class']} {d['conf']:.2f}",
                            )
                        _draw_rois(frame_b, self.rois, roi_counts)
                        last_b_frame = frame_b
                    elif last_b_frame is not None:
                        frame_b = last_b_frame

                # ---- Alerts (loiter / wrong-way, model A) ----
                new_alerts = engine.update(detections_a, now)
                for a in new_alerts:
                    self.alert.emit({
                        "track_id": a.track_id,
                        "rule": a.rule,
                        "message": a.message,
                        "timestamp": a.timestamp,
                    })
                    cv2.putText(
                        frame_a, f"! {a.rule.upper()}",
                        (10, 30 + 24 * (frame_count % 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_BGR, 2,
                    )

                # ---- ROI events ----
                for ev in roi_events:
                    self.alert.emit(ev)
                    if self.record_on_event:
                        self._event_record_until = max(
                            self._event_record_until,
                            now + self.event_clip_seconds,
                        )
                        with self._lock:
                            if self._record_request is None:
                                self._record_request = True

                # ---- Recording state machine ----
                want_record = False
                with self._lock:
                    req = self._record_request
                if req is True:
                    want_record = True
                elif self.record_on_event and now < self._event_record_until:
                    want_record = True
                elif req is False:
                    want_record = False

                if want_record:
                    self._ensure_writers(frame_a, frame_b)
                else:
                    if self._writer_a is not None:
                        self._close_writers()

                # ---- Overlays (FPS, REC) burnt onto outgoing frames ----
                _draw_fps_overlay(frame_a, fps_ema, infer_ema_a, label="A")
                if want_record:
                    _draw_rec_indicator(frame_a)
                if frame_b is not None:
                    _draw_fps_overlay(frame_b, fps_ema, infer_ema_b or 0.0, label="B")
                    if want_record:
                        _draw_rec_indicator(frame_b)

                # Write to recorders (after overlays — operator gets full UI)
                if self._writer_a is not None:
                    self._writer_a.write(frame_a)
                if self._writer_b is not None and frame_b is not None:
                    self._writer_b.write(frame_b)

                # ---- Snapshot request ----
                with self._lock:
                    snap = self._snapshot_request
                if snap:
                    self._write_snapshot(frame_a, frame_b)

                # ---- Build & emit stats ----
                stats_a = self._build_stats(
                    frame_count, detections_a, roi_counts,
                    fps_ema, infer_ema_a, "A",
                )
                self._last_frame_a = frame_a
                self.frame_ready.emit(bgr_to_qimage(frame_a), stats_a)

                if frame_b is not None:
                    stats_b = self._build_stats(
                        frame_count, detections_b, roi_counts,
                        fps_ema, infer_ema_b, "B",
                    )
                    self._last_frame_b = frame_b
                    self.frame_ready_b.emit(bgr_to_qimage(frame_b), stats_b)

            try:
                cap.release()
            except Exception:
                pass
            if self._writer_a is not None:
                self._close_writers()
            self.stopped.emit()

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.failed.emit(str(e))

    # ------------------------------------------------------------------
    # Stats helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_stats(
        frame_count: int,
        detections: list[dict],
        roi_counts: dict,
        fps: float,
        infer_ms: float,
        model_label: str,
    ) -> dict:
        by_class: dict[str, int] = {}
        for d in detections:
            by_class[d["class"]] = by_class.get(d["class"], 0) + 1
        return {
            "frame": frame_count,
            "detections": len(detections),
            "by_class": by_class,
            "roi_counts": dict(roi_counts),
            "fps": fps,
            "inference_ms": infer_ms,
            "model": model_label,
        }
