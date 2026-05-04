"""
Live video source — RTSP / webcam / file with YOLO detection + tracking
in a Qt-friendly worker.

Emits annotated frames as QImage so they can be displayed without
synchronization headaches.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from cctv_yolo.alerts import AlertEngine, Alert


VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
CLASS_COLORS = {
    "car": (128, 255, 0),
    "truck": (0, 128, 255),
    "bus": (255, 128, 0),
    "motorcycle": (0, 255, 255),
    "bicycle": (255, 0, 128),
}


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
    return QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)


class LiveStreamWorker(QThread):
    """Pull frames from any cv2.VideoCapture source, run detection,
    emit annotated frames + alerts."""

    frame_ready = Signal(QImage, dict)  # QImage, stats
    alert = Signal(dict)                # serialized Alert
    failed = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        source: str,
        model_path: str,
        models_dir: Path,
        conf: float = 0.3,
        loiter_seconds: float = 30.0,
        wrong_way_dx_threshold: float = -10.0,
        max_fps: int = 15,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.model_path = model_path
        self.models_dir = Path(models_dir)
        self.conf = conf
        self.loiter_seconds = loiter_seconds
        self.wrong_way_dx_threshold = wrong_way_dx_threshold
        self.max_fps = max(1, int(max_fps))
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from ultralytics import YOLO

            local = self.models_dir / self.model_path
            model = YOLO(str(local) if local.exists() else self.model_path)
            model.to(_device())

            # OpenCV accepts both URLs and ints for webcams
            cap_source = int(self.source) if self.source.isdigit() else self.source
            cap = cv2.VideoCapture(cap_source)
            if not cap.isOpened():
                self.failed.emit(f"Cannot open source: {self.source}")
                return

            engine = AlertEngine(
                loiter_seconds=self.loiter_seconds,
                wrong_way_dx_threshold=self.wrong_way_dx_threshold,
            )

            min_dt = 1.0 / self.max_fps
            last_t = 0.0
            frame_count = 0

            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    self.failed.emit("Stream ended or read error")
                    break

                now = time.time()
                if now - last_t < min_dt:
                    # Throttle
                    continue
                last_t = now
                frame_count += 1

                # Track on this frame
                results = model.track(
                    source=frame,
                    conf=self.conf,
                    classes=list(VEHICLE_CLASSES.keys()),
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )

                detections = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy().tolist()
                        cid = int(boxes.cls[i].item())
                        cname = VEHICLE_CLASSES.get(cid, "unknown")
                        conf = float(boxes.conf[i].item())
                        tid = int(boxes.id[i].item()) if boxes.id is not None else -1
                        detections.append({
                            "track_id": tid,
                            "class": cname,
                            "conf": conf,
                            "bbox": bbox,
                        })

                # Draw
                for d in detections:
                    x1, y1, x2, y2 = [int(c) for c in d["bbox"]]
                    color = CLASS_COLORS.get(d["class"], (200, 200, 200))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"#{d['track_id']} {d['class']} {d['conf']:.2f}"
                    cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Alerts
                new_alerts = engine.update(detections, now)
                for a in new_alerts:
                    self.alert.emit({
                        "track_id": a.track_id,
                        "rule": a.rule,
                        "message": a.message,
                        "timestamp": a.timestamp,
                    })
                    # Burn alert label on frame for visibility
                    cv2.putText(frame, f"! {a.rule.upper()}",
                                (10, 30 + 24 * (frame_count % 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                stats = {
                    "frame": frame_count,
                    "detections": len(detections),
                    "by_class": {},
                }
                for d in detections:
                    stats["by_class"][d["class"]] = stats["by_class"].get(d["class"], 0) + 1

                self.frame_ready.emit(bgr_to_qimage(frame), stats)

            cap.release()
            self.stopped.emit()
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.failed.emit(str(e))
