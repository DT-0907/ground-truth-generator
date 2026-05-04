"""
Model comparison — run two YOLO models on the same video and produce
a diff: counts, mean confidence, per-class breakdown, side-by-side
visualization.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal


VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}


def _device():
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_model_track(
    video_path: Path,
    model_path: str,
    conf: float,
    models_dir: Path,
    progress_callback=None,
    stride: int = 1,
) -> dict:
    """Run YOLO detect+track on a video. Returns track summary stats
    suitable for comparison."""
    from ultralytics import YOLO

    local = Path(models_dir) / model_path
    model = YOLO(str(local) if local.exists() else model_path)
    model.to(_device())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    results = model.track(
        source=str(video_path),
        conf=conf,
        classes=list(VEHICLE_CLASSES.keys()),
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
        vid_stride=stride,
    )

    tracks = defaultdict(lambda: {"frames": 0, "confs": [], "class": None})
    last_pct = -1

    for frame_idx, result in enumerate(results):
        if progress_callback and total_frames > 0:
            pct = int((frame_idx + 1) / (total_frames / max(1, stride)) * 100)
            pct = min(pct, 99)
            if pct != last_pct:
                progress_callback(pct)
                last_pct = pct

        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes
        if boxes.id is None:
            continue
        for i in range(len(boxes)):
            tid = int(boxes.id[i].item())
            conf_v = float(boxes.conf[i].item())
            cid = int(boxes.cls[i].item())
            cname = VEHICLE_CLASSES.get(cid, "unknown")
            tracks[tid]["frames"] += 1
            tracks[tid]["confs"].append(conf_v)
            tracks[tid]["class"] = cname

    stats = {
        "total_tracks": len(tracks),
        "by_class": defaultdict(int),
        "mean_conf": 0.0,
        "median_track_length": 0.0,
        "total_detections": 0,
    }
    confs = []
    lengths = []
    for tid, t in tracks.items():
        stats["by_class"][t["class"]] += 1
        confs.extend(t["confs"])
        lengths.append(t["frames"])
        stats["total_detections"] += t["frames"]

    if confs:
        stats["mean_conf"] = round(sum(confs) / len(confs), 3)
    if lengths:
        lengths.sort()
        stats["median_track_length"] = lengths[len(lengths) // 2]
    stats["by_class"] = dict(stats["by_class"])
    return stats


def diff_stats(a: dict, b: dict) -> dict:
    """Diff two stats dicts. Returns delta (b - a)."""
    delta = {
        "total_tracks": b["total_tracks"] - a["total_tracks"],
        "mean_conf": round(b["mean_conf"] - a["mean_conf"], 3),
        "total_detections": b["total_detections"] - a["total_detections"],
        "median_track_length": b["median_track_length"] - a["median_track_length"],
        "by_class": {},
    }
    classes = set(a["by_class"].keys()) | set(b["by_class"].keys())
    for c in sorted(classes):
        delta["by_class"][c] = b["by_class"].get(c, 0) - a["by_class"].get(c, 0)
    return delta


class ModelCompareWorker(QThread):
    """Run two models sequentially and report the diff."""

    log_line = Signal(str)
    progress = Signal(str, int)  # which model (A/B), pct
    finished_ok = Signal(dict)   # {"a": stats, "b": stats, "delta": stats}
    failed = Signal(str)

    def __init__(
        self,
        video_path: Path,
        model_a: str,
        model_b: str,
        models_dir: Path,
        conf: float = 0.25,
        stride: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.model_a = model_a
        self.model_b = model_b
        self.models_dir = Path(models_dir)
        self.conf = conf
        self.stride = stride

    def run(self):
        try:
            self.log_line.emit(f"[A] running {self.model_a}…")
            stats_a = run_model_track(
                self.video_path, self.model_a, self.conf, self.models_dir,
                progress_callback=lambda p: self.progress.emit("A", p),
                stride=self.stride,
            )
            self.log_line.emit(f"[A] tracks={stats_a['total_tracks']} "
                               f"mean_conf={stats_a['mean_conf']}")
            self.log_line.emit(f"[B] running {self.model_b}…")
            stats_b = run_model_track(
                self.video_path, self.model_b, self.conf, self.models_dir,
                progress_callback=lambda p: self.progress.emit("B", p),
                stride=self.stride,
            )
            self.log_line.emit(f"[B] tracks={stats_b['total_tracks']} "
                               f"mean_conf={stats_b['mean_conf']}")
            self.finished_ok.emit({
                "a": stats_a, "b": stats_b, "delta": diff_stats(stats_a, stats_b),
                "model_a": self.model_a, "model_b": self.model_b,
            })
        except Exception as e:
            import traceback
            self.log_line.emit(traceback.format_exc())
            self.failed.emit(str(e))
