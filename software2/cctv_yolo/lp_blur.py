"""
License-plate blur — runs an optional plate detector on vehicle crops
and blurs the matched regions in-place.

Drop a plate-detection .pt into the app's models/ folder (resolved via
``cctv_yolo.paths.get_models_dir()``) named ``license_plate.pt`` (or set
the CCTV_YOLO_LP_MODEL env var to point at any path) to enable.
The blurrer is a graceful no-op if the model can't be loaded.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_LP_MODEL = "license_plate.pt"
LP_MODEL_ENV = "CCTV_YOLO_LP_MODEL"


class LicensePlateBlurrer:
    """Detect plates inside vehicle bboxes and apply a heavy gaussian blur.

    Falls back to blurring the bottom 25% of each vehicle bbox if the
    detector model can't be loaded — better than nothing for privacy.
    """

    def __init__(self, model_path: str | Path | None = None):
        self.model = None
        self._fallback = True

        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        if os.environ.get(LP_MODEL_ENV):
            candidates.append(Path(os.environ[LP_MODEL_ENV]))
        from cctv_yolo.paths import get_models_dir
        candidates.append(get_models_dir() / DEFAULT_LP_MODEL)

        for p in candidates:
            if p.exists():
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(str(p))
                    self._fallback = False
                    print(f"[LP] Loaded plate detector: {p}")
                    break
                except Exception as e:
                    print(f"[LP] Failed to load plate detector {p}: {e}")

        if self._fallback:
            print("[LP] Using bottom-25% fallback blur (no plate model found)")

    def blur(self, frame: np.ndarray, vehicle_bboxes: Iterable[list]) -> int:
        """Apply blur in-place. Returns number of regions blurred."""
        h, w = frame.shape[:2]
        n = 0
        for bbox in vehicle_bboxes:
            x1, y1, x2, y2 = [int(round(c)) for c in bbox]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if self.model is not None:
                try:
                    results = self.model.predict(
                        source=crop, conf=0.35, verbose=False, imgsz=320
                    )
                    if not results:
                        continue
                    for r in results:
                        if r.boxes is None:
                            continue
                        for box in r.boxes.xyxy.cpu().numpy().tolist():
                            px1, py1, px2, py2 = [int(round(c)) for c in box]
                            self._blur_region(frame, x1 + px1, y1 + py1,
                                              x1 + px2, y1 + py2)
                            n += 1
                except Exception as e:
                    print(f"[LP] inference error: {e}")
            else:
                # Fallback — blur bottom 25% of the vehicle bbox
                bh = y2 - y1
                py1 = y1 + int(bh * 0.55)
                py2 = y2
                self._blur_region(frame, x1, py1, x2, py2)
                n += 1
        return n

    @staticmethod
    def _blur_region(frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return
        region = frame[y1:y2, x1:x2]
        # Heavy blur — kernel needs to be odd
        k = max(15, min(75, ((max(x2 - x1, y2 - y1) // 4) | 1)))
        if k % 2 == 0:
            k += 1
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
