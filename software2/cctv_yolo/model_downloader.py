"""
Async YOLOv8 model downloader (PRD C1 / K2-6).

Ultralytics' ``YOLO()`` constructor auto-fetches the requested ``.pt`` from
GitHub releases on first use. The file lands in one of a handful of platform
locations — we probe each, then copy it into the user's models folder so the
rest of the app can find it predictably.

Cross-platform cache locations:
- macOS  : ``~/.config/Ultralytics/`` or cwd
- Windows: ``%APPDATA%/Ultralytics/`` or cwd
- Linux  : ``~/.config/Ultralytics/`` or cwd
"""
from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal


# Display label → (filename, approximate MB, blurb)
YOLO_VARIANTS: list[tuple[str, str, int, str]] = [
    ("yolov8n.pt", "yolov8n.pt", 6,   "fastest"),
    ("yolov8s.pt", "yolov8s.pt", 22,  "balanced"),
    ("yolov8m.pt", "yolov8m.pt", 52,  "recommended"),
    ("yolov8l.pt", "yolov8l.pt", 87,  "accurate"),
    ("yolov8x.pt", "yolov8x.pt", 136, "most accurate"),
]


def variant_labels() -> list[str]:
    """Human-readable picker entries: ``yolov8n.pt  (6 MB · fastest)``"""
    return [f"{name}  ({mb} MB · {blurb})" for name, _, mb, blurb in YOLO_VARIANTS]


def variant_from_label(label: str) -> str:
    """Strip the size hint back down to the bare filename."""
    return label.split()[0]


def candidate_cache_paths(model_name: str) -> list[Path]:
    """Every place Ultralytics might have dropped the freshly-downloaded .pt."""
    cands: list[Path] = []
    cands.append(Path.cwd() / model_name)
    cands.append(Path.home() / ".config" / "Ultralytics" / model_name)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            cands.append(Path(appdata) / "Ultralytics" / model_name)
    return cands


class ModelDownloadWorker(QThread):
    """Download a YOLOv8 variant and copy it into ``dest_dir``."""

    done = Signal(str)    # path to downloaded model
    failed = Signal(str)

    def __init__(self, model_name: str, dest_dir: Path, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.dest_dir = Path(dest_dir)

    def run(self):
        try:
            from ultralytics import YOLO

            # First-time construction triggers the download.
            m = YOLO(self.model_name)

            candidates: list[Path] = []
            if hasattr(m, "ckpt_path") and m.ckpt_path:
                candidates.append(Path(m.ckpt_path))
            candidates.extend(candidate_cache_paths(self.model_name))

            src = next((c for c in candidates if c.exists()), None)
            if src is None:
                self.failed.emit(
                    f"Downloaded {self.model_name} but couldn't locate the file. "
                    f"Tried: {', '.join(str(c) for c in candidates)}"
                )
                return

            self.dest_dir.mkdir(parents=True, exist_ok=True)
            dest = self.dest_dir / self.model_name
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            self.done.emit(str(dest))
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))
