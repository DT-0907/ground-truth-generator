"""Single source of truth for runtime data paths.

The app stores all runtime data (videos, tracks, corrections, exports,
models, config, logs) in **the same folder as the app** — so the install
is portable: copy the folder, all your work comes with it. No more
`~/Documents/CCTV-YOLO/` separate-folder confusion.

Resolution order (first writable candidate wins):

    1. $CCTV_YOLO_DATA_DIR env var          — explicit override
    2. If frozen (PyInstaller):
        - Windows / Linux : folder containing CCTV-YOLO.exe
        - macOS .app      : folder containing the .app bundle
    3. Running from source : repo root (the parent of software2/)
    4. ~/Documents/cctv-yolo/               — last-resort fallback when
                                              the install location is
                                              read-only (e.g. installed
                                              under C:\\Program Files\\)

The choice is cached on first call.
"""
from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_writable(p: Path) -> bool:
    """True if we can create p and write a probe file inside it."""
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except (OSError, PermissionError):
        return False


def _candidate_roots() -> list[Path]:
    """Resolve candidate data roots, in priority order."""
    candidates: list[Path] = []

    # 1. Environment override
    env = os.environ.get("CCTV_YOLO_DATA_DIR")
    if env:
        candidates.append(Path(env).expanduser().resolve())
        # Env var is explicit — honor it even if write fails later.
        return candidates

    # 2. Frozen (PyInstaller) build
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).resolve()
        if sys.platform == "darwin":
            # macOS .app: walk up to find the .app, use its parent dir
            for parent in exe.parents:
                if parent.name.endswith(".app"):
                    candidates.append(parent.parent)
                    break
            else:
                candidates.append(exe.parent)
        else:
            # Windows / Linux: data lives next to the executable
            candidates.append(exe.parent)
    else:
        # 3. Source mode: this file is at software2/cctv_yolo/paths.py
        #    parent.parent.parent = the repo root (e.g. cctv-yolo/).
        candidates.append(Path(__file__).resolve().parent.parent.parent)

    # 4. Last-resort fallback
    fallback = Path.home() / "Documents" / "cctv-yolo"
    if fallback not in candidates:
        candidates.append(fallback)
    return candidates


@lru_cache(maxsize=1)
def get_data_root() -> Path:
    """Return the data root. Cached after first call.

    Picks the first writable candidate. Logs which one was chosen so the
    user can find their data via Help → Show Log Folder.
    """
    cands = _candidate_roots()
    for c in cands:
        if _is_writable(c):
            logger.info("Data root resolved to: %s", c)
            return c
    # No candidate is writable — return the first one anyway so callers
    # get a deterministic path and the OSError fires loudly downstream.
    logger.warning(
        "No writable data-root candidate; falling back to %s (writes may fail).",
        cands[0],
    )
    return cands[0]


# Convenience accessors — every consumer should go through these so the
# choice in get_data_root() is honored everywhere.

def get_data_dir() -> Path:        return get_data_root() / "data"
def get_videos_dir() -> Path:      return get_data_dir() / "videos"
def get_tracks_dir() -> Path:      return get_data_dir() / "tracks"
def get_corrections_dir() -> Path: return get_data_dir() / "corrections"
def get_exports_dir() -> Path:     return get_data_dir() / "exports"
def get_config_dir() -> Path:      return get_data_root() / "config"
def get_models_dir() -> Path:      return get_data_root() / "models"
def get_logs_dir() -> Path:        return get_data_root() / "logs"

def get_log_file() -> Path:        return get_logs_dir() / "app.log"
def get_first_run_marker() -> Path: return get_data_root() / ".first_run_complete"
def get_crash_log() -> Path:       return get_data_root() / "crash.log"
