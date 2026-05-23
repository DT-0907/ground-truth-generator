"""Single source of truth for runtime data paths.

Goal: the `cctv-yolo/` data folder is portable. Move it around (Documents
→ Desktop → external drive), and the app finds it on next launch.

The installed binary (.exe / .app) lives wherever the OS conventions say
(`/Applications`, `C:\\Program Files\\`, …) — data does NOT live there.
Data always lives in a folder named `cctv-yolo` (or whatever the user
configures) somewhere the user can reach.

Resolution order:

    1. $CCTV_YOLO_DATA_DIR env var          — explicit override
    2. Remembered path from per-user config — set the last time the app
                                              picked a root (so a one-time
                                              search becomes a permanent
                                              choice)
    3. Source mode only — repo root         — when running `python run.py`
                                              from the working tree, use
                                              the repo itself
    4. First existing folder named one of:  — search candidates so a moved
        - ~/Documents/cctv-yolo                folder is auto-detected
        - ~/Desktop/cctv-yolo
        - ~/cctv-yolo
        - ~/Documents/CCTV-YOLO  (legacy)
    5. Default: create ~/Documents/cctv-yolo/

Per-user config (the "remembered path") lives in the OS-standard app
config location, NOT in the data folder itself — that way the marker
survives even if the data folder is moved or deleted.

    macOS    : ~/Library/Application Support/CCTV-YOLO/data_root.txt
    Windows  : %APPDATA%\\CCTV-YOLO\\data_root.txt
    Linux    : ~/.config/CCTV-YOLO/data_root.txt

If the user moves the data folder, they can update the remembered path
either by:
    - setting CCTV_YOLO_DATA_DIR before launching, or
    - deleting the marker file (the next launch re-searches and saves
      whatever it finds), or
    - eventually, via Settings → Data Folder (UI not yet wired).
"""
from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


# Folder names we'll search for, in priority order.
_FOLDER_NAMES = ("cctv-yolo", "CCTV-YOLO")


def _settings_dir() -> Path:
    """Per-user app config directory (OS-standard)."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "CCTV-YOLO"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "CCTV-YOLO"
        return Path.home() / "AppData" / "Roaming" / "CCTV-YOLO"
    # Linux / other
    return Path.home() / ".config" / "CCTV-YOLO"


def _remembered_marker() -> Path:
    return _settings_dir() / "data_root.txt"


def _read_remembered() -> Path | None:
    p = _remembered_marker()
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    return Path(text).expanduser()


def _write_remembered(path: Path) -> None:
    """Persist the chosen root so next launch picks the same location."""
    try:
        d = _settings_dir()
        d.mkdir(parents=True, exist_ok=True)
        _remembered_marker().write_text(str(path), encoding="utf-8")
    except OSError as e:
        logger.warning("Couldn't save remembered data root to %s: %s",
                       _remembered_marker(), e)


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


def _search_candidates() -> list[Path]:
    """Folders we look at to auto-detect an existing cctv-yolo data dir."""
    home = Path.home()
    out: list[Path] = []
    for name in _FOLDER_NAMES:
        out.append(home / "Documents" / name)
        out.append(home / "Desktop" / name)
        out.append(home / name)
    return out


def _looks_like_data_root(p: Path) -> bool:
    """A folder is treated as an existing data root if any of the standard
    runtime subfolders are present. We don't require all of them — the
    first launch creates them.
    """
    if not p.is_dir():
        return False
    markers = ("data", "models", "config", ".first_run_complete", ".cctv-yolo")
    return any((p / m).exists() for m in markers)


@lru_cache(maxsize=1)
def get_data_root() -> Path:
    """Return the data root. Cached after first call."""
    # 1. Env override always wins
    env = os.environ.get("CCTV_YOLO_DATA_DIR")
    if env:
        root = Path(env).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        logger.info("Data root resolved from $CCTV_YOLO_DATA_DIR: %s", root)
        return root

    # 2. Remembered path from previous launch — but only if it still exists
    #    (the user may have moved the folder).
    remembered = _read_remembered()
    if remembered is not None and remembered.is_dir():
        logger.info("Data root resolved from remembered marker: %s", remembered)
        return remembered.resolve()

    # 3. Dev mode: when running from source, the repo root is the natural
    #    data folder. paths.py lives at software2/cctv_yolo/paths.py, so
    #    going up three levels gives the repo root.
    if not getattr(sys, "frozen", False):
        root = Path(__file__).resolve().parent.parent.parent
        root.mkdir(parents=True, exist_ok=True)
        _write_remembered(root)
        logger.info("Data root resolved to repo root (dev mode): %s", root)
        return root

    # 4. Frozen build: search common locations for an existing cctv-yolo
    #    folder. First hit wins.
    for cand in _search_candidates():
        if _looks_like_data_root(cand) and _is_writable(cand):
            cand = cand.resolve()
            _write_remembered(cand)
            logger.info("Data root auto-detected at: %s", cand)
            return cand

    # 5. No existing folder found — create the default location.
    default = Path.home() / "Documents" / "cctv-yolo"
    default.mkdir(parents=True, exist_ok=True)
    if _is_writable(default):
        default = default.resolve()
        _write_remembered(default)
        logger.info("Data root created at default location: %s", default)
        return default

    # 6. Last-resort fallback (can't write to ~/Documents for some reason)
    fallback = Path.home() / "cctv-yolo"
    fallback.mkdir(parents=True, exist_ok=True)
    logger.warning("Default location unwritable; falling back to %s", fallback)
    return fallback.resolve()


def set_data_root(path: Path | str) -> Path:
    """Override the data root and persist the choice.

    Call this from a Settings UI when the user picks a new location.
    Clears the lru_cache so subsequent get_data_root() calls see the change.
    """
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    _write_remembered(p)
    get_data_root.cache_clear()
    logger.info("Data root explicitly set to: %s", p)
    return p


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
