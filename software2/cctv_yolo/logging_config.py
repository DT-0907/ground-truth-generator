"""Central logging configuration (PRD C4).

Configures the root logger ONCE at app startup with:
- A rotating file handler at ~/Documents/CCTV-YOLO/logs/app.log (5 MB × 5)
- A stream handler that prints to stdout (visible in Windows console=True
  build + during dev runs)

Every module gets a logger via ``logging.getLogger(__name__)``. No more
``print()`` for errors.

The log folder is exposed via Help → Show Log Folder (wired in main_window).
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
BACKUP_COUNT = 5              # keep 5 rotated copies

_configured = False


def configure_logging(data_root: Path | None = None, *, level: int = logging.INFO) -> Path:
    """Set up the root logger. Returns the log file path so callers can show
    it in About dialog / Help menu.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _configured

    # Resolve the log directory. Default = ~/Documents/CCTV-YOLO/logs/
    if data_root is None:
        data_root = Path.home() / "Documents" / "CCTV-YOLO"
    log_dir = Path(data_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    if _configured:
        return log_file

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Rotating file handler — survives crashes, capped at 25 MB total
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Stream handler — visible in dev runs + Windows console=True frozen exe
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    root = logging.getLogger()
    # Reset any handlers Qt or third-parties may have installed
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root.setLevel(level)

    # Silence chatty third-party loggers we don't care about
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).info("Logging configured. File=%s", log_file)
    return log_file


def get_log_file_path(data_root: Path | None = None) -> Path:
    """Return the path to app.log without configuring logging (for menu links)."""
    if data_root is None:
        data_root = Path.home() / "Documents" / "CCTV-YOLO"
    return Path(data_root) / "logs" / "app.log"
