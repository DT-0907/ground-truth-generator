"""
Batch processing queue with persistent state, watch-folder ingest,
priority/scheduling, and resume-on-crash.

Persistent queue file: {data_root}/config/batch_queue.json
Schema (per item):
    {
      "session_id": str,
      "video_path": str,        # absolute, may be NAS-relative
      "model": str,             # YOLO model filename
      "conf": float,
      "priority": int,          # higher runs first
      "scheduled_at": str|null, # ISO datetime, null = run ASAP
      "added_at": str,
      "status": "queued"|"processing"|"done"|"error"|"paused",
      "progress": int,          # 0-100
      "error": str|null,
      "processing_roi": dict|null,
    }
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal


# ---------------------------------------------------------------------------
# Persistent queue file
# ---------------------------------------------------------------------------

QUEUE_FILENAME = "batch_queue.json"


class BatchQueueStore:
    """File-backed queue store. Pure data layer, no Qt."""

    def __init__(self, config_dir: Path):
        self.path = Path(config_dir) / QUEUE_FILENAME
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r") as f:
                items = json.load(f)
            if not isinstance(items, list):
                return []
            return items
        except (json.JSONDecodeError, OSError):
            return []

    def save(self, items: list[dict]):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(items, f, indent=2)
        tmp.replace(self.path)

    def reset_in_progress_to_queued(self):
        """Resume-on-crash: anything that was processing when the app
        died gets put back in the queue."""
        items = self.load()
        changed = False
        for it in items:
            if it.get("status") == "processing":
                it["status"] = "queued"
                it["progress"] = 0
                changed = True
        if changed:
            self.save(items)
        return items


# ---------------------------------------------------------------------------
# Watch folder
# ---------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


class WatchFolderWorker(QThread):
    """Polls a folder periodically and emits *new_video* signals
    when previously-unseen video files appear.

    Signals
    -------
    new_video(video_path: str)
    """

    new_video = Signal(str)

    def __init__(self, folder: Path, poll_seconds: int = 5, parent=None):
        super().__init__(parent)
        self.folder = Path(folder)
        self.poll_seconds = max(1, int(poll_seconds))
        self._stop = False
        self._seen: set[str] = set()
        # Seed with files already present so existing videos aren't
        # re-queued every restart.
        if self.folder.exists():
            for p in self.folder.rglob("*"):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    self._seen.add(str(p.resolve()))

    def stop(self):
        self._stop = True

    def run(self):
        while not self._stop:
            try:
                if self.folder.exists():
                    for p in self.folder.rglob("*"):
                        if not p.is_file():
                            continue
                        if p.suffix.lower() not in VIDEO_EXTS:
                            continue
                        key = str(p.resolve())
                        if key in self._seen:
                            continue
                        # Wait for the file to stop growing (likely still
                        # being copied if size changes within 1s).
                        try:
                            s1 = p.stat().st_size
                            time.sleep(0.8)
                            s2 = p.stat().st_size
                            if s1 != s2:
                                continue  # try again next poll
                        except OSError:
                            continue
                        self._seen.add(key)
                        self.new_video.emit(key)
            except Exception as e:
                # Keep watcher alive even on transient errors
                print(f"[WatchFolderWorker] error: {e}")
            for _ in range(self.poll_seconds * 2):
                if self._stop:
                    return
                time.sleep(0.5)


# ---------------------------------------------------------------------------
# Queue manager (Qt-aware orchestrator)
# ---------------------------------------------------------------------------

class BatchQueueManager(QObject):
    """Coordinates the persistent queue, watch folders, and the
    ProcessingWorker.

    Signals
    -------
    queue_changed()
        Emitted whenever the queue contents change (add/remove/status).
    item_progress(session_id: str, percent: int)
    item_finished(session_id: str)
    item_error(session_id: str, error_message: str)
    """

    queue_changed = Signal()
    item_progress = Signal(str, int)
    item_finished = Signal(str)
    item_error = Signal(str, str)

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.store = BatchQueueStore(data_manager.config_dir)
        self.items: list[dict] = self.store.reset_in_progress_to_queued()
        self._current_worker = None
        self._current_session: Optional[str] = None
        self._paused = False
        self._watcher: Optional[WatchFolderWorker] = None

    # ----- Persistence -----
    def _persist(self):
        self.store.save(self.items)
        self.queue_changed.emit()

    # ----- Public API -----
    def add(
        self,
        video_path: str,
        model: str = "yolov8m.pt",
        conf: float = 0.25,
        priority: int = 0,
        scheduled_at: Optional[str] = None,
        processing_roi: Optional[dict] = None,
    ) -> Optional[str]:
        """Add a video to the queue. Returns the session_id, or None if
        the video doesn't exist or is already queued/processing."""
        p = Path(video_path)
        if not p.exists():
            return None

        # Build a session_id consistent with DataManager
        if self.dm.active_mode == "nas":
            try:
                session_id = self.dm.build_session_id(p, self.dm.videos_dir)
            except ValueError:
                session_id = p.stem
        else:
            session_id = p.stem

        # Skip if already in the queue and not finished
        for it in self.items:
            if it["session_id"] == session_id and it["status"] in ("queued", "processing", "paused"):
                return None

        item = {
            "session_id": session_id,
            "video_path": str(p),
            "model": model,
            "conf": conf,
            "priority": int(priority),
            "scheduled_at": scheduled_at,
            "added_at": datetime.now().isoformat(timespec="seconds"),
            "status": "queued",
            "progress": 0,
            "error": None,
            "processing_roi": processing_roi,
        }
        self.items.append(item)
        self._persist()
        return session_id

    def remove(self, session_id: str):
        if self._current_session == session_id:
            return  # cannot remove what's running
        self.items = [it for it in self.items if it["session_id"] != session_id]
        self._persist()

    def clear_finished(self):
        self.items = [it for it in self.items if it["status"] in ("queued", "processing", "paused")]
        self._persist()

    def set_priority(self, session_id: str, priority: int):
        for it in self.items:
            if it["session_id"] == session_id and it["status"] == "queued":
                it["priority"] = int(priority)
        self._persist()

    def pause_all(self):
        self._paused = True
        for it in self.items:
            if it["status"] == "queued":
                it["status"] = "paused"
        self._persist()

    def resume_all(self):
        self._paused = False
        for it in self.items:
            if it["status"] == "paused":
                it["status"] = "queued"
        self._persist()
        self.start_next_if_idle()

    def get_items(self) -> list[dict]:
        return list(self.items)

    # ----- Watch folder -----
    def start_watch_folder(self, folder: Path, model: str, conf: float, poll_seconds: int = 5):
        self.stop_watch_folder()
        self._watcher = WatchFolderWorker(folder, poll_seconds)
        self._watcher.new_video.connect(
            lambda path, m=model, c=conf: self._on_watch_new_video(path, m, c)
        )
        self._watcher.start()

    def stop_watch_folder(self):
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher.wait(2000)
            self._watcher = None

    def is_watching(self) -> bool:
        return self._watcher is not None and self._watcher.isRunning()

    def _on_watch_new_video(self, path: str, model: str, conf: float):
        self.add(path, model=model, conf=conf, priority=0)
        self.start_next_if_idle()

    # ----- Scheduler / runner -----
    def _next_runnable(self) -> Optional[dict]:
        """Pick the next queued item, respecting priority and
        ``scheduled_at`` (don't run anything scheduled in the future)."""
        if self._paused:
            return None
        now = datetime.now()
        candidates = []
        for it in self.items:
            if it["status"] != "queued":
                continue
            sched = it.get("scheduled_at")
            if sched:
                try:
                    s_dt = datetime.fromisoformat(sched)
                    if s_dt > now:
                        continue
                except ValueError:
                    pass
            candidates.append(it)
        if not candidates:
            return None
        # Higher priority first, then oldest (FIFO)
        candidates.sort(key=lambda it: (-it.get("priority", 0), it["added_at"]))
        return candidates[0]

    def start_next_if_idle(self):
        if self._current_worker is not None:
            return
        item = self._next_runnable()
        if item is None:
            return
        self._launch(item)

    def _launch(self, item: dict):
        from cctv_yolo.processing import ProcessingWorker

        item["status"] = "processing"
        item["progress"] = 0
        item["error"] = None
        self._current_session = item["session_id"]
        self._persist()

        worker = ProcessingWorker(
            video_path=item["video_path"],
            tracks_dir=str(self.dm.tracks_dir),
            model=item["model"],
            conf=item["conf"],
            session_id=item["session_id"],
            models_dir=str(self.dm.models_dir),
            processing_roi=item.get("processing_roi"),
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        self._current_worker = worker
        worker.start()

    def _on_progress(self, session_id: str, pct: int):
        for it in self.items:
            if it["session_id"] == session_id:
                it["progress"] = int(pct)
                break
        # No persist on every tick — too chatty. Just notify UI.
        self.item_progress.emit(session_id, int(pct))

    def _on_finished(self, session_id: str):
        for it in self.items:
            if it["session_id"] == session_id:
                it["status"] = "done"
                it["progress"] = 100
                break
        self._current_worker = None
        self._current_session = None
        self._persist()
        self.item_finished.emit(session_id)
        self.start_next_if_idle()

    def _on_error(self, session_id: str, error_message: str):
        for it in self.items:
            if it["session_id"] == session_id:
                it["status"] = "error"
                it["error"] = error_message[:500]
                break
        self._current_worker = None
        self._current_session = None
        self._persist()
        self.item_error.emit(session_id, error_message)
        # Continue processing the rest of the queue
        self.start_next_if_idle()

    def shutdown(self):
        """Best-effort shutdown for app close."""
        self.stop_watch_folder()
        if self._current_worker is not None:
            # Mark as paused so resume-on-crash logic puts it back as queued
            for it in self.items:
                if it["session_id"] == self._current_session:
                    it["status"] = "queued"
                    break
            self._persist()
