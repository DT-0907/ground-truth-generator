"""
Batch processing queue with persistent state, parallel scheduling,
watch-folder ingest, and resume-on-crash.

Persistent queue file: {data_root}/config/batch_queue.json
Schema (per item):
    {
      "session_id": str,
      "video_path": str,        # absolute path on disk
      "source_folder": str|None,# top-level folder this video was discovered under
                                #   (used to compute the hash-based session_id)
      "model": str,             # YOLO model filename
      "conf": float,
      "priority": int,          # higher runs first
      "scheduled_at": str|null, # ISO datetime, null = run ASAP
      "added_at": str,
      "status": "queued"|"processing"|"done"|"error"|"paused"|"cancelled",
      "progress": int,          # 0-100
      "error": str|null,
      "processing_roi": dict|null,
    }
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, QThreadPool, Signal


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
        # PRD C3: route through DataManager's atomic helper (fsync + os.replace).
        from cctv_yolo.data_manager import _atomic_write_json
        _atomic_write_json(self.path, items)

    def reset_in_progress_to_queued(self, tracks_dir: Path | None = None):
        """Resume-on-crash: anything that was processing when the app
        died gets put back in the queue.

        PRD E5-11: also wipe any partial ``tracks/<sid>.json`` so the next
        run starts from a clean slate (a half-written tracks file would
        otherwise confuse the rest of the app).
        """
        items = self.load()
        changed = False
        for it in items:
            if it.get("status") == "processing":
                it["status"] = "queued"
                it["progress"] = 0
                changed = True
                if tracks_dir is not None:
                    sid = it.get("session_id")
                    if sid:
                        partial = Path(tracks_dir) / f"{sid}.json"
                        try:
                            partial.unlink(missing_ok=True)
                        except OSError:
                            pass
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
                        try:
                            s1 = p.stat().st_size
                            time.sleep(0.8)
                            s2 = p.stat().st_size
                            if s1 != s2:
                                continue
                        except OSError:
                            continue
                        self._seen.add(key)
                        self.new_video.emit(key)
            except Exception as e:
                print(f"[WatchFolderWorker] error: {e}")
            for _ in range(self.poll_seconds * 2):
                if self._stop:
                    return
                time.sleep(0.5)


# ---------------------------------------------------------------------------
# Queue manager (Qt-aware orchestrator with QThreadPool)
# ---------------------------------------------------------------------------

class BatchQueueManager(QObject):
    """Coordinates the persistent queue, watch folders, and parallel processing.

    PRD E5-4: jobs run as ``ProcessingRunnable`` instances inside a
    ``QThreadPool`` whose max thread count is user-controlled (1–100).

    Signals
    -------
    queue_changed()
        Emitted whenever the queue contents change (add/remove/status).
    item_progress(session_id: str, percent: int)
    item_finished(session_id: str)
    item_error(session_id: str, error_message: str)
    item_cancelled(session_id: str)
    stats_changed(dict)
        Aggregate counts: {total, processed, processing, queued, errors, cancelled}.
    """

    queue_changed = Signal()
    item_progress = Signal(str, int)
    item_finished = Signal(str)
    item_error = Signal(str, str)
    item_cancelled = Signal(str)
    stats_changed = Signal(dict)

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.store = BatchQueueStore(data_manager.config_dir)
        # PRD E5-11: clean up partial outputs left behind by a crash.
        self.items: list[dict] = self.store.reset_in_progress_to_queued(
            tracks_dir=data_manager.tracks_dir
        )
        self._paused = False
        self._watcher: Optional[WatchFolderWorker] = None

        # Active jobs — session_id -> ProcessingRunnable
        self._active: dict[str, object] = {}
        # Set of session_ids whose cancel was requested by the user.
        self._cancelled_ids: set[str] = set()
        self._cancel_all_active = False

        # Run-window limit (PRD E5-11b)
        self._stop_after_n: Optional[int] = None  # None = unlimited
        self._count_failed_toward_limit = True
        self._completed_this_run = 0
        # Resume any "queued" items left from a previous session
        self._run_active = bool([it for it in self.items if it["status"] == "queued"])

        # Thread pool — single shared instance per manager. Default 4 workers.
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(4)
        self._max_workers = 4

    # ----- Persistence -----
    def _persist(self):
        self.store.save(self.items)
        self.queue_changed.emit()
        self._emit_stats()

    def _emit_stats(self):
        stats = {
            "total": len(self.items),
            "processed": sum(1 for it in self.items if it["status"] == "done"),
            "processing": sum(1 for it in self.items if it["status"] == "processing"),
            "queued": sum(1 for it in self.items if it["status"] in ("queued", "paused")),
            "errors": sum(1 for it in self.items if it["status"] == "error"),
            "cancelled": sum(1 for it in self.items if it["status"] == "cancelled"),
        }
        self.stats_changed.emit(stats)

    # ----- Public API -----
    def add(
        self,
        video_path: str,
        model: str = "yolov8m.pt",
        conf: float = 0.25,
        priority: int = 0,
        scheduled_at: Optional[str] = None,
        processing_roi: Optional[dict] = None,
        source_folder: Optional[str] = None,
    ) -> Optional[str]:
        """Add a video to the queue. Returns the session_id, or None if
        the video doesn't exist or is already queued/processing."""
        p = Path(video_path)
        if not p.exists():
            return None

        # PRD E3-c: hash-based session_id when we know the source folder so
        # two files with the same name in different folders never collide.
        if source_folder:
            session_id = self.dm.build_session_id_for_batch(p, Path(source_folder))
            try:
                self.dm.register_batch_session(session_id, p)
            except Exception as e:
                print(f"[BatchQueueManager] register_batch_session failed: {e}")
        elif self.dm.active_mode == "nas":
            try:
                session_id = self.dm.build_session_id(p, self.dm.videos_dir)
            except ValueError:
                session_id = p.stem
        else:
            session_id = p.stem

        # Skip if already in the queue and not finished/cancelled
        for it in self.items:
            if it["session_id"] == session_id and it["status"] in ("queued", "processing", "paused"):
                return None

        item = {
            "session_id": session_id,
            "video_path": str(p),
            "source_folder": str(source_folder) if source_folder else None,
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
        # Replace stale duplicate (done/error/cancelled with same sid)
        self.items = [it for it in self.items if it["session_id"] != session_id]
        self.items.append(item)
        self._persist()
        return session_id

    def remove(self, session_id: str):
        if session_id in self._active:
            return  # cannot remove what's running
        self.items = [it for it in self.items if it["session_id"] != session_id]
        try:
            self.dm.unregister_batch_session(session_id)
        except Exception:
            pass
        self._persist()

    def clear_finished(self):
        keep = []
        for it in self.items:
            if it["status"] in ("queued", "processing", "paused"):
                keep.append(it)
            else:
                # Drop session-map entries for finished items so they
                # don't accumulate forever.
                try:
                    self.dm.unregister_batch_session(it["session_id"])
                except Exception:
                    pass
        self.items = keep
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
        self._run_active = True
        self._completed_this_run = 0
        self._persist()
        self._fill_pool()

    def get_items(self) -> list[dict]:
        return list(self.items)

    def get_status(self, session_id: str) -> Optional[str]:
        for it in self.items:
            if it["session_id"] == session_id:
                return it["status"]
        return None

    def status_for_path(self, abs_path: str) -> str:
        """Lookup status by absolute video path (used by the tree's status column)."""
        for it in self.items:
            if it.get("video_path") == abs_path:
                return it.get("status", "queued")
        return ""

    # ----- Worker pool sizing -----
    def set_max_workers(self, n: int):
        """Adjust concurrency mid-run. Pool size can be changed anytime."""
        n = max(1, int(n))
        self._max_workers = n
        self._pool.setMaxThreadCount(n)
        # If we just got bigger, kick the pool to fill the new slots.
        if self._run_active:
            self._fill_pool()

    def max_workers(self) -> int:
        return self._max_workers

    def set_stop_after_n(self, n: Optional[int]):
        """Cap how many videos this run will process. None / 0 = unlimited."""
        if not n or n <= 0:
            self._stop_after_n = None
        else:
            self._stop_after_n = int(n)

    def set_count_failed_toward_limit(self, on: bool):
        self._count_failed_toward_limit = bool(on)

    def start_all(self):
        """User pressed Start All — kick off / refill the pool."""
        self._paused = False
        self._run_active = True
        self._completed_this_run = 0
        self._cancel_all_active = False
        for it in self.items:
            if it["status"] == "paused":
                it["status"] = "queued"
        self._persist()
        self._fill_pool()

    # ----- Cancellation -----
    def cancel_all(self):
        """Atomically cancel everything queued + in-flight.

        PRD E5-6: pending runnables are dropped from the pool, in-flight ones
        observe ``cancel_requested`` on their next per-frame check. Partial
        ``tracks/<sid>.json`` files are deleted in the cancel callback.
        """
        self._cancel_all_active = True
        self._run_active = False
        # Drop everything pending in the pool first.
        try:
            self._pool.clear()
        except Exception:
            pass
        # Mark queued items as cancelled.
        for it in self.items:
            if it["status"] in ("queued", "paused"):
                it["status"] = "cancelled"
                it["progress"] = 0
        # Ask in-flight runnables to stop.
        for sid, runnable in list(self._active.items()):
            self._cancelled_ids.add(sid)
            try:
                runnable.cancel()
            except Exception:
                pass
        self._persist()

    def cancel(self, session_id: str):
        """Cancel a single item (queued or in-flight)."""
        for it in self.items:
            if it["session_id"] != session_id:
                continue
            if it["status"] in ("queued", "paused"):
                it["status"] = "cancelled"
                it["progress"] = 0
                self._persist()
                return
            if it["status"] == "processing":
                self._cancelled_ids.add(session_id)
                runnable = self._active.get(session_id)
                if runnable is not None:
                    try:
                        runnable.cancel()
                    except Exception:
                        pass
                return

    # ----- Watch folder -----
    def start_watch_folder(self, folder: Path, model: str, conf: float, poll_seconds: int = 5):
        self.stop_watch_folder()
        self._watcher = WatchFolderWorker(folder, poll_seconds)
        self._watcher.new_video.connect(
            lambda path, m=model, c=conf, sf=str(folder):
                self._on_watch_new_video(path, m, c, sf)
        )
        self._watcher.start()

    def stop_watch_folder(self):
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher.wait(2000)
            self._watcher = None

    def is_watching(self) -> bool:
        return self._watcher is not None and self._watcher.isRunning()

    def _on_watch_new_video(self, path: str, model: str, conf: float, source_folder: str):
        self.add(path, model=model, conf=conf, priority=0, source_folder=source_folder)
        self._fill_pool()

    # ----- Scheduler -----
    def start_next_if_idle(self):
        """Kept for backwards compat with old callers — equivalent to fill."""
        self._fill_pool()

    def _next_runnable(self) -> Optional[dict]:
        """Pick the next queued item, respecting priority, schedule, and
        the stop-after-N cap."""
        if self._paused or self._cancel_all_active:
            return None
        if self._stop_after_n is not None and self._completed_this_run >= self._stop_after_n:
            return None
        now = datetime.now()
        candidates = []
        for it in self.items:
            if it["status"] != "queued":
                continue
            if it["session_id"] in self._active:
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
        candidates.sort(key=lambda it: (-it.get("priority", 0), it["added_at"]))
        return candidates[0]

    def _fill_pool(self):
        """Dispatch queued items until we hit max_workers in flight."""
        if not self._run_active:
            return
        while len(self._active) < self._max_workers:
            item = self._next_runnable()
            if item is None:
                return
            self._launch(item)

    def _launch(self, item: dict):
        from cctv_yolo.processing import ProcessingRunnable

        item["status"] = "processing"
        item["progress"] = 0
        item["error"] = None
        self._persist()

        runnable = ProcessingRunnable(
            video_path=item["video_path"],
            tracks_dir=str(self.dm.tracks_dir),  # PRD E5-7: always central tracks dir
            model=item["model"],
            conf=item["conf"],
            session_id=item["session_id"],
            models_dir=str(self.dm.models_dir),
            processing_roi=item.get("processing_roi"),
        )
        runnable.signals.progress.connect(self._on_progress)
        runnable.signals.finished.connect(self._on_finished)
        runnable.signals.error.connect(self._on_error)
        runnable.signals.cancelled.connect(self._on_cancelled)

        self._active[item["session_id"]] = runnable
        self._pool.start(runnable)

    # ----- Signal handlers (worker -> manager) -----
    def _on_progress(self, session_id: str, pct: int):
        for it in self.items:
            if it["session_id"] == session_id:
                it["progress"] = int(pct)
                break
        # Don't persist on every tick — UI gets a signal that's cheaper.
        self.item_progress.emit(session_id, int(pct))

    def _on_finished(self, session_id: str):
        for it in self.items:
            if it["session_id"] == session_id:
                it["status"] = "done"
                it["progress"] = 100
                break
        self._active.pop(session_id, None)
        self._completed_this_run += 1
        self._persist()
        self.item_finished.emit(session_id)
        self._fill_pool()

    def _on_error(self, session_id: str, error_message: str):
        for it in self.items:
            if it["session_id"] == session_id:
                it["status"] = "error"
                it["error"] = error_message[:500]
                break
        self._active.pop(session_id, None)
        if self._count_failed_toward_limit:
            self._completed_this_run += 1
        self._persist()
        self.item_error.emit(session_id, error_message)
        self._fill_pool()

    def _on_cancelled(self, session_id: str):
        # PRD E5-6: clean up partial output so a re-run starts fresh.
        try:
            partial = Path(self.dm.tracks_dir) / f"{session_id}.json"
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        for it in self.items:
            if it["session_id"] == session_id:
                # If the user explicitly cancelled this item, mark it
                # cancelled. Otherwise (cancel_all rolled in) it stays as
                # cancelled. We always wind progress back to 0.
                it["status"] = "cancelled"
                it["progress"] = 0
                break
        self._active.pop(session_id, None)
        self._cancelled_ids.discard(session_id)
        self._persist()
        self.item_cancelled.emit(session_id)
        # If cancel_all is finishing, only fill once everything drained.
        if self._cancel_all_active and not self._active:
            self._cancel_all_active = False
        else:
            self._fill_pool()

    # ----- Shutdown -----
    def shutdown(self):
        """Best-effort shutdown for app close.

        Mark active jobs as queued (so resume-on-crash logic will retry),
        request cooperative cancel, then wait briefly for the pool.
        """
        self.stop_watch_folder()
        for sid, runnable in list(self._active.items()):
            try:
                runnable.cancel()
            except Exception:
                pass
            for it in self.items:
                if it["session_id"] == sid:
                    it["status"] = "queued"
                    break
        self._persist()
        # Give in-flight runnables ~3s to notice the cancel flag.
        self._pool.waitForDone(3000)
