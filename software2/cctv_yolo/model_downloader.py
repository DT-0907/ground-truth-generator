"""
Async YOLOv8 model downloader (PRD C1 / K2-6).

We bypass Ultralytics' built-in fetcher (which shells out to curl and emits
opaque "Curl return value 56" / similar errors on flaky networks) in favor
of a direct urllib download with:
  - Multiple retries with exponential backoff (1s, 2s, 4s)
  - Two mirrors per file: GitHub Releases (primary), HuggingFace (fallback)
  - Plain-English error messages for the common curl / urllib failure modes
  - Byte-level progress so the progress dialog can show real status
  - Resume support: a partial .pt.tmp left by a previous failed run is
    re-used and the download continues from where it stopped (Range header)

Cross-platform: pure-Python stdlib (urllib.request + socket). No curl
dependency at all.
"""
from __future__ import annotations

import os
import shutil
import socket
import time
import traceback
import urllib.error
import urllib.request
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
    """Every place Ultralytics might have dropped the freshly-downloaded .pt.

    Used as a last-resort fallback — if our direct download fails AND
    Ultralytics' built-in fetch has already succeeded at some point, we
    can still find the file.
    """
    cands: list[Path] = []
    cands.append(Path.cwd() / model_name)
    cands.append(Path.home() / ".config" / "Ultralytics" / model_name)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            cands.append(Path(appdata) / "Ultralytics" / model_name)
    return cands


# Two mirrors per file, in priority order. Both serve the same checkpoints.
def _mirror_urls(model_name: str) -> list[str]:
    return [
        f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}",
        f"https://huggingface.co/Ultralytics/YOLOv8/resolve/main/{model_name}",
    ]


def _humanize_download_error(err: Exception) -> str:
    """Turn an opaque network error into something a user can act on."""
    msg = str(err) if err else "unknown"
    lower = msg.lower()

    if any(s in lower for s in ("10054", "connection reset", "remote end closed",
                                "incomplete read", "curl return value 56")):
        return (
            "Network connection was reset during the download. This usually "
            "means a firewall, antivirus, VPN, or unstable Wi-Fi interrupted "
            "the transfer.\n\n"
            "Try one of:\n"
            "  - Disable VPN temporarily\n"
            "  - Whitelist github.com and huggingface.co in your antivirus\n"
            "  - Switch to a more stable network (wired, or a different Wi-Fi)\n"
            "  - Retry the download — the partial file is kept so the next "
            "    attempt resumes where this one stopped.\n\n"
            f"Underlying error: {msg}"
        )
    if "timeout" in lower or "timed out" in lower:
        return (
            "Download timed out. Check your internet connection and try "
            "again. Retries will resume from where the previous attempt "
            "stopped.\n\n"
            f"Underlying error: {msg}"
        )
    if any(s in lower for s in ("getaddrinfo", "name or service not known",
                                "name resolution", "nodename nor servname",
                                "no address associated")):
        return (
            "DNS lookup failed for github.com or huggingface.co. Check your "
            "internet connection and DNS settings (try switching to 1.1.1.1 "
            "or 8.8.8.8 if your ISP DNS is failing).\n\n"
            f"Underlying error: {msg}"
        )
    if "ssl" in lower or "certificate" in lower:
        return (
            "SSL/TLS error while contacting the download server. This often "
            "means your system clock is wrong, or a corporate proxy is "
            "intercepting traffic. Fix the clock or whitelist github.com / "
            "huggingface.co in your proxy.\n\n"
            f"Underlying error: {msg}"
        )
    if "403" in msg or "forbidden" in lower:
        return (
            "Server returned 403 Forbidden — the URL may have moved or the "
            "release tag is no longer available. Update Ultralytics or use "
            "a custom model file (Models tab → Import .pt).\n\n"
            f"Underlying error: {msg}"
        )
    return f"Download failed: {msg}"


class ModelDownloadWorker(QThread):
    """Download a YOLOv8 variant and copy it into ``dest_dir``."""

    done     = Signal(str)         # path to downloaded model
    failed   = Signal(str)         # human-readable error message
    progress = Signal(int, int)    # (bytes_downloaded, total_bytes); -1 if unknown

    # Per-file network timeout (seconds). HTTP keep-alive will reset within
    # this window if the server stalls.
    READ_TIMEOUT = 30
    CONNECT_TIMEOUT = 15
    MAX_ATTEMPTS_PER_URL = 3
    CHUNK_SIZE = 64 * 1024

    def __init__(self, model_name: str, dest_dir: Path, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.dest_dir = Path(dest_dir)
        self._cancel = False

    def cancel(self) -> None:
        """Request cooperative cancellation. Checked between chunks."""
        self._cancel = True

    # ------------------------------------------------------------------

    def run(self):
        try:
            self.dest_dir.mkdir(parents=True, exist_ok=True)
            dest = self.dest_dir / self.model_name

            # If the final file already exists (and is non-empty), we're done.
            if dest.exists() and dest.stat().st_size > 0:
                self.done.emit(str(dest))
                return

            # Try our mirrors with retries.
            last_err: Exception | None = None
            for url in _mirror_urls(self.model_name):
                for attempt in range(1, self.MAX_ATTEMPTS_PER_URL + 1):
                    if self._cancel:
                        self.failed.emit("Cancelled.")
                        return
                    try:
                        self._download_to(url, dest)
                        self.done.emit(str(dest))
                        return
                    except Exception as e:
                        last_err = e
                        if attempt < self.MAX_ATTEMPTS_PER_URL:
                            # Exponential backoff between attempts on the
                            # same URL: 1s, 2s, 4s.
                            time.sleep(2 ** (attempt - 1))
                # Try the next mirror.

            # All mirrors exhausted. Last fallback: check whether Ultralytics
            # has cached this file from a previous successful run elsewhere.
            for cached in candidate_cache_paths(self.model_name):
                if cached.exists() and cached.stat().st_size > 0:
                    shutil.copy2(cached, dest)
                    self.done.emit(str(dest))
                    return

            self.failed.emit(_humanize_download_error(last_err))
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(_humanize_download_error(e))

    # ------------------------------------------------------------------

    def _download_to(self, url: str, dest: Path) -> None:
        """Download `url` to `dest`, resuming a previous partial transfer
        via `dest.with_suffix(dest.suffix + ".tmp")` if present.
        """
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        # If we have a partial download, request the remainder via Range.
        existing = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "CCTV-YOLO model-downloader/2.0",
                **({"Range": f"bytes={existing}-"} if existing else {}),
            },
        )

        socket.setdefaulttimeout(self.CONNECT_TIMEOUT)
        try:
            with urllib.request.urlopen(req, timeout=self.CONNECT_TIMEOUT) as resp:
                # Determine total size (for progress bar). Includes already-
                # downloaded bytes when Range was used.
                content_len = resp.headers.get("Content-Length")
                total = int(content_len) + existing if content_len else -1

                # 206 = Partial Content, 200 = full body. If the server
                # ignored our Range header (200), we have to start over.
                if resp.status == 200 and existing:
                    existing = 0
                    if tmp.exists():
                        tmp.unlink()

                mode = "ab" if existing else "wb"
                downloaded = existing
                self.progress.emit(downloaded, total)

                with open(tmp, mode) as f:
                    while True:
                        if self._cancel:
                            raise RuntimeError("Cancelled by user.")
                        # urlopen returns a file-like; set socket read timeout
                        # so a stalled server eventually errors instead of
                        # hanging forever.
                        try:
                            chunk = resp.read(self.CHUNK_SIZE)
                        except socket.timeout as e:
                            raise TimeoutError(f"Read timed out: {e}")
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(downloaded, total)
        except urllib.error.HTTPError as e:
            # 416 = "Requested range not satisfiable" — usually means our
            # partial file is already complete. Reset and try fresh.
            if e.code == 416 and existing:
                if tmp.exists():
                    tmp.unlink()
                raise RuntimeError("Partial file invalid; will retry fresh.")
            raise

        # Sanity check: the file shouldn't be tiny if we expected MB-sized
        # model weights. Catches "downloaded an HTML error page" bugs.
        if tmp.stat().st_size < 100 * 1024:  # 100 KB minimum
            tmp.unlink()
            raise RuntimeError(
                f"Downloaded file is too small ({tmp.stat().st_size} bytes); "
                "the server may have returned an error page."
            )

        # Atomic move from .tmp to final destination.
        if dest.exists():
            dest.unlink()
        os.replace(str(tmp), str(dest))
