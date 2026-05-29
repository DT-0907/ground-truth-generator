"""First-run GPU acceleration — download a CUDA build of torch on demand.

The Windows app ships with **CPU** torch baked in (staged as a data tree, see
cctv_yolo.spec), so it always works out of the box. If an NVIDIA GPU is
present, this module downloads the matching CUDA build of torch + torchvision
into a per-user folder; ``runtime_hook.py`` puts that folder first on
``sys.path`` on the next launch, so the app then runs on the GPU.

Design facts (verified against the live PyTorch index 2026-05-29):
  * Windows CUDA torch wheels are SELF-CONTAINED — the CUDA runtime DLLs live
    in ``torch/lib`` inside the wheel, so unzip-and-import works with no pip
    and no separate nvidia-* packages (those are Linux-only).
  * torch <-> torchvision are tightly coupled (tv minor = torch minor - 2) and
    must share the same +cuXXX build, so we PIN known-good pairs.
  * cu128 (torch 2.8.0 / tv 0.23.0) covers Ampere (sm_86), Ada (sm_89) AND
    Blackwell (sm_120). cu118 (torch 2.7.1 / tv 0.22.1) is the broad fallback
    for older drivers (no Blackwell). cu127 does NOT exist.
  * torch cannot be hot-swapped once imported, so a fresh install needs an app
    RESTART to take effect.

Stdlib only (no torch, no pip) so this can run before any ``import torch``.
Windows-only: macOS uses MPS with the baked torch; Linux CUDA wheels aren't
self-contained, so the feature is gated to ``sys.platform == 'win32'``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Python + platform tags for the wheel we need (derived from the running build
# so it stays correct if the bundled Python ever changes).
PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"   # e.g. cp312
PLATFORM_TAG = "win_amd64"
READY_MARKER = ".torch_ready"          # written LAST, after a clean install
DECLINED_MARKER = ".gpu_setup_declined"

# Pinned, mutually-compatible (torch, torchvision) versions per CUDA variant.
# Verified present (with sha256) on download.pytorch.org for cp312/win_amd64.
# NOTE: the baked CPU baseline is torch 2.8.0 (build_windows.bat) and its
# pure-Python deps are frozen for that version (cctv_yolo.spec _TORCH_PY_DEPS).
# The cu118 pin (torch 2.7.1) is older but verified compatible with those same
# deps. If you bump the cu128/baseline torch, re-validate the cu118 pin too.
_PINS: dict[str, dict[str, str]] = {
    "cu128": {"torch": "2.8.0", "torchvision": "0.23.0"},
    "cu118": {"torch": "2.7.1", "torchvision": "0.22.1"},
}

_INDEX = "https://download.pytorch.org/whl/{variant}/{pkg}/"
_UA = {"User-Agent": "Mozilla/5.0 (CCTV-YOLO model-setup)"}
_SSL = ssl.create_default_context()
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class GpuSetupCancelled(Exception):
    """Raised when the user cancels a download in progress."""


# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------
def _app_config_dir() -> Path:
    """Per-user app config dir (NOT inside the portable data folder)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "CCTV-YOLO"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "CCTV-YOLO"
    return Path.home() / ".local" / "share" / "CCTV-YOLO"


def runtime_dir() -> Path:
    """Where a downloaded GPU torch is unpacked. Matches runtime_hook.py."""
    return _app_config_dir() / "torch_runtime"


def _declined_marker() -> Path:
    return _app_config_dir() / DECLINED_MARKER


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
def is_installed() -> bool:
    """True if a complete GPU torch is present (ready-marker + torch/lib)."""
    rd = runtime_dir()
    return (rd / READY_MARKER).is_file() and (rd / "torch" / "lib").is_dir()


def installed_info() -> dict | None:
    """The {variant, torch, torchvision} record written at install time."""
    m = runtime_dir() / READY_MARKER
    if not m.is_file():
        return None
    try:
        return json.loads(m.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_declined() -> bool:
    return _declined_marker().is_file()


def mark_declined() -> None:
    try:
        d = _app_config_dir()
        d.mkdir(parents=True, exist_ok=True)
        _declined_marker().write_text("user chose not to install GPU support", encoding="utf-8")
    except OSError as e:
        logger.warning("Couldn't write GPU-declined marker: %s", e)


def clear_declined() -> None:
    try:
        _declined_marker().unlink(missing_ok=True)
    except OSError:
        pass


# --------------------------------------------------------------------------
# GPU / variant detection (nvidia-smi; no torch needed)
# --------------------------------------------------------------------------
def _run_smi(args: list[str]) -> str:
    try:
        out = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=10, creationflags=_NO_WINDOW,
        )
        return out.stdout.decode("utf-8", "replace")
    except Exception:
        return ""


@lru_cache(maxsize=1)
def _driver_cuda_code() -> int:
    """Parse 'CUDA Version: 12.8' from nvidia-smi -> 1208. 0 if no NVIDIA GPU.

    Cached: nvidia-smi is slow-ish and the driver can't change mid-process, so
    should_offer()/the dialog/Settings all reuse one query instead of stalling
    the GUI thread repeatedly.
    """
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", _run_smi(["nvidia-smi"]))
    if not m:
        return 0
    return int(m.group(1)) * 100 + int(m.group(2))


@lru_cache(maxsize=1)
def gpu_name() -> str:
    """First NVIDIA GPU name, or '' if none. Cached (see _driver_cuda_code)."""
    txt = _run_smi(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return txt.strip().splitlines()[0].strip() if txt.strip() else ""


def desired_variant() -> str | None:
    """Pick the CUDA wheel variant for this machine's driver. Windows only.

    cu128 needs a driver advertising CUDA >= 12.8 (covers RTX 50/40/30);
    cu118 is the broad fallback (CUDA >= 11.8, no Blackwell). None = no usable
    NVIDIA GPU.
    """
    if sys.platform != "win32":
        return None
    code = _driver_cuda_code()
    if code <= 0:
        return None
    if code >= 1208:
        return "cu128"
    if code >= 1108:
        return "cu118"
    return None


def should_offer() -> tuple[bool, str | None]:
    """(offer?, variant). Offer GPU setup only when: Windows, a FROZEN build
    (a downloaded torch only takes effect via the bundle's runtime hook — in a
    dev run it can't), an NVIDIA GPU is present, nothing is installed yet, and
    the user hasn't declined."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return (False, None)
    if is_installed() or is_declined():
        return (False, None)
    variant = desired_variant()
    return (variant is not None, variant)


# --------------------------------------------------------------------------
# Wheel resolution + download + install
# --------------------------------------------------------------------------
def _resolve_wheel(pkg: str, variant: str, version: str) -> tuple[str, str]:
    """Return (download_url, sha256) for the exact pinned wheel on the variant
    index, matching this Python tag + win_amd64. Raises if not found.

    The index is a PEP 503 flat list of
        <a href="https://.../<file>#sha256=<64hex>">pkg-ver+variant-...whl</a>
    The href encodes '+' as %2B; the link TEXT shows a literal '+'.
    """
    url = _INDEX.format(variant=variant, pkg=pkg)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, context=_SSL, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    pat = re.compile(
        r'<a\s+href="(?P<url>[^"#]+)'
        r'(?:#sha256=(?P<sha>[0-9a-fA-F]{64}))?"'
        r'[^>]*>'
        rf'{re.escape(pkg)}-{re.escape(version)}\+{re.escape(variant)}'
        rf'-{PY_TAG}-{PY_TAG}-{PLATFORM_TAG}\.whl',
        re.IGNORECASE,
    )
    m = pat.search(html)
    if not m:
        raise RuntimeError(
            f"No {pkg} {version} ({variant}/{PY_TAG}/{PLATFORM_TAG}) wheel found "
            f"on the PyTorch index. The pinned version may have been removed."
        )
    return m.group("url"), (m.group("sha") or "").lower()


def resolve_plan(variant: str) -> list[dict]:
    """Resolve URLs + hashes for the pinned torch+torchvision of `variant`.

    Does network I/O (two small index fetches). Returns a list of dicts:
    {pkg, version, url, sha256}.
    """
    if variant not in _PINS:
        raise ValueError(f"Unknown CUDA variant: {variant}")
    pin = _PINS[variant]
    plan = []
    for pkg in ("torch", "torchvision"):
        version = pin[pkg]
        wheel_url, sha = _resolve_wheel(pkg, variant, version)
        plan.append({"pkg": pkg, "version": version, "url": wheel_url, "sha256": sha})
    return plan


def estimated_download_mb(variant: str) -> int:
    """Rough total download size for the variant (for UI copy)."""
    return 3400 if variant == "cu128" else 2900


def _download(url: str, sha256: str, dest: Path,
              progress=None, cancel=None) -> None:
    """Stream `url` to `dest`, verifying sha256. Raises GpuSetupCancelled if
    `cancel()` becomes true. Atomic via a .part file + os.replace."""
    req = urllib.request.Request(url, headers=_UA)
    h = hashlib.sha256()
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f, \
                urllib.request.urlopen(req, context=_SSL, timeout=120) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                if cancel is not None and cancel():
                    raise GpuSetupCancelled()
                chunk = r.read(1 << 20)   # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
        if sha256 and h.hexdigest() != sha256:
            raise RuntimeError("Downloaded file is corrupt (sha256 mismatch).")
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def install(variant: str, progress=None, status=None, cancel=None) -> dict:
    """Download + unzip the pinned torch & torchvision for `variant` into
    runtime_dir(). Writes the ready-marker LAST so an interrupted install is
    never treated as complete (runtime_hook falls back to the CPU baseline).

    progress(done_bytes, total_bytes) — per-file byte progress.
    status(message)                   — coarse stage messages for the UI.
    cancel() -> bool                  — cooperative cancel check.

    Returns the manifest dict. Raises on failure (caller cleans up / reports).
    """
    rd = runtime_dir()
    # Always start clean — a half-extracted dir must never look ready.
    if rd.exists():
        shutil.rmtree(rd, ignore_errors=True)
    rd.mkdir(parents=True, exist_ok=True)

    if status:
        status("Finding the right PyTorch build…")
    plan = resolve_plan(variant)            # network

    wheels: list[Path] = []
    for item in plan:
        if status:
            status(f"Downloading {item['pkg']} {item['version']} ({variant})…")
        dest = rd / f"{item['pkg']}-{item['version']}-{variant}.whl"
        _download(item["url"], item["sha256"], dest, progress=progress, cancel=cancel)
        wheels.append(dest)

    for whl in wheels:
        if cancel is not None and cancel():
            raise GpuSetupCancelled()
        if status:
            status(f"Installing {whl.stem}…")
        with zipfile.ZipFile(whl) as z:
            z.extractall(rd)
        try:
            whl.unlink()
        except OSError:
            pass

    manifest = {
        "variant": variant,
        "torch": plan[0]["version"],
        "torchvision": plan[1]["version"],
        "py_tag": PY_TAG,
    }
    # Marker LAST — its presence is the "this install is complete" signal.
    (rd / READY_MARKER).write_text(json.dumps(manifest), encoding="utf-8")
    if status:
        status("Done. Restart CCTV-YOLO to use your GPU.")
    return manifest
