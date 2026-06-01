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

import ctypes
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
@lru_cache(maxsize=1)
def _nvidia_smi() -> str:
    """Absolute path to nvidia-smi.exe, or the bare name as a last resort.

    Modern drivers drop nvidia-smi.exe into System32 (always on PATH), but some
    laptop OEM driver packages only place it under
    ``Program Files\\NVIDIA Corporation\\NVSMI`` — which is NOT on PATH — so a
    bare ``nvidia-smi`` invocation silently finds nothing and we wrongly report
    "no GPU" on a machine that clearly has one. Probe the known locations.
    """
    exe = "nvidia-smi.exe" if sys.platform == "win32" else "nvidia-smi"
    found = shutil.which(exe) or shutil.which("nvidia-smi")
    if found:
        return found
    candidates = []
    sysroot = os.environ.get("SystemRoot") or r"C:\Windows"
    candidates.append(Path(sysroot) / "System32" / exe)
    for pf in (os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles"),
               r"C:\Program Files"):
        if pf:
            candidates.append(Path(pf) / "NVIDIA Corporation" / "NVSMI" / exe)
    for c in candidates:
        try:
            if c.is_file():
                return str(c)
        except OSError:
            continue
    return exe  # nothing found; let the OS try to resolve it at call time


def _run_smi(args: list[str]) -> str:
    if args and args[0] == "nvidia-smi":
        args = [_nvidia_smi(), *args[1:]]
    try:
        out = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=4, creationflags=_NO_WINDOW,
        )
        return out.stdout.decode("utf-8", "replace")
    except Exception:
        return ""


@lru_cache(maxsize=1)
def _nvcuda_probe() -> tuple[int, str]:
    """Detect the NVIDIA GPU directly via the CUDA driver DLL (``nvcuda.dll``).

    Returns ``(cuda_driver_code, gpu_name)`` — e.g. ``(1208, "NVIDIA GeForce
    RTX 4090 Laptop GPU")`` — or ``(0, "")`` if there's no usable NVIDIA GPU.

    This is the PRIMARY detector: it's torch-free, nvidia-smi-free, needs no
    subprocess, and queries the installed driver directly, so it works on the
    machines where ``nvidia-smi`` silently returns nothing — off-PATH, or a
    laptop whose Optimus dGPU was parked when queried (the exact failure a
    colleague hit on an RTX 4090 laptop: card present and working, yet "Set up
    GPU" reported "no NVIDIA GPU"). We require a real CUDA device
    (``cuInit`` + ``cuDeviceGetCount >= 1``) so a driver-without-GPU never
    registers as a false positive. ``cuDriverGetVersion`` returns the max CUDA
    the driver supports (e.g. 12080 == 12.8), which is exactly what picks
    cu128 vs cu118.
    """
    if sys.platform != "win32":
        return (0, "")
    try:
        lib = ctypes.WinDLL("nvcuda.dll")
    except OSError:
        return (0, "")  # no NVIDIA driver installed
    try:
        if lib.cuInit(0) != 0:
            return (0, "")                       # CUDA_ERROR_NO_DEVICE etc.
        cnt = ctypes.c_int(0)
        if lib.cuDeviceGetCount(ctypes.byref(cnt)) != 0 or cnt.value < 1:
            return (0, "")
        ver = ctypes.c_int(0)
        code = 0
        if lib.cuDriverGetVersion(ctypes.byref(ver)) == 0 and ver.value > 0:
            code = (ver.value // 1000) * 100 + (ver.value % 1000) // 10  # 12080 -> 1208
        dev = ctypes.c_int(0)
        name = ""
        if lib.cuDeviceGet(ctypes.byref(dev), 0) == 0:
            buf = ctypes.create_string_buffer(256)
            if lib.cuDeviceGetName(buf, 256, dev) == 0:
                name = buf.value.decode("utf-8", "replace").strip()
        logger.info("nvcuda probe: cuda_code=%s name=%r", code, name or "NVIDIA GPU")
        return (code, name or "NVIDIA GPU")
    except Exception:
        logger.debug("nvcuda probe failed", exc_info=True)
        return (0, "")


def _torch_cuda_view() -> tuple[str | None, str | None]:
    """(gpu_name, wheel_variant) inferred from an ALREADY-LOADED torch that can
    see CUDA, or (None, None).

    Last-resort fallback: if a downloaded CUDA build is already active, there is
    unquestionably a usable NVIDIA GPU. IMPORTANT: this never *imports* torch —
    it only inspects ``sys.modules`` — so it can't trigger a multi-second torch
    import on the GUI thread (the Settings panel calls into here while building).
    cu128 covers every arch we ship kernels for (sm_70..sm_120).
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return (None, None)
    try:
        if not bool(torch.cuda.is_available()):
            return (None, None)
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = None
    except Exception:
        return (None, None)
    return (name or "NVIDIA GPU", "cu128")


@lru_cache(maxsize=1)
def _driver_cuda_code() -> int:
    """CUDA driver version as ``major*100+minor`` (e.g. 1208 for CUDA 12.8), or
    0 if no usable NVIDIA GPU.

    Prefers the CUDA driver DLL (fast, reliable, no subprocess) and only falls
    back to parsing ``nvidia-smi`` if that DLL probe comes up empty. Cached.
    """
    code, _ = _nvcuda_probe()
    if code > 0:
        return code
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", _run_smi(["nvidia-smi"]))
    if m:
        return int(m.group(1)) * 100 + int(m.group(2))
    return 0


@lru_cache(maxsize=1)
def gpu_name() -> str:
    """First NVIDIA GPU name, or '' if none. Cached.

    nvcuda.dll first (works when nvidia-smi can't be reached), then nvidia-smi,
    then an already-loaded torch.
    """
    _, name = _nvcuda_probe()
    if name and name != "NVIDIA GPU":
        return name
    txt = _run_smi(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    smi_name = txt.strip().splitlines()[0].strip() if txt.strip() else ""
    if smi_name:
        return smi_name
    if name:                       # generic "NVIDIA GPU" from nvcuda
        return name
    tname, _ = _torch_cuda_view()
    return tname or ""


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
        logger.info("desired_variant: no NVIDIA GPU detected (driver code=0)")
        return None
    variant = "cu128" if code >= 1208 else ("cu118" if code >= 1108 else None)
    logger.info("desired_variant: driver cuda code=%s -> %s", code, variant)
    return variant


def variant_for_setup() -> str | None:
    """Best CUDA variant for the Settings 'Set up / repair' button.

    desired_variant() trusts nvidia-smi alone, which gives a false "no GPU" on
    machines where nvidia-smi is off-PATH or the laptop dGPU was asleep when
    queried — even when a CUDA torch is already installed and running. Widen the
    evidence: live driver query, then the already-installed manifest, then what
    torch currently reports. Returns None only when nothing anywhere indicates
    an NVIDIA GPU.
    """
    if sys.platform != "win32":
        return None
    v = desired_variant()
    if v:
        return v
    info = installed_info()
    if info and info.get("variant") in _PINS:
        return info["variant"]
    _name, tv = _torch_cuda_view()
    return tv


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
