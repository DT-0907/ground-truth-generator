#!/usr/bin/env python3
"""Pick the right PyTorch CUDA wheel index for THIS machine.

build_windows.bat calls this with the build interpreter to choose which
torch wheels to install. Printing one of:

    cu128  -> CUDA 12.8 wheels  (RTX 50-series / Blackwell, driver CUDA >= 12.8)
    cu126  -> CUDA 12.6 wheels  (driver CUDA >= 12.6)
    cu124  -> CUDA 12.4 wheels  (driver CUDA >= 12.4)
    cu121  -> CUDA 12.1 wheels  (driver CUDA >= 12.1)
    cu118  -> CUDA 11.8 wheels  (older NVIDIA GPUs / drivers)
    cpu    -> no usable NVIDIA GPU detected

Why this exists: the previous default of cu118 cannot drive a Blackwell
GPU (RTX 5070, compute sm_120) — those need cu128. And parsing nvidia-smi
inside a .bat is fragile, so we do it here in stdlib Python (no torch
required — torch isn't installed yet at this point in the build).

Stdlib only. Never raises; prints 'cpu' on any failure so the build can
fall back gracefully.
"""
from __future__ import annotations

import re
import subprocess
import sys

# Driver max-CUDA (major*100+minor) -> wheel variant. Highest match wins.
# A driver advertising CUDA X.Y can run any wheel built for <= X.Y.
_CUDA_TO_VARIANT = [
    (1208, "cu128"),
    (1206, "cu126"),
    (1204, "cu124"),
    (1201, "cu121"),
    (1108, "cu118"),
]

# GPU name fragments that REQUIRE cu128 regardless of the parsed driver CUDA
# (belt-and-suspenders for Blackwell, whose sm_120 kernels only ship in
# cu128+). Matched case-insensitively against the GPU name. We use SPECIFIC
# model numbers, NOT a broad "rtx 50" substring — that would false-match
# "RTX 5000 Ada" (an Ada card that needs cu124, not cu128).
_BLACKWELL_NAMES = ("5090", "5080", "5070", "5060", "5050",
                    "b100", "b200", "gb200", "blackwell")


def _run(args: list[str]) -> str:
    """Run a command, return stdout (or '' on any error). Never raises."""
    try:
        out = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
            # Suppress a console flash if ever called from a windowed parent.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return out.stdout.decode("utf-8", "replace")
    except Exception:
        return ""


def _driver_cuda_code(smi_text: str) -> int:
    """Parse 'CUDA Version: 12.8' from nvidia-smi header -> 1208. 0 if absent."""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi_text)
    if not m:
        return 0
    return int(m.group(1)) * 100 + int(m.group(2))


def choose_variant(smi_text: str, gpu_names: str) -> str:
    """Pure decision function — unit-testable without a real GPU."""
    if not smi_text.strip():
        return "cpu"  # nvidia-smi produced nothing -> no NVIDIA GPU
    names = (gpu_names or "").lower()
    if any(frag in names for frag in _BLACKWELL_NAMES):
        return "cu128"
    code = _driver_cuda_code(smi_text)
    for threshold, variant in _CUDA_TO_VARIANT:
        if code >= threshold:
            return variant
    # nvidia-smi ran (so a driver/GPU exists) but we couldn't read a CUDA
    # version >= 11.8. cu118 is the safe floor for any modern NVIDIA driver.
    return "cu118"


def detect() -> str:
    smi = _run(["nvidia-smi"])
    names = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return choose_variant(smi, names)


if __name__ == "__main__":
    sys.stdout.write(detect())
