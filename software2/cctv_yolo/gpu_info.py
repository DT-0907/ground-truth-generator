"""GPU / CUDA introspection.

Centralizes "what device will torch use, and if it can't use CUDA, why"
so it can be surfaced in the status bar, About dialog, and processor
logs in one consistent voice. The colleague who reported "running CPU
even though I have an NVIDIA GPU" needs a single string that tells them
WHICH of the three failure modes they hit:

  1. CPU-only torch was bundled                 -> rebuild with CUDA torch
  2. CUDA torch bundled but no NVIDIA driver    -> install/update driver
  3. CUDA torch bundled but driver too old      -> update NVIDIA driver
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    device: str             # "cuda:0" | "mps" | "cpu"
    label: str              # e.g. "NVIDIA GeForce RTX 3060 (CUDA 11.8)"
    torch_cuda_build: str   # e.g. "11.8" or "" if CPU-only torch
    reason: str             # explanation when device == "cpu"; "" otherwise


def detect_device() -> DeviceInfo:
    """Pick the best device and explain *why* if CPU was the only option.

    Never raises — falls back to CPU with a generic reason on any error.
    """
    try:
        import torch
    except Exception as e:
        return DeviceInfo("cpu", "CPU", "", f"PyTorch not importable: {e}")

    torch_cuda_build = getattr(torch.version, "cuda", None) or ""

    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False

    if cuda_ok:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "NVIDIA GPU"
        label = f"{name} (CUDA {torch_cuda_build or '?'})"
        return DeviceInfo("cuda:0", label, torch_cuda_build, "")

    # MPS (Apple Silicon)
    try:
        mps_ok = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception:
        mps_ok = False
    if mps_ok:
        return DeviceInfo("mps", "Apple Metal (MPS)", torch_cuda_build, "")

    # CPU fallback — figure out *why*.
    if not torch_cuda_build:
        reason = (
            "PyTorch was installed without CUDA support (CPU-only build). "
            "Rebuild with build_windows.bat — it defaults to CUDA torch."
        )
    else:
        # CUDA-built torch but is_available() is False. Most common cause
        # on Windows: the NVIDIA driver is older than the wheel's bundled
        # CUDA runtime. cu118 wheels need driver >= 452.39 (Windows);
        # cu121 needs >= 528.33. Less common: no NVIDIA hardware at all.
        reason = (
            f"PyTorch was built for CUDA {torch_cuda_build} but no usable "
            "NVIDIA GPU is visible. Update your NVIDIA driver (the "
            "bundled CUDA runtime needs a recent driver), or rebuild "
            "with a CUDA variant matching your driver "
            "(set CCTV_YOLO_TORCH_VARIANT=cu118 for the broadest "
            "compatibility, then re-run build_windows.bat)."
        )
    return DeviceInfo("cpu", "CPU", torch_cuda_build, reason)


def short_summary(info: DeviceInfo) -> str:
    """One-line status string for status bar / startup banner."""
    if info.device.startswith("cuda"):
        return f"GPU: {info.label}"
    if info.device == "mps":
        return f"GPU: {info.label}"
    return "GPU: not in use (CPU mode)"
