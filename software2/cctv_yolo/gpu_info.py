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


def _cuda_capability_and_arches():
    """Return (capability tuple, compiled sm_* arch list, device name).

    Any element may be None/empty if torch can't report it. Never raises.
    """
    try:
        import torch
    except Exception:
        return None, [], None
    try:
        cap = torch.cuda.get_device_capability(0)
    except Exception:
        cap = None
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = None
    try:
        arches = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    except Exception:
        arches = []
    return cap, arches, name


def _arch_supported(cap, arch_list) -> bool:
    """True if the GPU's compute capability is covered by the bundled wheel.

    Classic failure: an RTX 50-series (Blackwell, sm_120 / major 12) GPU with
    a cu118/cu121 wheel whose newest kernel is sm_90 (major 9). torch reports
    cuda.is_available()==True (the driver is new enough) but the first kernel
    launch dies with "no kernel image is available for execution on the
    device". NVIDIA GPUs JIT forward *within* a major version, so the GPU is
    usable only if the wheel compiled at least one arch whose major >= the
    GPU's major. If the newest compiled major is older than the GPU's, it
    cannot run. Conservative: unknowns return True so we never false-alarm.
    """
    try:
        major, _minor = cap
    except Exception:
        return True
    majors = []
    for a in arch_list:
        try:
            n = int(a.split("_", 1)[1].rstrip("a+"))  # 'sm_90a' -> 90
            majors.append(n // 10)
        except Exception:
            continue
    if not majors:
        return True
    return major <= max(majors)


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
        # cuda.is_available() can be True yet the bundled wheel still lack
        # kernels for this GPU's architecture (RTX 50-series / Blackwell +
        # cu118). Catch that here instead of returning a green "GPU active"
        # that crashes on the first inference.
        cap, arches, name = _cuda_capability_and_arches()
        if cap is not None and arches and not _arch_supported(cap, arches):
            sm = f"sm_{cap[0]}{cap[1]}"
            reason = (
                f"Your GPU ({name or 'NVIDIA GPU'}, compute {sm}) is newer "
                f"than the active PyTorch CUDA build (CUDA "
                f"{torch_cuda_build or '?'}; kernels: {', '.join(arches)}). "
                "Use Settings -> 'Set up / repair GPU acceleration' to install "
                "the matching build (RTX 50-series / Blackwell needs cu128). "
                "Running on CPU for now to avoid a hard crash."
            )
            return DeviceInfo("cpu", "CPU", torch_cuda_build, reason)
        if not name:
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
            "PyTorch is running in CPU-only mode. On Windows, enable GPU "
            "acceleration from Settings -> 'Set up / repair GPU acceleration' "
            "(it downloads the matching CUDA build for your card)."
        )
    else:
        # CUDA-built torch but is_available() is False. Most common cause
        # on Windows: the NVIDIA driver is older than the wheel's bundled
        # CUDA runtime. Less common: no NVIDIA hardware at all.
        reason = (
            f"PyTorch was built for CUDA {torch_cuda_build} but no usable "
            "NVIDIA GPU is visible. Update your NVIDIA driver, or (on Windows) "
            "use Settings -> 'Set up / repair GPU acceleration' to install a "
            "CUDA build matching your card."
        )
    return DeviceInfo("cpu", "CPU", torch_cuda_build, reason)


def short_summary(info: DeviceInfo) -> str:
    """One-line status string for status bar / startup banner."""
    if info.device.startswith("cuda"):
        return f"GPU: {info.label}"
    if info.device == "mps":
        return f"GPU: {info.label}"
    return "GPU: not in use (CPU mode)"
