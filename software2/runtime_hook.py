"""
PyInstaller runtime hook — runs before main() in the frozen exe.

Sets environment variables required for torch + numpy + Qt to coexist inside a
single PyInstaller bundle, and (on Windows) selects which torch build to use:
the downloaded GPU build if one is installed, otherwise the CPU baseline that
ships inside the bundle. See cctv_yolo/gpu_runtime.py and cctv_yolo.spec.
"""
import os
import sys

# OpenMP duplicate-library fix. On Windows, numpy (MKL) and torch both pull in
# libiomp5md.dll; without this the process aborts on `import torch` with:
#   "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
#    already initialized."
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Pin native math libraries to a single thread. Detection/tracking runs on Qt
# worker threads in the frozen app; combined with the duplicate-OpenMP shim
# above, multi-threaded torch/OpenCV on a 2nd worker thread heap-corrupts the
# process on Windows (0xC0000374). Set before torch is ever imported. Batch
# throughput is preserved (parallel workers use one core each). See processor.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Suppress stray console windows. The production exe is windowed (no console of
# its own), but library code we call (ultralytics, and any tool that shells out)
# can spawn console subprocesses that flash a black terminal over the GUI — and
# a lingering one can take the app down with it when closed. Patch Popen's
# default creationflags to add CREATE_NO_WINDOW so no child ever allocates a
# console. This is a no-op for GUI children (e.g. explorer for "reveal in
# folder"), so the OpenLocationBar still works. Must run before any subprocess.
if sys.platform == "win32":
    try:
        import subprocess as _sp
        _CREATE_NO_WINDOW = 0x08000000
        _orig_popen_init = _sp.Popen.__init__

        def _quiet_popen_init(self, *args, **kwargs):
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | _CREATE_NO_WINDOW
            _orig_popen_init(self, *args, **kwargs)

        _sp.Popen.__init__ = _quiet_popen_init
    except Exception:
        pass


def _select_torch_dir():
    """Which directory should provide torch/torchvision (Windows frozen only).

    On Windows, torch is NOT frozen into the bundle (it's excluded in the spec)
    so a *downloaded* GPU build can override the baked CPU one — frozen modules
    in the PYZ would otherwise always win over sys.path. We return the dir to
    place FIRST on sys.path:

      1. %LocalAppData%\\CCTV-YOLO\\torch_runtime  — the downloaded GPU build,
         but only if its ``.torch_ready`` marker is present (an interrupted
         install must never be used), else
      2. <bundle>/torch_cpu_baseline               — the baked CPU build.

    Returns None on macOS/Linux and in dev runs, where torch resolves normally
    (on macOS it's baked into the bundle and uses MPS).
    """
    if sys.platform != "win32" or not hasattr(sys, "_MEIPASS"):
        return None
    local = os.environ.get("LOCALAPPDATA") or os.path.join(
        os.path.expanduser("~"), "AppData", "Local")
    gpu = os.path.join(local, "CCTV-YOLO", "torch_runtime")
    if os.path.isfile(os.path.join(gpu, ".torch_ready")) and \
            os.path.isdir(os.path.join(gpu, "torch", "lib")):
        return gpu
    baseline = os.path.join(sys._MEIPASS, "torch_cpu_baseline")
    if os.path.isdir(os.path.join(baseline, "torch")):
        return baseline
    return None


_torch_dir = _select_torch_dir()
if _torch_dir:
    # MUST be index 0 so `import torch` resolves to this dir and nowhere else.
    if _torch_dir not in sys.path:
        sys.path.insert(0, _torch_dir)
    # torch's own __init__ adds torch/lib to the DLL search on Windows, but we
    # add it too (belt-and-suspenders) so the bundled CUDA DLLs resolve.
    _libdir = os.path.join(_torch_dir, "torch", "lib")
    if os.path.isdir(_libdir):
        os.environ["PATH"] = _libdir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory") and sys.platform == "win32":
            try:
                os.add_dll_directory(_libdir)
            except OSError:
                pass
elif sys.platform == "win32" and hasattr(sys, "_MEIPASS"):
    # Frozen Windows build, but neither a downloaded GPU torch nor the baked
    # CPU baseline was found — the bundle is incomplete. Surface it (visible in
    # CCTV-YOLO-debug.exe) instead of a bare ModuleNotFoundError later.
    try:
        sys.stderr.write(
            "CCTV-YOLO: bundled CPU torch is missing from this build — "
            "reinstall the app.\n")
    except Exception:
        pass

# Make adjacent bundle DLLs (c10.dll/torch_cpu.dll for the baked CPU build, and
# crucially the MSVC runtime — vcruntime140/msvcp140 — that torch loads at
# import) discoverable. PyInstaller drops them in _MEIPASS, which isn't on the
# native DLL search path by default.
if hasattr(sys, "_MEIPASS"):
    bundle = sys._MEIPASS
    os.environ["PATH"] = bundle + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory") and sys.platform == "win32":
        try:
            os.add_dll_directory(bundle)
        except OSError:
            pass
