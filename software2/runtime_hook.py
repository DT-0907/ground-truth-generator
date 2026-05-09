"""
PyInstaller runtime hook — runs before main() in the frozen exe.

Sets environment variables required for torch + numpy + Qt to coexist
inside a single PyInstaller bundle on Windows.
"""
import os
import sys

# OpenMP duplicate-library fix. On Windows, numpy (MKL) and torch both pull
# in libiomp5md.dll; without this the process aborts on import torch with:
#   "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
#    already initialized."
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Make adjacent DLLs (c10.dll, torch_cpu.dll) discoverable. PyInstaller drops
# them in _MEIPASS on Windows but Python's DLL search path doesn't include it
# by default for transitive native loads.
if hasattr(sys, "_MEIPASS"):
    bundle = sys._MEIPASS
    os.environ["PATH"] = bundle + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory") and sys.platform == "win32":
        try:
            os.add_dll_directory(bundle)
        except OSError:
            pass
