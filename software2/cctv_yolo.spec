# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for CCTV-YOLO v2 native desktop application.

Builds:
  macOS  -> CCTV-YOLO.app  (then wrapped in .dmg by build_mac.sh)
  Windows -> CCTV-YOLO.exe  (single-folder distribution)
"""

import ast
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

sys.setrecursionlimit(10000)


def _read_version_constants():
    """Pull __version__ etc. out of cctv_yolo/__version__.py via AST so we don't
    have to import the cctv_yolo package (which would drag torch/Qt into the
    spec process). PRD C2 — single source of truth for the version string.
    """
    tree = ast.parse(Path("cctv_yolo/__version__.py").read_text())
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and isinstance(node.value, ast.Constant):
                consts[tgt.id] = node.value.value
    return consts


_VC          = _read_version_constants()
APP_VERSION  = _VC["__version__"]
APP_NAME     = _VC["__app_name__"]
BUNDLE_ID    = _VC["__bundle_id__"]

# ultralytics ships YAML configs (bytetrack.yaml, default.yaml) that aren't
# importable Python — collect_data_files grabs them so YOLO can find them
# at runtime. PySide6 / cv2 / torch are deliberately NOT manually collected:
# PyInstaller has well-tested hooks for those, and double-collecting them
# can produce conflicting copies of qwindows.dll or c10.dll.
ultralytics_datas = collect_data_files('ultralytics')
ultralytics_hiddenimports = collect_submodules('ultralytics')

# Pull every cctv_yolo module — tabs are loaded via dynamic imports so
# PyInstaller's static analysis misses about 30 of them.
cctv_hiddenimports = collect_submodules('cctv_yolo')

IS_WIN = sys.platform == 'win32'

# torch's pure-Python deps (GPU-independent). Normally pulled in transitively,
# but on Windows we EXCLUDE torch from analysis (below), which cuts that import
# chain — so list them explicitly so BOTH the baked CPU torch and a downloaded
# GPU torch import cleanly.
_TORCH_PY_DEPS = [
    'filelock', 'typing_extensions', 'sympy', 'mpmath',
    'networkx', 'jinja2', 'markupsafe', 'fsspec', 'setuptools',
]

if IS_WIN:
    # HYBRID packaging (see cctv_yolo/gpu_runtime.py): torch is NOT frozen into
    # the PYZ — a frozen torch can't be overridden by a downloaded GPU build,
    # because PyInstaller's frozen importer beats sys.path. Instead we EXCLUDE
    # torch/torchvision and ship the CPU build as a data tree under
    # torch_cpu_baseline/. runtime_hook.py puts the active one (downloaded GPU
    # build if installed, else this CPU baseline) first on sys.path.
    from PyInstaller.building.datastruct import Tree
    _torch_extra_hidden = _TORCH_PY_DEPS
    _torch_excludes = ['torch', 'torchvision']
    _site = Path(sys.prefix) / 'Lib' / 'site-packages'
    _torch_cpu_tree = []
    for _pkg in ('torch', 'torchvision'):
        _src = _site / _pkg
        if not _src.is_dir():
            raise SystemExit(
                f"cctv_yolo.spec: CPU {_pkg} not found at {_src}. The hybrid "
                f"Windows build stages the CPU torch — run build_windows.bat "
                f"(it installs torch+torchvision from the cpu index first).")
        _torch_cpu_tree += Tree(str(_src), prefix='torch_cpu_baseline/' + _pkg)
        # CRITICAL: stage the sibling *.dist-info NEXT TO the package, mirroring
        # a normal site-packages layout. ultralytics runs
        # importlib.metadata.version("torchvision") at IMPORT time; with no
        # dist-info on sys.path that raises PackageNotFoundError and crashes
        # `from ultralytics import YOLO` on every fresh CPU install. (The
        # downloaded GPU wheels already carry their own dist-info, so that path
        # is fine — this only affects the baked CPU baseline.)
        _di = sorted(_site.glob(_pkg + '-*.dist-info'))
        if not _di:
            raise SystemExit(
                f"cctv_yolo.spec: no {_pkg}-*.dist-info beside {_src} — "
                f"importlib.metadata.version('{_pkg}') would crash at runtime.")
        _torch_cpu_tree += Tree(str(_di[0]), prefix='torch_cpu_baseline/' + _di[0].name)

    # Sibling top-level packages that ship INSIDE the torch wheel (next to
    # torch/ in site-packages) and are imported by torch AT IMPORT TIME — e.g.
    # torch.utils._python_dispatch does `import torchgen`. They carry no
    # separate dist-info. Because torch is EXCLUDED from analysis (above),
    # PyInstaller never follows torch's imports and so never discovers them,
    # and they are NOT inside the torch/ tree staged above — so without this
    # the frozen app dies on first launch with
    # "ModuleNotFoundError: No module named 'torchgen'".
    for _pkg in ('torchgen', 'functorch'):
        _src = _site / _pkg
        if _src.is_dir():
            _torch_cpu_tree += Tree(str(_src), prefix='torch_cpu_baseline/' + _pkg)

    # CRITICAL: torch's DLLs (torch_cpu.dll/c10.dll/fbgemm.dll) dynamically link
    # the MSVC C/C++ runtime — msvcp140.dll, vcruntime140.dll, and especially
    # vcruntime140_1.dll (the VS2019+ EH runtime). These are NOT inside the
    # torch wheel. Because we EXCLUDE torch from analysis, PyInstaller never
    # inspects torch's DLLs and so never learns to collect them — they'd reach
    # the bundle only by luck of another dep referencing them. Bundle them
    # EXPLICITLY so `import torch` works on clean Windows boxes with no system
    # VC++ redistributable (the whole target audience).
    _sys32 = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32'
    _vc_dlls = []
    for _d in ('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'vcomp140.dll'):
        _p = _sys32 / _d
        if _p.is_file():
            _vc_dlls.append((str(_p), '.'))
else:
    # macOS / Linux: bake torch normally (macOS uses MPS — one variant, no
    # download needed; the GPU-download feature is gated to win32).
    _torch_extra_hidden = ['torch', 'torchvision']
    _torch_excludes = []
    _torch_cpu_tree = []
    _vc_dlls = []

a = Analysis(
    ['cctv_yolo/main.py'],
    pathex=[],
    binaries=_vc_dlls,
    datas=ultralytics_datas,
    hiddenimports=cctv_hiddenimports + ultralytics_hiddenimports + [
        'cv2',
        'numpy',
        'tqdm',
        'ultralytics',
        # NOTE: torch/torchvision are added via _torch_extra_hidden below — on
        # Windows they are EXCLUDED from the bundle and shipped as a data tree
        # instead (hybrid GPU packaging), so they must NOT be hardcoded here.
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        # Transitive deps of newer Ultralytics versions. Without these
        # the Windows .exe crashes on first model load with
        # "ModuleNotFoundError: No module named 'matplotlib'" (raised
        # from ultralytics.models.yolo.semantic.train).
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_agg',
        'pandas',
        'scipy',
        'scipy.spatial',
        'scipy.ndimage',
        'yaml',                  # ByteTrack tracker config
        'psutil',                # ultralytics.utils.checks
        'seaborn',               # ultralytics.utils.plotting
        'PIL',                   # ultralytics image utils
        'PIL.Image',
    ] + _torch_extra_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[
        # NOTE: do NOT exclude matplotlib/pandas/scipy here — newer
        # Ultralytics versions import them eagerly via
        # ultralytics.models.yolo.semantic.train and ultralytics.utils.
        # Excluding them broke the Windows .exe with cryptic
        # "ModuleNotFoundError: No module named 'matplotlib'" on first
        # model load.
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'sklearn',
        'PIL.ImageQt',
        'PyQt5',
        'PyQt6',
    ] + _torch_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

if sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='CCTV-YOLO',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon='cctv_yolo/resources/icon.icns' if Path('cctv_yolo/resources/icon.icns').exists() else None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        name='CCTV-YOLO',
    )
    app = BUNDLE(
        coll,
        name=f'{APP_NAME}.app',
        icon='cctv_yolo/resources/icon.icns' if Path('cctv_yolo/resources/icon.icns').exists() else None,
        bundle_identifier=BUNDLE_ID,
        info_plist={
            'CFBundleName': APP_NAME,
            'CFBundleDisplayName': APP_NAME,
            'CFBundleVersion': APP_VERSION,
            'CFBundleShortVersionString': APP_VERSION,
            'NSHighResolutionCapable': True,
            # Match the real floor of the bundled PySide6 6.5+/torch wheels
            # (and the README's stated "macOS 12+"). The old 10.15 made Launch
            # Services try to run on Catalina, then dyld-crash mid-import.
            'LSMinimumSystemVersion': '12.0',
        },
    )
else:
    # Windows / Linux. Two exes ship in the same bundle:
    #   CCTV-YOLO.exe        -> windowed (no console). Users double-click this.
    #                           Cleaner UI: no black cmd window flashes alongside
    #                           the GUI.
    #   CCTV-YOLO-debug.exe  -> console=True. CCTV-YOLO-debug.bat calls this so
    #                           native crash output (missing DLL, OMP errors,
    #                           Qt platform plugin failures) is visible.
    # PyInstaller supports multiple EXE entries that share one COLLECT bundle —
    # the .exe stubs are tiny, the heavy binaries/datas are referenced once.
    icon_path = ('cctv_yolo/resources/icon.ico'
                 if Path('cctv_yolo/resources/icon.ico').exists() else None)
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='CCTV-YOLO',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon=icon_path,
    )
    exe_debug = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='CCTV-YOLO-debug',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        icon=icon_path,
    )
    coll = COLLECT(
        exe,
        exe_debug,
        a.binaries,
        a.zipfiles,
        a.datas,
        _torch_cpu_tree,   # CPU torch/torchvision staged under torch_cpu_baseline/
        strip=False,
        upx=False,
        name='CCTV-YOLO',
    )
