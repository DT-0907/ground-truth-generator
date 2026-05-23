# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for CCTV-YOLO v2 native desktop application.

Builds:
  macOS  -> CCTV-YOLO.app  (then wrapped in .dmg by build_mac.sh)
  Windows -> CCTV-YOLO.exe  (single-folder distribution)
"""

import ast
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

a = Analysis(
    ['cctv_yolo/main.py'],
    pathex=[],
    binaries=[],
    datas=ultralytics_datas,
    hiddenimports=cctv_hiddenimports + ultralytics_hiddenimports + [
        'cv2',
        'numpy',
        'tqdm',
        'ultralytics',
        'torch',
        'torchvision',
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
    ],
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
    ],
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
            'LSMinimumSystemVersion': '10.15',
        },
    )
else:
    # Windows / Linux. console=True keeps the console open so any startup
    # error (missing DLL, import failure) is visible instead of a silent
    # flash — and crash.log captures it for next time too.
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
        console=True,
        icon='cctv_yolo/resources/icon.ico' if Path('cctv_yolo/resources/icon.ico').exists() else None,
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
