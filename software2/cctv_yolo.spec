# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for CCTV-YOLO v2 native desktop application.

Builds:
  macOS  -> CCTV-YOLO.app  (then wrapped in .dmg by build_mac.sh)
  Windows -> CCTV-YOLO.exe  (single-folder distribution)
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Bump recursion limit — torch + ultralytics + PySide6 hit the default 1000.
sys.setrecursionlimit(10000)

# ---- Collection helpers ---------------------------------------------------
# Each of these libs has its own native files (Qt plugins, DLLs, YAML configs)
# that PyInstaller's analyzer doesn't pick up automatically. Without these,
# the exe builds but crashes silently at startup with "Could not find Qt
# platform plugin 'windows'" or "ModuleNotFoundError: ultralytics.cfg".
ultralytics_datas = collect_data_files('ultralytics')
ultralytics_hiddenimports = collect_submodules('ultralytics')

pyside_datas = collect_data_files('PySide6')
pyside_hiddenimports = collect_submodules('PySide6')

cv2_datas = collect_data_files('cv2')

# Pull every cctv_yolo module — the package is loaded via dynamic tab imports
# so PyInstaller's static analysis misses about 30 of them.
cctv_hiddenimports = collect_submodules('cctv_yolo')

# Torch ships its DLLs as binaries; collect_dynamic_libs handles that, but on
# Windows we also need the version file and a few configs.
try:
    from PyInstaller.utils.hooks import collect_dynamic_libs
    torch_binaries = collect_dynamic_libs('torch')
except Exception:
    torch_binaries = []

a = Analysis(
    ['cctv_yolo/main.py'],
    pathex=[],
    binaries=torch_binaries,
    datas=ultralytics_datas + pyside_datas + cv2_datas,
    hiddenimports=cctv_hiddenimports + ultralytics_hiddenimports + pyside_hiddenimports + [
        'cv2',
        'numpy',
        'tqdm',
        'ultralytics',
        'torch',
        'torchvision',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Cut analysis time + binary size — none of these are imported.
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pandas',
        'scipy',
        'sklearn',
        'PIL.ImageQt',
        # PyQt5/6 collide with PySide6 if both are present in the venv.
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
        name='CCTV-YOLO.app',
        icon='cctv_yolo/resources/icon.icns' if Path('cctv_yolo/resources/icon.icns').exists() else None,
        bundle_identifier='com.cctv-yolo.app',
        info_plist={
            'CFBundleName': 'CCTV-YOLO',
            'CFBundleDisplayName': 'CCTV-YOLO',
            'CFBundleVersion': '2.0.0',
            'CFBundleShortVersionString': '2.0.0',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '10.15',
        },
    )
else:
    # Windows / Linux. console=True keeps the console open so the user sees
    # any startup error instead of a silent flash. Flip to False once stable.
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
