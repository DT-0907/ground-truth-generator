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

sys.setrecursionlimit(10000)

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
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pandas',
        'scipy',
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
