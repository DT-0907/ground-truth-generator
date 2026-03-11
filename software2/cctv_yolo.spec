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

# Collect ultralytics YAML configs (bytetrack.yaml, default.yaml, etc.)
ultralytics_datas = collect_data_files('ultralytics')
ultralytics_hiddenimports = collect_submodules('ultralytics')

a = Analysis(
    ['cctv_yolo/main.py'],
    pathex=[],
    binaries=[],
    datas=ultralytics_datas,
    hiddenimports=[
        'cctv_yolo',
        'cctv_yolo.main_window',
        'cctv_yolo.review_window',
        'cctv_yolo.video_canvas',
        'cctv_yolo.track_sidebar',
        'cctv_yolo.dialogs',
        'cctv_yolo.data_manager',
        'cctv_yolo.nas_manager',
        'cctv_yolo.processing',
        'cctv_yolo.sessions_tab',
        'cctv_yolo.videos_tab',
        'cctv_yolo.settings_tab',
        'cctv_yolo.processor',
        'cctv_yolo.feedback',
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'cv2',
        'numpy',
        'tqdm',
        'ultralytics',
        'torch',
        'torchvision',
    ] + ultralytics_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
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
