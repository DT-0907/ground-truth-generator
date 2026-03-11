# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for CCTV-YOLO desktop application.

Builds:
  macOS  -> CCTV-YOLO.app  (then wrapped in .dmg by build_mac.sh)
  Windows -> CCTV-YOLO.exe  (single-folder distribution)
"""

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['cctv_yolo/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('cctv_yolo/templates', 'cctv_yolo/templates'),
    ],
    hiddenimports=[
        'cctv_yolo',
        'cctv_yolo.server',
        'cctv_yolo.processor',
        'cctv_yolo.feedback',
        'flask',
        'flask_cors',
        'jinja2',
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
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if sys.platform == 'darwin':
    # ---- macOS: .app bundle ----
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
        icon='assets/icon.icns' if Path('assets/icon.icns').exists() else None,
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
        icon='assets/icon.icns' if Path('assets/icon.icns').exists() else None,
        bundle_identifier='com.cctv-yolo.app',
        info_plist={
            'CFBundleName': 'CCTV-YOLO',
            'CFBundleDisplayName': 'CCTV-YOLO',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '10.15',
        },
    )
else:
    # ---- Windows / Linux: single-folder exe ----
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
        console=False,  # No console window on Windows
        icon='assets/icon.ico' if Path('assets/icon.ico').exists() else None,
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
