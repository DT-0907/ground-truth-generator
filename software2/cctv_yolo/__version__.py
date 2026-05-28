"""Single source of truth for the app version.

main.py reads this for setApplicationVersion. cctv_yolo.spec reads it to
populate CFBundleVersion / CFBundleShortVersionString on macOS. The Inno
Setup installer for Windows gets a generated header from this same value
(see build_windows.bat).
"""

__version__ = "2.0.4"
__app_name__ = "CCTV-YOLO"
__org_name__ = "CCTV-YOLO"
__bundle_id__ = "com.cctv-yolo.app"
