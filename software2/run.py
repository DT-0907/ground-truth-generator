#!/usr/bin/env python3
"""Dev quick-launch — runs the app without building (PRD B).

Requires the local ``build_venv`` to be populated with all deps.
If you see an ImportError on PySide6 or Ultralytics, run ``./build_mac.sh``
(or ``build_windows.bat``) once to set up the venv, then invoke this script
via ``./build_venv/bin/python run.py``.

Usage:
    ./build_venv/bin/python run.py        # macOS / Linux
    .\\build_venv\\Scripts\\python.exe run.py  # Windows
"""
from cctv_yolo.main import main

if __name__ == "__main__":
    main()
