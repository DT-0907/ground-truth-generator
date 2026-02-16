"""
CCTV-YOLO Desktop Application Entry Point.

Starts the Flask server and opens the default browser.
Creates data directories in ~/Documents/CCTV-YOLO/ on first run.
"""

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path


def get_app_data_dir():
    """Get the application data directory in the user's Documents folder."""
    return Path.home() / "Documents" / "CCTV-YOLO"


def ensure_directories(base):
    """Create all necessary data directories on first run."""
    for sub in [
        "data/videos",
        "data/tracks",
        "data/corrections",
        "data/exports",
        "config",
    ]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def get_resource_path(relative_path):
    """Get absolute path to resource — works for both dev and PyInstaller bundle."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).parent / relative_path


def main():
    port = 5005
    app_dir = get_app_data_dir()
    ensure_directories(app_dir)

    # Tell the Flask app where to find data
    os.environ["CCTV_YOLO_DATA_DIR"] = str(app_dir)

    # Import after env is set so server.py picks it up at module load
    from cctv_yolo.server import app

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    print("=" * 50)
    print("  CCTV-YOLO  —  Vehicle Track Correction")
    print("=" * 50)
    print(f"  Data folder : {app_dir}")
    print(f"  Server      : http://localhost:{port}")
    print(f"  Press Ctrl+C to stop")
    print("=" * 50)

    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
