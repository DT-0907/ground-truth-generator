# Ground Truth Generator

Vehicle detection, tracking, and correction system for generating ground-truth traffic data from CCTV footage. Uses YOLOv8 + ByteTrack for automatic detection and provides a correction UI for human review.

## Directory Structure

```
.
├── software/          # Desktop app v1 — Flask + browser
├── software2/         # Desktop app v2 — native PySide6/Qt
├── test/              # Original development codebase (Flask web app)
├── claude-work/       # Development workspace
├── data/              # Shared local data (videos, tracks, corrections, exports)
└── config/            # Application configuration
```

## Versions

### `software2/` — Native Desktop App (Recommended)

The latest version. A native desktop application built with **PySide6/Qt** that runs as a standalone program (no browser needed).

**Features:**
- Native window with tabbed interface (Sessions, Videos, Settings)
- Video processing with YOLOv8 + ByteTrack in background threads
- Track editing: select, draw boxes, delete, merge, split, change class
- ROI drawing (rectangle and polygon) with per-ROI vehicle counting
- Undo/redo system (Ctrl+Z / Ctrl+Y)
- NAS integration via Tailscale SMB
- COCO format export
- Builds to `.dmg` (macOS) and `.exe` (Windows) via PyInstaller

**Quick Start:**
```bash
cd software2
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
python run.py
```

**Build Installers:**
```bash
# macOS → .dmg
chmod +x build_mac.sh && ./build_mac.sh

# Windows → .exe
build_windows.bat
```

**Key Modules:**
| Module | Purpose |
|--------|---------|
| `main.py` | App entry point, dark Fusion theme |
| `main_window.py` | Main window with tabs, menu bar, status bar |
| `review_window.py` | Track correction editor with toolbar and canvas |
| `video_canvas.py` | OpenCV frame rendering on Qt canvas |
| `track_sidebar.py` | Track list, filters, ROI panel |
| `sessions_tab.py` | Processed sessions with stats cards |
| `videos_tab.py` | Video management and processing |
| `settings_tab.py` | Local folders and NAS connection config |
| `data_manager.py` | Data access layer (files, sessions, tracks) |
| `processor.py` | YOLOv8 detection + ByteTrack tracking |
| `feedback.py` | Correction analysis + COCO export |
| `processing.py` | Background processing worker (QThread) |
| `nas_manager.py` | SMB mount/unmount for NAS shares |
| `dialogs.py` | Class change, merge, new track, ROI naming dialogs |

---

### `software/` — Desktop App v1 (Flask + Browser)

The first packaged version. Runs a local Flask server and opens the UI in your default browser.

**Quick Start:**
```bash
cd software
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
# Opens http://localhost:5005 in your browser
```

**How it works:**
- `main.py` starts Flask on port 5005 and opens the browser
- `server.py` handles all HTTP routes (API + HTML templates)
- Data stored in `~/Documents/CCTV-YOLO/`
- Same processing pipeline (YOLOv8 + ByteTrack) as v2

---

### `test/` — Original Development Codebase

The original Flask web application with the full detection pipeline, correction UI, and lane-based counting logic. This is the development version that `software/` and `software2/` were built from.

```bash
cd test
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python ui/app.py
# Server runs on http://localhost:5005
```

**Structure:**
- `src/` — Core processing (detection, tracking, correction, counting, pipeline)
- `ui/` — Flask app with Jinja templates (`index.html`, `review.html`)
- `config/` — Camera and ROI configurations
- `data/` — Videos, tracks, corrections, exports

---

### `claude-work/` — Development Workspace

Working directory used during development. Contains the Flask backend (`ui/app.py`), processing modules (`src/`), and a Python virtual environment.

## Data Format

All versions use the same data format:

- **Videos:** `.mp4`, `.mov`, `.avi`, `.mkv` files
- **Tracks:** JSON files with per-frame bounding boxes `[x1, y1, x2, y2]` (YOLO xyxy format)
- **Corrections:** JSON files with corrected tracks + ROI definitions (non-destructive, saved separately)
- **Exports:** COCO format JSON for training data

Data is stored in `~/Documents/CCTV-YOLO/data/` by default (desktop apps) or `./data/` (Flask dev server).

## Keyboard Shortcuts (Review Window)

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| Left / Right | Step frame |
| V | Select mode |
| B | Draw box mode |
| D | Delete track |
| C | Change class |
| M | Merge tracks |
| S | Split track |
| N | Copy box to next frame |
| P | Copy box to previous frame |
| R | Next review session |
| Shift+R | ROI rectangle mode |
| Shift+P | ROI polygon mode |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+S | Save corrections |
| Escape | Cancel / Select mode |

## Requirements

- Python 3.10+
- YOLOv8 model (auto-downloaded by ultralytics on first run)
- macOS or Windows (for desktop builds)
- Optional: Tailscale + NAS for remote video access

## License

Private repository.
