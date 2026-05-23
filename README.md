# CCTV-YOLO — Ground Truth Generator

Vehicle detection, tracking, and correction system for generating ground-truth
traffic data from CCTV footage. Native desktop app built on PySide6/Qt with
YOLOv8 + ByteTrack for automatic detection, a tabbed correction UI for human
review, and an end-to-end iterative learning loop.

## Repository layout

```
.
├── software2/         # The app (PySide6/Qt native desktop, v2.x)
├── docs/              # Long-form docs + planning archive
├── .planning/         # In-flight planning artifacts (PRDs, checkpoints)
├── UPDATE.md          # Historical changelog (waves 1 & 2)
└── README.md          # You are here
```

The `data/`, `config/`, `models/`, and `logs/` folders at the repo root are
created by the app at runtime when launched in dev mode from inside this
checkout (macOS case-insensitive FS quirk — see
`software2/cctv_yolo/data_manager.py`). They're all gitignored.

## What's in the app (v2.x)

Tabbed PySide6 desktop app with nine tabs and a global menu-bar OpenLocationBar:

| # | Tab          | What it does                                                                                       |
|---|--------------|----------------------------------------------------------------------------------------------------|
| 1 | Preprocessing| Single-video detect+track with cancel/ETA/presets, ROI drawing, recent runs sidebar                |
| 2 | Batch        | Pick any folder, tree view, parallel scheduler (1–100 workers), atomic cancel, Stop-after-N         |
| 3 | Correction   | Track editor + ROI panel + 9 export formats (COCO/YOLO/CVAT/MOT/CSV/Annotated MP4/Stills/PDF/Zip)   |
| 4 | Performance  | Per-session stats, Model A/B compare, Before/After renderer, Confusion matrix, Groups aggregation   |
| 5 | Analytics    | 9 inline-rendered sections (heatmap, OD matrix, speed, direction, clips, before/after, HTML report) |
| 6 | Insights     | 4 sub-tabs: Session / Group / Dataset / Multi — anomaly detection, dataset health, confusion matrix |
| 7 | Training     | Active-learning queue, "Build from unused corrections", manual+combine dataset modes, promote prompt|
| 8 | Models       | List/import/download/delete/rename, model A/B compare on single video or dataset val split          |
| 9 | Live         | Webcam/RTSP, A/B model side-by-side, per-ROI live counts + alerts, on-event recording               |

Cross-cutting:

- **Groups** — group sessions by attribute (Snow, Night, etc.); aggregated stats roll up through Performance / Analytics / Insights / Training.
- **ROIs** — drawn in Preprocessing or Correction; respected by every downstream tab + every export.
- **Iterative learning loop** — every save in Correction immediately surfaces in Training; "Build from unused corrections" only includes sessions newer than the last successful train.
- **First-run wizard** — picks data folder, optionally downloads yolov8n, walks the new user through the tabs.
- **Help → About** shows version, platform, data + log folder shortcuts.
- **Help → Show Log Folder** opens `~/Documents/CCTV-YOLO/logs/` (rotating 5 MB × 5 file handler).

Storage layout (`~/Documents/CCTV-YOLO/`):

```
data/{videos,tracks,corrections,exports,training,live}/
config/         model_config.json, nas.json, ui_state.json, session_groups.json,
                training_history.json, batch_registry.json, batch_session_map.json
models/         *.pt + sidecar .meta.json for trained models
logs/           app.log (rotating)
```

## Quick start (dev)

```bash
cd software2
./build_venv/bin/python run.py        # macOS / Linux

# Windows:
# .\build_venv\Scripts\python.exe run.py
```

If `build_venv/` doesn't exist yet, run one of the build scripts below first
(they create it as a side-effect even if you never produce an installer).

## Building installers

```bash
# macOS → dist/CCTV-YOLO.dmg
cd software2 && ./build_mac.sh

# Windows → dist\CCTV-YOLO\CCTV-YOLO.exe (+ optional installer via Inno Setup)
cd software2 && build_windows.bat
```

The Windows script now auto-detects Python compatibility and falls back from
CUDA to CPU torch wheels automatically if no GPU wheel exists for your Python
version. Recommended Python: 3.10, 3.11, or 3.12.

If `CCTV-YOLO.exe` won't open after building, run `CCTV-YOLO-debug.bat`
(shipped next to the exe) — it captures the real startup error to
`startup-output.log`.

## Data format

- **Videos:** `.mp4`, `.mov`, `.avi`, `.mkv`
- **Tracks JSON:** per-frame bboxes `[x1, y1, x2, y2]` (YOLO xyxy)
- **Corrections JSON:** non-destructive overlay of tracks + ROIs +
  `"_version": 2` schema field + `.bak` rotating backups
- **Exports:** COCO / YOLO / CVAT XML 1.1 / MOT Challenge / CSV per-track /
  CSV per-frame / Annotated MP4 / per-track stills / PDF summary /
  Review Pack zip

## Keyboard shortcuts (Review window)

| Key            | Action                       |
|----------------|------------------------------|
| Space          | Play / Pause                 |
| ← / →          | Step frame                   |
| V / B          | Select / draw mode           |
| D / C / M / S  | Delete / Class / Merge / Split track |
| N / P          | Copy box to next / prev frame|
| R              | Next review session          |
| Shift+R        | ROI rectangle                |
| Shift+P        | ROI polygon (double-click closes) |
| Ctrl+Z / Y     | Undo / Redo                  |
| Ctrl+S         | Save corrections             |
| Ctrl+1..9      | Jump to tab N (Main window)  |

## Requirements

- Python 3.10–3.12 (CPython, not PyPy)
- macOS 12+ or Windows 10/11
- Optional: NVIDIA GPU + CUDA drivers (auto-detected) for ~20× faster
  inference; Apple Silicon uses MPS automatically for ~10× speedup
- Optional: Tailscale + SMB share for NAS-mode video access

## License

Private research repository.
