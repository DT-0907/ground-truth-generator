# Vehicle Detection & Correction System - Setup Guide

## Overview

This system:
1. **Detects** vehicles in traffic camera videos using YOLOv8
2. **Tracks** them across frames (one ID per vehicle) using ByteTrack
3. **Lets you correct** mistakes via a web UI (merge lost tracks, draw ROIs, count vehicles)
4. **Exports** corrected data as COCO-format datasets for fine-tuning

---

## New User Setup (any computer)

### Prerequisites

- **Python 3.9+** (check with `python3 --version`)
- **pip** (usually comes with Python)
- **~2GB disk space** for dependencies + YOLO model
- **GPU optional** but recommended (10-50x faster processing)

### Setup (4 commands)

```bash
# 1. Navigate to project
cd path/to/claude-work

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the app
python ui/app.py
# Open: http://localhost:5005
```

That's it. No database, no external services. Everything runs locally.

The YOLO model (`yolov8m.pt`, ~50MB) downloads automatically on first use.

### What gets installed

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 detection + ByteTrack tracking |
| `opencv-python` | Video frame extraction |
| `flask` + `flask-cors` | Web UI server |
| `tqdm` | Progress bars for CLI processing |
| `numpy` | Numerical operations |

---

## Usage

### Option A: Process via Web UI (recommended)

```bash
python ui/app.py
```

1. Open **http://localhost:5005**
2. Click the **Videos** tab
3. All video files in `data/videos/` appear as cards with thumbnails
4. Click **Process** on any video — runs YOLO in the background
5. When done, click **Review** to correct tracks

### Option B: Process via CLI

```bash
# Single video
python process.py data/videos/traffic.mp4

# All videos in a directory
python process.py --batch data/videos/

# Use smaller/faster model
python process.py data/videos/traffic.mp4 --model yolov8n.pt

# Adjust confidence threshold
python process.py data/videos/traffic.mp4 --conf 0.3
```

**Processing time (5-minute video):**

| Hardware | Time |
|----------|------|
| NVIDIA GPU | ~2-5 min |
| Apple M1/M2/M3 | ~5-10 min |
| CPU only | ~15-30 min |

---

## Where Results Are Stored

```
data/
├── videos/                          # INPUT: your video files go here
│   └── traffic.mp4
│
├── tracks/                          # AUTO-GENERATED: raw YOLO detections
│   └── traffic.json                 #   bounding boxes + track IDs per frame
│
├── corrections/                     # USER-GENERATED: your corrected tracks
│   └── traffic.json                 #   saved when you hit Ctrl+S in the UI
│                                    #   also contains ROI definitions
│
└── exports/                         # OPTIONAL: COCO dataset for fine-tuning
    ├── images/                      #   extracted video frames as JPEGs
    │   ├── traffic_frame_000000.jpg
    │   ├── traffic_frame_000010.jpg
    │   └── ...
    └── annotations.json             #   COCO format bounding box annotations
```

### What each file contains

**`data/tracks/{video}.json`** — Raw processing output:
- Video metadata (fps, resolution, frame count)
- Array of tracks, each with: track_id, vehicle class, bounding boxes per frame
- Bounding box format: `[x1, y1, x2, y2]` (top-left to bottom-right, pixels)
- Tracks flagged `needs_review` if short (<5 frames) or low confidence (<0.4)

**`data/corrections/{video}.json`** — Your corrected version:
- Same structure as tracks, but with your edits applied
- Merged tracks include interpolated frames (marked `"interpolated": true`)
- Also stores ROI definitions (regions of interest with vehicle counts)
- Original track file is never modified

**`data/exports/`** — Only created when you run `python process.py --export`:
- Actual JPEG images extracted from videos
- COCO-format `annotations.json` with bounding boxes
- Ready for YOLO fine-tuning: `yolo detect train data=data/exports model=yolov8m.pt epochs=50`

---

## Correction UI Features

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| Left/Right | Previous/Next frame |
| V | Select mode |
| B | Draw box mode |
| D | Delete selected track |
| C | Change class |
| M | Merge tracks (with gap interpolation) |
| S | Split at current frame |
| N | Copy box to next frame |
| P | Copy box to previous frame |
| R | Jump to next track needing review |
| Shift+R | Draw rectangular ROI |
| Shift+P | Draw polygon ROI |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+S | Save corrections |
| Escape | Cancel/exit current mode |

### Track Merging (for lost-and-found objects)

When a vehicle disappears and reappears with a new track ID:
1. Select the first track
2. Press **M** (or click Merge)
3. Choose the second track from the dropdown
4. The UI shows how many gap frames will be interpolated
5. After merge: gap frames get linearly interpolated bounding boxes (shown as dashed lines)
6. The vehicle is now fully tracked across the entire video

### Regions of Interest (ROI)

- **Shift+R**: Draw a rectangular ROI (click and drag)
- **Shift+P**: Draw a polygon ROI (click vertices, double-click to finish)
- Each ROI shows in the sidebar with: total vehicle count + breakdown by class
- ROIs are saved with corrections and persist across sessions
- Click ROI name to rename, X to delete

### Videos Tab

- Shows all videos in `data/videos/` with thumbnails
- Process videos directly from the browser (runs in background)
- Choose YOLO model size and confidence threshold
- "Process All Unprocessed" button for batch processing
- Status polling — cards update automatically when processing finishes

---

## Workflow

```
1. ADD VIDEOS
   Put .mp4/.mov/.avi/.mkv files in data/videos/

2. PROCESS (either way)
   Via UI:  Videos tab → click Process
   Via CLI: python process.py data/videos/video.mp4
   Creates: data/tracks/video.json

3. CORRECT
   Sessions tab → click Review
   - Delete false positives (D)
   - Fix class labels (C)
   - Merge lost-and-found tracks (M) — auto-interpolates gaps
   - Split incorrectly merged tracks (S)
   - Draw new bounding boxes (B)
   - Draw ROIs for vehicle counting (Shift+R, Shift+P)
   - Save with Ctrl+S → data/corrections/video.json

4. BUILD FEEDBACK (optional, after correcting several videos)
   python process.py --feedback
   Creates: config/feedback.json
   Future processing runs automatically use adjusted thresholds

5. EXPORT COCO DATASET (optional, for fine-tuning)
   python process.py --export
   Creates: data/exports/images/ + data/exports/annotations.json
```

---

## Project Structure

```
claude-work/
├── process.py              # CLI entry point (process, feedback, export)
├── requirements.txt        # Python dependencies
├── SETUP.md                # This file
├── yolov8m.pt              # YOLO model (auto-downloaded on first use)
│
├── src/
│   ├── processor.py        # YOLOv8 detection + ByteTrack tracking pipeline
│   └── feedback.py         # Correction analysis + COCO export
│
├── ui/
│   ├── app.py              # Flask web server (port 5005)
│   └── templates/
│       ├── index.html      # Home page (Sessions + Videos tabs)
│       └── review.html     # Track correction interface
│
├── data/
│   ├── videos/             # INPUT: your video files
│   ├── tracks/             # Raw YOLO output (JSON)
│   ├── corrections/        # User corrections (JSON)
│   └── exports/            # COCO dataset (when exported)
│
├── config/
│   └── feedback.json       # Auto-generated confidence adjustments
│
└── venv/                   # Python virtual environment (not committed)
```

---

## Optional: NAS Integration (Tailscale + UGREEN)

If you have a UGREEN NAS and Tailscale:

1. Install Tailscale on the NAS (via Docker or UGOS app center)
2. Enable SMB sharing on the NAS, create a `cctv-yolo/videos/` folder
3. Mount it on your machine:
   ```bash
   # macOS
   mount_smbfs //user:pass@100.x.x.x/cctv-yolo/videos /Volumes/nas-videos

   # Then symlink into the project
   mv data/videos data/videos_backup
   ln -s /Volumes/nas-videos data/videos
   ```
4. Videos on the NAS now appear in the Videos tab automatically

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'ultralytics'"**
Make sure you activated the virtual environment: `source venv/bin/activate`

**"Video not found" in UI**
Check that the video file is in `data/videos/` and the filename matches the track file stem

**Processing is very slow**
Try a smaller model: select YOLOv8n in the Videos tab, or `--model yolov8n.pt` on CLI

**No tracks detected**
Lower the confidence threshold: set to 0.15 in the Videos tab, or `--conf 0.15` on CLI

**GPU not being used**
Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

---

## YOLO Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n.pt | 6MB | Fastest | Good |
| yolov8s.pt | 22MB | Fast | Better |
| yolov8m.pt | 52MB | Medium | Best (default) |
| yolov8l.pt | 88MB | Slow | Best |

## Vehicle Classes Detected

Car, Truck, Bus, Motorcycle, Bicycle (from COCO dataset)
