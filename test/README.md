# Vehicle Counting System - Lane-Based Traffic Analysis

## Project Overview

This system processes **your own 5-minute traffic camera video clips** to produce ground-truth lane-based vehicle counts by class through:
1. **Automatic Detection**: YOLO detects vehicles per frame
2. **Tracking**: Multi-object tracking assigns consistent IDs across frames
3. **Lane Counting**: ROI + lane-based counting logic
4. **Human Correction UI**: Fast review and correction tool
5. **Structured Outputs**: Per-object tracks, per-lane counts, time summaries

## System Architecture

```
┌─────────────────┐
│  Video Input    │ (Your 5-minute clips)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Frame Extract  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  YOLO Detection │──────▶│  Detections  │
└────────┬────────┘      └──────┬───────┘
         │                      │
         ▼                      ▼
┌─────────────────┐      ┌──────────────┐
│  MOT Tracking   │──────▶│   Tracks     │
└────────┬────────┘      └──────┬───────┘
         │                      │
         ▼                      ▼
┌─────────────────┐      ┌──────────────┐
│ Lane Assignment │──────▶│ Lane Counts │
└────────┬────────┘      └──────┬───────┘
         │                      │
         ▼                      ▼
┌─────────────────┐      ┌──────────────┐
│  Correction UI  │──────▶│ Ground Truth │
└─────────────────┘      └──────────────┘
```

## Project Structure

```
test/
├── config/                 # Configuration files
│   ├── cameras/           # Per-camera ROI/lane configs
│   └── models/            # Model configs
├── src/                   # Source code
│   ├── detection/         # YOLO detection module
│   ├── tracking/          # MOT tracking module
│   ├── counting/          # Lane counting logic
│   ├── correction/        # Correction UI (web app)
│   ├── utils/             # Utilities
│   └── pipeline/          # Main processing pipeline
├── data/                  # Data storage
│   ├── videos/            # Input videos (place your clips here)
│   ├── detections/        # Detection outputs (JSON)
│   ├── tracks/            # Tracking outputs (JSON)
│   ├── annotations/       # Ground truth annotations
│   └── analytics/         # Count summaries (CSV/Parquet)
├── models/                # Model weights (gitignored)
│   └── yolo/             # YOLO model files
├── ui/                    # Web UI frontend
│   ├── static/           # CSS, JS
│   ├── templates/        # HTML templates
│   └── app.py            # Flask/FastAPI backend
├── scripts/               # Utility scripts
│   ├── process_video.py  # Process single video
│   ├── batch_process.py  # Process multiple videos
│   ├── view_counts.py    # View results
│   ├── verify_setup.py   # Verify installation
│   └── download_models.py # Pre-download YOLO models
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### Installation (5 minutes)

```bash
# 1. Install Python packages
cd "/Users/delberttran/Documents/cctv yolo test/test"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Verify installation
python scripts/verify_setup.py
```

**See `INSTALLATION.md` for detailed setup instructions.**

### Process Your Videos

```bash
# 1. Place your 5-minute video clips
cp /path/to/your/video.mp4 data/videos/cam_001_1.mp4

# 2. Process a video
python scripts/process_video.py \
    --video-path data/videos/cam_001_1.mp4 \
    --camera-id cam_001

# 3. View results & correct
python scripts/view_counts.py --camera-id cam_001

# Launch correction UI (fully implemented!)
python ui/app.py
# Open: http://localhost:5000
```

**See `GETTING_STARTED.md` for detailed workflow.**

## 📦 What to Download

**Automatic (no action needed):**
- ✅ Python packages (~1.5-2GB) via `pip install -r requirements.txt`
- ✅ YOLO models (~50MB) on first use

**Manual (not required):**
- ✅ Tracking: Built into Ultralytics (no install needed)

**System tools (check):**
- FFmpeg: Usually pre-installed (`ffmpeg -version`)

**See `WHAT_TO_DOWNLOAD.md` for complete details.**

## 📚 Documentation

- **`INSTALLATION.md`** - Complete installation guide ⭐ **START HERE**
- **`GETTING_STARTED.md`** - Quick start for video processing
- **`VIDEO_WORKFLOW.md`** - Detailed video processing guide
- **`CORRECTION_UI.md`** - Correction UI user guide ⭐
- **`DESIGN_DOCUMENT.md`** - Complete technical design document 📘
- **`SLIDESHOW_OUTLINE.md`** - Presentation outline for slides 📊
- **`QUICK_START.md`** - Quick reference
- **`WHAT_TO_DOWNLOAD.md`** - What you need to download
- **`INSTALL_TRACKING.md`** - Tracking details (Ultralytics built-in)
- **`PROJECT_PLAN.md`** - Complete system architecture

## Data Formats

### Camera Configuration
See `config/cameras/example_camera.json`

### Detection Output
JSON per frame with bboxes, classes, confidences

### Track Output
JSON with track_id, class, frames array

### Analytics Output
CSV/Parquet with counts by lane/class/time

## Processing Time

For a **5-minute video** (~9,000 frames at 30 FPS):

| Hardware | Time |
|----------|------|
| GPU (NVIDIA) | ~2-5 minutes |
| CPU (Mac M1/M2) | ~10-20 minutes |
| CPU (Intel) | ~20-40 minutes |

## Next Steps

1. **Install dependencies** (see `INSTALLATION.md`)
2. **Place your videos** in `data/videos/`
3. **Process your first video** (see `GETTING_STARTED.md`)
4. **Review results** and use correction UI (see `CORRECTION_UI.md`)

## Support

- Check `INSTALLATION.md` for setup issues
- Check `VIDEO_WORKFLOW.md` for processing questions
- Check `CORRECTION_UI.md` for UI troubleshooting ⭐
- Check `PROJECT_PLAN.md` for architecture details
