# Setup Guide - What to Download Before Starting

> **Note**: For the most up-to-date installation instructions, see `INSTALLATION.md`

## 📦 Required Downloads

### 1. Python Packages (Automatic via pip)

All Python packages will be installed via `pip install -r requirements.txt`. No manual downloads needed.

**Key packages:**
- `ultralytics` - YOLO detection (includes YOLOv8 models)
- `opencv-python` - Video/image processing
- `flask` - Web UI for correction tool
- `pandas` - Data processing
- `torch` - PyTorch (for YOLO)

**Note**: ByteTrack is NOT in requirements.txt - install separately (see below)

### 2. YOLO Model Weights (Automatic Download)

**Good news:** Ultralytics YOLO automatically downloads model weights on first use. You don't need to manually download them.

**However, if you want to pre-download or use specific models:**

#### YOLOv8 Models (Recommended)
- **YOLOv8n (nano)** - Fastest, smallest: ~6MB
  - Best for: Real-time processing, low-resource systems
  - URL: Auto-downloaded, or from https://github.com/ultralytics/assets/releases
  
- **YOLOv8s (small)** - Balanced: ~22MB
  - Best for: Good accuracy/speed balance
  
- **YOLOv8m (medium)** - Better accuracy: ~52MB
  - Best for: Production accuracy (recommended)
  
- **YOLOv8l (large)** - High accuracy: ~88MB
- **YOLOv8x (xlarge)** - Best accuracy: ~136MB

**Download command:**
```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')  # Downloads automatically if not present
```

**Manual download (optional):**
- Visit: https://github.com/ultralytics/assets/releases
- Download `yolov8m.pt` (or your preferred size)
- Place in: `models/yolo/yolov8m.pt`

### 3. System Dependencies

#### macOS (your system)
```bash
# FFmpeg (usually pre-installed, but if not):
brew install ffmpeg

# Python 3.8+ (check with: python3 --version)
# If needed: brew install python3
```

#### Linux
```bash
sudo apt-get update
sudo apt-get install ffmpeg python3-pip
```

#### Windows
- Download FFmpeg from: https://ffmpeg.org/download.html
- Add to PATH

### 4. Sample Videos (Optional, for Testing)

You'll need your own CCTV videos, but for testing you can use:
- Any MP4/MOV video file
- Recommended: 1080p or 720p, 15-30 FPS
- Daytime traffic scenes work best initially

## 🚀 Installation Steps

### Step 1: Create Virtual Environment (Recommended)
```bash
cd "/Users/delberttran/Documents/cctv yolo test/test"
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR on Windows: venv\Scripts\activate
```

### Step 2: Install Python Packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected download sizes:**
- Core packages: ~500MB-1GB (includes PyTorch)
- YOLO model (first use): ~50MB (yolov8m.pt)
- Total: ~1-2GB depending on your system

### Step 3: Install ByteTrack (Required)
```bash
pip install git+https://github.com/ifzhang/ByteTrack.git
```

**Alternative** (if GitHub install fails):
```bash
pip install bytetrack
```

### Step 4: Verify Installation
```bash
python scripts/verify_setup.py
```

This will:
- Check all imports
- Download YOLO model if needed
- Test video reading
- Verify ByteTrack tracking library

## 📋 Pre-Download Checklist

Before running the pipeline, ensure:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] `requirements.txt` packages installed
- [ ] FFmpeg available (check with `ffmpeg -version`)
- [ ] At least 2GB free disk space
- [ ] Sample video file ready (for testing)

## 🔍 What Gets Downloaded Automatically

When you first run the code:

1. **YOLO model weights** (if not pre-downloaded)
   - Location: `~/.ultralytics/weights/` or `models/yolo/`
   - Size: 50-150MB depending on model

2. **COCO class names** (built into ultralytics)
   - Vehicle classes: car, truck, bus, motorcycle

3. **Tracking dependencies** (via pip)

## ⚠️ Important Notes

### Model Selection
- Start with **YOLOv8m** (medium) for best balance
- If processing is too slow, switch to **YOLOv8n** (nano)
- If accuracy is insufficient, try **YOLOv8l** (large)

### GPU vs CPU
- **GPU (CUDA)**: Much faster (10-50x speedup)
  - Requires: NVIDIA GPU + CUDA toolkit
  - PyTorch will auto-detect if available
- **CPU**: Works fine, just slower
  - No additional setup needed

### Internet Connection
- First run needs internet to download model weights
- After that, works offline

## 🎯 Next Steps After Setup

1. **Place your videos:**
   ```bash
   cp /path/to/your/video.mp4 data/videos/cam_001_1.mp4
   ```

2. **Process a video:**
   ```bash
   python scripts/process_video.py --video-path data/videos/cam_001_1.mp4 --camera-id cam_001
   ```

3. **View results:**
   ```bash
   python scripts/view_counts.py --camera-id cam_001
   ```

4. **Launch correction UI (fully implemented & debugged!):**
   ```bash
   python ui/app.py
   # Open: http://localhost:5000
   ```
   
   **Features:**
   - Video player with track overlays
   - Delete, merge, split tracks
   - Change track classes
   - Save corrected annotations
   - Keyboard shortcuts for fast editing
   - Error handling and validation
   - Fully debugged and tested
   
   **See `CORRECTION_UI.md` for complete user guide.**

## ❓ Troubleshooting

### "Module not found" errors
- Ensure virtual environment is activated
- Re-run: `pip install -r requirements.txt`

### "CUDA out of memory"
- Use smaller YOLO model (yolov8n instead of yolov8m)
- Process videos in smaller chunks
- Reduce batch size in config

### "FFmpeg not found"
- Install FFmpeg: `brew install ffmpeg` (macOS)
- Verify: `ffmpeg -version`

### Model download fails
- Check internet connection
- Manually download from: https://github.com/ultralytics/assets/releases
- Place in `models/yolo/` directory

