# 📦 What You Need to Download - Complete List

## TL;DR - Quick Answer

**You don't need to manually download anything!** Everything is automatic:

1. ✅ **Python packages** → `pip install -r requirements.txt` (automatic)
2. ✅ **YOLO models** → Auto-download on first use (or pre-download with script)
3. ✅ **FFmpeg** → Usually pre-installed on macOS (check with `ffmpeg -version`)

## Detailed Breakdown

### 1. Python Packages (Automatic via pip)

**Command**: `pip install -r requirements.txt`

**What gets installed**:

| Package | Size | Purpose | Auto-download? |
|---------|------|---------|---------------|
| `ultralytics` | ~500MB | YOLO detection | ✅ Yes |
| `opencv-python` | ~50MB | Video/image processing | ✅ Yes |
| `byte-track` | ~10MB | Multi-object tracking | ✅ Yes |
| `flask` | ~5MB | Web UI framework | ✅ Yes |
| `pandas` | ~30MB | Data processing | ✅ Yes |
| `numpy` | ~20MB | Numerical operations | ✅ Yes |
| `torch` | ~500MB | PyTorch (for YOLO) | ✅ Yes |
| Other deps | ~100MB | Utilities | ✅ Yes |
| **TOTAL** | **~1-2GB** | | ✅ All automatic |

**No manual downloads needed** - pip handles everything!

### 2. YOLO Model Weights (Automatic or Optional Pre-download)

#### Option A: Automatic (Recommended)
Models download automatically when you first use them:

```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')  # Downloads automatically if not present
```

**Location**: `~/.ultralytics/weights/` or `models/yolo/`

#### Option B: Pre-download (Optional)
If you want to download models before running:

```bash
python scripts/download_models.py
```

Or manually:
- **URL**: https://github.com/ultralytics/assets/releases
- **Recommended**: `yolov8m.pt` (~52MB)
- **Fastest**: `yolov8n.pt` (~6MB)
- **Best accuracy**: `yolov8l.pt` (~88MB)

**Place in**: `models/yolo/yolov8m.pt`

### 3. System Tools

#### FFmpeg (Video Processing)
- **macOS**: Usually pre-installed
- **Check**: `ffmpeg -version`
- **If missing**: `brew install ffmpeg`
- **Size**: ~50MB

#### Python 3.8+
- **Check**: `python3 --version`
- **If missing**: `brew install python3` (macOS)

### 4. Sample Videos (Your Own)

You'll need your own CCTV videos:
- **Format**: MP4, MOV, AVI
- **Recommended**: 1080p or 720p, 15-30 FPS
- **Location**: `data/videos/`

## Installation Checklist

Before starting, ensure:

- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] Virtual environment created (`python3 -m venv venv`)
- [ ] Virtual environment activated (`source venv/bin/activate`)
- [ ] FFmpeg available (`ffmpeg -version`)
- [ ] Internet connection (for first-time downloads)
- [ ] ~2GB free disk space

## Step-by-Step Installation

### Step 1: Setup Environment
```bash
cd "/Users/delberttran/Documents/cctv yolo test/test"
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This will:**
- Download and install all Python packages
- Take 5-15 minutes depending on internet speed
- Download ~1-2GB total

### Step 3: Verify Installation
```bash
python scripts/verify_setup.py
```

**This will:**
- Check all packages are installed
- Test YOLO (downloads model if needed)
- Verify FFmpeg
- Check tracking library

### Step 4: (Optional) Pre-download Models
```bash
python scripts/download_models.py
```

**This will:**
- Download `yolov8n.pt` and `yolov8m.pt`
- Save to `models/yolo/`
- Takes ~1-2 minutes

## What Happens on First Run

When you first run the detection code:

1. **YOLO model downloads** (if not pre-downloaded)
   - Size: ~50MB (yolov8m.pt)
   - Location: `~/.ultralytics/weights/yolov8m.pt`
   - Time: ~30 seconds (depends on internet)

2. **COCO class names loaded** (built into ultralytics)
   - Vehicle classes: car (class 2), truck (class 7), bus (class 5), motorcycle (class 3)

3. **Tracking library initializes** (already installed via pip)

## Internet Requirements

**First-time setup:**
- Download Python packages: ~1-2GB
- Download YOLO model: ~50MB
- **Total**: ~1-2GB

**After setup:**
- Works offline (models cached locally)

## Storage Requirements

- **Python packages**: ~1-2GB
- **YOLO models**: ~50-150MB (depending on model size)
- **Data/videos**: Varies (your videos)
- **Total minimum**: ~2GB

## GPU vs CPU

### CPU (Default)
- ✅ Works out of the box
- ✅ No additional downloads
- ⚠️ Slower (but still usable)

### GPU (Optional, Faster)
- Requires: NVIDIA GPU + CUDA toolkit
- PyTorch auto-detects if available
- **10-50x faster** processing
- No additional downloads needed (CUDA support included in PyTorch)

## Summary

### ✅ Automatic (No Action Needed)
- All Python packages
- YOLO models (on first use)
- COCO class names

### 📥 Optional Manual Downloads
- YOLO models (if you want to pre-download)
- Sample test videos

### 🔧 System Requirements
- Python 3.8+ (usually pre-installed)
- FFmpeg (usually pre-installed on macOS)
- ~2GB disk space

## Quick Command Reference

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate

# 2. Install everything
pip install -r requirements.txt

# 3. Verify
python scripts/verify_setup.py

# 4. (Optional) Pre-download models
python scripts/download_models.py
```

**That's it!** Everything else is automatic. 🎉


