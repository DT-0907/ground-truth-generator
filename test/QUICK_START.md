# Quick Start Guide

## 🚀 Installation (5 minutes)

### Step 1: Install Python Packages
```bash
cd "/Users/delberttran/Documents/cctv yolo test/test"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Verify
```bash
python scripts/verify_setup.py
```

**That's it!** See `INSTALLATION.md` for detailed instructions.

---

## 📹 Process Your Videos

### 1. Place Your Videos
```bash
# Copy your 5-minute video clips here:
cp /path/to/your/video.mp4 data/videos/cam_001_1.mp4
```

### 2. Process a Video
```bash
python scripts/process_video.py \
    --video-path data/videos/cam_001_1.mp4 \
    --camera-id cam_001
```

### 3. View Results & Correct
```bash
# View counts
python scripts/view_counts.py --camera-id cam_001

# Launch correction UI (fully implemented & debugged!)
python ui/app.py
# Open: http://localhost:5000
```

**Correction UI:** Fully functional with error handling. See `CORRECTION_UI.md` for details.

---

## 📦 What Gets Downloaded

**Automatic (no action needed):**
- ✅ Python packages (~1.5-2GB) via `pip install -r requirements.txt`
- ✅ YOLO models (~50MB) on first use
- ✅ COCO classes (built-in)

**Manual (not required):**
- ✅ Tracking: Built into Ultralytics (no install needed)

**System tools (check):**
- FFmpeg: Usually pre-installed (`ffmpeg -version`)

---

## ⚡ Quick Reference

```bash
# Install everything
pip install -r requirements.txt

# Process video
python scripts/process_video.py --video-path data/videos/your_video.mp4 --camera-id cam_001

# View results
python scripts/view_counts.py --camera-id cam_001
```

---

## 📚 More Information

- **Full installation**: See `INSTALLATION.md`
- **Video workflow**: See `VIDEO_WORKFLOW.md`
- **Correction UI**: See `CORRECTION_UI.md` ⭐
- **Getting started**: See `GETTING_STARTED.md`
- **What to download**: See `WHAT_TO_DOWNLOAD.md`

