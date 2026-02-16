# Complete Installation Guide

## 🎯 What You Need to Download

**Short answer**: Almost nothing! Everything downloads automatically.

### ✅ Automatic (No Manual Downloads)

1. **Python packages** → `pip install -r requirements.txt`
2. **YOLO models** → Auto-download on first use
3. **COCO classes** → Built into ultralytics

### 📥 Manual Install (Not Required!)

**No manual installs needed!** Ultralytics includes built-in tracking.

### 🔧 System Tools (Check if Installed)

- **FFmpeg** → Usually pre-installed on macOS
- **Python 3.8+** → Usually pre-installed

---

## 📋 Step-by-Step Installation

### Step 1: Navigate to Project

```bash
cd "/Users/delberttran/Documents/cctv yolo test/test"
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

**Note**: You should see `(venv)` in your terminal prompt.

### Step 3: Install Python Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This downloads and installs** (~1.5-2GB, takes 5-15 minutes):
- Ultralytics YOLO (~500MB)
- OpenCV (~50MB)
- PyTorch (~500MB)
- Pandas, NumPy, Flask, etc. (~500MB)

### Step 4: Verify Installation

```bash
python scripts/verify_setup.py
```

This checks:
- ✓ All Python packages installed
- ✓ YOLO model (downloads if needed)
- ✓ OpenCV working
- ✓ Tracking capability (Ultralytics built-in)
- ✓ FFmpeg available

### Step 5: (Optional) Pre-download YOLO Models

```bash
python scripts/download_models.py
```

Downloads `yolov8n.pt` and `yolov8m.pt` (~60MB total).

---

## 📊 What Gets Downloaded

| Item | Size | When | Location |
|------|------|------|----------|
| Python packages | ~1.5-2GB | Step 3 | `venv/` directory |
| YOLO model | ~50MB | First use or Step 5 | `~/.ultralytics/weights/` |

**Total disk space needed**: ~2GB

---

## ✅ Installation Checklist

Before you start processing videos:

- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] Virtual environment created and activated
- [ ] `pip install -r requirements.txt` completed
- [ ] `python scripts/verify_setup.py` passes all checks
- [ ] FFmpeg available (`ffmpeg -version`)
- [ ] At least 2GB free disk space

---

## 🚀 Quick Start After Installation

### 1. Place Your Videos

```bash
# Copy your 5-minute video clips to:
cp /path/to/your/video.mp4 data/videos/cam_001_1.mp4
```

### 2. Process a Video

```bash
python scripts/process_video.py \
    --video-path data/videos/cam_001_1.mp4 \
    --camera-id cam_001
```

### 3. View Results & Correct Tracks

```bash
# View count summaries
python scripts/view_counts.py --camera-id cam_001

# Launch correction UI (fully implemented & debugged!)
python ui/app.py
# Open: http://localhost:5000
```

**Correction UI Features:**
- ✅ Video player with frame-by-frame navigation
- ✅ Track overlay on video (bounding boxes with IDs)
- ✅ Track list with filtering and sorting
- ✅ Delete, merge, split tracks
- ✅ Change track classes
- ✅ Save corrected annotations
- ✅ Keyboard shortcuts (D/C/S/M keys)
- ✅ Error handling and validation

**See `CORRECTION_UI.md` for complete user guide.**

---

## ❓ Troubleshooting

### "Module not found" after installation

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall packages
pip install -r requirements.txt
```

### "FFmpeg not found"

```bash
# macOS
brew install ffmpeg

# Verify
ffmpeg -version
```

### "YOLO model download fails"

- Check internet connection
- Models auto-download on first use
- Or manually download from: https://github.com/ultralytics/assets/releases
- Place in `models/yolo/yolov8m.pt`

### "CUDA out of memory" (GPU users)

- Use smaller model: `--model yolov8n.pt`
- Process shorter video clips
- Reduce batch size

### "Correction UI not working"

**Common issues:**
- **"No sessions found"**: Process videos first to generate track files
- **"Video won't play"**: Check video codec (H.264 recommended)
- **"Track file not found"**: Make sure video was processed successfully
- **"Changes not saving"**: Check that `data/annotations/` directory exists and is writable

**See `CORRECTION_UI.md` for detailed troubleshooting.**

---

## 🎯 Summary

**You need to run these 2 commands:**

```bash
# 1. Install packages
pip install -r requirements.txt

# 2. Verify
python scripts/verify_setup.py
```

**That's it!** Everything else is automatic.

---

## 📚 Next Steps

- See `GETTING_STARTED.md` for video processing workflow
- See `VIDEO_WORKFLOW.md` for detailed video guide
- See `CORRECTION_UI.md` for correction UI user guide ⭐
- See `PROJECT_PLAN.md` for system architecture


