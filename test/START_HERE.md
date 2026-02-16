# 🚀 START HERE - Installation & Setup

## What You Need to Download

**Almost nothing!** Everything downloads automatically.

### ✅ Automatic Downloads
- Python packages (~1.5-2GB) → `pip install -r requirements.txt`
- YOLO models (~50MB) → Auto-downloads on first use
- Tracking → Built into Ultralytics (no install needed)

### 🔧 Check System Tools
- FFmpeg → Usually pre-installed (`ffmpeg -version`)
- Python 3.8+ → Usually pre-installed (`python3 --version`)

---

## Quick Installation (3 Commands)

```bash
# 1. Navigate to project
cd "/Users/delberttran/Documents/cctv yolo test/test"

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install everything
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify
python scripts/verify_setup.py
```

**That's it!** Takes 5-15 minutes depending on internet speed.

---

## Process Your Videos

```bash
# 1. Place your 5-minute video clips
cp /path/to/your/video.mp4 data/videos/cam_001_1.mp4

# 2. Process
python scripts/process_video.py \
    --video-path data/videos/cam_001_1.mp4 \
    --camera-id cam_001

# 3. View results
python scripts/view_counts.py --camera-id cam_001

# 4. Launch correction UI (fully implemented!)
python ui/app.py
# Open: http://localhost:5000
```

---

## 📚 Full Documentation

- **`INSTALLATION.md`** - Complete installation guide ⭐
- **`GETTING_STARTED.md`** - Video processing workflow
- **`VIDEO_WORKFLOW.md`** - Detailed video guide
- **`WHAT_TO_DOWNLOAD.md`** - What to download details

---

## ❓ Troubleshooting

**"Module not found"**
```bash
source venv/bin/activate  # Make sure venv is active
pip install -r requirements.txt
```

**"FFmpeg not found"**
```bash
brew install ffmpeg  # macOS
```

---

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] `pip install -r requirements.txt` completed
- [ ] `python scripts/verify_setup.py` passes
- [ ] FFmpeg available
- [ ] ~2GB free disk space

**Ready to process videos!** 🎉


