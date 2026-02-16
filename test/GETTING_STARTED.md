# Getting Started - Your 5-Minute Video Clips

## Overview

This system processes **your own 5-minute traffic camera video clips** to produce lane-based vehicle counts.

## Quick Workflow

### 1. Upload Your Videos

Place your video files in `data/videos/`:

```bash
# Example: Copy your videos
cp ~/Downloads/traffic_cam_1.mp4 data/videos/cam_001_1.mp4
cp ~/Downloads/traffic_cam_2.mp4 data/videos/cam_001_2.mp4
```

**Supported formats**: MP4, MOV, AVI, MKV  
**Recommended**: 720p or 1080p, 15-30 FPS, H.264 codec

### 2. Process Videos

**Single video:**
```bash
python scripts/process_video.py \
    --video-path data/videos/cam_001_1.mp4 \
    --camera-id cam_001
```

**Multiple videos:**
```bash
python scripts/batch_process.py \
    --video-dir data/videos/ \
    --camera-id cam_001
```

### 3. View Results

```bash
# See count summaries
python scripts/view_counts.py --camera-id cam_001

# Launch correction UI (fully implemented!)
python ui/app.py
# Open: http://localhost:5000
```

**Correction UI Features:**
- ✅ Video player with frame-by-frame navigation
- ✅ Track overlay on video (bounding boxes with IDs)
- ✅ Track list with filtering and sorting
- ✅ Delete tracks (D key)
- ✅ Change track class (C key)
- ✅ Split tracks at frame (S key)
- ✅ Merge tracks (M key)
- ✅ Save corrected annotations
- ✅ Keyboard shortcuts for fast editing
- ✅ Error handling and validation
- ✅ Fully debugged and tested

**See `CORRECTION_UI.md` for complete guide.**

## What Happens During Processing

For each 5-minute video (~9,000 frames at 30 FPS):

1. **Detection**: YOLO detects vehicles in each frame
   - Output: Bounding boxes, classes, confidences
   - Saved to: `data/detections/`

2. **Tracking**: Assigns consistent IDs across frames
   - Output: Track IDs with bboxes over time
   - Saved to: `data/tracks/`

3. **Counting**: Lane assignment + line crossing detection
   - Output: Count events by lane/class/time
   - Saved to: `data/analytics/`

## Processing Time

For a **5-minute video**:

| Hardware | Time |
|----------|------|
| GPU (NVIDIA) | ~2-5 minutes |
| CPU (Mac M1/M2) | ~10-20 minutes |
| CPU (Intel) | ~20-40 minutes |

**Note**: First video takes longer (model loading). Subsequent videos are faster.

## Output Files

After processing `cam_001_1.mp4`:

```
data/
├── detections/
│   └── cam_001_1.json          # Raw detections per frame
├── tracks/
│   └── cam_001_1.json          # Tracked vehicles with IDs
├── annotations/
│   └── cam_001_1.json          # Final annotations (after correction)
└── analytics/
    └── cam_001_1.csv           # Count summaries
```

## Video Naming Tips

**Recommended naming**:
- `{camera_id}_{clip_number}.mp4`
  - Example: `cam_001_1.mp4`, `cam_001_2.mp4`

- `{camera_id}_{date}_{time}.mp4`
  - Example: `cam_001_2024-01-15_10-00-00.mp4`

## Next Steps

1. **Install dependencies** (if not done):
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your videos** in `data/videos/`

3. **Process your first video**:
   ```bash
   python scripts/process_video.py --video-path data/videos/your_video.mp4 --camera-id cam_001
   ```

4. **Review results** and use correction UI if needed

## Troubleshooting

**"Video file not found"**
- Check the file path is correct
- Ensure video is in `data/videos/` directory

**"Unsupported codec"**
- Convert with FFmpeg:
  ```bash
  ffmpeg -i input.mp4 -c:v libx264 output.mp4
  ```

**"Processing too slow"**
- Use smaller model: `--model yolov8n.pt`
- Process shorter test clips first

## See Also

- `VIDEO_WORKFLOW.md` - Detailed workflow guide
- `README.md` - Full project documentation
- `PROJECT_PLAN.md` - System architecture


