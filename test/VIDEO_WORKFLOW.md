# Video Processing Workflow

## Overview

This system is designed to process **your own 5-minute traffic camera video clips**.

## Step 1: Upload Your Videos

Place your video files in the `data/videos/` directory:

```bash
data/videos/
├── cam_001_clip_1.mp4
├── cam_001_clip_2.mp4
├── cam_002_clip_1.mp4
└── ...
```

**Supported formats**: MP4, MOV, AVI, MKV

**Recommended**:
- Resolution: 720p or 1080p
- Frame rate: 15-30 FPS
- Length: 5-minute clips (as you mentioned)
- Codec: H.264 (most common)

## Step 2: Process a Video

### Quick Process (Single Video)

```bash
python scripts/process_video.py --video-path data/videos/your_video.mp4 --camera-id cam_001
```

This will:
1. Run YOLO detection on all frames
2. Track vehicles across frames
3. Assign lanes and count vehicles
4. Save results to `data/tracks/` and `data/analytics/`

### Batch Process (Multiple Videos)

```bash
python scripts/batch_process.py --video-dir data/videos/ --camera-id cam_001
```

Processes all videos in a directory.

## Step 3: Review Results

### View Counts
```bash
python scripts/view_counts.py --camera-id cam_001
```

### Launch Correction UI
```bash
python ui/app.py
```

Then open: http://localhost:5000

**Correction UI Features:**
- Video player with frame-by-frame controls
- Track bounding boxes overlaid on video
- Track list panel (filter, sort, select)
- Quick actions:
  - **Delete track** (D key or button)
  - **Change class** (C key or button)
  - **Split track** at current frame (S key or button)
  - **Merge tracks** (M key or button)
- Save corrected annotations to `data/annotations/`
- Keyboard shortcuts for efficient editing
- Error handling and validation
- Fully debugged and tested

**See `CORRECTION_UI.md` for complete user guide.**

## Video Naming Convention

**Recommended naming**:
- `{camera_id}_{clip_number}.mp4`
- Example: `cam_001_1.mp4`, `cam_001_2.mp4`

**Or**:
- `{camera_id}_{date}_{time}.mp4`
- Example: `cam_001_2024-01-15_10-00-00.mp4`

## Processing Time Estimates

For a **5-minute video** (30 FPS = 9,000 frames):

| Hardware | Processing Time |
|----------|----------------|
| GPU (NVIDIA) | ~2-5 minutes |
| CPU (Mac M1/M2) | ~10-20 minutes |
| CPU (Intel) | ~20-40 minutes |

**Note**: First video takes longer (model loading). Subsequent videos are faster.

## Output Files

After processing, you'll get:

```
data/
├── detections/
│   └── cam_001_1.json          # Raw detections per frame
├── tracks/
│   └── cam_001_1.json          # Tracked vehicles with IDs
├── annotations/
│   └── cam_001_1.json          # Final annotations (after correction)
└── analytics/
    └── cam_001_1.csv           # Count summaries by lane/class/time
```

## Example: Processing Your First Video

```bash
# 1. Place your video
cp /path/to/your/video.mp4 data/videos/cam_001_1.mp4

# 2. Process it
python scripts/process_video.py \
    --video-path data/videos/cam_001_1.mp4 \
    --camera-id cam_001

# 3. View results
python scripts/view_counts.py --camera-id cam_001
```

## Tips for Best Results

1. **Daytime videos work best** (better detection accuracy)
2. **Stable camera angles** (less tracking errors)
3. **Clear lane markings** (easier lane assignment)
4. **Good lighting** (improves detection confidence)

## Troubleshooting

**"Video file not found"**
- Check file path is correct
- Ensure video is in `data/videos/` directory

**"Unsupported codec"**
- Convert video: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

**"Processing too slow"**
- Use smaller YOLO model: `--model yolov8n.pt`
- Process shorter clips first to test


