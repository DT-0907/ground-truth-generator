# Vehicle Counting System - Complete Design Document

## Executive Summary

This system transforms raw CCTV traffic camera videos into accurate, lane-based vehicle counts by class through an automated pipeline with human-in-the-loop correction. The system processes 5-minute video clips, detects vehicles using YOLO, tracks them across frames, assigns lanes, counts crossings, and provides a web-based correction interface for quality assurance.

**Key Value Proposition:**
- **Automated Processing**: 70-90% accuracy on first pass
- **Human Efficiency**: 5-20x faster than manual labeling
- **Scalable**: Designed for hundreds of cameras
- **Structured Outputs**: Clean data for training and analytics

---

## 1. System Architecture

### 1.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                              │
│  Raw CCTV Video Clips (5-minute MP4 files)                  │
│  Camera Configurations (ROI, lanes, count lines)            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  PROCESSING LAYER                           │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Detection   │───▶│  Tracking    │───▶│  Counting    │   │ ### mapping to corresponding lane 
│  │  (YOLO)      │    │  (ByteTrack) │    │  (Lane/Line) │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                             │
└────────────────────┬────────────────────────────────────────┘
                     │ ### output each image label (thousands of images) + corresponding objects (bounding box of object)
                     | ### get id, type of object (car, truck, bus, pedestrian), check coco definition of objects + ids
                     | ### edge case of vehicle overlap
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  CORRECTION LAYER                           │
│  Web-based UI for human review and correction               │
│  - Delete false positives                                   │
│  - Merge/split tracks                                       │
│  - Change class labels                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  OUTPUT LAYER                               │
│  - Ground truth annotations (COCO/MOT format)               │
│  - Analytics tables (counts by lane/class/time)             │
│  - Training datasets                                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Component Breakdown

#### A. Detection Module (`src/detection/`)
**Purpose**: Identify vehicles in each video frame

**Technology**: Ultralytics YOLOv8
- Pre-trained on COCO dataset
- Vehicle classes: car, truck, bus, motorcycle, bicycle
- Configurable confidence thresholds (default: 0.25)

**Input**: Video frames (numpy arrays)
**Output**: Per-frame detections
```json
{
  "frame": 120,
  "timestamp": 4.0,
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "class": "car",
      "confidence": 0.88,
      "class_id": 2
    }
  ]
}
```

**Key Functions**:
- `detect_frame(frame, model, conf_threshold)` - Run YOLO on single frame
- `filter_by_roi(detections, roi_polygon)` - Filter detections inside ROI
- `filter_by_class(detections, class_ids)` - Filter by vehicle classes

**Configuration**:
- Model selection: yolov8n/m/l/x (speed vs accuracy tradeoff)
- Confidence threshold: 0.25-0.4 (tunable per camera)
- ROI polygon: Per-camera region of interest

#### B. Tracking Module (`src/tracking/`)
**Purpose**: Assign consistent IDs to vehicles across frames

**Technology**: Ultralytics built-in tracking (ByteTrack-based)
- No external dependencies required
- Handles occlusions and re-identification
- Kalman filtering for smooth trajectories

**Input**: Detections per frame
**Output**: Tracked objects with consistent IDs
```json
{
  "tracks": [
    {
      "track_id": 91,
      "class": "truck",
      "start_frame": 120,
      "end_frame": 180,
      "frames": [
        {"frame": 120, "bbox": [x1,y1,x2,y2], "conf": 0.88},
        {"frame": 121, "bbox": [x1,y1,x2,y2], "conf": 0.90}
      ]
    }
  ]
}
```

**Key Functions**:
- `update_tracks(detections, tracker)` - Update tracks with new detections
- `get_track_history(track_id)` - Retrieve full track history
- `smooth_trajectory(track)` - Apply Kalman filtering (optional)

**Why Tracks Matter**:
- Eliminates duplicates (one vehicle = one track, not 60 detections)
- Enables robust counting (count once per crossing)
- Makes correction efficient (edit one track, not 60 boxes)

#### C. Counting Module (`src/counting/`)
**Purpose**: Assign lanes and count vehicle crossings

**Technology**: Geometric algorithms (polygon intersection, line crossing)

**Input**: Tracks + camera configuration
**Output**: Lane assignments + count events
```json
{
  "count_events": [
    {
      "timestamp": 12.43,
      "lane_id": 2,
      "class": "truck",
      "track_id": 91,
      "direction": "downstream"
    }
  ],
  "lane_assignments": {
    "91": 2
  }
}
```

**Key Functions**:
- `assign_lane(track, lane_polygons)` - Assign track to lane
- `detect_crossing(track, count_line, direction)` - Detect line crossing
- `count_vehicles(tracks, camera_config)` - Main counting logic

**Lane Assignment Methods**:
1. **Centroid-based**: Where track centroid spends most time
2. **Polygon intersection**: Track bbox intersects lane polygon
3. **First stable position**: Lane at first stable detection

**Counting Logic**:
- Line crossing detection per lane
- Direction filtering (prevent back-and-forth counting)
- Hysteresis (ignore brief reversals)
- One count per track crossing

#### D. Correction UI (`ui/`)
**Purpose**: Human-in-the-loop quality assurance

**Technology**: Flask (backend) + HTML5/Canvas (frontend)

**Features**:
- **Session Discovery**: Automatically finds videos with track files
- **Video Player**: HTML5 video with frame-by-frame navigation
- **Track Overlay**: Canvas-based bbox rendering on video
- **Track List**: Filterable, sortable list of all tracks
- **Correction Actions**:
  - Delete false positives
  - Merge incorrectly split tracks
  - Split incorrectly merged tracks
  - Change class labels
- **Save Annotations**: Export corrected tracks to ground truth

**Data Flow**:
```
data/tracks/video.json (original)
    ↓
[User corrections in UI]
    ↓
data/annotations/video.json (ground truth)
```

**Keyboard Shortcuts**:
- `D` - Delete selected track
- `C` - Change track class
- `S` - Split track at current frame
- `M` - Merge tracks
- `Space` - Play/Pause
- `←/→` - Previous/Next frame

#### E. Pipeline Module (`src/pipeline/`)
**Purpose**: Orchestrate end-to-end processing

**Workflow**:
1. Load camera configuration
2. Extract frames from video (or stream decode)
3. Run detection on each frame
4. Run tracking across frames
5. Assign lanes to tracks
6. Count line crossings
7. Export results (detections, tracks, counts)

**Output Files**:
- `data/detections/{camera_id}_{video}.json` - Raw detections
- `data/tracks/{camera_id}_{video}.json` - Tracked objects
- `data/analytics/{camera_id}_{video}.csv` - Count summaries

---

## 2. Data Models

### 2.1 Camera Configuration

**Location**: `config/cameras/{camera_id}.json`

```json
{
  "camera_id": "cam_001",
  "video_path": "data/videos/cam_001_1.mp4",
  "fps": 30,
  "resolution": [1920, 1080],
  "roi_polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "lanes": [
    {
      "lane_id": 1,
      "name": "Left Lane",
      "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "direction": "downstream"
    }
  ],
  "count_lines": [
    {
      "lane_id": 1,
      "line": {"p1": [x1, y1], "p2": [x2, y2]},
      "direction": "downstream"
    }
  ],
  "detection_config": {
    "model": "yolov8m.pt",
    "conf_threshold": 0.25,
    "classes": [2, 3, 5, 7]
  },
  "tracking_config": {
    "tracker_type": "ultralytics",
    "frame_rate": 30
  }
}
```

### 2.2 Track Data Format

**Location**: `data/tracks/{session_id}.json`

```json
{
  "camera_id": "cam_001",
  "video_path": "data/videos/cam_001_1.mp4",
  "fps": 30,
  "processed_at": "2024-01-15T10:00:00",
  "tracks": [
    {
      "track_id": 91,
      "class": "truck",
      "lane_id": 2,
      "start_frame": 120,
      "end_frame": 180,
      "frames": [
        {
          "frame": 120,
          "bbox": [x1, y1, x2, y2],
          "conf": 0.88
        }
      ]
    }
  ]
}
```

### 2.3 Analytics Output

**Location**: `data/analytics/{session_id}.csv`

```csv
camera_id,date,time_bin_start,lane_id,class,count
cam_001,2024-01-15,10:00:00,1,car,45
cam_001,2024-01-15,10:00:00,1,truck,12
cam_001,2024-01-15,10:00:00,2,car,38
```

---

## 3. Processing Pipeline

### 3.1 Video Processing Flow

```
1. INPUT VALIDATION
   ├─ Check video file exists
   ├─ Load camera configuration
   └─ Verify ROI/lane polygons

2. FRAME EXTRACTION
   ├─ Open video with OpenCV
   ├─ Extract frames (or stream decode)
   └─ Get video metadata (FPS, resolution)

3. DETECTION PHASE
   ├─ Load YOLO model
   ├─ For each frame:
   │   ├─ Run YOLO inference
   │   ├─ Filter by ROI
   │   ├─ Filter by confidence
   │   └─ Store detections
   └─ Export detections JSON

4. TRACKING PHASE
   ├─ Initialize tracker
   ├─ For each frame:
   │   ├─ Update tracker with detections
   │   └─ Store track updates
   ├─ Post-process tracks (smooth, filter short tracks)
   └─ Export tracks JSON

5. COUNTING PHASE
   ├─ Load lane configuration
   ├─ For each track:
   │   ├─ Assign lane (centroid/polygon method)
   │   ├─ Detect line crossings
   │   └─ Generate count events
   └─ Export analytics CSV

6. OUTPUT GENERATION
   ├─ Save detections to data/detections/
   ├─ Save tracks to data/tracks/
   └─ Save analytics to data/analytics/
```

### 3.2 Performance Characteristics

**For a 5-minute video (9,000 frames at 30 FPS):**

| Component | Time (GPU) | Time (CPU) |
|-----------|------------|------------|
| Detection | ~1-2 min | ~5-10 min |
| Tracking | ~30 sec | ~2-5 min |
| Counting | ~5 sec | ~30 sec |
| **Total** | **~2-3 min** | **~8-16 min** |

**Bottlenecks**:
- Detection: YOLO inference (GPU helps significantly)
- Tracking: Frame-by-frame processing (CPU-bound)
- Counting: Fast (geometric calculations)

---

## 4. Correction Workflow

### 4.1 Human Review Process

```
1. LAUNCH UI
   └─ python ui/app.py → http://localhost:5000

2. SELECT SESSION
   ├─ View list of processed videos
   ├─ See which have tracks/annotations
   └─ Click "Review" on desired session

3. REVIEW TRACKS
   ├─ Play video to see track overlays
   ├─ Navigate frame-by-frame
   ├─ Filter/sort tracks in list
   └─ Identify errors:
       ├─ False positives (delete)
       ├─ Incorrect splits (merge)
       ├─ Incorrect merges (split)
       └─ Wrong class labels (change)

4. CORRECT ERRORS
   ├─ Select track in list
   ├─ Perform action (delete/merge/split/change)
   └─ Verify correction in video

5. SAVE ANNOTATIONS
   ├─ Click "Save Annotations"
   └─ Corrected tracks saved to data/annotations/
```

### 4.2 Quality Assurance Metrics

**Before Correction**:
- Detection accuracy: ~70-90% (depends on camera quality)
- Tracking accuracy: ~80-95% (depends on occlusion)
- Class accuracy: ~85-95% (depends on vehicle similarity)

**After Correction**:
- Target: >95% accuracy on corrected tracks
- Human time: <5 minutes per 5-minute video
- Efficiency gain: 5-20x faster than manual labeling

---

## 5. Scalability Design

### 5.1 Multi-Camera Processing

**Architecture**:
- Each camera has independent configuration
- Videos processed independently
- Results aggregated in analytics

**Storage Structure**:
```
data/
├── videos/
│   ├── cam_001_1.mp4
│   ├── cam_001_2.mp4
│   ├── cam_002_1.mp4
│   └── ...
├── tracks/
│   ├── cam_001_1.json
│   ├── cam_001_2.json
│   └── ...
└── analytics/
    ├── cam_001_1.csv
    └── ...
```

### 5.2 Batch Processing

**Script**: `scripts/batch_process.py`
- Processes all videos in a directory
- Handles multiple cameras
- Generates unified analytics

**Job Queue Pattern** (Future):
- Each video = one job
- GPU workers for detection
- CPU workers for tracking/counting
- Parallel processing for scale

### 5.3 Versioning

**Tracked Versions**:
- Model version (YOLO model used)
- Config version (ROI/lane polygons)
- Tool version (code version)
- Ground truth version (annotation version)

**Purpose**: Reproducibility and audit trail

---

## 6. Output Formats

### 6.1 Ground Truth Annotations

**Formats Supported**:
- **COCO**: Standard detection format
- **MOT**: Standard tracking format
- **Custom JSON**: Project-specific format

**Use Cases**:
- Model fine-tuning
- Dataset creation
- Benchmarking

### 6.2 Analytics Tables

**Format**: CSV/Parquet

**Schema**:
- `camera_id`: Camera identifier
- `date`: Date of recording
- `time_bin_start`: Time bucket (10s, 1min, etc.)
- `lane_id`: Lane number
- `class`: Vehicle class
- `count`: Number of vehicles

**Use Cases**:
- Traffic flow analysis
- Peak hour identification
- Lane utilization statistics
- Class distribution analysis

### 6.3 Event Logs

**Format**: JSON

**Content**:
- Individual crossing events
- Track metadata
- Confidence scores
- Timestamps

**Use Cases**:
- Detailed analysis
- Debugging
- Audit trails

---

## 7. Technology Stack

### 7.1 Core Technologies

**Detection**:
- Ultralytics YOLOv8 (Python)
- PyTorch (deep learning framework)
- CUDA (GPU acceleration, optional)

**Tracking**:
- Ultralytics built-in tracking (ByteTrack-based)
- No external dependencies

**Video Processing**:
- OpenCV (frame extraction, video I/O)
- FFmpeg (codec support)

**Web UI**:
- Flask (backend web framework)
- HTML5 Video API (video playback)
- Canvas API (bbox overlay)
- JavaScript (frontend logic)

**Data Processing**:
- Pandas (analytics tables)
- NumPy (numerical operations)
- JSON (data serialization)

### 7.2 System Requirements

**Minimum**:
- Python 3.8+
- 4GB RAM
- 2GB disk space
- CPU (slower but works)

**Recommended**:
- Python 3.9+
- 8GB+ RAM
- 5GB+ disk space
- NVIDIA GPU with CUDA (10-50x faster)

---

## 8. Configuration Management

### 8.1 Per-Camera Configuration

Each camera requires:
- ROI polygon (region of interest)
- Lane polygons or lane separator lines
- Count lines per lane
- Detection parameters (confidence threshold)
- Tracking parameters (if custom)

**Storage**: `config/cameras/{camera_id}.json`

### 8.2 Model Configuration

**YOLO Models Available**:
- `yolov8n.pt` - Nano (fastest, ~6MB)
- `yolov8s.pt` - Small (balanced, ~22MB)
- `yolov8m.pt` - Medium (recommended, ~52MB)
- `yolov8l.pt` - Large (high accuracy, ~88MB)
- `yolov8x.pt` - XLarge (best accuracy, ~136MB)

**Selection Criteria**:
- Speed requirement → smaller model
- Accuracy requirement → larger model
- Resource constraints → smaller model

---

## 9. Error Handling & Validation

### 9.1 Input Validation

**Video Files**:
- Format check (MP4, MOV, AVI, MKV)
- Codec validation
- Resolution/FPS extraction

**Configuration Files**:
- JSON schema validation
- Polygon coordinate validation
- Required field checks

### 9.2 Processing Errors

**Detection Errors**:
- Model loading failures → fallback to CPU
- Memory errors → use smaller model
- Frame decode errors → skip frame, log warning

**Tracking Errors**:
- Track ID conflicts → reassign IDs
- Missing detections → interpolate or drop
- Long gaps → split track

**Counting Errors**:
- Invalid lane assignments → default to nearest lane
- Missing count lines → skip counting for lane
- Direction ambiguity → use majority direction

### 9.3 UI Error Handling

**Backend**:
- File not found → 404 with helpful message
- Invalid JSON → 400 with validation errors
- Server errors → 500 with error details

**Frontend**:
- Video load failures → show error message
- Missing tracks → show empty state
- Network errors → retry with user notification

---

## 10. Future Enhancements

### 10.1 Active Learning

**Priority**: High

**Implementation**:
- Flag low-confidence tracks for review
- Identify unusual patterns (short tracks, unstable classes)
- Prioritize review queue by expected error impact
- Use corrected data to fine-tune models

**Benefits**:
- Reduce human review time by 80-95%
- Focus effort on high-impact corrections
- Continuous model improvement

### 10.2 Model Fine-Tuning

**Priority**: Medium

**Implementation**:
- Use corrected annotations as training data
- Fine-tune YOLO on camera-specific viewpoints
- Train custom classifier for vehicle types

**Benefits**:
- Improve accuracy for specific camera angles
- Reduce false positives
- Better class discrimination

### 10.3 Advanced Tracking

**Priority**: Low

**Options**:
- DeepSORT (appearance embeddings for better occlusion handling)
- BoT-SORT (stronger variant)
- Custom re-ID models

**When Needed**:
- High occlusion scenarios
- Camera shake
- Poor lighting conditions

### 10.4 Batch Processing Infrastructure

**Priority**: Medium

**Components**:
- Job queue (Celery, RQ, or custom)
- Worker pool (GPU + CPU workers)
- Progress tracking
- Failure recovery

**Benefits**:
- Process hundreds of cameras in parallel
- Automatic retry on failures
- Resource optimization

---

## 11. Success Metrics

### 11.1 Accuracy Targets

- **Detection**: >90% recall, >85% precision
- **Tracking**: >90% track accuracy
- **Counting**: >95% accuracy after correction
- **Class**: >90% class accuracy

### 11.2 Efficiency Targets

- **Processing Speed**: <10 min per 5-min video (GPU)
- **Human Review Time**: <5 min per 5-min video
- **Throughput**: 100+ videos per day (with infrastructure)

### 11.3 Quality Targets

- **Ground Truth Quality**: >95% accuracy
- **Annotation Consistency**: <2% inter-annotator disagreement
- **Data Completeness**: >98% of videos processed successfully

---

## 12. Risk Mitigation

### 12.1 Technical Risks

**Camera Shake / Different Viewpoints**
- **Mitigation**: Per-camera ROI/lane configuration
- **Solution**: Store and reuse camera configs

**Occlusions**
- **Mitigation**: Better tracker + UI merge/split tools
- **Solution**: DeepSORT if needed, manual correction

**Night / Glare / Rain**
- **Mitigation**: Domain fine-tuning + augmentation
- **Solution**: Train on diverse conditions

**Class Confusion**
- **Mitigation**: Second-stage classifier if needed
- **Solution**: Manual correction in UI

**Stopped Vehicles**
- **Mitigation**: Direction + hysteresis rules
- **Solution**: Time-based filtering

### 12.2 Operational Risks

**Scale Challenges**
- **Mitigation**: Batch processing, job queues
- **Solution**: Containerization, cloud deployment

**Data Quality**
- **Mitigation**: Validation, error handling
- **Solution**: Human review workflow

**Version Control**
- **Mitigation**: Version tracking system
- **Solution**: Config versioning, model versioning

---

## 13. Deployment Considerations

### 13.1 Development Environment

**Setup**:
- Local development with venv
- Test on sample videos
- Iterate on UI/UX

### 13.2 Production Deployment

**Options**:
- **Local Server**: Single machine processing
- **Cloud**: AWS/GCP with GPU instances
- **Hybrid**: Local processing + cloud storage

**Requirements**:
- GPU access for speed
- Sufficient storage for videos/results
- Network for data transfer (if cloud)

### 13.3 Maintenance

**Ongoing Tasks**:
- Monitor processing success rates
- Update models as needed
- Refine camera configurations
- Handle edge cases

---

## 14. Conclusion

This system provides a complete solution for transforming raw CCTV videos into accurate, lane-based vehicle counts. The combination of automated processing (YOLO + tracking) and human-in-the-loop correction creates an efficient workflow that scales to hundreds of cameras while maintaining high accuracy.

**Key Strengths**:
- Automated first pass (70-90% accuracy)
- Fast human correction (5-20x faster than manual)
- Structured outputs (ready for training/analytics)
- Scalable architecture (hundreds of cameras)
- No hardcoded values (fully generic)

**Next Steps**:
1. Process initial video set
2. Collect correction data
3. Fine-tune models on corrected data
4. Scale to production volumes

