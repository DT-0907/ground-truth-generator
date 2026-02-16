# Project Plan - Vehicle Counting System

## High-Level Overview

**Goal**: Transform raw CCTV videos into accurate lane-based vehicle counts by class through an automated pipeline with human-in-the-loop correction.

**Input**: Your own 5-minute traffic camera video clips (uploaded to `data/videos/`)

**Key Features**:
- Automatic YOLO detection (vehicles per frame)
- Multi-object tracking (consistent IDs across frames)
- Lane-based counting with ROI/zone logic
- Web-based correction UI for fast human review
- Structured outputs (tracks, counts, analytics)
- Batch processing for multiple videos
- Scalable to hundreds of cameras

## Architecture Layers

### Layer 1: Detection (YOLO)
- **Input**: Video frames
- **Output**: Bounding boxes + classes + confidences per frame
- **Technology**: Ultralytics YOLOv8
- **Classes**: car, truck, bus, motorcycle, bicycle (COCO classes)

### Layer 2: Tracking (MOT)
- **Input**: Detections per frame
- **Output**: Track IDs with consistent bboxes across frames
- **Technology**: ByteTrack (recommended) or DeepSORT
- **Purpose**: Eliminate duplicates (one vehicle = one track)

### Layer 3: Lane Assignment & Counting
- **Input**: Tracks + camera configuration (ROI, lanes, count lines)
- **Output**: Lane assignments + count events
- **Logic**: 
  - Lane assignment: centroid-based or polygon intersection
  - Counting: line-crossing detection with direction filtering

### Layer 4: Correction UI
- **Input**: Video + tracks + annotations
- **Output**: Corrected ground truth
- **Features**: Merge/split tracks, change class, delete FPs, add boxes

## Detailed Component Breakdown

### 1. Detection Module (`src/detection/`)

**Files**:
- `yolo_detector.py` - YOLO wrapper, frame-by-frame detection
- `filter.py` - ROI filtering, confidence thresholding
- `config.py` - Detection parameters

**Key Functions**:
```python
def detect_frame(frame, model, conf_threshold=0.25):
    """Detect vehicles in a single frame."""
    # Returns: List[Detection] with bbox, class, confidence

def filter_by_roi(detections, roi_polygon):
    """Filter detections inside ROI."""
    # Returns: Filtered detections
```

**Output Format**:
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

### 2. Tracking Module (`src/tracking/`)

**Files**:
- `tracker.py` - ByteTrack/DeepSORT wrapper
- `track_manager.py` - Track lifecycle management
- `kalman.py` - Kalman filtering for smooth bboxes (optional)

**Key Functions**:
```python
def update_tracks(detections, tracker):
    """Update tracks with new detections."""
    # Returns: List[Track] with track_id, bbox, class

def get_track_history(track_id):
    """Get full history of a track."""
    # Returns: List of bboxes over time
```

**Output Format**:
```json
{
  "tracks": [
    {
      "track_id": 91,
      "class": "truck",
      "frames": [
        {"frame": 120, "bbox": [x1,y1,x2,y2], "conf": 0.88},
        {"frame": 121, "bbox": [x1,y1,x2,y2], "conf": 0.90}
      ],
      "start_frame": 120,
      "end_frame": 180
    }
  ]
}
```

### 3. Counting Module (`src/counting/`)

**Files**:
- `lane_assigner.py` - Assign tracks to lanes
- `line_crossing.py` - Detect line crossings
- `counter.py` - Main counting logic
- `config_loader.py` - Load camera configs

**Key Functions**:
```python
def assign_lane(track, lane_polygons):
    """Assign track to lane based on centroid."""
    # Returns: lane_id

def detect_crossing(track, count_line, direction):
    """Detect if track crossed count line."""
    # Returns: crossing_event or None

def count_vehicles(tracks, camera_config):
    """Main counting function."""
    # Returns: List[CountEvent]
```

**Output Format**:
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
    "91": 2  # track_id -> lane_id
  }
}
```

### 4. Camera Configuration (`config/cameras/`)

**Format**: JSON per camera

```json
{
  "camera_id": "cam_102",
  "video_path": "data/videos/cam_102.mp4",
  "fps": 30,
  "resolution": [1920, 1080],
  "roi_polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "lanes": [
    {
      "lane_id": 1,
      "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "direction": "downstream"
    },
    {
      "lane_id": 2,
      "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "direction": "downstream"
    }
  ],
  "count_lines": [
    {
      "lane_id": 1,
      "line": {"p1": [x1, y1], "p2": [x2, y2]},
      "direction": "downstream"
    },
    {
      "lane_id": 2,
      "line": {"p1": [x1, y1], "p2": [x2, y2]},
      "direction": "downstream"
    }
  ]
}
```

### 5. Correction UI (`ui/`)

**Technology**: Flask (lightweight) or FastAPI (modern)

**Features**:
- Video player with frame scrubber
- Overlay tracks/bboxes on video
- Track list panel (sortable)
- Quick actions:
  - Delete track (D key)
  - Change class (1/2/3 keys)
  - Split track at frame (S key)
  - Merge tracks (M key)
  - Add bbox (click-drag)
- Save corrected annotations

**API Endpoints**:
```
GET  /api/video/<camera_id>/<frame>     - Get frame + tracks
POST /api/track/delete                  - Delete track
POST /api/track/split                   - Split track
POST /api/track/merge                   - Merge tracks
POST /api/track/update_class            - Change class
POST /api/annotation/save               - Save corrections
```

### 6. Pipeline (`src/pipeline/`)

**Main Script**: `process_video.py`

**Workflow**:
1. Load camera config
2. Extract frames (or stream decode)
3. Run detection per frame
4. Run tracking
5. Assign lanes
6. Count crossings
7. Export results

**Outputs**:
- `data/detections/{camera_id}.json` - Raw detections
- `data/tracks/{camera_id}.json` - Track data
- `data/annotations/{camera_id}.json` - Final annotations
- `data/analytics/{camera_id}.csv` - Count summaries

### 7. Analytics (`src/analytics/`)

**Functions**:
- Generate time-binned counts (per 10s, 1min, etc.)
- Aggregate by lane/class
- Export to CSV/Parquet
- Generate reports

**Output Format**:
```csv
camera_id,date,time_bin_start,lane_id,class,count
cam_102,2024-01-15,10:00:00,1,car,45
cam_102,2024-01-15,10:00:00,1,truck,12
cam_102,2024-01-15,10:00:00,2,car,38
```

## Implementation Phases

### Phase 1: Baseline (Days 1-3)
**Goal**: Get working detection + tracking + basic counting

**Tasks**:
1. ✅ Setup project structure
2. ✅ Install dependencies
3. ✅ Create video processing workflow
4. Implement YOLO detection wrapper
5. Implement ByteTrack tracking
6. Implement basic lane assignment
7. Implement line-crossing counting
8. Test on your 5-minute video clips
9. Export basic counts

**Deliverable**: Script that processes your videos → outputs counts

### Phase 2: Correction UI (Days 4-7)
**Goal**: Build web UI for human review

**Tasks**:
1. Setup Flask/FastAPI server
2. Video player component
3. Track overlay rendering
4. Track list panel
5. Implement correction actions (delete, merge, split, etc.)
6. Save/load corrected annotations
7. Re-run counting on corrected data

**Deliverable**: Working web UI for correction

### Phase 3: Refinement (Days 8-10)
**Goal**: Improve accuracy and efficiency

**Tasks**:
1. Implement review queue (prioritize low-confidence tracks)
2. Auto-fix rules (drop short tracks, smooth bboxes, etc.)
3. Better lane assignment logic
4. Analytics generation
5. Export formats (COCO, MOT)

**Deliverable**: Production-ready system

### Phase 4: Scaling (Days 11+)
**Goal**: Handle hundreds of cameras

**Tasks**:
1. Batch processing pipeline
2. Job queue system
3. Containerization (Docker)
4. Versioning system
5. Active learning integration

**Deliverable**: Scalable production system

## Data Flow

```
Video File
    ↓
[Frame Extraction]
    ↓
[YOLO Detection] → detections.json
    ↓
[Tracking] → tracks.json
    ↓
[Lane Assignment] → lane_assignments.json
    ↓
[Counting] → count_events.json
    ↓
[Correction UI] → corrected_tracks.json
    ↓
[Re-counting] → final_counts.csv
    ↓
[Analytics] → summary_report.csv
```

## Key Design Decisions

### Why Tracks as First-Class Objects?
- Eliminates duplicates (one vehicle = one track)
- Makes counting robust (count once per crossing)
- Enables efficient correction (edit track, not 60 boxes)

### Why Lane-Based Counting?
- More accurate than zone-based (knows which lane)
- Enables lane-specific analytics
- Handles multi-lane roads naturally

### Why Human-in-the-Loop?
- YOLO + tracking is ~70-90% accurate
- Human correction on flagged segments is 5-20x faster than manual labeling
- Creates ground truth for fine-tuning

### Why ByteTrack over DeepSORT?
- Faster (no appearance embeddings)
- Simpler (fewer dependencies)
- Good enough for most traffic scenarios
- Can switch to DeepSORT later if needed

## Success Metrics

- **Accuracy**: >95% correct counts after human review
- **Speed**: Process 1 hour video in <10 minutes (GPU)
- **Human Time**: <5 minutes to correct 1 hour of video
- **Scalability**: Process 100 cameras in parallel

## Next Steps

1. **Install dependencies** (see SETUP_GUIDE.md)
2. **Download YOLO models** (automatic or manual)
3. **Start with Phase 1**: Detection + Tracking
4. **Test on sample video**
5. **Iterate based on results**

