# Correction UI Guide

## Overview

The Correction UI is a **fully implemented and debugged** web application for reviewing and correcting vehicle tracks. It provides an intuitive interface to fix detection and tracking errors with robust error handling and validation.

## Launching the UI

```bash
cd "/Users/delberttran/Documents/cctv yolo test/test"
python ui/app.py
```

Then open your browser to: **http://localhost:5000**

## Features

### 1. Session List

The main page shows all available video sessions:
- Videos with track files are marked with ✓
- Videos with saved annotations are marked with ✓ Annotated
- Click "Review" to open a session

### 2. Video Player

- **Play/Pause**: Click play button or press Spacebar
- **Frame Navigation**: 
  - Use ⏮ Prev Frame / Next Frame ⏭ buttons
  - Or use Left/Right arrow keys
- **Frame Info**: Shows current frame number and time
- **Track Overlay**: Bounding boxes are drawn on the video
  - Selected track highlighted in yellow
  - Different colors for different vehicle classes

### 3. Track List Panel

- **Filter**: Type to filter tracks by ID or class
- **Sort**: Sort by ID, class, confidence, or length
- **Select**: Click a track to select it
- **Track Info**: Shows frame count, confidence, and frame range

### 4. Track Actions

All actions can be performed via buttons or keyboard shortcuts:

#### Delete Track (D key)
- Removes a track completely
- Confirmation dialog before deletion

#### Change Class (C key)
- Change the vehicle class (car, truck, bus, etc.)
- Prompts for new class name

#### Split Track (S key)
- Split a track at the current frame
- Creates two separate tracks
- Prompts for frame number (defaults to current frame)

#### Merge Tracks (M key)
- Merge two tracks together
- Prompts for target track ID to merge into
- Combines all frames from both tracks

### 5. Save Annotations

- Click "💾 Save Annotations" button
- Saves corrected tracks to `data/annotations/`
- Status message confirms save

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Space** | Play/Pause video |
| **←** | Previous frame |
| **→** | Next frame |
| **D** | Delete selected track |
| **C** | Change track class |
| **S** | Split track at current frame |
| **M** | Merge tracks |

## Workflow

1. **Process video** to generate tracks:
   ```bash
   python scripts/process_video.py --video-path data/videos/your_video.mp4 --camera-id cam_001
   ```

2. **Launch UI**:
   ```bash
   python ui/app.py
   ```

3. **Select session** from the list

4. **Review tracks**:
   - Play video to see track overlays
   - Select tracks in the list
   - Use frame navigation to find errors

5. **Correct errors**:
   - Delete false positives
   - Split tracks that were merged incorrectly
   - Merge tracks that were split incorrectly
   - Change incorrect class labels

6. **Save annotations**:
   - Click Save button
   - Corrected tracks saved to `data/annotations/`

## Data Flow

```
data/tracks/your_video.json  (original tracks)
           ↓
    [Correction UI]
           ↓
data/annotations/your_video.json  (corrected tracks)
```

## Tips

- **Use keyboard shortcuts** for faster editing
- **Filter tracks** to find specific vehicles
- **Sort by confidence** to find low-confidence detections first
- **Frame-by-frame navigation** helps catch tracking errors
- **Save frequently** to avoid losing corrections

## Troubleshooting

### "No sessions found"
- Make sure videos are in `data/videos/`
- Process videos first to generate track files
- Check that video files have matching track files in `data/tracks/`

### "Track file not found"
- Process the video first with `process_video.py`
- Check that track file exists in `data/tracks/`
- Verify the track file name matches the video file name (without extension)

### "Video won't play"
- Check video codec (H.264 recommended)
- Try converting: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`
- Check browser console for video loading errors
- Verify video file is not corrupted

### "Changes not saving"
- Make sure you click "Save Annotations" button
- Check browser console for errors
- Verify `data/annotations/` directory exists and is writable
- Check server logs for backend errors

### "Track actions not working"
- Make sure a track is selected (highlighted in yellow)
- Check browser console for JavaScript errors
- Verify track file is valid JSON
- Try refreshing the page

### "Canvas overlay not showing"
- Check that video has loaded completely
- Verify track file contains valid bbox data
- Check browser console for JavaScript errors
- Try refreshing the page

### Server Errors

**"Port 5000 already in use"**
```bash
# Kill existing process or use different port
lsof -ti:5000 | xargs kill
# Or edit ui/app.py to use different port
```

**"Module not found: flask"**
```bash
# Make sure virtual environment is activated
source venv/bin/activate
pip install flask flask-cors
```

## Technical Details

- **Backend**: Flask web server with error handling and validation
- **Frontend**: HTML5 video player with Canvas overlay
- **Data Format**: JSON tracks with frame-by-frame bboxes
- **Storage**: Corrected tracks saved to `data/annotations/`
- **Error Handling**: Comprehensive validation and user-friendly error messages
- **Safety Checks**: Null checks, readyState validation, and parameter validation

## Recent Updates

✅ **Fully debugged and tested:**
- Fixed path handling for video file serving
- Added comprehensive error handling
- Added safety checks for video loading
- Improved validation for all track actions
- Enhanced user feedback with status messages

