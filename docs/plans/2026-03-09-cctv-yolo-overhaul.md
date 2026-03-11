# CCTV-YOLO Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix software2 bugs (video playback, YOLO processing), add offline model loading, ROI for processing/correction, bbox resizing, 3-page layout (Preprocessing | Correction | Performance), then port all changes to software (Flask v1) and test (dev Flask).

**Architecture:** Fix critical bugs in software2 first, then restructure UI to 3 tabs, add new features, finally port to web versions. All versions share the same processor.py and data format.

**Tech Stack:** PySide6/Qt (software2), Flask + Jinja + vanilla JS (software, test), YOLOv8/Ultralytics, OpenCV, ByteTrack

---

## Task 1: Fix software2 Video Playback

**Problem:** When playing video, frames appear to not update (looks like same frame). On pause, jumps to last frame. The QTimer fires but the canvas doesn't visually refresh properly.

**Files:**
- Modify: `software2/cctv_yolo/video_canvas.py` (set_frame method, lines 99-115)
- Modify: `software2/cctv_yolo/review_window.py` (_on_play_tick, lines 590-594; _toggle_play, lines 572-576)

**Step 1:** Fix `video_canvas.py:set_frame()` — ensure it seeks correctly, reads the frame, and forces a repaint.

The bug is likely that `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES)` doesn't always seek accurately, or `self.update()` isn't being called. Fix:

```python
def set_frame(self, frame_num: int):
    if self._cap is None or not self._cap.isOpened():
        return
    frame_num = max(0, min(frame_num, self._total_frames - 1))
    self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, bgr = self._cap.read()
    if not ret:
        return
    self._current_frame_num = frame_num
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    self._pixmap = QPixmap.fromImage(qimg)
    self.update()  # force repaint
```

**Step 2:** Fix `review_window.py:_on_play_tick()` — ensure it increments frame AND updates all UI elements (canvas, sidebar, slider, spinbox):

```python
def _on_play_tick(self):
    next_frame = self.current_frame + 1
    if next_frame >= self.total_frames:
        self._pause()
        return
    self._go_to_frame(next_frame)
```

Verify `_go_to_frame` calls `canvas.set_frame()`, updates `self.current_frame`, updates slider/spinbox, and refreshes sidebar. If it doesn't, fix it.

**Step 3:** Fix `_toggle_play` / `_play` / `_pause` to properly manage the QTimer interval:

```python
def _play(self):
    if self.playing:
        return
    self.playing = True
    interval = max(16, int(1000 / self.fps))
    self.play_timer.start(interval)
    self._play_btn.setText("Pause")

def _pause(self):
    self.playing = False
    self.play_timer.stop()
    self._play_btn.setText("Play")
```

**Step 4:** Test manually — open a video in review, press play, verify frames advance visually like a video. Press pause, verify it stops on current frame (not jumping).

**Step 5:** Commit: `fix: software2 video playback - ensure frames update on each tick`

---

## Task 2: Fix software2 YOLO Processing

**Problem:** YOLO model fails to load/run in the Qt app. Videos aren't being processed.

**Files:**
- Modify: `software2/cctv_yolo/processor.py` (lines 24-192)
- Modify: `software2/cctv_yolo/processing.py` (ProcessingWorker, lines 9-61)
- Modify: `software2/cctv_yolo/videos_tab.py` (_process_video, lines 546-573)

**Step 1:** Debug processor.py model loading. The issue is likely:
- Model path resolution when running from different working directories
- The `models_dir` check using `Path.home() / "Documents" / "CCTV-YOLO" / "models"` may not match the actual data dir

Fix processor.py to accept an explicit models_dir parameter and fall back correctly:

```python
def process_video(video_path, output_dir="data/tracks", model_name="yolov8m.pt",
                  conf_threshold=0.25, feedback_file=None, session_id=None,
                  progress_callback=None, models_dir=None):
    # Try explicit models_dir first, then default location
    search_dirs = []
    if models_dir:
        search_dirs.append(Path(models_dir))
    search_dirs.append(Path.home() / "Documents" / "CCTV-YOLO" / "models")

    model_path = None
    for d in search_dirs:
        candidate = d / model_name
        if candidate.exists():
            model_path = str(candidate)
            break

    if model_path is None:
        # Use model_name directly — Ultralytics will download or find in cache
        model_path = model_name

    model = YOLO(model_path)
```

**Step 2:** Fix ProcessingWorker to pass data_manager paths to processor:

```python
def run(self):
    try:
        from .processor import process_video
        result = process_video(
            video_path=self._video_path,
            output_dir=self._tracks_dir,
            model_name=self._model,
            conf_threshold=self._conf,
            session_id=self._session_id,
            progress_callback=self._report_progress,
        )
        self.finished.emit(self._session_id)
    except Exception as e:
        self.error.emit(self._session_id, str(e))
```

**Step 3:** Fix videos_tab.py `_process_video` to pass correct paths from data_manager:

Ensure `tracks_dir` is passed as `str(self.data_manager.tracks_dir)` and `video_path` is resolved via `self.data_manager.get_video_path(session_id)`.

**Step 4:** Add error reporting — ensure processing errors are displayed in the UI (not silently swallowed). Connect `worker.error` signal to show a message box or status label.

**Step 5:** Test manually — select a video, click Process, verify it completes without error and creates a tracks JSON.

**Step 6:** Commit: `fix: software2 YOLO processing - correct model/path resolution and error handling`

---

## Task 3: Offline YOLO Model Loading (software2)

**Problem:** Need to let users browse and select custom `.pt` model files. No internet required.

**Files:**
- Modify: `software2/cctv_yolo/processor.py` — accept model_path (not just model_name)
- Modify: `software2/cctv_yolo/data_manager.py` — add methods to list/manage models
- Modify: `software2/cctv_yolo/videos_tab.py` — add model file picker UI (will become preprocessing_tab.py in Task 5)

**Step 1:** Add model management to data_manager.py:

```python
def get_models_dir(self) -> Path:
    d = Path.home() / "Documents" / "CCTV-YOLO" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d

def list_models(self) -> list:
    """List all .pt files in models directory."""
    models_dir = self.get_models_dir()
    return sorted([f.name for f in models_dir.glob("*.pt")])

def get_model_path(self, model_name: str) -> Path | None:
    """Get full path to a model file, or None if not found."""
    p = self.get_models_dir() / model_name
    return p if p.exists() else None
```

**Step 2:** Update processor.py to accept a full model path:

Change `model_name` parameter to also accept absolute paths. If the path exists as-is, use it directly. Otherwise search models_dir then fall back to Ultralytics download.

**Step 3:** Add model picker to the UI:

- Dropdown showing models from `list_models()`
- "Browse..." button opening `QFileDialog.getOpenFileName(filter="YOLO Models (*.pt)")`
- When a model is browsed, copy it to models_dir (or just use the path directly)
- Store last-used model in config

**Step 4:** Test — place a `.pt` file in ~/Documents/CCTV-YOLO/models/, verify it appears in dropdown and can process a video.

**Step 5:** Commit: `feat: offline YOLO model loading with file browser`

---

## Task 4: Three-Page Layout (software2)

**Problem:** Restructure from (Sessions, Videos, Settings) tabs to (Preprocessing, Correction, Performance) with Settings as a menu dialog.

**Files:**
- Create: `software2/cctv_yolo/preprocessing_tab.py` — combines video management + model selection + ROI drawing
- Create: `software2/cctv_yolo/correction_tab.py` — sessions list that opens review windows
- Create: `software2/cctv_yolo/performance_tab.py` — traffic counts by vehicle type and lane
- Modify: `software2/cctv_yolo/main_window.py` — swap tab setup (lines 105-131)
- Modify: `software2/cctv_yolo/settings_tab.py` — convert to a QDialog instead of QWidget tab

**Step 1:** Create `preprocessing_tab.py`:
- Merge functionality from videos_tab.py (video listing, processing controls)
- Add model picker (dropdown + browse button from Task 3)
- Add confidence threshold slider
- Add ROI drawing on video preview frame (Task 6)
- Layout: top bar with model/conf controls, grid of video cards below

**Step 2:** Create `correction_tab.py`:
- Move session listing from sessions_tab.py
- Show sessions that have been processed (have tracks)
- "Review" button opens ReviewWindow
- Show correction progress stats
- Filter: All / Needs Review / Corrected

**Step 3:** Create `performance_tab.py`:
- Select a session from dropdown
- Show traffic counts by vehicle type (table + simple bar display)
- Show counts by ROI (if ROIs are defined)
- Show detection confidence distribution
- Export stats button (CSV)

**Step 4:** Convert settings_tab.py to `SettingsDialog(QDialog)`:
- Move NAS config and local folders into a dialog
- Access via File > Settings menu item

**Step 5:** Update main_window.py:
- Replace 3 old tabs with 3 new tabs
- Add Settings to File menu
- Update keyboard shortcuts (Ctrl+1/2/3 for new tabs)

**Step 6:** Test — verify all 3 tabs render, Settings dialog opens from menu, review window still works.

**Step 7:** Commit: `feat: restructure to 3-page layout (Preprocessing, Correction, Performance)`

---

## Task 5: ROI for Processing (software2)

**Problem:** Let users draw an ROI on a preview frame before processing, so YOLO only keeps detections inside the ROI.

**Files:**
- Modify: `software2/cctv_yolo/preprocessing_tab.py` — add preview canvas with ROI drawing
- Modify: `software2/cctv_yolo/processor.py` — accept processing_roi parameter and filter detections
- Modify: `software2/cctv_yolo/processing.py` — pass ROI to processor

**Step 1:** Add video preview to preprocessing_tab.py:
- When user selects a video, show first frame in a small VideoCanvas
- Enable ROI drawing on that canvas (rect or polygon)
- Store the ROI coordinates with the processing request

**Step 2:** Update processor.py to accept and apply ROI filtering:

```python
def _point_in_polygon(px, py, polygon):
    """Ray-casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _bbox_center_in_roi(bbox, roi):
    """Check if bbox center is inside ROI."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    if roi['type'] == 'rect':
        pts = roi['points']
        x1, y1 = pts[0]['x'], pts[0]['y']
        x2, y2 = pts[1]['x'], pts[1]['y']
        return x1 <= cx <= x2 and y1 <= cy <= y2
    else:  # polygon
        poly = [(p['x'], p['y']) for p in roi['points']]
        return _point_in_polygon(cx, cy, poly)
```

In the detection loop, after getting each detection, skip if ROI is defined and bbox center is outside:

```python
if processing_roi and not _bbox_center_in_roi(bbox_list, processing_roi):
    continue
```

**Step 3:** Pass ROI through ProcessingWorker to processor.

**Step 4:** Test — draw ROI on preview, process video, verify only detections inside ROI are kept.

**Step 5:** Commit: `feat: ROI for processing - filter detections by region of interest`

---

## Task 6: Bounding Box Resizing (software2)

**Problem:** Users need to drag handles to resize bounding boxes (make them smaller/larger).

**Files:**
- Modify: `software2/cctv_yolo/video_canvas.py` — add resize handles and drag logic
- Modify: `software2/cctv_yolo/review_window.py` — handle bbox resize events

**Step 1:** Add resize handle detection to video_canvas.py:

In select mode, when a track is selected and has a bbox on the current frame, draw 8 handles (4 corners + 4 edge midpoints). Each handle is a small square (8x8px).

```python
HANDLE_SIZE = 8  # pixels in canvas space

def _get_bbox_handles(self, bbox, scale_x, scale_y, dr):
    """Return 8 handle rects for a bbox. bbox is [x1,y1,x2,y2] in video coords."""
    x1c = dr.x() + bbox[0] * scale_x
    y1c = dr.y() + bbox[1] * scale_y
    x2c = dr.x() + bbox[2] * scale_x
    y2c = dr.y() + bbox[3] * scale_y
    mx = (x1c + x2c) / 2
    my = (y1c + y2c) / 2
    hs = HANDLE_SIZE / 2
    # Returns dict: handle_name -> QRectF
    return {
        'tl': QRectF(x1c-hs, y1c-hs, HANDLE_SIZE, HANDLE_SIZE),
        'tc': QRectF(mx-hs, y1c-hs, HANDLE_SIZE, HANDLE_SIZE),
        'tr': QRectF(x2c-hs, y1c-hs, HANDLE_SIZE, HANDLE_SIZE),
        'ml': QRectF(x1c-hs, my-hs, HANDLE_SIZE, HANDLE_SIZE),
        'mr': QRectF(x2c-hs, my-hs, HANDLE_SIZE, HANDLE_SIZE),
        'bl': QRectF(x1c-hs, y2c-hs, HANDLE_SIZE, HANDLE_SIZE),
        'bc': QRectF(mx-hs, y2c-hs, HANDLE_SIZE, HANDLE_SIZE),
        'br': QRectF(x2c-hs, y2c-hs, HANDLE_SIZE, HANDLE_SIZE),
    }
```

**Step 2:** Handle mouse events for resizing:

In `mousePressEvent`, if in select mode and clicking on a handle, start resize drag. Store which handle and original bbox.

In `mouseMoveEvent`, during resize drag, update the bbox based on which handle is being dragged:
- Corner handles move 2 edges
- Edge handles move 1 edge

In `mouseReleaseEvent`, finalize the resize and emit a signal.

**Step 3:** Add `bbox_resized = Signal(list)` signal to VideoCanvas. Connect it in ReviewWindow to update the track's bbox for the current frame.

**Step 4:** Draw handles in `paintEvent` when a track is selected and has a bbox on the current frame:

```python
# In _paint_tracks, after drawing selected bbox:
if track['track_id'] == self.selected_track_id:
    handles = self._get_bbox_handles(bbox, scale_x, scale_y, dr)
    painter.setBrush(QColor(255, 255, 255))
    painter.setPen(QPen(QColor(0, 0, 0), 1))
    for rect in handles.values():
        painter.drawRect(rect)
```

**Step 5:** In ReviewWindow, handle the resize signal:

```python
def _on_bbox_resized(self, new_bbox):
    track = self._find_track(self.selected_track_id)
    if not track:
        return
    self._push_undo()
    frame_data = self._get_track_frame(track, self.current_frame)
    if frame_data:
        frame_data['bbox'] = new_bbox
    self._mark_unsaved()
    self._refresh_all()
```

**Step 6:** Test — select a track, verify handles appear, drag a corner to resize, verify bbox updates.

**Step 7:** Commit: `feat: bounding box resize with drag handles`

---

## Task 7: ROI Filtering for Correction (software2)

**Problem:** Let users filter which tracks are visible in the correction page based on ROI.

**Files:**
- Modify: `software2/cctv_yolo/track_sidebar.py` — add "Show only in ROI" toggle
- Modify: `software2/cctv_yolo/review_window.py` — filter tracks by ROI when toggle is on

**Step 1:** Add a checkbox to track_sidebar.py filter bar:

```python
self._roi_filter_check = QCheckBox("In ROI only")
self._roi_filter_check.toggled.connect(lambda: self.filter_changed.emit(self._current_filter))
```

**Step 2:** In review_window.py, when refreshing the track list, if "In ROI only" is checked and ROIs exist, filter tracks to only those with at least one frame bbox center inside any ROI.

**Step 3:** Use the same `_point_in_polygon` / `_bbox_center_in_roi` logic from Task 5.

**Step 4:** The canvas should still show ROI boundaries. Tracks outside ROI are hidden from the sidebar list and their bboxes are dimmed/hidden on canvas.

**Step 5:** Test — draw ROI, enable filter, verify only tracks inside ROI are shown.

**Step 6:** Commit: `feat: ROI-based track filtering in correction page`

---

## Task 8: Port Changes to software (Flask Desktop v1)

**Files:**
- Modify: `software/cctv_yolo/server.py` — add model picker endpoint, ROI processing, performance routes
- Modify: `software/cctv_yolo/processor.py` — same fixes as software2 processor
- Modify: `software/cctv_yolo/templates/index.html` — restructure to 3 tabs
- Modify: `software/cctv_yolo/templates/review.html` — add bbox resize handles, ROI filtering, fix playback

**Step 1:** Update processor.py with same changes from Task 2-3 (offline model loading, ROI filtering).

**Step 2:** Add new API routes to server.py:
- `GET /api/models` — list available .pt files
- `POST /api/models/upload` — upload a model file
- `GET /api/performance/<session_id>` — get traffic counts and stats

**Step 3:** Restructure index.html tabs:
- Tab 1: Preprocessing (video list + model picker + confidence + process buttons)
- Tab 2: Correction (session list with review links)
- Tab 3: Performance (session picker + stats display)
- Settings: move to a modal dialog triggered from header

**Step 4:** Update review.html:
- Fix video playback: use `requestAnimationFrame` + frame-by-frame fetch instead of HTML5 video element (to match software2 approach of loading individual frames)
- Add bbox resize handles on canvas (JS implementation of same concept)
- Add ROI filter toggle

**Step 5:** Add bbox resize in review.html canvas:
```javascript
// In canvas mousedown: check if click is near a handle of selected bbox
// In canvas mousemove: drag handle to resize
// In canvas mouseup: finalize and save new bbox
```

**Step 6:** Test all features in browser.

**Step 7:** Commit: `feat: port all changes to software (Flask desktop v1)`

---

## Task 9: Port Changes to test (Original Dev Flask)

**Files:**
- Modify: `test/ui/app.py` — add model/performance routes
- Modify: `test/ui/templates/index.html` — restructure to 3 tabs
- Modify: `test/ui/templates/review.html` — add features
- Modify: `test/ui/static/app.js` — add bbox resize, ROI filter, fix playback
- Modify: `test/ui/static/style.css` — style new elements
- Copy processor improvements if test has its own processor

**Step 1:** The test version has a simpler structure (app.py is only 287 lines). Add:
- Model listing/selection routes
- Processing routes (currently placeholder scripts)
- Performance stats route

**Step 2:** Update index.html with 3-tab layout (currently just a simple session table).

**Step 3:** Update review.html and app.js with same features as software version.

**Step 4:** Fix playback in app.js — the test version uses HTML5 `<video>` element with `timeupdate`. This may actually work better than frame-by-frame fetch, but ensure frames sync correctly with canvas overlay.

**Step 5:** Test in browser.

**Step 6:** Commit: `feat: port all changes to test (original dev version)`

---

## Task 10: Final Integration Testing

**Step 1:** Test software2 end-to-end:
- Launch app, verify 3 tabs show
- Preprocessing: select model, draw ROI, process video
- Correction: review tracks, resize bbox, filter by ROI, save
- Performance: view counts by type

**Step 2:** Test software end-to-end:
- Launch app, open browser, verify 3 tabs
- Same flow as above

**Step 3:** Test test version:
- Launch Flask server, verify 3 tabs
- Same flow as above

**Step 4:** Commit: `chore: final integration testing complete`
