"""
Review window — the track correction editor.

Opens as a separate QMainWindow with:
- Toolbar: mode buttons, undo/redo, copy, next review, ROI tools
- Center: VideoCanvas
- Right: TrackSidebar (via QSplitter)
- Bottom: frame controls (play, step, slider)
"""
import copy
import json
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QIcon, QShortcut
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QToolBar,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QMessageBox,
    QStatusBar,
    QSizePolicy,
)

from cctv_yolo.video_canvas import VideoCanvas
from cctv_yolo.track_sidebar import TrackSidebar
from cctv_yolo.dialogs import (
    ClassChangeDialog,
    MergeDialog,
    NewTrackDialog,
    RoiNameDialog,
    RenameRoiDialog,
)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
BG = "#1a1a2e"
PANEL = "#16213e"
BORDER = "#2d3a5a"
ACCENT = "#4ecca3"
TEXT = "#eeeeee"

STYLE = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
}}
QToolBar {{
    background-color: {PANEL};
    border-bottom: 1px solid {BORDER};
    spacing: 6px;
    padding: 4px 6px;
}}
QToolBar::separator {{
    width: 1px;
    background: {BORDER};
    margin: 4px 6px;
}}
QToolBar QToolButton {{
    background: transparent;
    color: {TEXT};
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 6px 10px;
    font-size: 12px;
}}
QToolBar QToolButton:hover {{
    background-color: rgba(78, 204, 163, 0.1);
    border: 1px solid {BORDER};
}}
QToolBar QToolButton:checked {{
    background-color: {ACCENT};
    color: #000;
    font-weight: bold;
}}
QPushButton {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 12px;
    min-height: 26px;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: #1b2844;
    border-color: #3d4e73;
}}
QPushButton:pressed {{
    background-color: {ACCENT};
    color: #000;
}}
QSlider::groove:horizontal {{
    background: #0d1525;
    height: 6px;
    border-radius: 3px;
    border: 1px solid {BORDER};
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2fa87e, stop:1 {ACCENT});
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 2px solid #2fa87e;
}}
QSlider::handle:horizontal:hover {{
    background: #6fe8c0;
    border: 2px solid {ACCENT};
}}
QSpinBox {{
    background-color: #0d1525;
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 12px;
}}
QSpinBox:focus {{
    border-color: {ACCENT};
}}
QLabel {{
    color: {TEXT};
}}
QStatusBar {{
    background-color: {PANEL};
    color: {TEXT};
    border-top: 1px solid {BORDER};
    padding: 2px 8px;
}}
"""

PLAY_BTN_PLAYING = f"""
QPushButton {{
    background-color: #2fa87e;
    color: #000;
    border: 1px solid {ACCENT};
    border-radius: 6px;
    padding: 5px 12px;
    min-height: 26px;
    font-size: 12px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: {ACCENT};
}}
"""

PLAY_BTN_PAUSED = f"""
QPushButton {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 12px;
    min-height: 26px;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: #1b2844;
    border-color: #3d4e73;
}}
"""

MAX_UNDO = 50

ROI_COLORS = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57", "#ff9ff3", "#54a0ff", "#5f27cd"]


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def interpolate_frames(early_frames, late_frames):
    """Linear bbox interpolation to fill the gap between two frame lists."""
    if not early_frames or not late_frames:
        return []
    last_early = early_frames[-1]
    first_late = late_frames[0]
    if first_late["frame"] <= last_early["frame"] + 1:
        return []
    gap_start = last_early["frame"] + 1
    gap_end = first_late["frame"] - 1
    total_gap = gap_end - gap_start + 1
    if total_gap <= 0:
        return []
    interpolated = []
    for i in range(total_gap):
        t = (i + 1) / (total_gap + 1)
        bbox = [
            round(
                last_early["bbox"][j]
                + (first_late["bbox"][j] - last_early["bbox"][j]) * t,
                1,
            )
            for j in range(4)
        ]
        interpolated.append(
            {"frame": gap_start + i, "bbox": bbox, "conf": 0, "interpolated": True}
        )
    return interpolated


# ---------------------------------------------------------------------------
# ReviewWindow
# ---------------------------------------------------------------------------

class ReviewWindow(QMainWindow):
    """Track correction editor window."""

    closed = Signal()  # emitted when the window closes

    def __init__(self, data_manager, session_id, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.session_id = session_id

        # State
        self.tracks = []
        self.rois = []
        self.fps = 30.0
        self.total_frames = 0
        self.resolution = ""
        self.video_name = ""
        self.video_path = None
        self.current_frame = 0
        self.selected_track_id = None
        self.playing = False
        self.unsaved = False

        # Undo/redo stacks
        self._undo_stack = []
        self._redo_stack = []

        self._setup_ui()
        self._setup_shortcuts()
        self._load_session()

        self.setStyleSheet(STYLE)
        self.setWindowTitle(f"Review — {self.session_id}")
        self.resize(1400, 900)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --- Toolbar ---
        self.toolbar = QToolBar("Tools")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(self.toolbar.iconSize())
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        # Mode buttons
        self.btn_select = self._add_tool_button("Select (V)", checkable=True, checked=True)
        self.btn_draw = self._add_tool_button("Draw Box (B)", checkable=True)
        self.toolbar.addSeparator()

        self.btn_undo = self._add_tool_button("Undo")
        self.btn_redo = self._add_tool_button("Redo")
        self.toolbar.addSeparator()

        self.btn_copy_next = self._add_tool_button("Copy Next (N)")
        self.btn_copy_prev = self._add_tool_button("Copy Prev (P)")
        self.toolbar.addSeparator()

        self.btn_next_review = self._add_tool_button("Next Review (R)")
        self.lbl_review_progress = QLabel("")
        self.lbl_review_progress.setStyleSheet(f"color: {ACCENT}; padding: 0 8px;")
        self.toolbar.addWidget(self.lbl_review_progress)
        self.toolbar.addSeparator()

        self.btn_roi_rect = self._add_tool_button("ROI Rect (Shift+R)", checkable=True)
        self.btn_roi_poly = self._add_tool_button("ROI Poly (Shift+P)", checkable=True)

        self.lbl_mode = QLabel("  Mode: Select")
        self.lbl_mode.setStyleSheet(f"color: {ACCENT}; font-weight: bold; padding: 0 8px;")
        self.toolbar.addWidget(self.lbl_mode)

        # Connect toolbar buttons (QAction uses .triggered, not .clicked)
        self.btn_select.triggered.connect(lambda: self._set_mode("select"))
        self.btn_draw.triggered.connect(lambda: self._set_mode("draw_box"))
        self.btn_undo.triggered.connect(self._undo)
        self.btn_redo.triggered.connect(self._redo)
        self.btn_copy_next.triggered.connect(self._copy_to_next)
        self.btn_copy_prev.triggered.connect(self._copy_to_prev)
        self.btn_next_review.triggered.connect(self._next_review)
        self.btn_roi_rect.triggered.connect(lambda: self._set_mode("roi_rect"))
        self.btn_roi_poly.triggered.connect(lambda: self._set_mode("roi_polygon"))

        # --- Splitter: canvas + sidebar ---
        self.splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(self.splitter, stretch=1)

        self.canvas = VideoCanvas()
        self.splitter.addWidget(self.canvas)

        self.sidebar = TrackSidebar()
        self.splitter.addWidget(self.sidebar)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)

        # --- Frame controls ---
        frame_bar = QWidget()
        frame_bar.setStyleSheet(f"""
            background-color: {PANEL};
            border-top: 1px solid {BORDER};
        """)
        frame_layout = QHBoxLayout(frame_bar)
        frame_layout.setContentsMargins(12, 6, 12, 6)
        frame_layout.setSpacing(6)

        self.btn_play = QPushButton("Play")
        self.btn_play.setFixedWidth(70)
        self.btn_play.setStyleSheet(PLAY_BTN_PAUSED)
        self.btn_play.clicked.connect(self._toggle_play)

        self.btn_step_back = QPushButton("<")
        self.btn_step_back.setFixedWidth(34)
        self.btn_step_back.setToolTip("Step back 1 frame")
        self.btn_step_back.clicked.connect(lambda: self._step_frame(-1))

        self.btn_back_10 = QPushButton("-10")
        self.btn_back_10.setFixedWidth(44)
        self.btn_back_10.setToolTip("Step back 10 frames")
        self.btn_back_10.clicked.connect(lambda: self._step_frame(-10))

        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(0)
        self.frame_spin.setFixedWidth(90)
        self.frame_spin.valueChanged.connect(self._on_frame_spin_changed)

        self.btn_fwd_10 = QPushButton("+10")
        self.btn_fwd_10.setFixedWidth(44)
        self.btn_fwd_10.setToolTip("Step forward 10 frames")
        self.btn_fwd_10.clicked.connect(lambda: self._step_frame(10))

        self.btn_step_fwd = QPushButton(">")
        self.btn_step_fwd.setFixedWidth(34)
        self.btn_step_fwd.setToolTip("Step forward 1 frame")
        self.btn_step_fwd.clicked.connect(lambda: self._step_frame(1))

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)

        self.lbl_frame_info = QLabel("Frame 0 / 0")
        self.lbl_frame_info.setMinimumWidth(150)
        self.lbl_frame_info.setStyleSheet(f"color: #8899aa; font-size: 12px; font-family: 'SF Mono', 'Menlo', monospace;")

        frame_layout.addWidget(self.btn_play)
        frame_layout.addWidget(self.btn_step_back)
        frame_layout.addWidget(self.btn_back_10)
        frame_layout.addWidget(self.frame_spin)
        frame_layout.addWidget(self.btn_fwd_10)
        frame_layout.addWidget(self.btn_step_fwd)
        frame_layout.addWidget(self.frame_slider, stretch=1)
        frame_layout.addWidget(self.lbl_frame_info)

        root_layout.addWidget(frame_bar)

        # --- Status bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # --- Play timer ---
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(100)
        self.play_timer.timeout.connect(self._on_play_tick)

        # --- Connect canvas signals ---
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self.canvas.bbox_resized.connect(self._on_bbox_resized)
        self.canvas.roi_rect_drawn.connect(self._on_roi_rect_drawn)
        self.canvas.roi_polygon_drawn.connect(self._on_roi_polygon_drawn)

        # --- Connect sidebar signals ---
        self.sidebar.track_selected.connect(self._on_track_selected)
        self.sidebar.track_double_clicked.connect(self._on_track_double_clicked)
        self.sidebar.delete_requested.connect(self._delete_track)
        self.sidebar.class_change_requested.connect(self._change_class)
        self.sidebar.merge_requested.connect(self._merge_tracks)
        self.sidebar.split_requested.connect(self._split_track)
        self.sidebar.save_requested.connect(self._save)
        self.sidebar.back_requested.connect(self.close)
        self.sidebar.roi_delete_requested.connect(self._delete_roi)

    def _add_tool_button(self, text, checkable=False, checked=False):
        """Add a button to the toolbar and return it."""
        action = QAction(text, self)
        action.setCheckable(checkable)
        action.setChecked(checked)
        self.toolbar.addAction(action)
        # Return the QToolButton widget for the action
        btn = self.toolbar.widgetForAction(action)
        return action

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _setup_shortcuts(self):
        shortcuts = [
            (Qt.Key_Space, self._toggle_play),
            (Qt.Key_Left, lambda: self._step_frame(-1)),
            (Qt.Key_Right, lambda: self._step_frame(1)),
            (Qt.Key_V, lambda: self._set_mode("select")),
            (Qt.Key_B, lambda: self._set_mode("draw_box")),
            (Qt.Key_D, self._delete_track),
            (Qt.Key_C, self._change_class),
            (Qt.Key_M, self._merge_tracks),
            (Qt.Key_S, self._split_track),
            (Qt.Key_N, self._copy_to_next),
            (Qt.Key_P, self._copy_to_prev),
            (Qt.Key_R, self._next_review),
            (Qt.Key_Escape, self._cancel_mode),
        ]
        for key, slot in shortcuts:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(slot)

        # Modifier shortcuts
        sc_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        sc_undo.activated.connect(self._undo)

        sc_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        sc_redo.activated.connect(self._redo)

        sc_save = QShortcut(QKeySequence("Ctrl+S"), self)
        sc_save.activated.connect(self._save)

        sc_roi_rect = QShortcut(QKeySequence("Shift+R"), self)
        sc_roi_rect.activated.connect(lambda: self._set_mode("roi_rect"))

        sc_roi_poly = QShortcut(QKeySequence("Shift+P"), self)
        sc_roi_poly.activated.connect(lambda: self._set_mode("roi_polygon"))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_session(self):
        """Load session data (corrections first, then tracks)."""
        data = self.data_manager.load_session_data(self.session_id)
        if data is None:
            QMessageBox.critical(self, "Error", f"No data found for session: {self.session_id}")
            QTimer.singleShot(0, self.close)
            return

        self.tracks = data.get("tracks", [])
        self.rois = data.get("rois", [])
        self.fps = data.get("fps", 30)
        self.total_frames = data.get("total_frames", 0)
        self.resolution = data.get("resolution", "")
        self.video_name = data.get("video_name", "")
        self.video_path = self.data_manager.get_video_path(self.session_id)

        # Set up frame controls
        max_frame = max(self.total_frames - 1, 0)
        self.frame_slider.setMaximum(max_frame)
        self.frame_spin.setMaximum(max_frame)

        # Open video in canvas
        if self.video_path and self.video_path.exists():
            self.canvas.open_video(str(self.video_path))
        else:
            self.status_bar.showMessage("Warning: video file not found")

        # Set play timer interval based on fps
        if self.fps > 0:
            self.play_timer.setInterval(max(int(1000 / self.fps), 16))

        # Push initial state to undo stack
        self._push_undo()

        # Update the UI
        self._go_to_frame(0)
        self._update_review_progress()
        self.setWindowTitle(f"Review — {self.video_name or self.session_id}")

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def _set_mode(self, mode):
        """Switch the editing mode."""
        self.btn_select.setChecked(mode == "select")
        self.btn_draw.setChecked(mode == "draw_box")
        self.btn_roi_rect.setChecked(mode == "roi_rect")
        self.btn_roi_poly.setChecked(mode == "roi_polygon")

        mode_labels = {
            "select": "Select",
            "draw_box": "Draw Box",
            "roi_rect": "ROI Rect",
            "roi_polygon": "ROI Polygon",
        }
        self.lbl_mode.setText(f"  Mode: {mode_labels.get(mode, mode)}")
        self.canvas.drawing_mode = mode
        self.canvas.set_cursor_for_mode()

    def _cancel_mode(self):
        """Cancel current drawing and return to select mode."""
        self.canvas.cancel_drawing()
        self._set_mode("select")

    # ------------------------------------------------------------------
    # Frame navigation
    # ------------------------------------------------------------------

    def _go_to_frame(self, frame_num, lightweight=False):
        """Navigate to a specific frame number.

        If lightweight=True, skip expensive sidebar refresh (used during playback).
        """
        frame_num = max(0, min(frame_num, self.total_frames - 1)) if self.total_frames > 0 else 0
        self.current_frame = frame_num

        # Update canvas properties BEFORE set_frame so the repaint has correct data
        self.canvas.tracks = self.tracks
        self.canvas.rois = self.rois
        self.canvas.selected_track_id = self.selected_track_id
        self.canvas.current_frame = frame_num
        self.canvas.set_frame(frame_num)

        # Update frame controls (block signals to avoid recursion)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_num)
        self.frame_slider.blockSignals(False)

        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(frame_num)
        self.frame_spin.blockSignals(False)

        self.lbl_frame_info.setText(f"Frame {frame_num} / {self.total_frames}")

        # Update sidebar — lightweight during playback to avoid stutter
        self.sidebar.set_current_frame(frame_num)
        if not lightweight:
            self._refresh_sidebar()

    def _step_frame(self, delta):
        """Step forward or backward by delta frames."""
        self._go_to_frame(self.current_frame + delta)

    def _on_slider_changed(self, value):
        self._go_to_frame(value)

    def _on_frame_spin_changed(self, value):
        self._go_to_frame(value)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _toggle_play(self):
        if self.playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if self.total_frames <= 0:
            return
        self.playing = True
        self.btn_play.setText("Pause")
        self.btn_play.setStyleSheet(PLAY_BTN_PLAYING)
        self.play_timer.start()

    def _pause(self):
        self.playing = False
        self.btn_play.setText("Play")
        self.btn_play.setStyleSheet(PLAY_BTN_PAUSED)
        self.play_timer.stop()
        self._refresh_sidebar()  # full refresh when paused

    def _on_play_tick(self):
        if self.current_frame >= self.total_frames - 1:
            self._pause()
            self._refresh_sidebar()  # full refresh on stop
            return
        self._go_to_frame(self.current_frame + 1, lightweight=True)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def _snapshot(self):
        """Return a JSON string snapshot of current tracks and rois."""
        return json.dumps({"tracks": self.tracks, "rois": self.rois})

    def _restore_snapshot(self, snapshot_str):
        """Restore tracks and rois from a JSON string snapshot."""
        data = json.loads(snapshot_str)
        self.tracks = data["tracks"]
        self.rois = data["rois"]
        self._mark_unsaved()
        self._refresh_all()

    def _push_undo(self):
        """Push current state onto the undo stack."""
        snapshot = self._snapshot()
        if self._undo_stack and self._undo_stack[-1] == snapshot:
            return  # no change
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _undo(self):
        if len(self._undo_stack) <= 1:
            self.status_bar.showMessage("Nothing to undo")
            return
        current = self._undo_stack.pop()
        self._redo_stack.append(current)
        self._restore_snapshot(self._undo_stack[-1])
        self.status_bar.showMessage("Undo")

    def _redo(self):
        if not self._redo_stack:
            self.status_bar.showMessage("Nothing to redo")
            return
        snapshot = self._redo_stack.pop()
        self._undo_stack.append(snapshot)
        self._restore_snapshot(snapshot)
        self.status_bar.showMessage("Redo")

    # ------------------------------------------------------------------
    # Track selection
    # ------------------------------------------------------------------

    def _on_track_selected(self, track_id):
        self.selected_track_id = track_id
        self.canvas.selected_track_id = track_id
        self.canvas.update()

    def _on_track_double_clicked(self, track_id):
        """Jump to the first frame of the double-clicked track."""
        track = self._find_track(track_id)
        if track and track.get("frames"):
            first_frame = track["frames"][0]["frame"]
            self._go_to_frame(first_frame)

    def _find_track(self, track_id):
        """Find a track by its ID."""
        for t in self.tracks:
            if t.get("track_id") == track_id:
                return t
        return None

    def _get_track_frame(self, track, frame_num):
        """Get the frame data for a track at a specific frame number."""
        if not track:
            return None
        for fd in track.get("frames", []):
            if fd["frame"] == frame_num:
                return fd
        return None

    # ------------------------------------------------------------------
    # Track operations
    # ------------------------------------------------------------------

    def _delete_track(self):
        """Delete the currently selected track."""
        if self.selected_track_id is None:
            self.status_bar.showMessage("No track selected")
            return
        reply = QMessageBox.question(
            self,
            "Delete Track",
            f"Delete track #{self.selected_track_id}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.tracks = [t for t in self.tracks if t.get("track_id") != self.selected_track_id]
        self.selected_track_id = None
        self._push_undo()
        self._mark_unsaved()
        self._refresh_all()
        self.status_bar.showMessage("Track deleted")

    def _change_class(self):
        """Change the class of the selected track."""
        if self.selected_track_id is None:
            self.status_bar.showMessage("No track selected")
            return
        track = self._find_track(self.selected_track_id)
        if not track:
            return
        dlg = ClassChangeDialog(track.get("class", "car"), self)
        if dlg.exec() == ClassChangeDialog.Accepted:
            new_class = dlg.selected_class()
            track["class"] = new_class
            self._push_undo()
            self._mark_unsaved()
            self._refresh_all()
            self.status_bar.showMessage(f"Track #{self.selected_track_id} class changed to {new_class}")

    def _merge_tracks(self):
        """Merge selected track with another track (with gap interpolation)."""
        if self.selected_track_id is None:
            self.status_bar.showMessage("No track selected")
            return
        dlg = MergeDialog(self.selected_track_id, self.tracks, self)
        if dlg.exec() != MergeDialog.Accepted:
            return
        target_id = dlg.selected_target_id()
        if target_id is None:
            return

        source = self._find_track(self.selected_track_id)
        target = self._find_track(target_id)
        if not source or not target:
            return

        # Combine frames
        all_frames = source.get("frames", []) + target.get("frames", [])
        all_frames.sort(key=lambda f: f["frame"])

        # Deduplicate — keep higher confidence per frame number
        seen = {}
        for fd in all_frames:
            fn = fd["frame"]
            if fn not in seen or fd.get("conf", 0) > seen[fn].get("conf", 0):
                seen[fn] = fd
        merged_frames = sorted(seen.values(), key=lambda f: f["frame"])

        # Find early and late segments for interpolation
        source_frames = sorted(source.get("frames", []), key=lambda f: f["frame"])
        target_frames = sorted(target.get("frames", []), key=lambda f: f["frame"])

        if source_frames and target_frames:
            if source_frames[-1]["frame"] < target_frames[0]["frame"]:
                interp = interpolate_frames(source_frames, target_frames)
                merged_frames = merged_frames + interp
            elif target_frames[-1]["frame"] < source_frames[0]["frame"]:
                interp = interpolate_frames(target_frames, source_frames)
                merged_frames = merged_frames + interp

        merged_frames.sort(key=lambda f: f["frame"])

        # Update target with merged frames, keep target class
        target["frames"] = merged_frames

        # Remove source track
        self.tracks = [t for t in self.tracks if t.get("track_id") != self.selected_track_id]
        self.selected_track_id = target_id

        self._push_undo()
        self._mark_unsaved()
        self._refresh_all()
        self.status_bar.showMessage(f"Merged into track #{target_id}")

    def _split_track(self):
        """Split the selected track at the current frame."""
        if self.selected_track_id is None:
            self.status_bar.showMessage("No track selected")
            return
        track = self._find_track(self.selected_track_id)
        if not track:
            return

        frames = track.get("frames", [])
        before = [f for f in frames if f["frame"] < self.current_frame]
        at_and_after = [f for f in frames if f["frame"] >= self.current_frame]

        if not before or not at_and_after:
            self.status_bar.showMessage("Cannot split — need frames on both sides of current frame")
            return

        # Generate new track ID
        max_id = max((t.get("track_id", 0) for t in self.tracks), default=0)
        new_id = max_id + 1

        # Modify existing track: keep only the "before" frames
        track["frames"] = before

        # Create new track with "at_and_after" frames
        new_track = {
            "track_id": new_id,
            "class": track.get("class", "car"),
            "frames": at_and_after,
        }
        self.tracks.append(new_track)

        self._push_undo()
        self._mark_unsaved()
        self._refresh_all()
        self.status_bar.showMessage(f"Split track #{self.selected_track_id} at frame {self.current_frame} -> new track #{new_id}")

    # ------------------------------------------------------------------
    # Copy bbox to adjacent frame
    # ------------------------------------------------------------------

    def _copy_to_next(self):
        """Copy the selected track's bbox at current frame to the next frame."""
        self._copy_box_to_frame(self.current_frame + 1)

    def _copy_to_prev(self):
        """Copy the selected track's bbox at current frame to the previous frame."""
        self._copy_box_to_frame(self.current_frame - 1)

    def _copy_box_to_frame(self, target_frame):
        """Copy the selected track's current frame bbox to target_frame."""
        if self.selected_track_id is None:
            self.status_bar.showMessage("No track selected")
            return
        if target_frame < 0 or target_frame >= self.total_frames:
            self.status_bar.showMessage("Target frame out of range")
            return

        track = self._find_track(self.selected_track_id)
        if not track:
            return

        current_fd = self._get_track_frame(track, self.current_frame)
        if not current_fd:
            self.status_bar.showMessage("No bbox at current frame for selected track")
            return

        # Check if target frame already has data
        existing = self._get_track_frame(track, target_frame)
        if existing:
            existing["bbox"] = list(current_fd["bbox"])
            existing["conf"] = 0
        else:
            new_fd = {
                "frame": target_frame,
                "bbox": list(current_fd["bbox"]),
                "conf": 0,
            }
            track["frames"].append(new_fd)
            track["frames"].sort(key=lambda f: f["frame"])

        self._push_undo()
        self._mark_unsaved()
        self._go_to_frame(target_frame)
        self.status_bar.showMessage(f"Copied bbox to frame {target_frame}")

    # ------------------------------------------------------------------
    # Drawing callbacks (from VideoCanvas signals)
    # ------------------------------------------------------------------

    def _on_box_drawn(self, bbox):
        """Handle a new bounding box drawn on the canvas."""
        if self.selected_track_id is not None:
            # Add/update bbox for existing track
            track = self._find_track(self.selected_track_id)
            if track:
                existing = self._get_track_frame(track, self.current_frame)
                if existing:
                    existing["bbox"] = bbox
                    existing["conf"] = 0
                else:
                    track["frames"].append({
                        "frame": self.current_frame,
                        "bbox": bbox,
                        "conf": 0,
                    })
                    track["frames"].sort(key=lambda f: f["frame"])
                self._push_undo()
                self._mark_unsaved()
                self._refresh_all()
                self.status_bar.showMessage(f"Updated bbox for track #{self.selected_track_id}")
                return

        # No track selected — create a new track
        dlg = NewTrackDialog(self)
        if dlg.exec() != NewTrackDialog.Accepted:
            return
        new_class = dlg.selected_class()
        max_id = max((t.get("track_id", 0) for t in self.tracks), default=0)
        new_id = max_id + 1
        new_track = {
            "track_id": new_id,
            "class": new_class,
            "frames": [
                {"frame": self.current_frame, "bbox": bbox, "conf": 0}
            ],
        }
        self.tracks.append(new_track)
        self.selected_track_id = new_id
        self._push_undo()
        self._mark_unsaved()
        self._refresh_all()
        self._set_mode("select")
        self.status_bar.showMessage(f"Created track #{new_id} ({new_class})")

    def _on_bbox_resized(self, new_bbox):
        """Handle a bounding box resize from the canvas."""
        track = self._find_track(self.selected_track_id)
        if not track:
            return
        self._push_undo()
        frame_data = self._get_track_frame(track, self.current_frame)
        if frame_data:
            frame_data['bbox'] = new_bbox
        self._mark_unsaved()
        self._refresh_all()
        self.status_bar.showMessage(f"Resized bbox for track #{self.selected_track_id}")

    # ------------------------------------------------------------------
    # ROI operations
    # ------------------------------------------------------------------

    def _on_roi_rect_drawn(self, top_left, bottom_right):
        """Handle a rectangular ROI drawn on the canvas."""
        dlg = RoiNameDialog(f"ROI {len(self.rois) + 1}", self)
        if dlg.exec() != RoiNameDialog.Accepted:
            return
        name = dlg.roi_name()
        roi = {
            "type": "rect",
            "name": name,
            "points": [top_left, bottom_right],
            "color": ROI_COLORS[len(self.rois) % len(ROI_COLORS)],
        }
        self.rois.append(roi)
        self._push_undo()
        self._mark_unsaved()
        self._refresh_all()
        self._set_mode("select")
        self.status_bar.showMessage(f"ROI '{name}' created")

    def _on_roi_polygon_drawn(self, points):
        """Handle a polygon ROI drawn on the canvas."""
        if len(points) < 3:
            self.status_bar.showMessage("Polygon needs at least 3 points")
            return
        dlg = RoiNameDialog(f"ROI {len(self.rois) + 1}", self)
        if dlg.exec() != RoiNameDialog.Accepted:
            return
        name = dlg.roi_name()
        roi = {
            "type": "polygon",
            "name": name,
            "points": points,
            "color": ROI_COLORS[len(self.rois) % len(ROI_COLORS)],
        }
        self.rois.append(roi)
        self._push_undo()
        self._mark_unsaved()
        self._refresh_all()
        self._set_mode("select")
        self.status_bar.showMessage(f"ROI '{name}' created")

    def _delete_roi(self, roi_index):
        """Delete an ROI by its index."""
        if 0 <= roi_index < len(self.rois):
            name = self.rois[roi_index].get("name", f"ROI {roi_index}")
            reply = QMessageBox.question(
                self,
                "Delete ROI",
                f"Delete ROI '{name}'?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.rois.pop(roi_index)
                self._push_undo()
                self._mark_unsaved()
                self._refresh_all()
                self.status_bar.showMessage(f"ROI '{name}' deleted")

    # ------------------------------------------------------------------
    # Next review
    # ------------------------------------------------------------------

    def _next_review(self):
        """Navigate to the next session that needs review."""
        next_id = self.data_manager.get_next_review_session(self.session_id)
        if next_id is None:
            self.status_bar.showMessage("No more sessions need review")
            return

        if self.unsaved:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Save before navigating?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._save()
            elif reply == QMessageBox.Cancel:
                return

        # Load the new session
        self.session_id = next_id
        self.selected_track_id = None
        self.unsaved = False
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._load_session()
        self.status_bar.showMessage(f"Navigated to session: {next_id}")

    def _update_review_progress(self):
        """Update the review progress label in the toolbar."""
        sessions = self.data_manager.get_sessions()
        total = len(sessions)
        needs = sum(1 for s in sessions if s.get("needs_review", 0) > 0 and not s.get("has_corrections", False))
        self.lbl_review_progress.setText(f"{needs} need review / {total} total")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self):
        """Save corrections to disk."""
        # Build the session data to save
        base_data = self.data_manager.load_session_data(self.session_id) or {}
        save_data = {
            "tracks": self.tracks,
            "rois": self.rois,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "resolution": self.resolution,
            "video_name": self.video_name,
        }
        # Preserve any extra fields from the original data
        for key in base_data:
            if key not in save_data:
                save_data[key] = base_data[key]

        self.data_manager.save_corrections(self.session_id, save_data)
        self.unsaved = False
        self.sidebar.set_unsaved(False)
        self.setWindowTitle(f"Review — {self.video_name or self.session_id}")
        self._update_review_progress()
        self.status_bar.showMessage("Corrections saved")

    # ------------------------------------------------------------------
    # UI refresh helpers
    # ------------------------------------------------------------------

    def _mark_unsaved(self):
        self.unsaved = True
        self.sidebar.set_unsaved(True)
        title = self.windowTitle()
        if not title.startswith("* "):
            self.setWindowTitle(f"* {title}")

    def _refresh_sidebar(self):
        """Refresh the sidebar with current state."""
        self.sidebar.set_tracks(self.tracks, self.current_frame, self.selected_track_id)
        self.sidebar.set_rois(self.rois, self.tracks)

    def _refresh_all(self):
        """Refresh both canvas and sidebar."""
        self.canvas.tracks = self.tracks
        self.canvas.rois = self.rois
        self.canvas.selected_track_id = self.selected_track_id
        self.canvas.current_frame = self.current_frame
        self.canvas.update()
        self._refresh_sidebar()

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Warn about unsaved changes before closing."""
        if self.unsaved:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._save()
                self._cleanup()
                event.accept()
            elif reply == QMessageBox.Discard:
                self._cleanup()
                event.accept()
            else:
                event.ignore()
                return
        else:
            self._cleanup()
            event.accept()

        self.closed.emit()

    def _cleanup(self):
        """Clean up resources."""
        self.play_timer.stop()
        self.canvas.close_video()
