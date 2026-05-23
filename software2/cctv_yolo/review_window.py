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
import uuid
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QIcon, QShortcut, QColor
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
    QMenu,
    QFileDialog,
    QColorDialog,
)

from cctv_yolo.video_canvas import VideoCanvas
from cctv_yolo.track_sidebar import TrackSidebar
from cctv_yolo.timeline_minimap import TimelineMinimap
from cctv_yolo.occlusion_dialog import OcclusionSuggestionsDialog
from cctv_yolo.dialogs import (
    ClassChangeDialog,
    MergeDialog,
    NewTrackDialog,
    RoiNameDialog,
    RenameRoiDialog,
)
from cctv_yolo.widgets.open_location_bar import OpenLocationBar

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
from cctv_yolo.theme import (
    INDIGO as BG, PANEL, BORDER, PURPLE as ACCENT, OFFWHITE as TEXT,
    roi_color as theme_roi_color,
    PINK,
    TEXT_MUTED,
    INDIGO,
    OFFWHITE,
    PURPLE,
)

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
    color: ;
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
    background-color: ;
    border-color: ;
}}
QPushButton:pressed {{
    background-color: {ACCENT};
    color: ;
}}
QSlider::groove:horizontal {{
    background: ;
    height: 6px;
    border-radius: 3px;
    border: 1px solid {BORDER};
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 , stop:1 {ACCENT});
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 2px solid ;
}}
QSlider::handle:horizontal:hover {{
    background: ;
    border: 2px solid {ACCENT};
}}
QSpinBox {{
    background-color: ;
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
    background-color: ;
    color: ;
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
    background-color: ;
    border-color: ;
}}
"""

MAX_UNDO = 50

# ROI colors come from the theme palette (PRD C11). Use ``theme_roi_color(i)``
# instead of hard-coded hex anywhere new.
def _roi_palette_color(index: int) -> str:
    return theme_roi_color(index)


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
        self.btn_occlusion = self._add_tool_button("Find Occlusions")
        self.lbl_review_progress = QLabel("")
        self.lbl_review_progress.setStyleSheet(f"color: {ACCENT}; padding: 0 8px;")
        self.toolbar.addWidget(self.lbl_review_progress)
        self.toolbar.addSeparator()

        self.btn_roi_rect = self._add_tool_button("ROI Rect (Shift+R)", checkable=True)
        self.btn_roi_poly = self._add_tool_button("ROI Poly (Shift+P)", checkable=True)
        self.toolbar.addSeparator()

        # Export menu (PRD F4) — single button opens grouped QMenu
        self.btn_export = QPushButton("Export…")
        self.btn_export.setMinimumHeight(28)
        self.btn_export.clicked.connect(self._show_export_menu)
        self.toolbar.addWidget(self.btn_export)

        self.lbl_mode = QLabel("  Mode: Select")
        self.lbl_mode.setStyleSheet(f"color: {ACCENT}; font-weight: bold; padding: 0 8px;")
        self.toolbar.addWidget(self.lbl_mode)

        # OpenLocationBar (PRD C12) — push right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)
        self.open_bar = OpenLocationBar(self)
        self.open_bar.add_file(
            "Session Video",
            lambda: self.data_manager.get_video_path(self.session_id),
        )
        self.open_bar.add_file(
            "Tracks JSON",
            lambda: self.data_manager.tracks_dir / f"{self.session_id}.json",
        )
        self.open_bar.add_file(
            "Corrections JSON",
            lambda: self.data_manager.corrections_dir / f"{self.session_id}.json",
        )
        self.open_bar.add_folder(
            "Exports",
            lambda: self.data_manager.exports_dir / self.session_id,
        )
        self.toolbar.addWidget(self.open_bar)

        # Connect toolbar buttons (QAction uses .triggered, not .clicked)
        self.btn_select.triggered.connect(lambda: self._set_mode("select"))
        self.btn_draw.triggered.connect(lambda: self._set_mode("draw_box"))
        self.btn_undo.triggered.connect(self._undo)
        self.btn_redo.triggered.connect(self._redo)
        self.btn_copy_next.triggered.connect(self._copy_to_next)
        self.btn_copy_prev.triggered.connect(self._copy_to_prev)
        self.btn_next_review.triggered.connect(self._next_review)
        self.btn_occlusion.triggered.connect(self._open_occlusion_dialog)
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
        self.lbl_frame_info.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; font-family: 'SF Mono', 'Menlo', monospace;")

        frame_layout.addWidget(self.btn_play)
        frame_layout.addWidget(self.btn_step_back)
        frame_layout.addWidget(self.btn_back_10)
        frame_layout.addWidget(self.frame_spin)
        frame_layout.addWidget(self.btn_fwd_10)
        frame_layout.addWidget(self.btn_step_fwd)
        frame_layout.addWidget(self.frame_slider, stretch=1)
        frame_layout.addWidget(self.lbl_frame_info)

        # --- Timeline minimap (above frame bar) ---
        self.minimap = TimelineMinimap()
        self.minimap.frame_clicked.connect(self._go_to_frame)
        root_layout.addWidget(self.minimap)

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
        self.canvas.bbox_moved.connect(self._on_bbox_moved)
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
        self.sidebar.roi_rename_requested.connect(self._rename_roi)
        self.sidebar.roi_recolor_requested.connect(self._recolor_roi)
        self.sidebar.roi_selection_changed.connect(self._on_roi_selection_changed)

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
        # Backfill missing UUID/created_at on legacy ROIs so the rest of
        # the app can always rely on rois[].id (PRD F3-1).
        for r in self.rois:
            if not r.get("id"):
                r["id"] = uuid.uuid4().hex[:12]
            if not r.get("created_at"):
                r["created_at"] = datetime.now().isoformat(timespec="seconds")
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
        self.minimap.set_data(
            {"tracks": self.tracks, "rois": self.rois},
            self.total_frames,
        )
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

        # Update minimap playhead
        if hasattr(self, "minimap"):
            self.minimap.set_current_frame(frame_num)

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
        dlg = ClassChangeDialog(
            track.get("class", "car"),
            track.get("subclass", "") or "",
            self,
        )
        if dlg.exec() == ClassChangeDialog.Accepted:
            new_class = dlg.selected_class()
            new_sub = dlg.selected_subclass()
            track["class"] = new_class
            if new_sub:
                track["subclass"] = new_sub
            else:
                track.pop("subclass", None)
            self._push_undo()
            self._mark_unsaved()
            self._refresh_all()
            label = new_class + ("/" + new_sub if new_sub else "")
            self.status_bar.showMessage(
                "Track #" + str(self.selected_track_id) + " class -> " + label
            )

    def _open_occlusion_dialog(self):
        """Run gap-candidate detection and let the reviewer accept merges."""
        track_data = {"tracks": self.tracks, "rois": self.rois}
        dlg = OcclusionSuggestionsDialog(track_data, self)
        dlg.apply_pair.connect(self._merge_pair_for_occlusion)
        dlg.show_pair.connect(self._focus_track)
        run = getattr(dlg, "exec")
        run()

    def _focus_track(self, track_id: int):
        """Select a track and seek to its first frame so the reviewer
        can eyeball it before accepting a merge."""
        track = self._find_track(track_id)
        if not track or not track.get("frames"):
            return
        self.selected_track_id = track_id
        first_frame = sorted(track["frames"], key=lambda f: f["frame"])[0]["frame"]
        self._go_to_frame(first_frame)

    def _merge_pair_for_occlusion(self, source_id: int, target_id: int):
        """Programmatically merge two tracks (used by the occlusion
        dialog). Identical to ``_merge_tracks`` but driven by IDs and
        marks the new interpolated frames as ``occluded``."""
        source = self._find_track(source_id)
        target = self._find_track(target_id)
        if not source or not target:
            return

        source_frames = sorted(source.get("frames", []), key=lambda f: f["frame"])
        target_frames = sorted(target.get("frames", []), key=lambda f: f["frame"])

        all_frames = source_frames + target_frames
        all_frames.sort(key=lambda f: f["frame"])
        seen = {}
        for fd in all_frames:
            fn = fd["frame"]
            if fn not in seen or fd.get("conf", 0) > seen[fn].get("conf", 0):
                seen[fn] = fd
        merged_frames = sorted(seen.values(), key=lambda f: f["frame"])

        if source_frames and target_frames:
            if source_frames[-1]["frame"] < target_frames[0]["frame"]:
                interp = interpolate_frames(source_frames, target_frames)
            elif target_frames[-1]["frame"] < source_frames[0]["frame"]:
                interp = interpolate_frames(target_frames, source_frames)
            else:
                interp = []
            # Mark every newly-added interpolated frame as occluded
            for fd in interp:
                fd["occluded"] = True
            merged_frames = merged_frames + interp

        merged_frames.sort(key=lambda f: f["frame"])
        target["frames"] = merged_frames
        self.tracks = [t for t in self.tracks if t.get("track_id") != source_id]
        if self.selected_track_id == source_id:
            self.selected_track_id = target_id

        self._push_undo()
        self._mark_unsaved()
        self._refresh_all()
        self.status_bar.showMessage(
            "Occlusion-merged #" + str(source_id) + " into #" + str(target_id)
        )

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

    def _on_bbox_moved(self, new_bbox):
        """Handle a bounding box drag-move from the canvas."""
        track = self._find_track(self.selected_track_id)
        if not track:
            return
        self._push_undo()
        frame_data = self._get_track_frame(track, self.current_frame)
        if frame_data:
            frame_data['bbox'] = new_bbox
        self._mark_unsaved()
        self._refresh_all()
        self.status_bar.showMessage(f"Moved bbox for track #{self.selected_track_id}")

    # ------------------------------------------------------------------
    # ROI operations
    # ------------------------------------------------------------------

    def _on_roi_rect_drawn(self, top_left, bottom_right):
        dlg = RoiNameDialog(f"ROI {len(self.rois) + 1}", self)
        if dlg.exec() != RoiNameDialog.Accepted:
            return
        name = dlg.roi_name()
        roi = {
            "id": uuid.uuid4().hex[:12],
            "type": "rect",
            "name": name,
            "points": [top_left, bottom_right],
            "color": _roi_palette_color(len(self.rois)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.rois.append(roi)
        self._push_undo()
        self._mark_unsaved()
        self._refresh_all()
        self._set_mode("select")
        self.status_bar.showMessage(f"ROI '{name}' created")

    def _on_roi_polygon_drawn(self, points):
        if len(points) < 3:
            self.status_bar.showMessage("Polygon needs at least 3 points")
            return
        dlg = RoiNameDialog(f"ROI {len(self.rois) + 1}", self)
        if dlg.exec() != RoiNameDialog.Accepted:
            return
        name = dlg.roi_name()
        roi = {
            "id": uuid.uuid4().hex[:12],
            "type": "polygon",
            "name": name,
            "points": points,
            "color": _roi_palette_color(len(self.rois)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.rois.append(roi)
        self._push_undo()
        self._mark_unsaved()
        self._refresh_all()
        self._set_mode("select")
        self.status_bar.showMessage(f"ROI '{name}' created")

    def _delete_roi(self, roi_index):
        if 0 <= roi_index < len(self.rois):
            name = self.rois[roi_index].get("name", f"ROI {roi_index}")
            reply = QMessageBox.question(
                self, "Delete ROI", f"Delete ROI '{name}'?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.rois.pop(roi_index)
                self._push_undo()
                self._mark_unsaved()
                self._refresh_all()
                self.status_bar.showMessage(f"ROI '{name}' deleted")

    def _rename_roi(self, roi_index):
        """Rename an existing ROI."""
        if 0 <= roi_index < len(self.rois):
            current_name = self.rois[roi_index].get("name", f"ROI {roi_index}")
            dlg = RenameRoiDialog(current_name, self)
            if dlg.exec() == RenameRoiDialog.Accepted:
                new_name = dlg.new_name()
                if new_name:
                    self.rois[roi_index]["name"] = new_name
                    self._push_undo()
                    self._mark_unsaved()
                    self._refresh_all()
                    self.status_bar.showMessage(f"ROI renamed to '{new_name}'")

    def _recolor_roi(self, roi_index):
        """Pick a new color for an ROI (PRD F3-1)."""
        if not (0 <= roi_index < len(self.rois)):
            return
        current = QColor(self.rois[roi_index].get("color", OFFWHITE))
        new_color = QColorDialog.getColor(current, self, "Choose ROI color")
        if not new_color.isValid():
            return
        self.rois[roi_index]["color"] = new_color.name()
        self._push_undo()
        self._mark_unsaved()
        self._refresh_all()
        self.status_bar.showMessage(
            f"ROI '{self.rois[roi_index].get('name', '?')}' recolored"
        )

    def _on_roi_selection_changed(self):
        """Handle ROI checkbox selection changes — update canvas dimming."""
        self._apply_roi_filter()

    def _apply_roi_filter(self):
        """Compute dimmed track IDs based on sidebar ROI filter state and update canvas."""
        if self.sidebar.is_roi_filter_active():
            roi_ids = self.sidebar.get_roi_track_ids()
            self.canvas.dimmed_track_ids = {
                t.get("track_id") for t in self.tracks
                if t.get("track_id") not in roi_ids
            }
        else:
            self.canvas.dimmed_track_ids = set()
        # PRD F3-1: pass selected ROI indices to the canvas so _paint_rois
        # knows which ROI gets the OFFWHITE focus stroke and which to dim.
        try:
            self.canvas.selected_roi_indices = set(
                self.sidebar.get_selected_roi_indices()
            )
        except Exception:
            self.canvas.selected_roi_indices = set()
        self.canvas.update()

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
        """Save corrections to disk.

        PRD F2-1f: any save error pops a toast and opens the corrections
        folder so the user can diagnose (e.g. permissions, full disk).
        """
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

        try:
            self.data_manager.save_corrections(self.session_id, save_data)
        except Exception as e:
            # Pop a toast then open the corrections folder so the user can
            # check what went wrong (permissions, no space, etc).
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Save failed: {e}\n\nFile location opened for diagnosis.",
            )
            try:
                self.data_manager.open_folder("corrections")
            except Exception:
                pass
            self.status_bar.showMessage(f"Save failed: {e}")
            return

        self.unsaved = False
        self.sidebar.set_unsaved(False)
        self.setWindowTitle(f"Review — {self.video_name or self.session_id}")
        self._update_review_progress()
        self.status_bar.showMessage("Corrections saved")

    # ------------------------------------------------------------------
    # Export menu (PRD F4)
    # ------------------------------------------------------------------

    def _active_roi_filter_id(self) -> str | None:
        """Return the ROI id (or name) the user has filtered to, or None.

        Honors the ROI panel's selected ROIs — single-select becomes the
        filter, multi-select disables the filter (no sensible filter).
        """
        try:
            sel = self.sidebar.get_selected_roi_indices()
        except Exception:
            return None
        if len(sel) != 1:
            return None
        idx = next(iter(sel))
        if 0 <= idx < len(self.rois):
            roi = self.rois[idx]
            return roi.get("id") or roi.get("name")
        return None

    def _show_export_menu(self):
        """Build and pop up the grouped Export menu."""
        menu = QMenu(self)
        # Annotation formats
        menu.addSection("Annotation Formats")
        menu.addAction("COCO JSON", self._export_coco)
        menu.addAction("YOLO format", self._export_yolo)
        menu.addAction("CVAT XML 1.1", self._export_cvat)
        menu.addAction("MOT Challenge", self._export_mot)
        # Spreadsheet
        menu.addSection("Spreadsheet")
        menu.addAction("CSV per-track stats", self._export_csv_per_track)
        menu.addAction("CSV per-frame detections", self._export_csv_per_frame)
        # Visuals
        menu.addSection("Visuals")
        menu.addAction("Annotated MP4", self._export_annotated_mp4)
        menu.addAction("Per-track frame stills", self._export_frame_stills)
        # Report
        menu.addSection("Report")
        menu.addAction("Summary PDF", self._export_summary_pdf)
        menu.addAction("Generate Review Pack (zip)", self._export_review_pack)
        # Show below the button
        menu.exec(self.btn_export.mapToGlobal(self.btn_export.rect().bottomLeft()))

    def _ensure_data_for_export(self) -> dict | None:
        """Build the data payload (tracks + rois + meta) for an export.

        Uses the in-memory session state so unsaved edits are exported too.
        """
        return {
            "tracks": self.tracks,
            "rois": self.rois,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "resolution": self.resolution,
            "video_name": self.video_name,
        }

    def _export_output_dir(self, fmt: str) -> Path:
        out = self.data_manager.exports_dir / self.session_id / fmt
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _export_done(self, label: str, path: Path):
        """Toast + open the folder containing the export."""
        self.status_bar.showMessage(f"{label} exported -> {path}")
        QMessageBox.information(
            self,
            "Export complete",
            f"{label} exported to:\n{path}",
        )

    def _export_failed(self, label: str, err: Exception):
        QMessageBox.critical(self, "Export failed", f"{label} export failed:\n{err}")
        self.status_bar.showMessage(f"{label} export failed: {err}")

    def _export_coco(self):
        try:
            roi_id = self._active_roi_filter_id()
            path = self.data_manager.export_coco(self.session_id, roi_id=roi_id)
            self._export_done("COCO JSON", path)
        except Exception as e:
            self._export_failed("COCO", e)

    def _export_yolo(self):
        try:
            from cctv_yolo.training import build_yolo_dataset
            out_root = self._export_output_dir("yolo")
            stats = build_yolo_dataset(
                self.data_manager,
                output_root=out_root,
                restrict_session_ids=[self.session_id],
            )
            self._export_done("YOLO dataset", out_root)
        except Exception as e:
            self._export_failed("YOLO", e)

    def _export_cvat(self):
        try:
            from cctv_yolo.exports.cvat_writer import write_cvat_xml
            out_dir = self._export_output_dir("cvat")
            path = out_dir / f"{self.session_id}.cvat.xml"
            w, h = 0, 0
            if self.resolution and "x" in str(self.resolution):
                try:
                    w, h = (int(x) for x in str(self.resolution).split("x"))
                except ValueError:
                    pass
            write_cvat_xml(
                path,
                self._ensure_data_for_export(),
                video_name=self.video_name or self.session_id,
                width=w,
                height=h,
                roi_id=self._active_roi_filter_id(),
            )
            self._export_done("CVAT XML", path)
        except Exception as e:
            self._export_failed("CVAT", e)

    def _export_mot(self):
        try:
            from cctv_yolo.exports.mot_writer import write_mot_txt
            out_dir = self._export_output_dir("mot")
            path = out_dir / "gt.txt"
            write_mot_txt(
                path,
                self._ensure_data_for_export(),
                roi_id=self._active_roi_filter_id(),
            )
            self._export_done("MOT Challenge", path)
        except Exception as e:
            self._export_failed("MOT", e)

    def _export_csv_per_track(self):
        try:
            from cctv_yolo.exports.csv_writer import write_per_track_csv
            out_dir = self._export_output_dir("csv")
            path = out_dir / f"{self.session_id}_per_track.csv"
            write_per_track_csv(
                path,
                self._ensure_data_for_export(),
                roi_id=self._active_roi_filter_id(),
            )
            self._export_done("Per-track CSV", path)
        except Exception as e:
            self._export_failed("CSV (per-track)", e)

    def _export_csv_per_frame(self):
        try:
            from cctv_yolo.exports.csv_writer import write_per_frame_csv
            out_dir = self._export_output_dir("csv")
            path = out_dir / f"{self.session_id}_per_frame.csv"
            write_per_frame_csv(
                path,
                self._ensure_data_for_export(),
                roi_id=self._active_roi_filter_id(),
            )
            self._export_done("Per-frame CSV", path)
        except Exception as e:
            self._export_failed("CSV (per-frame)", e)

    def _export_annotated_mp4(self):
        try:
            from cctv_yolo.annotated_export import annotate_video
            out_dir = self._export_output_dir("annotated")
            path = out_dir / f"{self.session_id}_annotated.mp4"
            video_path = self.data_manager.get_video_path(self.session_id)
            if not video_path:
                raise FileNotFoundError("Video not found for session")
            annotate_video(
                video_path=video_path,
                track_data=self._ensure_data_for_export(),
                output_path=path,
                roi_id=self._active_roi_filter_id(),
            )
            self._export_done("Annotated MP4", path)
        except Exception as e:
            self._export_failed("Annotated MP4", e)

    def _export_frame_stills(self):
        try:
            count = self.data_manager.export_labeled_images(self.session_id)
            out_dir = self.data_manager.exports_dir / self.session_id / "labeled"
            self._export_done(f"Per-track frame stills ({count})", out_dir)
        except Exception as e:
            self._export_failed("Frame stills", e)

    def _export_summary_pdf(self):
        try:
            from cctv_yolo.exports.review_pack import _build_summary_pdf
            out_dir = self._export_output_dir("report")
            path = out_dir / f"{self.session_id}_summary.pdf"
            ok = _build_summary_pdf(
                path,
                self._ensure_data_for_export(),
                self.session_id,
                self.video_name or "",
            )
            if not ok:
                raise RuntimeError(
                    "reportlab is not installed. Run: pip install reportlab"
                )
            self._export_done("Summary PDF", path)
        except Exception as e:
            self._export_failed("Summary PDF", e)

    def _export_review_pack(self):
        try:
            from cctv_yolo.exports.review_pack import build_review_pack
            zip_path = build_review_pack(
                self.data_manager,
                self.session_id,
                roi_id=self._active_roi_filter_id(),
            )
            self._export_done("Review Pack (zip)", zip_path)
        except Exception as e:
            self._export_failed("Review Pack", e)

    # ------------------------------------------------------------------
    # UI refresh helpers
    # ------------------------------------------------------------------

    def _mark_unsaved(self):
        self.unsaved = True
        self.sidebar.set_unsaved(True)
        title = self.windowTitle()
        # PRD F2-1c: bullet (● ) dirty indicator at the start of the title.
        if not title.startswith("● ") and not title.startswith("* "):
            self.setWindowTitle(f"● {title}")

    def _refresh_sidebar(self):
        """Refresh the sidebar with current state."""
        self.sidebar.set_tracks(self.tracks, self.current_frame, self.selected_track_id)
        self.sidebar.set_rois(self.rois, self.tracks)
        self._apply_roi_filter()

    def _refresh_all(self):
        """Refresh both canvas and sidebar."""
        self.canvas.tracks = self.tracks
        self.canvas.rois = self.rois
        self.canvas.selected_track_id = self.selected_track_id
        self.canvas.current_frame = self.current_frame
        self.canvas.update()
        self._refresh_sidebar()
        if hasattr(self, "minimap"):
            self.minimap.set_data(
                {"tracks": self.tracks, "rois": self.rois},
                self.total_frames,
            )

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
