"""
Preprocessing tab -- video grid with model picker and processing controls.

Combines the video management from videos_tab.py with a YOLO model picker
(dropdown + Browse button) and confidence slider.
"""
import shutil
import cv2
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QProgressBar,
    QMessageBox,
    QFileDialog,
    QCheckBox,
)

from cctv_yolo.processing import ProcessingWorker, ExportWorker
from cctv_yolo.video_canvas import VideoCanvas

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
BG = "#1a1a2e"
PANEL = "#16213e"
BORDER = "#2d3a5a"
ACCENT = "#4ecca3"
TEXT = "#eeeeee"

CARD_STYLE = f"""
QFrame#videoCard {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame#videoCard:hover {{
    border: 1px solid rgba(78, 204, 163, 0.4);
    background-color: #1b2844;
}}
"""

CARD_SELECTED_STYLE = f"""
QFrame#videoCard {{
    background-color: #1b2844;
    border: 2px solid {ACCENT};
    border-radius: 8px;
}}
"""

ROI_BTN_ACTIVE = f"""
QPushButton {{
    background-color: {ACCENT};
    color: #000;
    border: none;
    border-radius: 4px;
    padding: 4px 10px;
    font-weight: bold;
    font-size: 11px;
}}
"""

STAT_CARD_STYLE = f"""
QFrame {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1b2844, stop:1 {PANEL});
    border: 1px solid {BORDER};
    border-top: 2px solid {ACCENT};
    border-radius: 8px;
    padding: 12px;
}}
"""

ACTION_BTN = f"""
QPushButton {{
    background-color: {ACCENT};
    color: #000;
    border: none;
    border-radius: 4px;
    padding: 4px 10px;
    font-weight: bold;
    font-size: 11px;
}}
QPushButton:hover {{
    background-color: #3bbb91;
}}
QPushButton:disabled {{
    background-color: {BORDER};
    color: #666;
}}
"""

SECONDARY_BTN = f"""
QPushButton {{
    background-color: transparent;
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11px;
}}
QPushButton:hover {{
    background-color: {ACCENT};
    color: #000;
}}
QPushButton:disabled {{
    border-color: {BORDER};
    color: #666;
}}
"""

CONTROLS_STYLE = f"""
QComboBox {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    min-width: 100px;
}}
QComboBox::drop-down {{
    border: none;
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: #000;
}}
"""

PROCESS_ALL_BTN = f"""
QPushButton {{
    background-color: {ACCENT};
    color: #000;
    border: none;
    border-radius: 4px;
    padding: 8px 20px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: #3bbb91;
}}
QPushButton:disabled {{
    background-color: {BORDER};
    color: #666;
}}
"""

BROWSE_BTN = f"""
QPushButton {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11px;
}}
QPushButton:hover {{
    background-color: {BORDER};
}}
"""

REFRESH_BTN_STYLE = f"""
QPushButton {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 16px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {BORDER};
}}
"""

BADGE_PROCESSED = f"""
QLabel {{
    background-color: {ACCENT};
    color: #000;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: bold;
}}
"""

BADGE_UNPROCESSED = f"""
QLabel {{
    background-color: #e74c3c;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: bold;
}}
"""

BADGE_PROCESSING = f"""
QLabel {{
    background-color: #f39c12;
    color: #000;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: bold;
}}
"""

SLIDER_STYLE = f"""
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
"""


def _numpy_to_pixmap(bgr_array, max_width=300):
    """Convert a BGR numpy array to a QPixmap, scaled to max_width."""
    if bgr_array is None:
        return None
    rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimg)
    if pixmap.width() > max_width:
        pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
    return pixmap


class PreprocessingTab(QWidget):
    """Preprocessing tab -- model picker + video grid with processing controls."""

    review_requested = Signal(str)  # session_id

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self._workers = {}  # session_id -> ProcessingWorker or ExportWorker
        self._card_widgets = {}  # session_id -> dict of widgets in the card
        self._selected_session_id = None  # currently selected video for preview
        self._current_roi = None  # ROI dict for the selected video
        self._setup_ui()
        # Restore global ROI state
        self._global_roi = self.data_manager.get_global_processing_roi()
        if self._global_roi:
            self.chk_global_roi.setChecked(True)
        self._populate_models()
        self.refresh()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # --- Header row ---
        header_row = QHBoxLayout()
        title = QLabel("Preprocessing")
        title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {TEXT};")
        header_row.addWidget(title)
        header_row.addStretch()

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setStyleSheet(REFRESH_BTN_STYLE)
        self.btn_refresh.clicked.connect(self.refresh)
        header_row.addWidget(self.btn_refresh)
        layout.addLayout(header_row)

        # --- Model picker + confidence row ---
        model_row = QHBoxLayout()
        model_row.setSpacing(10)

        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet(CONTROLS_STYLE)
        self.model_combo.setMinimumWidth(180)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_row.addWidget(self.model_combo)

        self.btn_browse_model = QPushButton("Browse...")
        self.btn_browse_model.setStyleSheet(BROWSE_BTN)
        self.btn_browse_model.clicked.connect(self._browse_model)
        model_row.addWidget(self.btn_browse_model)

        model_row.addSpacing(20)

        model_row.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 100)
        self.conf_slider.setSingleStep(5)
        self.conf_slider.setPageStep(10)
        self.conf_slider.setStyleSheet(SLIDER_STYLE)
        self.conf_slider.setFixedWidth(200)

        # Load saved confidence or default to 0.25
        saved_conf = self.data_manager.get_last_confidence()
        self.conf_slider.setValue(int(saved_conf * 100))

        self.lbl_conf_value = QLabel(f"{saved_conf:.2f}")
        self.lbl_conf_value.setStyleSheet(f"color: {ACCENT}; font-weight: bold; min-width: 35px;")
        self.conf_slider.valueChanged.connect(self._on_conf_slider_changed)
        model_row.addWidget(self.conf_slider)
        model_row.addWidget(self.lbl_conf_value)

        model_row.addSpacing(20)

        self.chk_global_roi = QCheckBox("Use Global ROI")
        self.chk_global_roi.setStyleSheet(f"color: {TEXT}; font-size: 12px;")
        self.chk_global_roi.setToolTip("Apply a single ROI to all videos during processing")
        self.chk_global_roi.toggled.connect(self._on_global_roi_toggled)
        model_row.addWidget(self.chk_global_roi)

        model_row.addStretch()
        layout.addLayout(model_row)

        # --- Stat cards row ---
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)

        self.stat_total = self._make_stat_card("Total Videos", "0")
        self.stat_processed = self._make_stat_card("Processed", "0")
        self.stat_unprocessed = self._make_stat_card("Unprocessed", "0")

        stats_row.addWidget(self.stat_total["frame"])
        stats_row.addWidget(self.stat_processed["frame"])
        stats_row.addWidget(self.stat_unprocessed["frame"])
        layout.addLayout(stats_row)

        # --- Process All button row ---
        proc_row = QHBoxLayout()
        proc_row.addStretch()

        self.btn_process_all = QPushButton("Process All Unprocessed")
        self.btn_process_all.setStyleSheet(PROCESS_ALL_BTN)
        self.btn_process_all.clicked.connect(self._process_all_unprocessed)
        proc_row.addWidget(self.btn_process_all)

        layout.addLayout(proc_row)

        # --- Scrollable video grid ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background-color: {BG}; border: none; }}")

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(12)

        scroll.setWidget(self.grid_widget)

        # --- Main content: grid on top, ROI preview below ---
        # Use a splitter-like layout: grid gets more space, preview is collapsible
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        content_layout.addWidget(scroll, stretch=1)

        # --- ROI Preview Panel (hidden until a video is selected) ---
        self.roi_panel = QFrame()
        self.roi_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {PANEL};
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
        """)
        self.roi_panel.setVisible(False)

        roi_panel_layout = QVBoxLayout(self.roi_panel)
        roi_panel_layout.setContentsMargins(12, 8, 12, 8)
        roi_panel_layout.setSpacing(6)

        # Panel header row
        roi_header = QHBoxLayout()
        self.roi_title = QLabel("ROI Preview")
        self.roi_title.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {TEXT}; border: none;")
        roi_header.addWidget(self.roi_title)
        roi_header.addStretch()

        self.btn_roi_rect = QPushButton("Draw Rect ROI")
        self.btn_roi_rect.setStyleSheet(SECONDARY_BTN)
        self.btn_roi_rect.setCheckable(True)
        self.btn_roi_rect.clicked.connect(self._on_roi_rect_mode)
        roi_header.addWidget(self.btn_roi_rect)

        self.btn_roi_poly = QPushButton("Draw Polygon ROI")
        self.btn_roi_poly.setStyleSheet(SECONDARY_BTN)
        self.btn_roi_poly.setCheckable(True)
        self.btn_roi_poly.clicked.connect(self._on_roi_poly_mode)
        roi_header.addWidget(self.btn_roi_poly)

        self.btn_roi_clear = QPushButton("Clear ROI")
        self.btn_roi_clear.setStyleSheet(SECONDARY_BTN)
        self.btn_roi_clear.clicked.connect(self._on_roi_clear)
        roi_header.addWidget(self.btn_roi_clear)

        self.btn_roi_close = QPushButton("Close Preview")
        self.btn_roi_close.setStyleSheet(BROWSE_BTN)
        self.btn_roi_close.clicked.connect(self._close_preview)
        roi_header.addWidget(self.btn_roi_close)

        roi_panel_layout.addLayout(roi_header)

        # VideoCanvas for preview
        self.preview_canvas = VideoCanvas()
        self.preview_canvas.setMinimumHeight(240)
        self.preview_canvas.setMaximumHeight(360)
        self.preview_canvas.roi_rect_drawn.connect(self._on_roi_rect_drawn)
        self.preview_canvas.roi_polygon_drawn.connect(self._on_roi_polygon_drawn)
        roi_panel_layout.addWidget(self.preview_canvas)

        # ROI status label
        self.roi_status = QLabel("No ROI set. Detections from full frame will be used.")
        self.roi_status.setStyleSheet("color: #999; font-size: 11px; border: none;")
        roi_panel_layout.addWidget(self.roi_status)

        content_layout.addWidget(self.roi_panel)

        layout.addWidget(content_widget, stretch=1)

    def _make_stat_card(self, label_text, value_text):
        """Create a stat card widget and return dict with frame, value_label."""
        frame = QFrame()
        frame.setStyleSheet(STAT_CARD_STYLE)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        frame.setFixedHeight(95)

        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(16, 12, 16, 12)

        lbl = QLabel(label_text.upper())
        lbl.setStyleSheet("color: #8899aa; font-size: 11px; letter-spacing: 1px; border: none;")
        lbl.setAlignment(Qt.AlignLeft)

        val = QLabel(value_text)
        val.setStyleSheet(f"color: {ACCENT}; font-size: 36px; font-weight: bold; border: none;")
        val.setAlignment(Qt.AlignLeft)

        vbox.addWidget(lbl)
        vbox.addWidget(val)

        return {"frame": frame, "value_label": val}

    # ------------------------------------------------------------------
    # Model picker
    # ------------------------------------------------------------------

    def _populate_models(self):
        """Fill the model combo box with available models."""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        # Built-in model names (these will be auto-downloaded by ultralytics)
        builtin = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]

        # Scan the models dir for custom .pt files
        custom_models = self.data_manager.list_models()

        # Combine: builtin first, then any custom ones not already listed
        all_models = list(builtin)
        for m in custom_models:
            if m not in all_models:
                all_models.append(m)

        self.model_combo.addItems(all_models)

        # Restore last-used model
        last_model = self.data_manager.get_last_model()
        if last_model and last_model in all_models:
            self.model_combo.setCurrentText(last_model)
        else:
            self.model_combo.setCurrentText("yolov8m.pt")

        self.model_combo.blockSignals(False)

    def _on_model_changed(self, model_name):
        """Save the selected model name to config."""
        if model_name:
            self.data_manager.set_last_model(model_name)

    def _browse_model(self):
        """Open a file dialog to select a custom .pt model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "PyTorch Models (*.pt);;All Files (*)",
        )
        if not file_path:
            return

        src = Path(file_path)
        dest = self.data_manager.models_dir / src.name

        if dest.exists() and dest != src:
            reply = QMessageBox.question(
                self,
                "Model Exists",
                f"'{src.name}' already exists in the models directory. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        # Copy to models dir (unless it's already there)
        if dest != src:
            try:
                shutil.copy2(str(src), str(dest))
            except Exception as e:
                QMessageBox.critical(self, "Copy Error", f"Failed to copy model:\n{e}")
                return

        # Refresh the model list and select the new model
        self._populate_models()
        self.model_combo.setCurrentText(src.name)

    def _on_conf_slider_changed(self, value):
        """Update the confidence label and save to config."""
        conf = value / 100.0
        self.lbl_conf_value.setText(f"{conf:.2f}")
        self.data_manager.set_last_confidence(conf)

    # ------------------------------------------------------------------
    # ROI preview panel
    # ------------------------------------------------------------------

    def _select_video(self, session_id):
        """Select a video for ROI preview."""
        # Deselect previous card
        if self._selected_session_id and self._selected_session_id in self._card_widgets:
            prev = self._card_widgets[self._selected_session_id]
            prev["frame"].setStyleSheet(CARD_STYLE)

        self._selected_session_id = session_id

        # Highlight the selected card
        if session_id in self._card_widgets:
            self._card_widgets[session_id]["frame"].setStyleSheet(CARD_SELECTED_STYLE)

        # Open video in preview canvas
        video_path = self.data_manager.get_video_path(session_id)
        if not video_path or not video_path.exists():
            return

        self.preview_canvas.open_video(str(video_path))
        self.preview_canvas.set_frame(0)
        self.preview_canvas.drawing_mode = "select"

        # Load saved ROI for this session
        self._current_roi = self.data_manager.get_processing_roi(session_id)
        self._apply_roi_to_canvas()

        # Update ROI status label
        if self._current_roi:
            roi_type = self._current_roi["type"]
            n_pts = len(self._current_roi["points"])
            self.roi_status.setText(
                f"ROI set ({roi_type}, {n_pts} points). "
                f"Only detections inside the ROI will be kept during processing."
            )
        else:
            self.roi_status.setText("No ROI set. Detections from full frame will be used.")

        # Update title and show panel
        display_name = video_path.name
        self.roi_title.setText(f"ROI Preview — {display_name}")
        self.roi_panel.setVisible(True)

        # Reset button states
        self.btn_roi_rect.setChecked(False)
        self.btn_roi_poly.setChecked(False)
        self.btn_roi_rect.setStyleSheet(SECONDARY_BTN)
        self.btn_roi_poly.setStyleSheet(SECONDARY_BTN)

    def _close_preview(self):
        """Close the ROI preview panel."""
        self.preview_canvas.close_video()
        self.roi_panel.setVisible(False)

        # Deselect card
        if self._selected_session_id and self._selected_session_id in self._card_widgets:
            self._card_widgets[self._selected_session_id]["frame"].setStyleSheet(CARD_STYLE)
        self._selected_session_id = None
        self._current_roi = None

    def _on_roi_rect_mode(self, checked):
        """Toggle rect ROI drawing mode."""
        if checked:
            self.btn_roi_poly.setChecked(False)
            self.preview_canvas.drawing_mode = "roi_rect"
            self.preview_canvas.set_cursor_for_mode()
            self.btn_roi_rect.setStyleSheet(ROI_BTN_ACTIVE)
            self.btn_roi_poly.setStyleSheet(SECONDARY_BTN)
        else:
            self.preview_canvas.drawing_mode = "select"
            self.preview_canvas.set_cursor_for_mode()
            self.btn_roi_rect.setStyleSheet(SECONDARY_BTN)

    def _on_roi_poly_mode(self, checked):
        """Toggle polygon ROI drawing mode."""
        if checked:
            self.btn_roi_rect.setChecked(False)
            self.preview_canvas.drawing_mode = "roi_polygon"
            self.preview_canvas.set_cursor_for_mode()
            self.btn_roi_poly.setStyleSheet(ROI_BTN_ACTIVE)
            self.btn_roi_rect.setStyleSheet(SECONDARY_BTN)
        else:
            self.preview_canvas.drawing_mode = "select"
            self.preview_canvas.set_cursor_for_mode()
            self.btn_roi_poly.setStyleSheet(SECONDARY_BTN)

    def _on_roi_rect_drawn(self, p1, p2):
        """Handle rect ROI drawn on the preview canvas."""
        self._current_roi = {
            "type": "rect",
            "points": [p1, p2],
        }
        self._save_and_display_roi()

        # Exit drawing mode
        self.btn_roi_rect.setChecked(False)
        self.preview_canvas.drawing_mode = "select"
        self.preview_canvas.set_cursor_for_mode()
        self.btn_roi_rect.setStyleSheet(SECONDARY_BTN)

    def _on_roi_polygon_drawn(self, points):
        """Handle polygon ROI drawn on the preview canvas."""
        if len(points) < 3:
            return
        self._current_roi = {
            "type": "polygon",
            "points": points,
        }
        self._save_and_display_roi()

        # Exit drawing mode
        self.btn_roi_poly.setChecked(False)
        self.preview_canvas.drawing_mode = "select"
        self.preview_canvas.set_cursor_for_mode()
        self.btn_roi_poly.setStyleSheet(SECONDARY_BTN)

    def _on_roi_clear(self):
        """Clear the ROI for the selected video or global ROI."""
        self._current_roi = None
        if self.chk_global_roi.isChecked():
            self._global_roi = None
            self.data_manager.set_global_processing_roi(None)
            self.roi_status.setText("Global ROI cleared. Detections from full frame will be used.")
        elif self._selected_session_id:
            self.data_manager.set_processing_roi(self._selected_session_id, None)
            self.roi_status.setText("No ROI set. Detections from full frame will be used.")
        self._apply_roi_to_canvas()

    def _save_and_display_roi(self):
        """Save the current ROI and update the canvas overlay."""
        if self._current_roi:
            if self.chk_global_roi.isChecked():
                # Save as global ROI
                self._global_roi = self._current_roi
                self.data_manager.set_global_processing_roi(self._current_roi)
            elif self._selected_session_id:
                # Save as per-video ROI
                self.data_manager.set_processing_roi(
                    self._selected_session_id, self._current_roi
                )
        self._apply_roi_to_canvas()

        # Update status label
        if self._current_roi:
            roi_type = self._current_roi["type"]
            n_pts = len(self._current_roi["points"])
            scope = "all videos (global)" if self.chk_global_roi.isChecked() else "this video"
            self.roi_status.setText(
                f"ROI set ({roi_type}, {n_pts} points). "
                f"Only detections inside the ROI will be kept for {scope}."
            )

    def _apply_roi_to_canvas(self):
        """Update the preview canvas ROI overlay."""
        if self._current_roi:
            roi_display = dict(self._current_roi)
            roi_display["name"] = "Processing ROI"
            roi_display["color"] = "#ff6b6b"
            self.preview_canvas.rois = [roi_display]
        else:
            self.preview_canvas.rois = []
        self.preview_canvas.update()

    def _on_global_roi_toggled(self, checked):
        """Toggle global ROI usage."""
        if checked:
            if self._global_roi:
                self.roi_status.setText("Global ROI active. This ROI will be used for all videos.")
            else:
                self.roi_status.setText("No global ROI set. Draw a ROI on any video preview to set it as global.")
        else:
            if self._selected_session_id:
                roi = self.data_manager.get_processing_roi(self._selected_session_id)
                if roi:
                    self.roi_status.setText("Per-video ROI active.")
                else:
                    self.roi_status.setText("No ROI set. Detections from full frame will be used.")

    # ------------------------------------------------------------------
    # Refresh / populate
    # ------------------------------------------------------------------

    def refresh(self):
        """Reload video data and refresh the grid."""
        videos = self.data_manager.get_videos()

        # Update stats
        total = len(videos)
        processed = sum(1 for v in videos if v["status"] == "processed")
        unprocessed = total - processed

        self.stat_total["value_label"].setText(str(total))
        self.stat_processed["value_label"].setText(str(processed))
        self.stat_unprocessed["value_label"].setText(str(unprocessed))

        # Clear existing grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._card_widgets.clear()

        if not videos:
            empty = QLabel("No videos found. Add videos to the videos directory.")
            empty.setStyleSheet("color: #999; font-size: 14px; padding: 24px;")
            empty.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(empty, 0, 0, 1, 3)
            return

        # Build video cards in a grid (3 columns)
        cols = 3
        for i, video in enumerate(videos):
            row = i // cols
            col = i % cols
            card = self._make_video_card(video)
            self.grid_layout.addWidget(card, row, col)

        # Fill remaining space
        total_rows = (len(videos) + cols - 1) // cols
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid_layout.addWidget(spacer, total_rows, 0, 1, cols)

    def _make_video_card(self, video):
        """Build a single video card widget."""
        frame = QFrame()
        frame.setObjectName("videoCard")
        frame.setStyleSheet(CARD_STYLE)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        frame.setMinimumWidth(280)

        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(0, 0, 0, 10)
        vbox.setSpacing(6)

        session_id = video["session_id"]

        # --- Thumbnail ---
        thumb_label = QLabel()
        thumb_label.setFixedHeight(160)
        thumb_label.setAlignment(Qt.AlignCenter)
        thumb_label.setStyleSheet(
            "background-color: #0d1117; border-radius: 8px 8px 0 0; border: none;"
        )

        video_path = self.data_manager.get_video_path(session_id)
        if video_path and video_path.exists():
            thumb_bgr = self.data_manager.get_video_thumbnail_frame(video_path)
            pixmap = _numpy_to_pixmap(thumb_bgr, max_width=320)
            if pixmap:
                thumb_label.setPixmap(
                    pixmap.scaled(
                        320, 160,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
            else:
                thumb_label.setText("No Preview")
                thumb_label.setStyleSheet(
                    "background-color: #0d1117; border-radius: 8px 8px 0 0;"
                    " border: none; color: #555; font-size: 14px;"
                )
        else:
            thumb_label.setText("No Video")
            thumb_label.setStyleSheet(
                "background-color: #0d1117; border-radius: 8px 8px 0 0;"
                " border: none; color: #444; font-size: 14px; font-style: italic;"
            )
        vbox.addWidget(thumb_label)

        # --- Info section ---
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(12, 4, 12, 0)
        info_layout.setSpacing(4)

        # Name + badge row
        name_row = QHBoxLayout()
        name_lbl = QLabel(video.get("display_name", video["name"]))
        name_lbl.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {TEXT};")
        name_lbl.setWordWrap(True)
        name_row.addWidget(name_lbl, stretch=1)

        status = video.get("status", "unprocessed")
        badge = QLabel(status.capitalize())
        if status == "processed":
            badge.setStyleSheet(BADGE_PROCESSED)
        elif status == "processing":
            badge.setStyleSheet(BADGE_PROCESSING)
        else:
            badge.setStyleSheet(BADGE_UNPROCESSED)
        name_row.addWidget(badge)
        info_layout.addLayout(name_row)

        # Folder (if any)
        folder = video.get("folder", "")
        if folder:
            folder_lbl = QLabel(f"Folder: {folder}")
            folder_lbl.setStyleSheet("font-size: 11px; color: #777;")
            info_layout.addWidget(folder_lbl)

        # Metadata line
        meta_parts = []
        if video.get("size_mb"):
            meta_parts.append(f"{video['size_mb']} MB")
        if video.get("resolution") and video["resolution"] != "Unknown":
            meta_parts.append(video["resolution"])
        if video.get("duration"):
            mins = int(video["duration"] // 60)
            secs = int(video["duration"] % 60)
            meta_parts.append(f"{mins}m{secs:02d}s")
        if video.get("total_frames"):
            meta_parts.append(f"{video['total_frames']} frames")
        meta_lbl = QLabel("  |  ".join(meta_parts))
        meta_lbl.setStyleSheet("font-size: 11px; color: #999;")
        info_layout.addWidget(meta_lbl)

        # Track count (if processed)
        if status == "processed":
            track_lbl = QLabel(f"{video.get('track_count', 0)} tracks")
            track_lbl.setStyleSheet(f"font-size: 11px; color: {ACCENT};")
            info_layout.addWidget(track_lbl)

        vbox.addWidget(info_widget)

        # --- Progress bar (hidden by default) ---
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setFixedHeight(16)
        progress.setVisible(False)
        progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: #0d1525;
                border: 1px solid {BORDER};
                border-radius: 8px;
                text-align: center;
                color: {TEXT};
                font-size: 10px;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2fa87e, stop:0.5 {ACCENT}, stop:1 #6fe8c0);
                border-radius: 7px;
            }}
        """)
        vbox.addWidget(progress)

        # --- Action buttons ---
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(12, 0, 12, 0)
        btn_row.setSpacing(6)

        # Preview button (always shown)
        btn_preview = QPushButton("Preview")
        btn_preview.setStyleSheet(SECONDARY_BTN)
        btn_preview.clicked.connect(
            lambda checked=False, sid=session_id: self._select_video(sid)
        )
        btn_row.addWidget(btn_preview)

        if status == "unprocessed":
            btn_process = QPushButton("Process")
            btn_process.setStyleSheet(ACTION_BTN)
            btn_process.clicked.connect(
                lambda checked=False, sid=session_id: self._process_video(sid)
            )
            btn_row.addWidget(btn_process)
        elif status == "processing":
            btn_cancel = QPushButton("Processing...")
            btn_cancel.setStyleSheet(ACTION_BTN)
            btn_cancel.setEnabled(False)
            btn_row.addWidget(btn_cancel)
        elif status == "processed":
            btn_review = QPushButton("Review")
            btn_review.setStyleSheet(ACTION_BTN)
            btn_review.clicked.connect(
                lambda checked=False, sid=session_id: self.review_requested.emit(sid)
            )
            btn_row.addWidget(btn_review)

            btn_export = QPushButton("Export")
            btn_export.setStyleSheet(SECONDARY_BTN)
            btn_export.clicked.connect(
                lambda checked=False, sid=session_id: self._export_video(sid)
            )
            btn_row.addWidget(btn_export)

            btn_reprocess = QPushButton("Reprocess")
            btn_reprocess.setStyleSheet(SECONDARY_BTN)
            btn_reprocess.clicked.connect(
                lambda checked=False, sid=session_id: self._process_video(sid)
            )
            btn_row.addWidget(btn_reprocess)

        btn_row.addStretch()
        vbox.addLayout(btn_row)

        # Store references for progress updates
        self._card_widgets[session_id] = {
            "frame": frame,
            "progress": progress,
            "badge": badge,
        }

        return frame

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _process_video(self, session_id):
        """Start processing a single video."""
        if session_id in self._workers:
            return  # already running

        video_path = self.data_manager.get_video_path(session_id)
        if not video_path or not video_path.exists():
            QMessageBox.warning(self, "Error", f"Video not found for session: {session_id}")
            return

        model = self.model_combo.currentText()
        conf = self.conf_slider.value() / 100.0

        # Load ROI: global ROI takes precedence when enabled
        if self.chk_global_roi.isChecked() and self._global_roi:
            processing_roi = self._global_roi
        else:
            processing_roi = self.data_manager.get_processing_roi(session_id)

        self.data_manager.set_processing_status(session_id, "processing", progress=0)

        worker = ProcessingWorker(
            video_path=str(video_path),
            tracks_dir=str(self.data_manager.tracks_dir),
            model=model,
            conf=conf,
            session_id=session_id,
            models_dir=str(self.data_manager.models_dir),
            processing_roi=processing_roi,
        )
        worker.progress.connect(self._on_processing_progress)
        worker.finished.connect(self._on_processing_finished)
        worker.error.connect(self._on_processing_error)

        self._workers[session_id] = worker
        worker.start()

        # Show progress bar if card exists
        if session_id in self._card_widgets:
            self._card_widgets[session_id]["progress"].setVisible(True)
            self._card_widgets[session_id]["progress"].setValue(0)
            self._card_widgets[session_id]["badge"].setText("Processing")
            self._card_widgets[session_id]["badge"].setStyleSheet(BADGE_PROCESSING)

    def _on_processing_progress(self, session_id, percent):
        if session_id in self._card_widgets:
            self._card_widgets[session_id]["progress"].setValue(percent)

    def _on_processing_finished(self, session_id):
        self.data_manager.clear_processing_job(session_id)
        worker = self._workers.pop(session_id, None)
        if worker:
            worker.wait()
        self.refresh()

    def _on_processing_error(self, session_id, error_msg):
        self.data_manager.clear_processing_job(session_id)
        worker = self._workers.pop(session_id, None)
        if worker:
            worker.wait()
        self.refresh()
        display_msg = error_msg
        if len(display_msg) > 800:
            lines = display_msg.split('\n')
            first_line = lines[0]
            tail = '\n'.join(lines[-10:])
            display_msg = f"{first_line}\n\n... (truncated) ...\n\n{tail}"
        QMessageBox.critical(self, "Processing Error", f"Error processing {session_id}:\n\n{display_msg}")

    def _process_all_unprocessed(self):
        """Start processing all unprocessed videos."""
        videos = self.data_manager.get_videos()
        unprocessed = [v for v in videos if v["status"] == "unprocessed"]
        if not unprocessed:
            QMessageBox.information(self, "Nothing to Process", "All videos are already processed.")
            return

        reply = QMessageBox.question(
            self,
            "Process All",
            f"Process {len(unprocessed)} unprocessed video(s)?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        for video in unprocessed:
            self._process_video(video["session_id"])

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_video(self, session_id):
        """Start exporting labeled images for a session."""
        if session_id in self._workers:
            QMessageBox.information(self, "Busy", f"Session {session_id} is already being processed.")
            return

        self.data_manager.set_export_status(session_id, "exporting", progress=0)

        worker = ExportWorker(
            data_manager=self.data_manager,
            session_id=session_id,
            sample_rate=1,
        )
        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_finished)
        worker.error.connect(self._on_export_error)

        self._workers[session_id] = worker
        worker.start()

        if session_id in self._card_widgets:
            self._card_widgets[session_id]["progress"].setVisible(True)
            self._card_widgets[session_id]["progress"].setValue(0)

    def _on_export_progress(self, session_id, percent):
        if session_id in self._card_widgets:
            self._card_widgets[session_id]["progress"].setValue(percent)

    def _on_export_finished(self, session_id, count):
        self._workers.pop(session_id, None)
        if session_id in self._card_widgets:
            self._card_widgets[session_id]["progress"].setVisible(False)
        QMessageBox.information(self, "Export Complete", f"Exported {count} images for {session_id}.")
        self.refresh()

    def _on_export_error(self, session_id, error_msg):
        self._workers.pop(session_id, None)
        if session_id in self._card_widgets:
            self._card_widgets[session_id]["progress"].setVisible(False)
        QMessageBox.critical(self, "Export Error", f"Error exporting {session_id}:\n{error_msg}")
        self.refresh()
