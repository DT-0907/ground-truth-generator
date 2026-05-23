"""
Videos tab — shows all video files with grid layout, processing controls,
and action buttons per video card.
"""
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QProgressBar,
    QMessageBox,
)

from cctv_yolo.processing import ProcessingWorker, ExportWorker

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
from cctv_yolo.theme import (
    INDIGO as BG, PANEL, BORDER, PURPLE as ACCENT, OFFWHITE as TEXT,
)

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
QDoubleSpinBox {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
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


class VideosTab(QWidget):
    """Videos tab content — video grid with processing controls."""

    review_requested = Signal(str)  # session_id

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self._workers = {}  # session_id -> ProcessingWorker or ExportWorker
        self._card_widgets = {}  # session_id -> dict of widgets in the card
        self._setup_ui()
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
        title = QLabel("Videos")
        title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {TEXT};")
        header_row.addWidget(title)
        header_row.addStretch()

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setStyleSheet(REFRESH_BTN_STYLE)
        self.btn_refresh.clicked.connect(self.refresh)
        header_row.addWidget(self.btn_refresh)
        layout.addLayout(header_row)

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

        # --- Controls row ---
        controls_row = QHBoxLayout()
        controls_row.setSpacing(12)

        controls_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
        self.model_combo.setCurrentText("yolov8m.pt")
        self.model_combo.setStyleSheet(CONTROLS_STYLE)
        controls_row.addWidget(self.model_combo)

        controls_row.addWidget(QLabel("Confidence:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setStyleSheet(CONTROLS_STYLE)
        controls_row.addWidget(self.conf_spin)

        controls_row.addStretch()

        self.btn_process_all = QPushButton("Process All Unprocessed")
        self.btn_process_all.setStyleSheet(PROCESS_ALL_BTN)
        self.btn_process_all.clicked.connect(self._process_all_unprocessed)
        controls_row.addWidget(self.btn_process_all)

        layout.addLayout(controls_row)

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
        layout.addWidget(scroll, stretch=1)

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
            empty.setStyleSheet(f"color: #999; font-size: 14px; padding: 24px;")
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
            f"background-color: #0d1117; border-radius: 8px 8px 0 0; border: none;"
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
                    f"background-color: #0d1117; border-radius: 8px 8px 0 0;"
                    f" border: none; color: #555; font-size: 14px;"
                )
        else:
            thumb_label.setText("No Video")
            thumb_label.setStyleSheet(
                f"background-color: #0d1117; border-radius: 8px 8px 0 0;"
                f" border: none; color: #444; font-size: 14px; font-style: italic;"
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
        conf = self.conf_spin.value()

        self.data_manager.set_processing_status(session_id, "processing", progress=0)

        worker = ProcessingWorker(
            video_path=str(video_path),
            tracks_dir=str(self.data_manager.tracks_dir),
            model=model,
            conf=conf,
            session_id=session_id,
            models_dir=str(self.data_manager.models_dir),
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
        # Clear job entry — the track file on disk is the source of truth for "processed"
        self.data_manager.clear_processing_job(session_id)
        worker = self._workers.pop(session_id, None)
        if worker:
            worker.wait()  # ensure thread fully stops before refresh
        self.refresh()

    def _on_processing_error(self, session_id, error_msg):
        # Clear processing status so card reverts to "unprocessed" (not stuck on "processing")
        self.data_manager.clear_processing_job(session_id)
        worker = self._workers.pop(session_id, None)
        if worker:
            worker.wait()
        self.refresh()
        # Show error after refresh so the dialog doesn't block the UI update
        # Truncate very long tracebacks for the dialog but keep the key message
        display_msg = error_msg
        if len(display_msg) > 800:
            # Show the first line (exception message) and the last few lines (traceback tail)
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
