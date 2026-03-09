"""
Performance tab -- traffic counts and detection statistics.

Shows a session selector dropdown, total vehicle counts broken down by
vehicle type, per-ROI counts, and a CSV export button.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QFileDialog,
    QMessageBox,
    QGridLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
BG = "#1a1a2e"
PANEL = "#16213e"
BORDER = "#2d3a5a"
ACCENT = "#4ecca3"
TEXT = "#eeeeee"

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

CONTROLS_STYLE = f"""
QComboBox {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 10px;
    min-width: 250px;
    font-size: 13px;
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

EXPORT_BTN = f"""
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

TABLE_STYLE = f"""
QTableWidget {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    gridline-color: {BORDER};
    font-size: 13px;
}}
QTableWidget::item {{
    padding: 8px 12px;
    border-bottom: 1px solid {BORDER};
}}
QTableWidget::item:selected {{
    background-color: rgba(78, 204, 163, 0.2);
    color: {TEXT};
}}
QHeaderView::section {{
    background-color: #0d1525;
    color: {ACCENT};
    border: none;
    border-bottom: 2px solid {ACCENT};
    padding: 8px 12px;
    font-weight: bold;
    font-size: 12px;
}}
QTableWidget QTableCornerButton::section {{
    background-color: #0d1525;
    border: none;
}}
"""

SECTION_LABEL = f"""
QLabel {{
    color: {TEXT};
    font-size: 16px;
    font-weight: bold;
    padding: 8px 0 4px 0;
}}
"""

# Vehicle type display colors for stat cards
VEHICLE_COLORS = {
    "car": "#4ecca3",
    "truck": "#ff9f43",
    "bus": "#54a0ff",
    "motorcycle": "#feca57",
    "bicycle": "#ff6b6b",
}


class PerformanceTab(QWidget):
    """Performance tab -- traffic counts and stats for processed sessions."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self._current_stats = {}
        self._setup_ui()
        self._populate_sessions()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # --- Header row ---
        header_row = QHBoxLayout()
        title = QLabel("Performance")
        title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {TEXT};")
        header_row.addWidget(title)
        header_row.addStretch()

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setStyleSheet(REFRESH_BTN_STYLE)
        self.btn_refresh.clicked.connect(self.refresh)
        header_row.addWidget(self.btn_refresh)
        layout.addLayout(header_row)

        # --- Session selector row ---
        selector_row = QHBoxLayout()
        selector_row.setSpacing(10)

        selector_row.addWidget(QLabel("Session:"))
        self.session_combo = QComboBox()
        self.session_combo.setStyleSheet(CONTROLS_STYLE)
        self.session_combo.currentIndexChanged.connect(self._on_session_changed)
        selector_row.addWidget(self.session_combo)

        selector_row.addStretch()

        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.setStyleSheet(EXPORT_BTN)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self._export_csv)
        selector_row.addWidget(self.btn_export_csv)

        layout.addLayout(selector_row)

        # --- Scrollable content area ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background-color: {BG}; border: none; }}")

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(16)

        scroll.setWidget(self.content_widget)
        layout.addWidget(scroll, stretch=1)

        # Placeholder message
        self.lbl_placeholder = QLabel("Select a session to view performance statistics.")
        self.lbl_placeholder.setStyleSheet("color: #999; font-size: 14px; padding: 24px;")
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(self.lbl_placeholder)
        self.content_layout.addStretch()

    def _make_stat_card(self, label_text, value_text, color=None):
        """Create a stat card widget and return dict with frame, value_label."""
        accent = color or ACCENT
        frame = QFrame()
        style = f"""
        QFrame {{
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1b2844, stop:1 {PANEL});
            border: 1px solid {BORDER};
            border-top: 2px solid {accent};
            border-radius: 8px;
            padding: 12px;
        }}
        """
        frame.setStyleSheet(style)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        frame.setFixedHeight(95)

        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(16, 12, 16, 12)

        lbl = QLabel(label_text.upper())
        lbl.setStyleSheet("color: #8899aa; font-size: 11px; letter-spacing: 1px; border: none;")
        lbl.setAlignment(Qt.AlignLeft)

        val = QLabel(value_text)
        val.setStyleSheet(f"color: {accent}; font-size: 36px; font-weight: bold; border: none;")
        val.setAlignment(Qt.AlignLeft)

        vbox.addWidget(lbl)
        vbox.addWidget(val)

        return {"frame": frame, "value_label": val}

    # ------------------------------------------------------------------
    # Session selector
    # ------------------------------------------------------------------

    def _populate_sessions(self):
        """Fill the session combo box with sessions that have track data."""
        self.session_combo.blockSignals(True)
        self.session_combo.clear()

        self.session_combo.addItem("-- Select Session --", "")

        sessions = self.data_manager.get_sessions()
        for s in sessions:
            display = s.get("video_name", s.get("id", "Unknown"))
            track_count = s.get("track_count", 0)
            label = f"{display}  ({track_count} tracks)"
            self.session_combo.addItem(label, s.get("id", ""))

        self.session_combo.blockSignals(False)

    def _on_session_changed(self, index):
        """Load stats for the selected session."""
        session_id = self.session_combo.currentData()
        if not session_id:
            self._show_placeholder()
            self.btn_export_csv.setEnabled(False)
            return
        self._load_stats(session_id)

    # ------------------------------------------------------------------
    # Stats loading
    # ------------------------------------------------------------------

    def _load_stats(self, session_id):
        """Load and compute statistics for the given session."""
        data = self.data_manager.load_session_data(session_id)
        if data is None:
            self._show_placeholder("No data found for this session.")
            self.btn_export_csv.setEnabled(False)
            return

        tracks = data.get("tracks", [])
        rois = data.get("rois", [])

        # Compute per-class counts
        class_counts = defaultdict(int)
        total_tracks = len(tracks)
        for track in tracks:
            cls = track.get("class", "vehicle")
            class_counts[cls] += 1

        # Compute per-ROI counts (which tracks pass through each ROI)
        roi_counts = []
        for roi in rois:
            roi_name = roi.get("name", "Unknown")
            roi_type = roi.get("type", "rect")
            points = roi.get("points", [])
            count = self._count_tracks_in_roi(tracks, roi_type, points)
            roi_counts.append({
                "name": roi_name,
                "type": roi_type,
                "count": count,
            })

        self._current_stats = {
            "session_id": session_id,
            "total_tracks": total_tracks,
            "class_counts": dict(class_counts),
            "roi_counts": roi_counts,
        }

        self._render_stats()
        self.btn_export_csv.setEnabled(True)

    def _count_tracks_in_roi(self, tracks, roi_type, points):
        """Count how many tracks have at least one detection inside the ROI."""
        if not points:
            return 0

        count = 0
        for track in tracks:
            for fd in track.get("frames", []):
                bbox = fd.get("bbox", [0, 0, 0, 0])
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                if self._point_in_roi(cx, cy, roi_type, points):
                    count += 1
                    break  # count each track only once
        return count

    def _point_in_roi(self, x, y, roi_type, points):
        """Check if a point (x, y) is inside the given ROI."""
        if roi_type == "rect" and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
        elif roi_type == "polygon" and len(points) >= 3:
            # Ray-casting algorithm
            n = len(points)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = points[i]
                xj, yj = points[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside
        return False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _clear_content(self):
        """Remove all widgets from the content layout."""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_placeholder(self, message=None):
        """Show a placeholder message in the content area."""
        self._clear_content()
        self._current_stats = {}
        lbl = QLabel(message or "Select a session to view performance statistics.")
        lbl.setStyleSheet("color: #999; font-size: 14px; padding: 24px;")
        lbl.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(lbl)
        self.content_layout.addStretch()

    def _render_stats(self):
        """Render the computed stats into the content area."""
        self._clear_content()

        stats = self._current_stats
        class_counts = stats.get("class_counts", {})
        roi_counts = stats.get("roi_counts", [])

        # --- Total vehicles stat card ---
        total_section = QLabel("Detection Summary")
        total_section.setStyleSheet(SECTION_LABEL)
        self.content_layout.addWidget(total_section)

        # Stat cards row: total + per vehicle type
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)

        total_card = self._make_stat_card("Total Vehicles", str(stats.get("total_tracks", 0)))
        cards_row.addWidget(total_card["frame"])

        # Add per-class cards
        vehicle_order = ["car", "truck", "bus", "motorcycle", "bicycle"]
        for cls in vehicle_order:
            count = class_counts.get(cls, 0)
            color = VEHICLE_COLORS.get(cls, ACCENT)
            card = self._make_stat_card(cls.capitalize(), str(count), color=color)
            cards_row.addWidget(card["frame"])

        # Add any non-standard classes
        for cls, count in sorted(class_counts.items()):
            if cls not in vehicle_order:
                card = self._make_stat_card(cls.capitalize(), str(count))
                cards_row.addWidget(card["frame"])

        cards_container = QWidget()
        cards_container.setLayout(cards_row)
        self.content_layout.addWidget(cards_container)

        # --- Vehicle type breakdown table ---
        type_section = QLabel("Vehicle Type Breakdown")
        type_section.setStyleSheet(SECTION_LABEL)
        self.content_layout.addWidget(type_section)

        type_table = QTableWidget()
        type_table.setStyleSheet(TABLE_STYLE)
        type_table.setColumnCount(3)
        type_table.setHorizontalHeaderLabels(["Vehicle Type", "Count", "Percentage"])
        type_table.horizontalHeader().setStretchLastSection(True)
        type_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        type_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        type_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        type_table.verticalHeader().setVisible(False)
        type_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        type_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        total = stats.get("total_tracks", 0) or 1  # avoid division by zero

        # Combine standard + extra classes
        all_classes = []
        for cls in vehicle_order:
            if cls in class_counts:
                all_classes.append((cls, class_counts[cls]))
        for cls in sorted(class_counts.keys()):
            if cls not in vehicle_order:
                all_classes.append((cls, class_counts[cls]))

        type_table.setRowCount(len(all_classes))
        for row, (cls, count) in enumerate(all_classes):
            pct = (count / total) * 100.0
            type_item = QTableWidgetItem(cls.capitalize())
            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pct_item = QTableWidgetItem(f"{pct:.1f}%")
            pct_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            type_table.setItem(row, 0, type_item)
            type_table.setItem(row, 1, count_item)
            type_table.setItem(row, 2, pct_item)

        type_table.setFixedHeight(max(50, 30 + len(all_classes) * 35))
        self.content_layout.addWidget(type_table)

        # --- Per-ROI counts ---
        if roi_counts:
            roi_section = QLabel("ROI Counts")
            roi_section.setStyleSheet(SECTION_LABEL)
            self.content_layout.addWidget(roi_section)

            roi_table = QTableWidget()
            roi_table.setStyleSheet(TABLE_STYLE)
            roi_table.setColumnCount(3)
            roi_table.setHorizontalHeaderLabels(["ROI Name", "Type", "Vehicles"])
            roi_table.horizontalHeader().setStretchLastSection(True)
            roi_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            roi_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            roi_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            roi_table.verticalHeader().setVisible(False)
            roi_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            roi_table.setSelectionBehavior(QAbstractItemView.SelectRows)

            roi_table.setRowCount(len(roi_counts))
            for row, roi in enumerate(roi_counts):
                name_item = QTableWidgetItem(roi["name"])
                type_item = QTableWidgetItem(roi["type"].capitalize())
                count_item = QTableWidgetItem(str(roi["count"]))
                count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                roi_table.setItem(row, 0, name_item)
                roi_table.setItem(row, 1, type_item)
                roi_table.setItem(row, 2, count_item)

            roi_table.setFixedHeight(max(50, 30 + len(roi_counts) * 35))
            self.content_layout.addWidget(roi_table)
        else:
            no_roi_lbl = QLabel("No ROIs defined for this session. "
                                "Draw ROIs in the Review window to see per-region counts.")
            no_roi_lbl.setStyleSheet("color: #777; font-size: 12px; padding: 8px;")
            no_roi_lbl.setWordWrap(True)
            self.content_layout.addWidget(no_roi_lbl)

        # Bottom spacer
        self.content_layout.addStretch()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self):
        """Refresh the session list and re-render current stats."""
        current_id = self.session_combo.currentData()
        self._populate_sessions()

        # Try to restore the previously selected session
        if current_id:
            for i in range(self.session_combo.count()):
                if self.session_combo.itemData(i) == current_id:
                    self.session_combo.setCurrentIndex(i)
                    return

        # If no match, show placeholder
        self._show_placeholder()
        self.btn_export_csv.setEnabled(False)

    # ------------------------------------------------------------------
    # CSV Export
    # ------------------------------------------------------------------

    def _export_csv(self):
        """Export the current stats to a CSV file."""
        if not self._current_stats:
            return

        session_id = self._current_stats.get("session_id", "session")
        default_name = f"{session_id}_stats.csv"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Stats as CSV",
            str(self.data_manager.exports_dir / default_name),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(["CCTV-YOLO Performance Report"])
                writer.writerow(["Session", session_id])
                writer.writerow([])

                # Vehicle type breakdown
                writer.writerow(["Vehicle Type Breakdown"])
                writer.writerow(["Type", "Count", "Percentage"])

                total = self._current_stats.get("total_tracks", 0) or 1
                class_counts = self._current_stats.get("class_counts", {})

                for cls, count in sorted(class_counts.items()):
                    pct = (count / total) * 100.0
                    writer.writerow([cls.capitalize(), count, f"{pct:.1f}%"])

                writer.writerow(["Total", self._current_stats.get("total_tracks", 0), "100.0%"])
                writer.writerow([])

                # ROI counts
                roi_counts = self._current_stats.get("roi_counts", [])
                if roi_counts:
                    writer.writerow(["ROI Counts"])
                    writer.writerow(["ROI Name", "Type", "Vehicles"])
                    for roi in roi_counts:
                        writer.writerow([roi["name"], roi["type"].capitalize(), roi["count"]])

            QMessageBox.information(self, "Export Complete", f"Stats exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{e}")
