"""
Track sidebar — track list, filters, ROI panel, and action buttons.
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QScrollArea, QFrame, QSizePolicy,
                                QCheckBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor


SIDEBAR_STYLE = """
QWidget#sidebar {
    background: #16213e;
    border-left: 1px solid #2d3a5a;
}
QLabel {
    color: #eee;
}
QPushButton {
    padding: 8px 14px;
    border: none;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 500;
    background: #2d3a5a;
    color: #fff;
}
QPushButton:hover {
    background: #3d4a6a;
}
QPushButton:disabled {
    opacity: 0.5;
    background: #2d3a5a;
    color: #666;
}
QPushButton#delete_btn { background: #e74c3c; color: #fff; }
QPushButton#delete_btn:hover { background: #c0392b; }
QPushButton#class_btn { background: #3498db; color: #fff; }
QPushButton#class_btn:hover { background: #2980b9; }
QPushButton#merge_btn { background: #9b59b6; color: #fff; }
QPushButton#merge_btn:hover { background: #8e44ad; }
QPushButton#split_btn { background: #f39c12; color: #000; }
QPushButton#split_btn:hover { background: #d68910; }
QPushButton#save_btn { background: #4ecca3; color: #000; }
QPushButton#save_btn:hover { background: #3db892; }
QScrollArea { border: none; background: transparent; }
QCheckBox {
    color: #888;
    font-size: 11px;
    spacing: 4px;
}
QCheckBox:disabled {
    color: #555;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #2d3a5a;
    border-radius: 3px;
    background: transparent;
}
QCheckBox::indicator:checked {
    background: #4ecca3;
    border-color: #4ecca3;
}
QCheckBox::indicator:disabled {
    background: #1a1a2e;
    border-color: #2d3a5a;
}
"""

CLASS_COLORS = {
    "car": "#3498db",
    "truck": "#e74c3c",
    "bus": "#9b59b6",
    "motorcycle": "#f39c12",
    "bicycle": "#1abc9c",
    "unknown": "#95a5a6",
}


class TrackItem(QFrame):
    """A single track entry in the sidebar list.

    Displays the track ID, vehicle class badge, frame range, interpolation
    count, confidence percentage, and visibility status. Highlighted when
    selected, with an orange left border when the track needs review.
    """

    clicked = Signal(int)         # track_id
    double_clicked = Signal(int)  # track_id

    def __init__(self, track: dict, current_frame: int, is_selected: bool, parent=None):
        super().__init__(parent)
        self.track_id = track["track_id"]
        self.setCursor(Qt.PointingHandCursor)

        frames_list = track.get("frames", [])
        start = track.get("start_frame", frames_list[0]["frame"] if frames_list else 0)
        end = track.get("end_frame", frames_list[-1]["frame"] if frames_list else 0)
        is_visible = (current_frame >= start and current_frame <= end)
        needs_review = track.get("needs_review", False)

        bg = "#1f3a3a" if is_selected else ("#2a2a4a" if is_visible else "#1a1a2e")
        border = "2px solid #4ecca3" if is_selected else "2px solid transparent"
        left_border = "border-left: 3px solid #f39c12;" if needs_review else ""

        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border: {border};
                border-radius: 6px;
                padding: 10px 12px;
                margin-bottom: 3px;
                {left_border}
            }}
            QFrame:hover {{ background: #222244; }}
            QLabel {{ background: transparent; border: none; }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header row: track ID + class badge
        header = QHBoxLayout()
        id_label = QLabel(f"#{track['track_id']}")
        id_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #eee;")
        header.addWidget(id_label)
        header.addStretch()

        class_name = track.get("class", "unknown")
        class_color = CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"])
        class_label = QLabel(class_name.upper())
        class_label.setStyleSheet(f"""
            background: {class_color};
            color: {'#000' if class_name == 'motorcycle' else '#fff'};
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        """)
        header.addWidget(class_label)
        layout.addLayout(header)

        # Meta row: frame range, box count, interpolation, confidence, visibility
        interp_count = sum(1 for f in track.get("frames", []) if f.get("interpolated"))
        interp_text = f", {interp_count} interp" if interp_count > 0 else ""
        conf = round(track.get("avg_confidence", 0) * 100) if track.get("avg_confidence") else "?"
        visible_text = " | visible" if is_visible else ""
        meta = QLabel(
            f"F{start}-{end} "
            f"({len(track.get('frames', []))} boxes{interp_text}) | "
            f"{conf}% conf{visible_text}"
        )
        meta.setStyleSheet("font-size: 11px; color: #888;")
        meta.setWordWrap(True)
        layout.addWidget(meta)

    def mousePressEvent(self, event):
        self.clicked.emit(self.track_id)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self.track_id)


class TrackSidebar(QWidget):
    """Complete sidebar: header, filters, track list, ROI panel, action buttons.

    This widget sits on the right side of the review window. It provides:
    - A filterable list of all tracks in the current session
    - ROI region display with per-ROI vehicle counts
    - Action buttons for track operations (delete, class change, merge, split)
    - A save button for persisting corrections
    - Keyboard shortcut reference
    """

    # Signals
    track_selected = Signal(int)       # track_id
    track_double_clicked = Signal(int) # track_id
    delete_requested = Signal()
    class_change_requested = Signal()
    merge_requested = Signal()
    split_requested = Signal()
    save_requested = Signal()
    filter_changed = Signal(str)       # filter name
    back_requested = Signal()
    roi_delete_requested = Signal(int) # roi_id
    roi_rename_requested = Signal(int) # roi_id
    roi_selection_changed = Signal()   # emitted when ROI checkboxes change

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(380)
        self.setStyleSheet(SIDEBAR_STYLE)

        self._tracks = []
        self._current_frame = 0
        self._selected_track_id = None
        self._current_filter = "all"
        self._has_unsaved = False
        self._rois = []
        self._roi_track_ids = set()  # track IDs that pass through any ROI
        self._selected_roi_indices = set()  # indices of ROIs selected via checkboxes
        self._roi_checkboxes = []  # list of QCheckBox widgets for ROI items

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Header ---
        header = QWidget()
        header.setStyleSheet("background: #16213e; border-bottom: 1px solid #2d3a5a; padding: 15px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(15, 15, 15, 15)
        self._track_count_label = QLabel("Tracks (0)")
        self._track_count_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(self._track_count_label)
        self._unsaved_label = QLabel("*unsaved")
        self._unsaved_label.setStyleSheet("color: #f39c12; font-size: 12px;")
        self._unsaved_label.hide()
        header_layout.addWidget(self._unsaved_label)
        header_layout.addStretch()
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet("background: transparent; color: #4ecca3; font-size: 13px;")
        back_btn.clicked.connect(self.back_requested.emit)
        header_layout.addWidget(back_btn)
        layout.addWidget(header)

        # --- Filters ---
        filter_widget = QWidget()
        filter_widget.setStyleSheet("background: #16213e; border-bottom: 1px solid #2d3a5a;")
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(15, 10, 15, 10)
        filter_layout.setSpacing(6)
        self._filter_buttons = {}
        for name in ["all", "visible", "review", "car", "truck", "bus"]:
            btn = QPushButton(name.capitalize())
            btn.setStyleSheet("""
                QPushButton { padding: 4px 8px; border: 1px solid #2d3a5a; border-radius: 4px;
                              background: transparent; color: #888; font-size: 11px; }
                QPushButton:hover { border-color: #4ecca3; color: #4ecca3; }
            """)
            btn.clicked.connect(lambda checked, n=name: self._on_filter(n))
            self._filter_buttons[name] = btn
            filter_layout.addWidget(btn)
        filter_layout.addStretch()
        self._roi_filter_cb = QCheckBox("In ROI only")
        self._roi_filter_cb.setEnabled(False)
        self._roi_filter_cb.toggled.connect(self._on_roi_filter_toggled)
        filter_layout.addWidget(self._roi_filter_cb)
        self._update_filter_buttons()
        layout.addWidget(filter_widget)

        # --- Track list (scrollable) ---
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(10, 10, 10, 10)
        self._scroll_layout.setSpacing(3)
        self._scroll_layout.addStretch()
        self._scroll.setWidget(self._scroll_content)
        layout.addWidget(self._scroll, 1)

        # --- ROI Panel ---
        roi_widget = QWidget()
        roi_widget.setStyleSheet("border-top: 1px solid #2d3a5a; padding: 10px 15px;")
        roi_layout = QVBoxLayout(roi_widget)
        roi_layout.setContentsMargins(15, 10, 15, 10)
        roi_header = QHBoxLayout()
        roi_title = QLabel("REGIONS OF INTEREST")
        roi_title.setStyleSheet("font-size: 12px; color: #888; font-weight: bold;")
        roi_header.addWidget(roi_title)
        self._roi_count_label = QLabel("")
        self._roi_count_label.setStyleSheet("font-size: 10px; color: #555;")
        roi_header.addWidget(self._roi_count_label)
        roi_header.addStretch()
        roi_layout.addLayout(roi_header)
        self._roi_list_layout = QVBoxLayout()
        roi_layout.addLayout(self._roi_list_layout)
        roi_widget.setMaximumHeight(200)
        layout.addWidget(roi_widget)

        # --- Actions ---
        actions_widget = QWidget()
        actions_widget.setStyleSheet("border-top: 1px solid #2d3a5a;")
        actions_layout = QVBoxLayout(actions_widget)
        actions_layout.setContentsMargins(15, 15, 15, 15)
        actions_layout.setSpacing(8)

        title = QLabel("TRACK ACTIONS")
        title.setStyleSheet("font-size: 12px; color: #888; font-weight: bold;")
        actions_layout.addWidget(title)

        row1 = QHBoxLayout()
        self.delete_btn = QPushButton("Delete (D)")
        self.delete_btn.setObjectName("delete_btn")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self.delete_requested.emit)
        row1.addWidget(self.delete_btn)
        self.class_btn = QPushButton("Class (C)")
        self.class_btn.setObjectName("class_btn")
        self.class_btn.setEnabled(False)
        self.class_btn.clicked.connect(self.class_change_requested.emit)
        row1.addWidget(self.class_btn)
        actions_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.merge_btn = QPushButton("Merge (M)")
        self.merge_btn.setObjectName("merge_btn")
        self.merge_btn.setEnabled(False)
        self.merge_btn.clicked.connect(self.merge_requested.emit)
        row2.addWidget(self.merge_btn)
        self.split_btn = QPushButton("Split (S)")
        self.split_btn.setObjectName("split_btn")
        self.split_btn.setEnabled(False)
        self.split_btn.clicked.connect(self.split_requested.emit)
        row2.addWidget(self.split_btn)
        actions_layout.addLayout(row2)

        self.save_btn = QPushButton("Save Corrections (Ctrl+S)")
        self.save_btn.setObjectName("save_btn")
        self.save_btn.clicked.connect(self.save_requested.emit)
        actions_layout.addWidget(self.save_btn)

        layout.addWidget(actions_widget)

        # --- Shortcuts help ---
        shortcuts = QLabel(
            "Space Play  ←→ Frame  V Select  B Draw\n"
            "D Delete  C Class  M Merge  S Split\n"
            "N Copy→Next  P Copy→Prev  R Next Review  Ctrl+S Save\n"
            "Shift+R ROI Rect  Shift+P ROI Poly"
        )
        shortcuts.setStyleSheet(
            "font-size: 10px; color: #555; padding: 10px 15px; "
            "border-top: 1px solid #2d3a5a; line-height: 1.6;"
        )
        shortcuts.setWordWrap(True)
        layout.addWidget(shortcuts)

    # ------------------------------------------------------------------
    # Filter handling
    # ------------------------------------------------------------------

    def _on_filter(self, name):
        self._current_filter = name
        self._update_filter_buttons()
        self.filter_changed.emit(name)
        self.refresh_tracks()

    def _on_roi_filter_toggled(self, checked):
        """Handle the 'In ROI only' checkbox toggle."""
        self.filter_changed.emit(self._current_filter)
        self.refresh_tracks()
        self.roi_selection_changed.emit()

    def is_roi_filter_active(self):
        """Return True if the ROI filter checkbox is checked and ROIs exist,
        or if specific ROIs are selected via checkboxes."""
        if self._selected_roi_indices:
            return True
        return self._roi_filter_cb.isChecked() and bool(self._rois)

    def get_selected_roi_indices(self):
        """Return the set of currently selected ROI indices."""
        return set(self._selected_roi_indices)

    def _update_filter_buttons(self):
        for name, btn in self._filter_buttons.items():
            if name == self._current_filter:
                btn.setStyleSheet("""
                    QPushButton { padding: 4px 8px; border: 1px solid #4ecca3; border-radius: 4px;
                                  background: transparent; color: #4ecca3; font-size: 11px; }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton { padding: 4px 8px; border: 1px solid #2d3a5a; border-radius: 4px;
                                  background: transparent; color: #888; font-size: 11px; }
                    QPushButton:hover { border-color: #4ecca3; color: #4ecca3; }
                """)

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_tracks(self, tracks, current_frame, selected_track_id=None):
        """Set the full track data and refresh the list."""
        self._tracks = tracks
        self._current_frame = current_frame
        self._selected_track_id = selected_track_id
        self.refresh_tracks()
        self._update_action_buttons()

    def set_current_frame(self, frame):
        self._current_frame = frame

    def set_selected_track(self, track_id):
        self._selected_track_id = track_id
        self.refresh_tracks()
        self._update_action_buttons()

    def set_unsaved(self, unsaved: bool):
        self._has_unsaved = unsaved
        self._unsaved_label.setVisible(unsaved)

    def set_rois(self, rois: list, tracks: list):
        """Update the ROI panel with current ROIs and recompute vehicle counts."""
        self._rois = rois
        # Enable/disable the ROI filter checkbox
        has_rois = bool(rois)
        self._roi_filter_cb.setEnabled(has_rois)
        if not has_rois:
            self._roi_filter_cb.setChecked(False)
            self._selected_roi_indices.clear()
        else:
            # Prune selected indices that are out of range after ROI deletion
            self._selected_roi_indices = {i for i in self._selected_roi_indices if i < len(rois)}
        # Reset checkbox reference list
        self._roi_checkboxes = []
        # Clear existing ROI items
        while self._roi_list_layout.count():
            item = self._roi_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not rois:
            empty = QLabel("No ROIs. Use ROI Rect/Poly buttons.")
            empty.setStyleSheet("font-size: 11px; color: #555;")
            self._roi_list_layout.addWidget(empty)
            self._roi_count_label.setText("")
            return

        self._roi_count_label.setText(f"({len(rois)})")
        for i, roi in enumerate(rois):
            stats = self._compute_roi_stats(roi, tracks)
            item = self._create_roi_item(roi, stats, i)
            self._roi_list_layout.addWidget(item)

    # ------------------------------------------------------------------
    # ROI helpers
    # ------------------------------------------------------------------

    def _compute_roi_track_ids(self):
        """Return the set of track IDs whose bbox center falls inside the active ROIs.

        If specific ROIs are selected via checkboxes, only those are checked.
        Otherwise (global 'In ROI only' checkbox), all ROIs are checked.
        """
        active_rois = self._get_active_rois()
        ids = set()
        for track in self._tracks:
            for fd in track.get("frames", []):
                cx = (fd["bbox"][0] + fd["bbox"][2]) / 2
                cy = (fd["bbox"][1] + fd["bbox"][3]) / 2
                if self._point_in_any_roi(cx, cy, active_rois):
                    ids.add(track.get("track_id"))
                    break
        return ids

    def _get_active_rois(self):
        """Return the list of ROIs to use for filtering.

        If specific ROI checkboxes are selected, return only those.
        Otherwise return all ROIs.
        """
        if self._selected_roi_indices:
            return [self._rois[i] for i in self._selected_roi_indices if i < len(self._rois)]
        return self._rois

    def _point_in_any_roi(self, px, py, rois=None):
        """Check if a point falls inside any of the given ROIs."""
        if rois is None:
            rois = self._rois
        for roi in rois:
            if roi.get("type") == "rect":
                p1, p2 = roi["points"][0], roi["points"][1]
                x1, y1 = p1["x"], p1["y"]
                x2, y2 = p2["x"], p2["y"]
                if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2):
                    return True
            elif roi.get("type") == "polygon":
                if self._point_in_polygon(px, py, roi["points"]):
                    return True
        return False

    def get_roi_track_ids(self):
        """Return the set of track IDs that pass through any ROI (computed during last refresh)."""
        return self._roi_track_ids

    def _compute_roi_stats(self, roi, tracks):
        """Count how many tracks pass through the given ROI."""
        stats = {"total": 0, "by_class": {}}
        for track in tracks:
            passes = False
            for f in track.get("frames", []):
                cx = (f["bbox"][0] + f["bbox"][2]) / 2
                cy = (f["bbox"][1] + f["bbox"][3]) / 2
                if roi["type"] == "rect":
                    p1, p2 = roi["points"][0], roi["points"][1]
                    if p1["x"] <= cx <= p2["x"] and p1["y"] <= cy <= p2["y"]:
                        passes = True
                        break
                elif roi["type"] == "polygon":
                    if self._point_in_polygon(cx, cy, roi["points"]):
                        passes = True
                        break
            if passes:
                stats["total"] += 1
                cls = track.get("class", "unknown")
                stats["by_class"][cls] = stats["by_class"].get(cls, 0) + 1
        return stats

    def _point_in_polygon(self, px, py, polygon):
        """Ray-casting point-in-polygon test."""
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]["x"], polygon[i]["y"]
            xj, yj = polygon[j]["x"], polygon[j]["y"]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _create_roi_item(self, roi, stats, roi_index):
        """Create a QFrame widget for a single ROI entry with a selection checkbox."""
        frame = QFrame()
        is_selected = roi_index in self._selected_roi_indices
        border_style = f"border: 1px solid {roi.get('color', '#ff6b6b')};" if is_selected else ""
        frame.setStyleSheet(
            f"background: #1a1a2e; border-radius: 4px; padding: 6px 8px; margin-bottom: 2px; {border_style}"
        )
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        cb = QCheckBox()
        cb.setChecked(is_selected)
        cb.setToolTip("Select to filter canvas to this ROI")
        cb.toggled.connect(lambda checked, idx=roi_index: self._on_roi_checkbox_toggled(idx, checked))
        layout.addWidget(cb)
        # Keep reference for later access
        while len(self._roi_checkboxes) <= roi_index:
            self._roi_checkboxes.append(None)
        self._roi_checkboxes[roi_index] = cb

        color_dot = QLabel()
        color_dot.setFixedSize(10, 10)
        color_dot.setStyleSheet(
            f"background: {roi.get('color', '#ff6b6b')}; border-radius: 2px;"
        )
        layout.addWidget(color_dot)

        name = QLabel(roi.get("name", "ROI"))
        name.setStyleSheet("font-size: 12px; color: #eee;")
        layout.addWidget(name, 1)

        count = QLabel(f"{stats['total']} vehicles")
        count.setStyleSheet("font-size: 11px; color: #aaa;")
        layout.addWidget(count)

        rename_btn = QPushButton("edit")
        rename_btn.setFixedSize(30, 20)
        rename_btn.setStyleSheet(
            "background: transparent; color: #4ecca3; font-size: 10px; padding: 0;"
        )
        rename_btn.setToolTip("Rename this ROI")
        rename_btn.clicked.connect(lambda checked=False, idx=roi_index: self._on_roi_rename(idx))
        layout.addWidget(rename_btn)

        del_btn = QPushButton("x")
        del_btn.setFixedSize(20, 20)
        del_btn.setStyleSheet(
            "background: transparent; color: #e74c3c; font-size: 14px; padding: 0;"
        )
        del_btn.clicked.connect(lambda checked=False, idx=roi_index: self._on_roi_delete(idx))
        layout.addWidget(del_btn)

        return frame

    def _on_roi_checkbox_toggled(self, roi_index, checked):
        """Handle toggling of an individual ROI checkbox."""
        if checked:
            self._selected_roi_indices.add(roi_index)
        else:
            self._selected_roi_indices.discard(roi_index)
        self.refresh_tracks()
        self.roi_selection_changed.emit()

    def _on_roi_rename(self, roi_id):
        self.roi_rename_requested.emit(roi_id)

    def _on_roi_delete(self, roi_id):
        self.roi_delete_requested.emit(roi_id)

    # ------------------------------------------------------------------
    # Track list rendering
    # ------------------------------------------------------------------

    def refresh_tracks(self):
        """Re-render the track list based on the current filter and frame."""
        # Clear existing items
        while self._scroll_layout.count():
            item = self._scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        filtered = list(self._tracks)
        f = self._current_filter
        if f == "review":
            filtered = [t for t in filtered if t.get("needs_review")]
        elif f == "visible":
            filtered = [t for t in filtered if
                        self._current_frame >= t.get("start_frame",
                            t["frames"][0]["frame"] if t.get("frames") else 0) and
                        self._current_frame <= t.get("end_frame",
                            t["frames"][-1]["frame"] if t.get("frames") else 0)]
        elif f not in ("all",):
            # Class filter (car, truck, bus, etc.)
            filtered = [t for t in filtered if t.get("class") == f]

        # ROI filter: only keep tracks that pass through at least one ROI
        if self.is_roi_filter_active():
            self._roi_track_ids = self._compute_roi_track_ids()
            filtered = [t for t in filtered if t.get("track_id") in self._roi_track_ids]
        else:
            self._roi_track_ids = set()

        # Sort: visible tracks first, then by track ID
        filtered.sort(key=lambda t: (
            not (self._current_frame >= t.get("start_frame",
                    t["frames"][0]["frame"] if t.get("frames") else 0)
                 and self._current_frame <= t.get("end_frame",
                    t["frames"][-1]["frame"] if t.get("frames") else 0)),
            t.get("track_id", 0)
        ))

        for track in filtered:
            item = TrackItem(
                track, self._current_frame,
                track["track_id"] == self._selected_track_id
            )
            item.clicked.connect(lambda tid: self.track_selected.emit(tid))
            item.double_clicked.connect(lambda tid: self.track_double_clicked.emit(tid))
            self._scroll_layout.addWidget(item)

        self._scroll_layout.addStretch()
        self._track_count_label.setText(f"Tracks ({len(filtered)})")

    def _update_action_buttons(self):
        """Enable or disable action buttons based on whether a track is selected."""
        has_sel = self._selected_track_id is not None
        self.delete_btn.setEnabled(has_sel)
        self.class_btn.setEnabled(has_sel)
        self.merge_btn.setEnabled(has_sel)
        self.split_btn.setEnabled(has_sel)
