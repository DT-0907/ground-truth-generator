"""
Correction tab -- session list for reviewing and correcting tracks.

Shows stats cards (Total Sessions, Needs Review, Corrected) and a
scrollable list of session cards with filter buttons and review actions.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
    QSizePolicy,
)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
from cctv_yolo.theme import (
    INDIGO as BG, PANEL, BORDER, PURPLE as ACCENT, OFFWHITE as TEXT,
)
from cctv_yolo.widgets.open_location_bar import OpenLocationBar

CARD_STYLE_REVIEW = f"""
QFrame {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-left: 3px solid #F1C56B;
    border-radius: 8px;
}}
QFrame:hover {{
    background-color: #1E2050;
    border: 1px solid #2D2F60;
    border-left: 3px solid #F1C56B;
}}
"""

CARD_STYLE_CORRECTED = f"""
QFrame {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 8px;
}}
QFrame:hover {{
    background-color: #1E2050;
    border: 1px solid #2D2F60;
    border-left: 3px solid {ACCENT};
}}
"""

CARD_STYLE_DEFAULT = f"""
QFrame {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-left: 3px solid {BORDER};
    border-radius: 8px;
}}
QFrame:hover {{
    background-color: #1E2050;
    border: 1px solid #2D2F60;
    border-left: 3px solid {BORDER};
}}
"""

STAT_CARD_STYLE = f"""
QFrame {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1E2050, stop:1 {PANEL});
    border: 1px solid {BORDER};
    border-top: 2px solid {ACCENT};
    border-radius: 8px;
    padding: 12px;
}}
"""

BADGE_REVIEW = f"""
QLabel {{
    background-color: #FF6B7A;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}}
"""

BADGE_CORRECTED = f"""
QLabel {{
    background-color: {ACCENT};
    color: #15173D;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}}
"""

BADGE_MISSING = f"""
QLabel {{
    background-color: #A89BA8;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}}
"""

REVIEW_BTN_STYLE = f"""
QPushButton {{
    background-color: {ACCENT};
    color: #15173D;
    border: none;
    border-radius: 4px;
    padding: 6px 16px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: #E491C9;
}}
QPushButton:pressed {{
    background-color: #E491C9;
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

FILTER_BTN_ACTIVE = f"""
QPushButton {{
    background-color: {ACCENT};
    color: #15173D;
    border: none;
    border-radius: 4px;
    padding: 5px 14px;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: #E491C9;
}}
"""

FILTER_BTN_INACTIVE = f"""
QPushButton {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 14px;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: {BORDER};
}}
"""

NEXT_REVIEW_BTN = f"""
QPushButton {{
    background-color: #F1C56B;
    color: #15173D;
    border: none;
    border-radius: 4px;
    padding: 6px 16px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: #F1C56B;
}}
QPushButton:disabled {{
    background-color: {BORDER};
    color: #A89BA8;
}}
"""


class CorrectionTab(QWidget):
    """Correction tab -- session list for reviewing and correcting tracks."""

    review_requested = Signal(str)  # session_id

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self._current_filter = "all"  # "all", "needs_review", "corrected"
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
        title = QLabel("Correction")
        title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {TEXT};")
        header_row.addWidget(title)
        header_row.addStretch()

        self.btn_next_review = QPushButton("Next Review")
        self.btn_next_review.setStyleSheet(NEXT_REVIEW_BTN)
        self.btn_next_review.clicked.connect(self._go_next_review)
        header_row.addWidget(self.btn_next_review)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setStyleSheet(REFRESH_BTN_STYLE)
        self.btn_refresh.clicked.connect(self.refresh)
        header_row.addWidget(self.btn_refresh)

        # OpenLocationBar (PRD C12) — quick jumps to the on-disk locations
        # this tab cares about. Tooltip shows the resolved path.
        self.open_bar = OpenLocationBar(self)
        self.open_bar.add_folder(
            "Corrections",
            lambda: self.data_manager.corrections_dir,
        )
        self.open_bar.add_folder(
            "Tracks",
            lambda: self.data_manager.tracks_dir,
        )
        self.open_bar.add_folder(
            "Exports",
            lambda: self.data_manager.exports_dir,
        )
        header_row.addWidget(self.open_bar)
        layout.addLayout(header_row)

        # --- Stat cards row ---
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)

        self.stat_total = self._make_stat_card("Total Sessions", "0")
        self.stat_review = self._make_stat_card("Needs Review", "0")
        self.stat_corrected = self._make_stat_card("Corrected", "0")

        stats_row.addWidget(self.stat_total["frame"])
        stats_row.addWidget(self.stat_review["frame"])
        stats_row.addWidget(self.stat_corrected["frame"])
        layout.addLayout(stats_row)

        # --- Filter row ---
        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)

        self.btn_filter_all = QPushButton("All")
        self.btn_filter_review = QPushButton("Needs Review")
        self.btn_filter_corrected = QPushButton("Corrected")

        for btn in [self.btn_filter_all, self.btn_filter_review, self.btn_filter_corrected]:
            btn.setStyleSheet(FILTER_BTN_INACTIVE)

        self.btn_filter_all.setStyleSheet(FILTER_BTN_ACTIVE)

        self.btn_filter_all.clicked.connect(lambda: self._set_filter("all"))
        self.btn_filter_review.clicked.connect(lambda: self._set_filter("needs_review"))
        self.btn_filter_corrected.clicked.connect(lambda: self._set_filter("corrected"))

        filter_row.addWidget(self.btn_filter_all)
        filter_row.addWidget(self.btn_filter_review)
        filter_row.addWidget(self.btn_filter_corrected)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        # --- Scrollable session list ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background-color: {BG}; border: none; }}")

        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(10)
        self.list_layout.addStretch()

        scroll.setWidget(self.list_widget)
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
        lbl.setStyleSheet("color: #A89BA8; font-size: 11px; letter-spacing: 1px; border: none; background: transparent;")
        lbl.setAlignment(Qt.AlignLeft)

        val = QLabel(value_text)
        val.setStyleSheet(f"color: {ACCENT}; font-size: 36px; font-weight: bold; border: none; background: transparent;")
        val.setAlignment(Qt.AlignLeft)

        vbox.addWidget(lbl)
        vbox.addWidget(val)

        return {"frame": frame, "value_label": val}

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def _set_filter(self, filter_name):
        """Change the active filter and refresh the list."""
        self._current_filter = filter_name

        self.btn_filter_all.setStyleSheet(
            FILTER_BTN_ACTIVE if filter_name == "all" else FILTER_BTN_INACTIVE
        )
        self.btn_filter_review.setStyleSheet(
            FILTER_BTN_ACTIVE if filter_name == "needs_review" else FILTER_BTN_INACTIVE
        )
        self.btn_filter_corrected.setStyleSheet(
            FILTER_BTN_ACTIVE if filter_name == "corrected" else FILTER_BTN_INACTIVE
        )

        self.refresh()

    # ------------------------------------------------------------------
    # Refresh / populate
    # ------------------------------------------------------------------

    def refresh(self):
        """Reload session data from data_manager and refresh UI."""
        sessions = self.data_manager.get_sessions()

        # Update stats
        total = len(sessions)
        needs_review = sum(
            1 for s in sessions
            if s.get("needs_review", 0) > 0 and not s.get("has_corrections", False)
        )
        corrected = sum(1 for s in sessions if s.get("has_corrections", False))

        self.stat_total["value_label"].setText(str(total))
        self.stat_review["value_label"].setText(str(needs_review))
        self.stat_corrected["value_label"].setText(str(corrected))

        # Enable/disable next review button
        self.btn_next_review.setEnabled(needs_review > 0)

        # Apply filter
        if self._current_filter == "needs_review":
            sessions = [
                s for s in sessions
                if s.get("needs_review", 0) > 0 and not s.get("has_corrections", False)
            ]
        elif self._current_filter == "corrected":
            sessions = [s for s in sessions if s.get("has_corrections", False)]

        # Clear existing cards
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not sessions:
            empty_messages = {
                "all": "No sessions found. Process some videos first.",
                "needs_review": "No sessions need review.",
                "corrected": "No corrected sessions found.",
            }
            empty = QLabel(empty_messages.get(self._current_filter, "No sessions found."))
            empty.setStyleSheet("color: #999; font-size: 14px; padding: 24px;")
            empty.setAlignment(Qt.AlignCenter)
            self.list_layout.insertWidget(0, empty)
            return

        # Build session cards
        for i, session in enumerate(sessions):
            card = self._make_session_card(session)
            self.list_layout.insertWidget(i, card)

    def _make_session_card(self, session):
        """Build a single session card widget."""
        frame = QFrame()
        needs_review = session.get("needs_review", 0)
        has_corrections = session.get("has_corrections", False)
        if has_corrections:
            frame.setStyleSheet(CARD_STYLE_CORRECTED)
        elif needs_review > 0:
            frame.setStyleSheet(CARD_STYLE_REVIEW)
        else:
            frame.setStyleSheet(CARD_STYLE_DEFAULT)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        frame.setFixedHeight(90)

        hbox = QHBoxLayout(frame)
        hbox.setContentsMargins(16, 12, 16, 12)
        hbox.setSpacing(14)

        # Left: session info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        name_lbl = QLabel(session.get("video_name", session.get("id", "Unknown")))
        name_lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {TEXT}; border: none;")
        info_layout.addWidget(name_lbl)

        detail_parts = [f"{session.get('track_count', 0)} tracks"]
        if session.get("processed_at") and session["processed_at"] != "Unknown":
            detail_parts.append(session["processed_at"])
        detail_lbl = QLabel("  |  ".join(detail_parts))
        detail_lbl.setStyleSheet("font-size: 12px; color: #999; border: none;")
        info_layout.addWidget(detail_lbl)

        hbox.addLayout(info_layout, stretch=1)

        # Center: badges
        badge_layout = QHBoxLayout()
        badge_layout.setSpacing(6)

        video_exists = session.get("video_exists", True)

        if needs_review > 0 and not has_corrections:
            badge = QLabel(f"Needs Review ({needs_review})")
            badge.setStyleSheet(BADGE_REVIEW)
            badge_layout.addWidget(badge)

        if has_corrections:
            badge = QLabel("Corrected")
            badge.setStyleSheet(BADGE_CORRECTED)
            badge_layout.addWidget(badge)

        if not video_exists:
            badge = QLabel("Video Missing")
            badge.setStyleSheet(BADGE_MISSING)
            badge_layout.addWidget(badge)

        hbox.addLayout(badge_layout)

        # Right: review button
        btn = QPushButton("Review")
        btn.setStyleSheet(REVIEW_BTN_STYLE)
        btn.setFixedWidth(90)
        session_id = session.get("id", "")
        btn.clicked.connect(lambda checked=False, sid=session_id: self.review_requested.emit(sid))
        hbox.addWidget(btn)

        return frame

    # ------------------------------------------------------------------
    # Next review
    # ------------------------------------------------------------------

    def _go_next_review(self):
        """Jump to the next session that needs review."""
        next_id = self.data_manager.get_next_review_session()
        if next_id:
            self.review_requested.emit(next_id)
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "All Done", "No more sessions need review.")
