"""
Main window — the primary application window with tab-based navigation.

Contains:
- Menu bar: File, View, Help
- QTabWidget: Sessions, Videos, Settings
- Status bar with mode badge
- Review window management
"""
import shutil
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QMenuBar,
    QMenu,
    QStatusBar,
    QLabel,
    QFileDialog,
    QMessageBox,
)

from cctv_yolo.data_manager import DataManager
from cctv_yolo.sessions_tab import SessionsTab
from cctv_yolo.videos_tab import VideosTab
from cctv_yolo.settings_tab import SettingsTab
from cctv_yolo.review_window import ReviewWindow

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
BG = "#1a1a2e"
PANEL = "#16213e"
BORDER = "#2d3a5a"
ACCENT = "#4ecca3"
TEXT = "#eeeeee"

STYLE = f"""
QMainWindow {{
    background-color: {BG};
    color: {TEXT};
}}
QStatusBar {{
    background-color: {PANEL};
    color: {TEXT};
    border-top: 1px solid {BORDER};
    font-size: 12px;
    padding: 2px 8px;
    min-height: 24px;
}}
"""

MODE_BADGE_LOCAL = f"""
QLabel {{
    background-color: {ACCENT};
    color: #000;
    border-radius: 10px;
    padding: 4px 14px;
    font-weight: bold;
    font-size: 11px;
    letter-spacing: 1px;
    margin: 2px 4px;
}}
"""

MODE_BADGE_NAS = f"""
QLabel {{
    background-color: #3498db;
    color: white;
    border-radius: 10px;
    padding: 4px 14px;
    font-weight: bold;
    font-size: 11px;
    letter-spacing: 1px;
    margin: 2px 4px;
}}
"""


class MainWindow(QMainWindow):
    """Primary application window with tabbed interface."""

    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self._review_windows = []  # keep references to open review windows

        self._setup_ui()
        self._setup_menu()
        self.setStyleSheet(STYLE)
        self.setWindowTitle("CCTV-YOLO  |  Vehicle Detection & Correction")
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        # Auto-reconnect NAS on startup
        self.settings_tab.check_auto_reconnect()
        self._update_mode_badge()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        # --- Tab widget ---
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.sessions_tab = SessionsTab(self.data_manager)
        self.videos_tab = VideosTab(self.data_manager)
        self.settings_tab = SettingsTab(self.data_manager)

        self.tabs.addTab(self.sessions_tab, "Sessions")
        self.tabs.addTab(self.videos_tab, "Videos")
        self.tabs.addTab(self.settings_tab, "Settings")

        # Connect signals
        self.sessions_tab.review_requested.connect(self.open_review)
        self.videos_tab.review_requested.connect(self.open_review)
        self.settings_tab.mode_changed.connect(self._on_mode_changed)

        # --- Status bar ---
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.mode_badge = QLabel("LOCAL")
        self.mode_badge.setStyleSheet(MODE_BADGE_LOCAL)
        self.status.addPermanentWidget(self.mode_badge)

        self.status.showMessage("Ready")

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _setup_menu(self):
        menubar = self.menuBar()

        # --- File menu ---
        file_menu = menubar.addMenu("File")

        action_import = QAction("Import Video...", self)
        action_import.setShortcut(QKeySequence("Ctrl+I"))
        action_import.triggered.connect(self._import_video)
        file_menu.addAction(action_import)

        action_open_data = QAction("Open Data Folder", self)
        action_open_data.triggered.connect(lambda: self.data_manager.open_folder("data"))
        file_menu.addAction(action_open_data)

        file_menu.addSeparator()

        action_quit = QAction("Quit", self)
        action_quit.setShortcut(QKeySequence("Ctrl+Q"))
        action_quit.triggered.connect(self.close)
        file_menu.addAction(action_quit)

        # --- View menu ---
        view_menu = menubar.addMenu("View")

        action_sessions = QAction("Sessions", self)
        action_sessions.setShortcut(QKeySequence("Ctrl+1"))
        action_sessions.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        view_menu.addAction(action_sessions)

        action_videos = QAction("Videos", self)
        action_videos.setShortcut(QKeySequence("Ctrl+2"))
        action_videos.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        view_menu.addAction(action_videos)

        action_settings = QAction("Settings", self)
        action_settings.setShortcut(QKeySequence("Ctrl+3"))
        action_settings.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        view_menu.addAction(action_settings)

        # --- Help menu ---
        help_menu = menubar.addMenu("Help")

        action_about = QAction("About", self)
        action_about.triggered.connect(self._show_about)
        help_menu.addAction(action_about)

        action_shortcuts = QAction("Keyboard Shortcuts", self)
        action_shortcuts.setShortcut(QKeySequence("Ctrl+/"))
        action_shortcuts.triggered.connect(self._show_shortcuts)
        help_menu.addAction(action_shortcuts)

    # ------------------------------------------------------------------
    # Review window
    # ------------------------------------------------------------------

    def open_review(self, session_id: str):
        """Open a review window for the given session."""
        # Check if already open
        for rw in self._review_windows:
            if rw.session_id == session_id and rw.isVisible():
                rw.raise_()
                rw.activateWindow()
                return

        review_win = ReviewWindow(self.data_manager, session_id, parent=None)
        review_win.closed.connect(lambda: self._on_review_closed(review_win))
        review_win.show()
        self._review_windows.append(review_win)
        self.status.showMessage(f"Opened review for: {session_id}")

    def _on_review_closed(self, review_win):
        """Clean up when a review window closes."""
        if review_win in self._review_windows:
            self._review_windows.remove(review_win)
        # Refresh tabs to reflect any saved corrections
        self.sessions_tab.refresh()
        self.videos_tab.refresh()

    # ------------------------------------------------------------------
    # Mode changes
    # ------------------------------------------------------------------

    def _on_mode_changed(self, mode):
        """Handle NAS connect/disconnect from settings tab."""
        self._update_mode_badge()
        # Refresh tabs with new data directories
        self.sessions_tab.refresh()
        self.videos_tab.refresh()
        self.status.showMessage(f"Switched to {mode.upper()} mode")

    def _update_mode_badge(self):
        """Update the mode badge in the status bar."""
        mode = self.data_manager.active_mode
        if mode == "nas":
            self.mode_badge.setText("NAS")
            self.mode_badge.setStyleSheet(MODE_BADGE_NAS)
        else:
            self.mode_badge.setText("LOCAL")
            self.mode_badge.setStyleSheet(MODE_BADGE_LOCAL)

    # ------------------------------------------------------------------
    # Import video
    # ------------------------------------------------------------------

    def _import_video(self):
        """Import a video file into the local videos directory."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Video",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)",
        )
        if not file_path:
            return

        src = Path(file_path)
        dest = self.data_manager.videos_dir / src.name

        if dest.exists():
            reply = QMessageBox.question(
                self,
                "File Exists",
                f"'{src.name}' already exists in the videos directory. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        try:
            shutil.copy2(str(src), str(dest))
            self.status.showMessage(f"Imported: {src.name}")
            self.videos_tab.refresh()
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import video:\n{e}")

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def _show_about(self):
        QMessageBox.about(
            self,
            "About CCTV-YOLO",
            "CCTV-YOLO v2.0\n\n"
            "Vehicle detection & correction tool\n"
            "YOLOv8 + ByteTrack\n\n"
            "Native desktop application built with PySide6.",
        )

    def _show_shortcuts(self):
        shortcuts_text = (
            "Review Window Shortcuts\n"
            "========================\n\n"
            "Space          Play / Pause\n"
            "Left / Right   Step frame\n"
            "V              Select mode\n"
            "B              Draw box mode\n"
            "D              Delete track\n"
            "C              Change class\n"
            "M              Merge tracks\n"
            "S              Split track\n"
            "N              Copy box to next frame\n"
            "P              Copy box to previous frame\n"
            "R              Next review session\n"
            "Shift+R        ROI rectangle mode\n"
            "Shift+P        ROI polygon mode\n"
            "Ctrl+Z         Undo\n"
            "Ctrl+Y         Redo\n"
            "Ctrl+S         Save corrections\n"
            "Escape         Cancel / Select mode\n\n"
            "Main Window Shortcuts\n"
            "=====================\n\n"
            "Ctrl+I         Import video\n"
            "Ctrl+Q         Quit\n"
            "Ctrl+1/2/3     Switch tabs\n"
            "Ctrl+/         Show this dialog\n"
        )
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Warn if any review windows have unsaved changes."""
        unsaved_reviews = [rw for rw in self._review_windows if rw.unsaved]
        if unsaved_reviews:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                f"There are {len(unsaved_reviews)} review window(s) with unsaved changes.\n"
                "Close all anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        # Close all review windows
        for rw in list(self._review_windows):
            rw.unsaved = False  # prevent individual close warnings
            rw.close()
        self._review_windows.clear()
        event.accept()
