"""
Main window -- the primary application window with tab-based navigation.

Contains:
- Menu bar: File, View, Help
- QTabWidget: Preprocessing, Correction, Performance
- Status bar with mode badge
- Review window management
- Settings dialog (File > Settings)
"""
import shutil
import sys
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
from cctv_yolo.preprocessing_tab import PreprocessingTab
from cctv_yolo.correction_tab import CorrectionTab
from cctv_yolo.performance_tab import PerformanceTab
from cctv_yolo.batch_tab import BatchTab
from cctv_yolo.analytics_tab import AnalyticsTab
from cctv_yolo.training_tab import TrainingTab
from cctv_yolo.models_tab import ModelsTab
from cctv_yolo.live_tab import LiveTab
from cctv_yolo.insights_tab import InsightsTab
from cctv_yolo.search_dialog import CrossSessionSearchDialog
from cctv_yolo.settings_dialog import SettingsDialog
from cctv_yolo.review_window import ReviewWindow

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
from cctv_yolo.theme import (
    INDIGO as BG, PANEL, BORDER, PURPLE as ACCENT, OFFWHITE as TEXT,
    PINK, INDIGO,
    OFFWHITE,
    PURPLE,
)

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
    color: ;
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
    background-color: {PINK};
    color: {INDIGO};
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
        self._settings_dialog = None  # lazy-created

        self._setup_ui()
        self._setup_menu()
        self.setStyleSheet(STYLE)
        self.setWindowTitle("CCTV-YOLO  |  Vehicle Detection & Correction")
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        # Auto-reconnect NAS on startup
        self._check_auto_reconnect()
        self._update_mode_badge()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        # --- Tab widget ---
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.preprocessing_tab = PreprocessingTab(self.data_manager)
        self.batch_tab = BatchTab(self.data_manager)
        self.correction_tab = CorrectionTab(self.data_manager)
        self.performance_tab = PerformanceTab(self.data_manager)
        self.analytics_tab = AnalyticsTab(self.data_manager)
        self.training_tab = TrainingTab(self.data_manager)
        self.models_tab = ModelsTab(self.data_manager)
        self.live_tab = LiveTab(self.data_manager)
        self.insights_tab = InsightsTab(self.data_manager)

        self.tabs.addTab(self.preprocessing_tab, "Preprocessing")
        self.tabs.addTab(self.batch_tab, "Batch")
        self.tabs.addTab(self.correction_tab, "Correction")
        self.tabs.addTab(self.performance_tab, "Performance")
        self.tabs.addTab(self.analytics_tab, "Analytics")
        self.tabs.addTab(self.insights_tab, "Insights")
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.models_tab, "Models")
        self.tabs.addTab(self.live_tab, "Live")

        # Connect signals
        self.preprocessing_tab.review_requested.connect(self.open_review)
        self.correction_tab.review_requested.connect(self.open_review)
        self.batch_tab.review_requested.connect(self.open_review)
        self.training_tab.review_requested.connect(self.open_review)
        self.insights_tab.review_requested.connect(self.open_review)

        # PRD J7 — "Promote after comparison" flow. When Training fires
        # compare_models_requested(model_a, model_b), jump to the Performance
        # tab with both models pre-selected in the A/B compare panel.
        self.training_tab.compare_models_requested.connect(
            self._open_perf_compare
        )

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

        # --- Global OpenLocationBar in the menu-bar corner (PRD C12). Gives
        # every tab access to the most-used data folders without per-tab plumbing.
        from cctv_yolo.widgets.open_location_bar import OpenLocationBar
        from cctv_yolo.logging_config import get_log_file_path
        global_bar = OpenLocationBar(self)
        global_bar.add_folder("Videos", self.data_manager.videos_dir)
        global_bar.add_folder("Tracks", self.data_manager.tracks_dir)
        global_bar.add_folder("Corrections", self.data_manager.corrections_dir)
        global_bar.add_folder("Exports", self.data_manager.exports_dir)
        global_bar.add_folder("Models", self.data_manager.models_dir)
        global_bar.add_folder("Logs", lambda: get_log_file_path().parent)
        menubar.setCornerWidget(global_bar, Qt.TopRightCorner)

        # --- File menu ---
        file_menu = menubar.addMenu("File")

        action_import = QAction("Import Video...", self)
        action_import.setShortcut(QKeySequence("Ctrl+I"))
        action_import.triggered.connect(self._import_video)
        file_menu.addAction(action_import)

        action_open_data = QAction("Open Data Folder", self)
        action_open_data.triggered.connect(lambda: self.data_manager.open_folder("data"))
        file_menu.addAction(action_open_data)

        action_search = QAction("Cross-session Search…", self)
        action_search.setShortcut(QKeySequence("Ctrl+F"))
        action_search.triggered.connect(self._open_search)
        file_menu.addAction(action_search)

        file_menu.addSeparator()

        action_settings = QAction("Settings...", self)
        action_settings.setShortcut(QKeySequence("Ctrl+,"))
        action_settings.triggered.connect(self._open_settings)
        file_menu.addAction(action_settings)

        file_menu.addSeparator()

        action_quit = QAction("Quit", self)
        action_quit.setShortcut(QKeySequence("Ctrl+Q"))
        action_quit.triggered.connect(self.close)
        file_menu.addAction(action_quit)

        # --- View menu ---
        view_menu = menubar.addMenu("View")

        action_preprocessing = QAction("Preprocessing", self)
        action_preprocessing.setShortcut(QKeySequence("Ctrl+1"))
        action_preprocessing.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        view_menu.addAction(action_preprocessing)

        action_correction = QAction("Correction", self)
        action_correction.setShortcut(QKeySequence("Ctrl+2"))
        action_correction.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        view_menu.addAction(action_correction)

        action_performance = QAction("Performance", self)
        action_performance.setShortcut(QKeySequence("Ctrl+3"))
        action_performance.triggered.connect(lambda: self.tabs.setCurrentIndex(3))
        view_menu.addAction(action_performance)

        action_batch = QAction("Batch", self)
        action_batch.setShortcut(QKeySequence("Ctrl+4"))
        action_batch.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        view_menu.addAction(action_batch)

        action_analytics = QAction("Analytics", self)
        action_analytics.setShortcut(QKeySequence("Ctrl+5"))
        action_analytics.triggered.connect(lambda: self.tabs.setCurrentIndex(4))
        view_menu.addAction(action_analytics)

        action_insights = QAction("Insights", self)
        action_insights.setShortcut(QKeySequence("Ctrl+9"))
        action_insights.triggered.connect(lambda: self.tabs.setCurrentIndex(5))
        view_menu.addAction(action_insights)

        action_training = QAction("Training", self)
        action_training.setShortcut(QKeySequence("Ctrl+6"))
        action_training.triggered.connect(lambda: self.tabs.setCurrentIndex(6))
        view_menu.addAction(action_training)

        action_models = QAction("Models", self)
        action_models.setShortcut(QKeySequence("Ctrl+7"))
        action_models.triggered.connect(lambda: self.tabs.setCurrentIndex(7))
        view_menu.addAction(action_models)

        action_live = QAction("Live", self)
        action_live.setShortcut(QKeySequence("Ctrl+8"))
        action_live.triggered.connect(lambda: self.tabs.setCurrentIndex(8))
        view_menu.addAction(action_live)

        # --- Help menu ---
        help_menu = menubar.addMenu("Help")

        action_about = QAction("About", self)
        action_about.triggered.connect(self._show_about)
        help_menu.addAction(action_about)

        action_shortcuts = QAction("Keyboard Shortcuts", self)
        action_shortcuts.setShortcut(QKeySequence("Ctrl+/"))
        action_shortcuts.triggered.connect(self._show_shortcuts)
        help_menu.addAction(action_shortcuts)

        help_menu.addSeparator()
        action_show_log = QAction("Show Log Folder", self)
        action_show_log.triggered.connect(self._show_log_folder)
        help_menu.addAction(action_show_log)

    def _show_log_folder(self):
        """PRD C4 — Help → Show Log Folder opens <data_root>/logs/."""
        import subprocess
        from cctv_yolo.logging_config import get_log_file_path
        log_path = get_log_file_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(log_path)] if log_path.exists()
                             else ["open", str(log_path.parent)])
        elif sys.platform == "win32":
            if log_path.exists():
                subprocess.Popen(["explorer", "/select,", str(log_path)])
            else:
                subprocess.Popen(["explorer", str(log_path.parent)])
        else:
            subprocess.Popen(["xdg-open", str(log_path.parent)])

    # ------------------------------------------------------------------
    # Settings dialog
    # ------------------------------------------------------------------

    def _open_settings(self):
        """Open the settings dialog."""
        dlg = SettingsDialog(self.data_manager, parent=self)
        dlg.mode_changed.connect(self._on_mode_changed)
        dlg.exec()

    def _check_auto_reconnect(self):
        """Auto-reconnect NAS on startup if previously connected."""
        from cctv_yolo.nas_manager import NasManager
        nas_manager = NasManager(self.data_manager.config_dir / "nas.json")
        mount_point = nas_manager.check_auto_reconnect()
        if mount_point:
            self.data_manager.switch_to_nas(mount_point)
            self._update_mode_badge()
            config = nas_manager.load_config()
            ip = config.get("ip", "?") if config else "?"
            share = config.get("share", "?") if config else "?"
            self.status.showMessage(f"Auto-reconnected to NAS //{ip}/{share}")

    # ------------------------------------------------------------------
    # Review window
    # ------------------------------------------------------------------

    def _open_perf_compare(self, model_a: str, model_b: str):
        """PRD J7 — handle Training's compare_models_requested signal.

        Switch to the Performance tab and pre-fill its A/B compare panel
        with the two models so the user can hit Run and see the comparison
        before deciding to promote.
        """
        try:
            self.performance_tab.prefill_compare(model_a, model_b)
        except Exception as e:
            self.status.showMessage(f"Couldn't pre-fill compare: {e}", 5000)
            return
        self.tabs.setCurrentWidget(self.performance_tab)
        self.status.showMessage(
            f"Pre-filled Performance compare: {model_a} vs {model_b}", 5000
        )

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
        self.preprocessing_tab.refresh()
        self.correction_tab.refresh()
        self.performance_tab.refresh()
        self.batch_tab.refresh()
        self.analytics_tab.refresh()
        self.training_tab.refresh()
        self.models_tab.refresh()
        self.live_tab.refresh()
        self.insights_tab.refresh()

    # ------------------------------------------------------------------
    # Mode changes
    # ------------------------------------------------------------------

    def _on_mode_changed(self, mode):
        """Handle NAS connect/disconnect from settings dialog."""
        self._update_mode_badge()
        # Refresh tabs with new data directories
        self.preprocessing_tab.refresh()
        self.correction_tab.refresh()
        self.performance_tab.refresh()
        self.batch_tab.refresh()
        self.analytics_tab.refresh()
        self.training_tab.refresh()
        self.models_tab.refresh()
        self.live_tab.refresh()
        self.insights_tab.refresh()
        self.status.showMessage(f"Switched to {mode.upper()} mode")

    def _open_search(self):
        """Open the cross-session track search dialog."""
        dlg = CrossSessionSearchDialog(self.data_manager, parent=self)
        dlg.open_review.connect(self.open_review)
        run = getattr(dlg, "exec")
        run()

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
        # Start the picker at the videos folder so the user can see what's
        # already imported (and so they're not picking from random spots).
        videos_dir = self.data_manager.videos_dir
        videos_dir.mkdir(parents=True, exist_ok=True)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Video",
            str(videos_dir),
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
            self.preprocessing_tab.refresh()
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import video:\n{e}")

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def _show_about(self):
        # PRD C8 — rich About dialog with version + platform + folder shortcuts.
        try:
            from cctv_yolo.about_dialog import AboutDialog
            dlg = AboutDialog(self.data_manager, parent=self)
            # Modal QDialog event loop (Qt method, not shell exec).
            getattr(dlg, "exec")()
        except Exception:
            # Fallback to plain QMessageBox if the dialog can't load.
            from cctv_yolo.__version__ import __version__
            QMessageBox.about(
                self,
                "About CCTV-YOLO",
                f"CCTV-YOLO v{__version__}\n\n"
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
            "Ctrl+,         Settings\n"
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

        # Stop batch queue + watch folders so the app exits cleanly
        try:
            self.batch_tab.shutdown()
        except Exception:
            pass

        event.accept()
