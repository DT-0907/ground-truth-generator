"""
Entry point for the CCTV-YOLO native desktop application.

Creates the QApplication, applies the dark Fusion palette,
ensures data directories exist, and launches the main window.
"""
import os
import sys
import traceback
from pathlib import Path

# OpenMP duplicate-library guard for Windows builds where torch + numpy both
# load libiomp5md.dll. Setting it here too (in addition to the runtime hook)
# covers `python -m cctv_yolo.main` dev runs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Pin native math libraries to a single thread BEFORE torch is imported.
# Detection runs on Qt worker threads; with the duplicate-OpenMP shim above,
# multi-threaded torch/OpenCV on a 2nd worker thread heap-corrupts the process
# on Windows (0xC0000374). Must be set before the first torch import — which in
# the GUI happens early via performance_tab -> model_compare. See processor.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Suppress stray console windows from child processes on Windows so no black
# terminal flashes over the GUI (the frozen build does this in runtime_hook.py;
# this covers dev runs: python run.py / -m cctv_yolo.main). CREATE_NO_WINDOW is
# a no-op for GUI children (explorer reveal), so it only hides console flashes.
if sys.platform == "win32":
    try:
        import subprocess as _sp
        _CREATE_NO_WINDOW = 0x08000000
        _orig_popen_init = _sp.Popen.__init__

        def _quiet_popen_init(self, *a, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _CREATE_NO_WINDOW
            _orig_popen_init(self, *a, **kw)

        _sp.Popen.__init__ = _quiet_popen_init
    except Exception:
        pass

# NOTE: PySide6 / cctv_yolo imports are deliberately NOT done at module scope.
# In the frozen Windows exe an import-time failure (missing DLL, missing hidden
# import) raised here would crash the process *before* the try/except in
# __main__ is reached — so no crash.log is written and the window just flashes
# and closes. Importing inside the functions below keeps every import under the
# crash handler.
#
# theme.py / __version__.py are safe to import at module scope — they contain
# only string constants, no Qt or heavy deps.
from cctv_yolo.theme import (
    INDIGO, PANEL, PANEL_HI, BORDER, PURPLE, PINK, OFFWHITE, TEXT_MUTED, ERROR,
    RADIUS, PAD,
)
from cctv_yolo.__version__ import __version__, __app_name__, __org_name__


def _make_dark_palette():
    """Build a dark QPalette using the C11 theme tokens."""
    from PySide6.QtGui import QPalette, QColor

    palette = QPalette()

    bg            = QColor(INDIGO)
    panel         = QColor(PANEL)
    border        = QColor(BORDER)
    accent        = QColor(PURPLE)
    highlight     = QColor(PINK)
    text          = QColor(OFFWHITE)
    dark_text     = QColor(INDIGO)
    disabled_text = QColor(TEXT_MUTED)
    bright_text   = QColor(OFFWHITE)

    # Active
    palette.setColor(QPalette.Window, bg)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, panel)
    palette.setColor(QPalette.AlternateBase, bg)
    palette.setColor(QPalette.ToolTipBase, panel)
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, panel)
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.BrightText, bright_text)
    palette.setColor(QPalette.Link, highlight)
    palette.setColor(QPalette.Highlight, accent)
    palette.setColor(QPalette.HighlightedText, dark_text)
    palette.setColor(QPalette.PlaceholderText, disabled_text)

    # Disabled
    palette.setColor(QPalette.Disabled, QPalette.WindowText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled_text)

    return palette


def main():
    """Application entry point."""
    # Heavy imports happen here — inside the crash-guarded path (see __main__),
    # so an import failure lands in crash.log instead of vanishing silently.
    from PySide6.QtWidgets import QApplication
    from cctv_yolo.data_manager import DataManager
    from cctv_yolo.logging_config import configure_logging
    from cctv_yolo.main_window import MainWindow

    # PRD C4 — configure logging FIRST so anything that follows is captured.
    configure_logging()

    app = QApplication(sys.argv)
    app.setApplicationName(__app_name__)
    app.setOrganizationName(__org_name__)
    app.setApplicationVersion(__version__)

    # Apply dark Fusion style
    app.setStyle("Fusion")
    app.setPalette(_make_dark_palette())

    # Global stylesheet for widgets that Fusion palette doesn't fully cover.
    # All colors come from theme tokens — no raw hex here. C11/C13 invariants:
    # PURPLE = action accent, PINK = hover/highlight, OFFWHITE = text.
    app.setStyleSheet(f"""
        /* --- Tooltips --- */
        QToolTip {{
            background-color: {PANEL};
            color: {OFFWHITE};
            border: 1px solid {PURPLE};
            border-radius: {RADIUS - 2}px;
            padding: 4px 8px;
            font-size: 12px;
        }}

        /* --- Scroll bars --- */
        QScrollBar:vertical {{
            background: {INDIGO};
            width: 10px;
            margin: 0;
            border: none;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical {{
            background: {BORDER};
            min-height: 30px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {PINK};
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height: 0;
            background: none;
            border: none;
        }}
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {{
            background: none;
        }}
        QScrollBar:horizontal {{
            background: {INDIGO};
            height: 10px;
            margin: 0;
            border: none;
            border-radius: 5px;
        }}
        QScrollBar::handle:horizontal {{
            background: {BORDER};
            min-width: 30px;
            border-radius: 5px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {PINK};
        }}
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {{
            width: 0;
            background: none;
            border: none;
        }}
        QScrollBar::add-page:horizontal,
        QScrollBar::sub-page:horizontal {{
            background: none;
        }}

        /* --- Group boxes --- */
        QGroupBox {{
            border: 1px solid {BORDER};
            border-top: 2px solid {PURPLE};
            border-radius: {RADIUS}px;
            margin-top: 16px;
            padding-top: 24px;
            font-weight: bold;
            color: {OFFWHITE};
            background-color: transparent;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 2px 10px;
            color: {PURPLE};
        }}

        /* --- Tab widget / tab bar --- */
        QTabWidget::pane {{
            border: none;
            background-color: {INDIGO};
        }}
        QTabBar {{
            background-color: {PANEL};
            qproperty-drawBase: 0;
        }}
        QTabBar::tab {{
            background-color: {PANEL};
            color: {TEXT_MUTED};
            border: none;
            border-bottom: 3px solid transparent;
            padding: 10px 24px;
            font-size: 13px;
            min-width: 80px;
        }}
        QTabBar::tab:hover {{
            color: {OFFWHITE};
            background-color: rgba(228, 145, 201, 0.10);  /* PINK @ 10% */
        }}
        QTabBar::tab:selected {{
            border-bottom: 3px solid {PURPLE};
            color: {OFFWHITE};
            font-weight: bold;
        }}

        /* --- Menu bar --- */
        QMenuBar {{
            background-color: {PANEL};
            color: {OFFWHITE};
            border-bottom: 1px solid {BORDER};
            padding: 2px 4px;
            font-size: 13px;
        }}
        QMenuBar::item {{
            padding: 6px 12px;
            border-radius: {RADIUS - 2}px;
        }}
        QMenuBar::item:selected {{
            background-color: rgba(152, 37, 152, 0.20);  /* PURPLE @ 20% */
            color: {OFFWHITE};
        }}
        QMenu {{
            background-color: {PANEL};
            color: {OFFWHITE};
            border: 1px solid {BORDER};
            border-radius: {RADIUS}px;
            padding: 4px 0;
        }}
        QMenu::item {{
            padding: 8px 28px 8px 20px;
        }}
        QMenu::item:selected {{
            background-color: {PURPLE};
            color: {OFFWHITE};
            border-radius: {RADIUS - 2}px;
            margin: 0 4px;
        }}
        QMenu::separator {{
            height: 1px;
            background: {BORDER};
            margin: 6px 12px;
        }}

        /* --- Message boxes --- */
        QMessageBox {{
            background-color: {INDIGO};
        }}
        QMessageBox QLabel {{
            color: {OFFWHITE};
        }}

        /* --- Buttons (C13: one primary style, secondary outlined) --- */
        QPushButton {{
            background-color: {PURPLE};
            color: {OFFWHITE};
            border: 1px solid {PURPLE};
            border-radius: {RADIUS}px;
            padding: 6px 14px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {PINK};
            border-color: {PINK};
            color: {INDIGO};
        }}
        QPushButton:pressed {{
            background-color: {PANEL_HI};
            border-color: {PANEL_HI};
            color: {OFFWHITE};
        }}
        QPushButton:disabled {{
            background-color: {BORDER};
            color: {TEXT_MUTED};
            border-color: {BORDER};
        }}

        /* --- Inputs --- */
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
            background-color: {PANEL};
            color: {OFFWHITE};
            border: 1px solid {BORDER};
            border-radius: {RADIUS - 2}px;
            padding: 4px 8px;
            selection-background-color: {PURPLE};
            selection-color: {OFFWHITE};
        }}
        QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {PINK};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 18px;
        }}

        /* --- Tables / tree / list views --- */
        QTableView, QTreeView, QListView {{
            background-color: {PANEL};
            alternate-background-color: {INDIGO};
            gridline-color: {BORDER};
            border: 1px solid {BORDER};
            border-radius: {RADIUS - 2}px;
            selection-background-color: {PURPLE};
            selection-color: {OFFWHITE};
        }}
        QHeaderView::section {{
            background-color: {PANEL};
            color: {PURPLE};
            border: none;
            border-bottom: 1px solid {BORDER};
            padding: 6px;
            font-weight: bold;
        }}

        /* --- Progress bars --- */
        QProgressBar {{
            background-color: {BORDER};
            border: none;
            border-radius: 3px;
            text-align: center;
            color: {OFFWHITE};
            min-height: 6px;
        }}
        QProgressBar::chunk {{
            background-color: {PURPLE};
            border-radius: 3px;
        }}

        /* --- Checkbox / radio --- */
        QCheckBox, QRadioButton {{
            color: {OFFWHITE};
            spacing: 6px;
        }}
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 14px;
            height: 14px;
            border: 1px solid {BORDER};
            background: {PANEL};
        }}
        QCheckBox::indicator {{ border-radius: 3px; }}
        QRadioButton::indicator {{ border-radius: 7px; }}
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background: {PURPLE};
            border-color: {PURPLE};
        }}
        QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
            border-color: {PINK};
        }}
    """)

    # First-run GPU acceleration (Windows + NVIDIA only). This only DOWNLOADS a
    # GPU torch and asks for a restart — runtime_hook.py selects the active
    # torch at the START of the next launch, so the in-session import order
    # doesn't matter here. Best-effort: the app always works on the baked CPU
    # torch if this is skipped or fails.
    try:
        from cctv_yolo import gpu_runtime
        offer, variant = gpu_runtime.should_offer()
        if offer and variant:
            from cctv_yolo.gpu_setup_dialog import maybe_run_gpu_setup
            maybe_run_gpu_setup(variant, parent=None)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("GPU setup check failed")

    # Ensure data directories exist
    dm = DataManager()

    # Create and show main window
    window = MainWindow(dm)
    window.show()

    # PRD C7 — first-run wizard. Only fires when the marker file is absent.
    try:
        from cctv_yolo.first_run import is_first_run, FirstRunWizard
        if is_first_run():
            wiz = FirstRunWizard(dm, parent=window)
            # Modal QWizard event loop.
            getattr(wiz, "exec")()
    except Exception:
        # The wizard is best-effort. If it can't open (e.g. headless),
        # don't let it block the rest of the app from launching.
        import logging
        logging.getLogger(__name__).exception("First-run wizard failed")

    sys.exit(app.exec())


def _write_crash_log(exc_text: str) -> Path:
    """Persist a crash traceback so the user can see what blew up.

    The exe runs with console=True on Windows, but if the user double-clicks
    they may miss the console flash — the log is the durable record.
    """
    try:
        from cctv_yolo.paths import get_crash_log
        log = get_crash_log()
    except Exception:
        # Worst case (paths.py failed to import for some reason) — fall
        # back to a stable known location so we still record the crash.
        log = Path.home() / "Documents" / "cctv-yolo" / "crash.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(exc_text)
    return log


def _show_crash_dialog(message: str) -> None:
    """Best-effort GUI popup so a double-clicked exe still tells the user."""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        instance = QApplication.instance()
        if instance is None:
            instance = QApplication(sys.argv)
        QMessageBox.critical(None, "CCTV-YOLO crashed", message)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        log_path = _write_crash_log(tb)
        _show_crash_dialog(f"{tb}\n\nFull log: {log_path}")
        print(tb, file=sys.stderr)
        print(f"\nCrash log written to: {log_path}", file=sys.stderr)
        sys.exit(1)
