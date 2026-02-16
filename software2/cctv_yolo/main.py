"""
Entry point for the CCTV-YOLO native desktop application.

Creates the QApplication, applies the dark Fusion palette,
ensures data directories exist, and launches the main window.
"""
import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

from cctv_yolo.data_manager import DataManager
from cctv_yolo.main_window import MainWindow

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
BG = "#1a1a2e"
PANEL = "#16213e"
BORDER = "#2d3a5a"
ACCENT = "#4ecca3"
TEXT = "#eeeeee"


def _make_dark_palette():
    """Build a dark QPalette based on the project color scheme."""
    palette = QPalette()

    bg = QColor(BG)
    panel = QColor(PANEL)
    border = QColor(BORDER)
    accent = QColor(ACCENT)
    text = QColor(TEXT)
    dark_text = QColor("#000000")
    disabled_text = QColor("#666666")
    bright_text = QColor("#ffffff")

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
    palette.setColor(QPalette.Link, accent)
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
    app = QApplication(sys.argv)
    app.setApplicationName("CCTV-YOLO")
    app.setOrganizationName("CCTV-YOLO")
    app.setApplicationVersion("2.0.0")

    # Apply dark Fusion style
    app.setStyle("Fusion")
    app.setPalette(_make_dark_palette())

    # Global stylesheet for widgets that Fusion palette doesn't fully cover
    app.setStyleSheet(f"""
        /* --- Tooltips --- */
        QToolTip {{
            background-color: {PANEL};
            color: {TEXT};
            border: 1px solid {ACCENT};
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
        }}

        /* --- Scroll bars --- */
        QScrollBar:vertical {{
            background: {BG};
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
            background: {ACCENT};
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
            background: {BG};
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
            background: {ACCENT};
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
            border-top: 2px solid {ACCENT};
            border-radius: 6px;
            margin-top: 14px;
            padding-top: 24px;
            font-weight: bold;
            color: {TEXT};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 2px 10px;
            color: {ACCENT};
        }}

        /* --- Tab widget / tab bar --- */
        QTabWidget::pane {{
            border: none;
            background-color: {BG};
        }}
        QTabBar {{
            background-color: {PANEL};
            qproperty-drawBase: 0;
        }}
        QTabBar::tab {{
            background-color: {PANEL};
            color: #aaaaaa;
            border: none;
            border-bottom: 3px solid transparent;
            padding: 10px 24px;
            font-size: 13px;
            min-width: 80px;
        }}
        QTabBar::tab:hover {{
            color: {TEXT};
            background-color: rgba(78, 204, 163, 0.08);
        }}
        QTabBar::tab:selected {{
            border-bottom: 3px solid {ACCENT};
            color: {ACCENT};
            font-weight: bold;
        }}

        /* --- Menu bar --- */
        QMenuBar {{
            background-color: {PANEL};
            color: {TEXT};
            border-bottom: 1px solid {BORDER};
            padding: 2px 4px;
            font-size: 13px;
        }}
        QMenuBar::item {{
            padding: 6px 12px;
            border-radius: 4px;
        }}
        QMenuBar::item:selected {{
            background-color: rgba(78, 204, 163, 0.15);
            color: {ACCENT};
        }}
        QMenu {{
            background-color: {PANEL};
            color: {TEXT};
            border: 1px solid {BORDER};
            border-radius: 6px;
            padding: 4px 0;
        }}
        QMenu::item {{
            padding: 8px 28px 8px 20px;
        }}
        QMenu::item:selected {{
            background-color: {ACCENT};
            color: #000;
            border-radius: 4px;
            margin: 0 4px;
        }}
        QMenu::separator {{
            height: 1px;
            background: {BORDER};
            margin: 6px 12px;
        }}

        /* --- Message boxes --- */
        QMessageBox {{
            background-color: {BG};
        }}
        QMessageBox QLabel {{
            color: {TEXT};
        }}
    """)

    # Ensure data directories exist
    dm = DataManager()

    # Create and show main window
    window = MainWindow(dm)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
