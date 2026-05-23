"""About dialog (PRD C8).

Shows app version + platform info + data/log folder locations with quick-open
buttons. Reachable from Help → About.
"""
from __future__ import annotations

import platform
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout,
    QFrame,
)

from cctv_yolo.__version__ import __version__, __app_name__
from cctv_yolo.paths import get_log_file
from cctv_yolo.theme import (
    INDIGO, PANEL, BORDER, PURPLE, PINK, OFFWHITE, TEXT_MUTED, RADIUS,
)
from cctv_yolo.widgets.open_location_bar import open_path


class AboutDialog(QDialog):
    """Read-only info dialog."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.setWindowTitle(f"About {__app_name__}")
        self.setMinimumWidth(480)
        self.setStyleSheet(
            f"QDialog {{ background: {INDIGO}; color: {OFFWHITE}; }}"
            f"QLabel {{ color: {OFFWHITE}; }}"
        )

        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 16)
        v.setSpacing(12)

        title = QLabel(__app_name__)
        title.setStyleSheet(f"color:{PINK}; font-size: 22px; font-weight: bold;")
        v.addWidget(title)

        ver = QLabel(f"Version {__version__}")
        ver.setStyleSheet(f"color:{TEXT_MUTED}; font-size: 12px;")
        v.addWidget(ver)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {BORDER}; background: {BORDER};")
        sep.setFixedHeight(1)
        v.addWidget(sep)

        # Info grid
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        rows = [
            ("Platform", f"{platform.system()} {platform.release()}"),
            ("Python", platform.python_version()),
            ("Architecture", platform.machine()),
            ("Data folder", str(self.dm.data_root)),
            ("Log file", str(get_log_file())),
        ]
        for i, (k, val) in enumerate(rows):
            kl = QLabel(k)
            kl.setStyleSheet(f"color:{TEXT_MUTED}; font-size: 12px;")
            vl = QLabel(val)
            vl.setStyleSheet(f"color:{OFFWHITE}; font-size: 12px;")
            vl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            vl.setWordWrap(True)
            grid.addWidget(kl, i, 0)
            grid.addWidget(vl, i, 1)
        v.addLayout(grid)

        # Action row
        btn_style = (
            f"QPushButton {{ background:{PURPLE}; color:{OFFWHITE}; "
            f"border:none; border-radius:{RADIUS}px; padding:6px 14px; "
            f"font-weight:bold; }}"
            f"QPushButton:hover {{ background:{PINK}; color:{INDIGO}; }}"
        )
        close_style = (
            f"QPushButton {{ background:transparent; color:{OFFWHITE}; "
            f"border:1px solid {BORDER}; border-radius:{RADIUS}px; "
            f"padding:6px 14px; }}"
            f"QPushButton:hover {{ border-color:{PINK}; color:{PINK}; }}"
        )

        h = QHBoxLayout()
        h.setSpacing(8)
        open_data_btn = QPushButton("📁  Open Data Folder")
        open_data_btn.setStyleSheet(btn_style)
        open_data_btn.clicked.connect(lambda: open_path(self.dm.data_root))
        h.addWidget(open_data_btn)

        open_log_btn = QPushButton("📁  Open Log Folder")
        open_log_btn.setStyleSheet(btn_style)
        open_log_btn.clicked.connect(
            lambda: open_path(get_log_file(), select=True)
            if get_log_file().exists() else open_path(get_log_file().parent)
        )
        h.addWidget(open_log_btn)

        h.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(close_style)
        close_btn.clicked.connect(self.accept)
        h.addWidget(close_btn)
        v.addLayout(h)
