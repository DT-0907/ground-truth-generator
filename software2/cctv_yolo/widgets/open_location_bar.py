"""OpenLocationBar — small button row that opens files/folders in Finder/Explorer.

PRD C12: every tab in the app gets one of these in its top-right corner so the
user can always jump straight to "where is this stuff stored?" with one click.

Each button is a small outlined QPushButton with an emoji icon + label. Tooltip
shows the absolute path that'll open. Buttons can either:

- Open a folder (``add_folder``)
- Open a folder AND highlight a specific file in it (``add_file``)

Cross-platform behavior:
- macOS  → ``open -R <file>``  (reveal & select in Finder), or ``open <dir>``
- Windows → ``explorer /select,<file>``, or ``explorer <dir>``
- Linux   → ``xdg-open <dir>`` (file-select isn't a standard concept)
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from cctv_yolo.theme import BORDER, OFFWHITE, PINK, PURPLE, RADIUS, TEXT_MUTED

logger = logging.getLogger(__name__)

# A path source can be a Path (resolved now) or a callable returning a Path
# (resolved at click time — useful when the path depends on UI state, e.g.
# "the currently active batch folder").
PathSource = Union[Path, str, Callable[[], Union[Path, str, None]]]


_BUTTON_STYLE = f"""
    QPushButton {{
        background-color: transparent;
        color: {OFFWHITE};
        border: 1px solid {BORDER};
        border-radius: {RADIUS - 2}px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: normal;
        min-height: 22px;
    }}
    QPushButton:hover {{
        color: {PINK};
        border-color: {PINK};
    }}
    QPushButton:pressed {{
        color: {PURPLE};
        border-color: {PURPLE};
    }}
    QPushButton:disabled {{
        color: {TEXT_MUTED};
        border-color: {BORDER};
    }}
"""


def _resolve(source: PathSource) -> Path | None:
    """Resolve a PathSource to a concrete Path (or None if the callable returns None)."""
    if callable(source):
        try:
            source = source()
        except Exception as e:
            logger.warning("OpenLocationBar callable failed: %s", e)
            return None
    if source is None:
        return None
    return Path(source)


def open_path(path: Path | str, *, select: bool = False) -> None:
    """Cross-platform open. If ``select=True`` and ``path`` is a file, opens
    its parent folder with the file highlighted."""
    path = Path(path)
    target = path
    use_select = select and path.is_file()

    # In a windowed (console=False) frozen build the parent's std handles are
    # invalid; a subprocess that inherits them raises WinError 6 and the
    # Explorer/Finder window never opens. Hand every child explicit DEVNULL
    # streams so it never reaches for the missing console.
    _quiet = dict(stdin=subprocess.DEVNULL,
                  stdout=subprocess.DEVNULL,
                  stderr=subprocess.DEVNULL)

    try:
        if sys.platform == "darwin":
            if use_select:
                subprocess.Popen(["open", "-R", str(target)], **_quiet)
            else:
                # If the path doesn't exist yet, fall back to parent
                if not target.exists() and target.parent.exists():
                    target = target.parent
                subprocess.Popen(["open", str(target)], **_quiet)
        elif sys.platform == "win32":
            if use_select:
                # explorer /select, requires a comma immediately after the flag
                subprocess.Popen(["explorer", f"/select,{target}"], **_quiet)
            else:
                if not target.exists() and target.parent.exists():
                    target = target.parent
                subprocess.Popen(["explorer", str(target)], **_quiet)
        else:
            # Linux: xdg-open doesn't support select; open the parent folder
            if use_select:
                target = target.parent
            elif not target.exists() and target.parent.exists():
                target = target.parent
            subprocess.Popen(["xdg-open", str(target)], **_quiet)
    except (FileNotFoundError, OSError) as e:
        logger.error("Couldn't open %s: %s", target, e)


class OpenLocationBar(QWidget):
    """Horizontal row of small icon-text buttons for opening files/folders.

    Add buttons via ``add_folder()`` / ``add_file()``. Each one becomes a
    small chip. Use this in every tab's header row.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._layout.addStretch(1)  # push buttons to the right

    def _add_button(
        self,
        label: str,
        source: PathSource,
        *,
        icon: str = "📁",
        select_file: bool = False,
    ) -> QPushButton:
        btn = QPushButton(f"{icon}  {label}")
        btn.setStyleSheet(_BUTTON_STYLE)
        btn.setCursor(Qt.PointingHandCursor)

        def _on_click():
            target = _resolve(source)
            if target is None:
                logger.info("OpenLocationBar: %s has no path", label)
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            open_path(target, select=select_file)

        def _update_tooltip():
            target = _resolve(source)
            btn.setToolTip(str(target) if target else f"{label} (no path set)")

        btn.clicked.connect(_on_click)
        btn.installEventFilter(self)  # for hover-to-update-tooltip

        # Resolve tooltip lazily on click; set an initial one too.
        _update_tooltip()

        # Insert before the trailing stretch
        self._layout.insertWidget(self._layout.count() - 1, btn)
        return btn

    def add_folder(self, label: str, source: PathSource, *, icon: str = "📁") -> QPushButton:
        """Add a button that opens a folder."""
        return self._add_button(label, source, icon=icon, select_file=False)

    def add_file(self, label: str, source: PathSource, *, icon: str = "🎯") -> QPushButton:
        """Add a button that opens a file's parent folder with the file selected."""
        return self._add_button(label, source, icon=icon, select_file=True)

    def clear(self) -> None:
        """Remove all buttons (keeps the trailing stretch)."""
        # Remove every widget except the trailing stretch (last item)
        for i in reversed(range(self._layout.count() - 1)):
            item = self._layout.takeAt(i)
            if item and item.widget():
                item.widget().deleteLater()
