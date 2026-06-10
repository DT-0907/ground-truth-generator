"""
Batch tree view — QTreeView + QFileSystemModel with a status-badge column.

PRD E5-1 / E5-2: present a folder's videos in their on-disk hierarchy with
per-row processing status rendered as a coloured chip.

Architecture
------------
``VideoFilterFSModel``  — QFileSystemModel subclass that hides non-video files
                          and exposes a synthetic "Status" column.
``StatusBadgeDelegate`` — QStyledItemDelegate that paints the status chip
                          using the theme palette.
``BatchTree``           — QWidget wrapping the tree + model, exposing a
                          ``status_provider`` callable so the BatchTab can
                          feed it live state (queued/processing/done/error)
                          per absolute path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QDir, QModelIndex, QRect, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QFileSystemModel,
    QHeaderView,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from cctv_yolo.theme import (
    BORDER,
    INDIGO,
    OFFWHITE,
    PANEL,
    PANEL_HI,
    PINK,
    PURPLE,
    RADIUS,
    STATUS_HOVER_OVERLAY,
    STATUS_SELECTED_OVERLAY,
    TEXT_MUTED,
    status_colors,
)


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}

# Synthetic status column index — appended after the 4 default FS columns
# (Name, Size, Type, Date Modified).
STATUS_COLUMN = 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VideoFilterFSModel(QFileSystemModel):
    """File-system model that hides non-video files and adds a status column.

    Lazy-loading is inherited from QFileSystemModel: directory contents are
    only listed when the user expands a node, so even huge folders open
    instantly.
    """

    def __init__(self, status_provider: Optional[Callable[[str], str]] = None, parent=None):
        super().__init__(parent)
        self._status_provider = status_provider or (lambda _abs: "")
        # Show files + folders, no dot dirs.
        self.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)
        # Only match video extensions for files. Folders ignore this filter
        # so the tree can be drilled into.
        self.setNameFilters([f"*{ext}" for ext in sorted({e.lower() for e in VIDEO_EXTS})])
        self.setNameFilterDisables(False)  # actually hide non-matches
        self.setReadOnly(True)

    def set_status_provider(self, provider: Callable[[str], str]):
        self._status_provider = provider

    def columnCount(self, parent=QModelIndex()) -> int:
        # Default 4 + our synthetic Status column.
        return super().columnCount(parent) + 1

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == STATUS_COLUMN:
                return "Status"
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if index.column() == STATUS_COLUMN:
            # Look up the status for this row's absolute path. Folders just
            # show nothing.
            name_index = self.index(index.row(), 0, index.parent())
            info = self.fileInfo(name_index)
            if info.isDir():
                return None
            abs_path = info.absoluteFilePath()
            if role in (Qt.DisplayRole, Qt.EditRole, Qt.UserRole):
                return self._status_provider(abs_path) or "queued"
            if role == Qt.TextAlignmentRole:
                return Qt.AlignCenter
            return None
        return super().data(index, role)

    def flags(self, index):
        f = super().flags(index)
        # Synthetic status column is not selectable on its own — the name
        # column drives selection so the user clicks the file, not the chip.
        if index.column() == STATUS_COLUMN:
            f &= ~Qt.ItemIsEditable
        return f


# ---------------------------------------------------------------------------
# Delegate
# ---------------------------------------------------------------------------

class StatusBadgeDelegate(QStyledItemDelegate):
    """Paints the synthetic Status column as a rounded chip.

    Colors come from ``theme.status_colors(state)``. No raw hex here.
    """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        if index.column() != STATUS_COLUMN:
            super().paint(painter, option, index)
            return

        state = (index.data(Qt.DisplayRole) or "").lower()
        if not state:
            # Folder row — paint nothing (no chip)
            return

        bg, fg = status_colors(state)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Slight inset so chip doesn't touch row edges.
        rect: QRect = option.rect.adjusted(6, 4, -6, -4)

        # Background chip (handle the "transparent" string for queued/idle)
        if bg and bg != "transparent":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(bg)))
            painter.drawRoundedRect(rect, RADIUS - 2, RADIUS - 2)
        else:
            # Outlined chip for queued state — keeps the row light.
            painter.setPen(QColor(BORDER))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect, RADIUS - 2, RADIUS - 2)

        # Label
        painter.setPen(QColor(fg))
        font: QFont = painter.font()
        font.setPointSize(max(8, font.pointSize() - 1))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, state.upper())

        painter.restore()

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        # Make sure the chip has room horizontally.
        s.setWidth(max(s.width(), 96))
        return s


# ---------------------------------------------------------------------------
# Tree widget
# ---------------------------------------------------------------------------

class BatchTree(QWidget):
    """Lightweight wrapper around a QTreeView showing one folder's videos.

    Signals
    -------
    file_activated(str)
        Emitted when the user double-clicks a video row (absolute path).
    selection_changed(list[str])
        Emitted with the list of currently-selected video absolute paths.
    expansion_changed(list[str])
        Emitted with the list of absolute paths of expanded folders. Used
        by BatchTab to persist UI state per-folder.
    """

    file_activated = Signal(str)
    selection_changed = Signal(list)
    expansion_changed = Signal(list)

    def __init__(self, status_provider: Optional[Callable[[str], str]] = None, parent=None):
        super().__init__(parent)
        self._model = VideoFilterFSModel(status_provider=status_provider)
        self._root_path: Optional[Path] = None

        self.view = QTreeView(self)
        self.view.setModel(self._model)
        self.view.setAnimated(False)
        self.view.setIndentation(16)
        self.view.setSortingEnabled(True)
        self.view.sortByColumn(0, Qt.AscendingOrder)
        self.view.setSelectionMode(QTreeView.ExtendedSelection)
        self.view.setUniformRowHeights(True)
        self.view.setAlternatingRowColors(False)
        self.view.setRootIsDecorated(True)
        self.view.setExpandsOnDoubleClick(False)
        self.view.setItemDelegateForColumn(STATUS_COLUMN, StatusBadgeDelegate(self.view))
        self.view.setStyleSheet(self._tree_qss())

        header = self.view.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        # Hide Size / Type / Date — they're noise for this use-case.
        for c in (1, 2, 3):
            self.view.setColumnHidden(c, True)
        header.setSectionResizeMode(STATUS_COLUMN, QHeaderView.Fixed)
        header.resizeSection(STATUS_COLUMN, 120)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        self.view.doubleClicked.connect(self._on_double_clicked)
        self.view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.view.expanded.connect(self._on_expansion_changed)
        self.view.collapsed.connect(self._on_expansion_changed)

    # ----- Public API -----
    def set_root(self, folder: Path | str | None):
        """Point the tree at a new source folder, or clear it if None."""
        if folder is None:
            self._root_path = None
            self._model.setRootPath("")
            self.view.setRootIndex(QModelIndex())
            return
        p = Path(folder)
        self._root_path = p
        root_index = self._model.setRootPath(str(p))
        self.view.setRootIndex(root_index)

    def root(self) -> Optional[Path]:
        return self._root_path

    def set_status_provider(self, provider: Callable[[str], str]):
        self._model.set_status_provider(provider)

    def selected_video_paths(self) -> list[str]:
        paths: list[str] = []
        for idx in self.view.selectionModel().selectedRows(0):
            info = self._model.fileInfo(idx)
            if info.isFile():
                paths.append(info.absoluteFilePath())
        return paths

    def refresh_statuses(self):
        """Repaint the Status column for every currently-visible row.

        The synthetic Status column is derived from live queue state the FS
        model knows nothing about, so we must tell the view its data changed.
        We walk each visible row (top → bottom, following expansion) and emit a
        per-row ``dataChanged``. Emitting one ``top → bottom`` range is wrong
        when the visible rows span multiple parent folders — ``dataChanged``
        requires topLeft/bottomRight to share a parent, so the range silently
        no-ops and chips stay "QUEUED". That is exactly the case here: videos
        live in nested ``loop N/`` subfolders, so each visible run of rows has a
        different parent.
        """
        view = self.view
        viewport_bottom = view.viewport().rect().bottom()
        idx = view.indexAt(view.viewport().rect().topLeft())
        guard = 0
        while idx.isValid() and guard < 100000:
            guard += 1
            status_idx = self._model.index(idx.row(), STATUS_COLUMN, idx.parent())
            if status_idx.isValid():
                self._model.dataChanged.emit(status_idx, status_idx, [Qt.DisplayRole])
            # Stop once we've painted past the bottom of the viewport.
            if view.visualRect(idx).top() > viewport_bottom:
                break
            idx = view.indexBelow(idx)

    def expanded_paths(self) -> list[str]:
        """Currently-expanded folder absolute paths (best-effort)."""
        out: list[str] = []
        if self._root_path is None:
            return out
        root_index = self._model.index(str(self._root_path))

        def walk(parent_index):
            rows = self._model.rowCount(parent_index)
            for r in range(rows):
                child = self._model.index(r, 0, parent_index)
                if not child.isValid():
                    continue
                if self._model.isDir(child) and self.view.isExpanded(child):
                    out.append(self._model.fileInfo(child).absoluteFilePath())
                    walk(child)

        walk(root_index)
        return out

    def restore_expanded_paths(self, paths: list[str]):
        for p in paths or []:
            idx = self._model.index(p)
            if idx.isValid():
                # Expand each ancestor up to the root so the path is visible.
                self.view.setExpanded(idx, True)

    # ----- Signal plumbing -----
    def _on_double_clicked(self, idx: QModelIndex):
        info = self._model.fileInfo(self._model.index(idx.row(), 0, idx.parent()))
        if info.isFile():
            self.file_activated.emit(info.absoluteFilePath())
        else:
            # Folders: just toggle expansion.
            self.view.setExpanded(idx, not self.view.isExpanded(idx))

    def _on_selection_changed(self, *_):
        self.selection_changed.emit(self.selected_video_paths())

    def _on_expansion_changed(self, *_):
        self.expansion_changed.emit(self.expanded_paths())

    # ----- Style -----
    @staticmethod
    def _tree_qss() -> str:
        return f"""
        QTreeView {{
            background-color: {PANEL};
            alternate-background-color: {PANEL_HI};
            color: {OFFWHITE};
            border: 1px solid {BORDER};
            border-radius: {RADIUS}px;
            padding: 4px;
            outline: 0;
            selection-background-color: {STATUS_SELECTED_OVERLAY[0]};
            selection-color: {OFFWHITE};
        }}
        QTreeView::item {{
            padding: 4px 6px;
            border: 0;
        }}
        QTreeView::item:hover {{
            background-color: {STATUS_HOVER_OVERLAY[0]};
        }}
        QTreeView::item:selected {{
            background-color: {STATUS_SELECTED_OVERLAY[0]};
            color: {OFFWHITE};
        }}
        QHeaderView::section {{
            background-color: {INDIGO};
            color: {PINK};
            border: 0;
            border-bottom: 1px solid {BORDER};
            padding: 6px 8px;
            font-weight: bold;
        }}
        QTreeView::branch:has-siblings:!adjoins-item,
        QTreeView::branch:has-siblings:adjoins-item,
        QTreeView::branch:!has-children:!has-siblings:adjoins-item {{
            border-image: none;
        }}
        QScrollBar:vertical {{
            background: {PANEL};
            width: 10px;
            border: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {BORDER};
            border-radius: 4px;
            min-height: 24px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {PURPLE}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        """


__all__ = [
    "BatchTree",
    "StatusBadgeDelegate",
    "VideoFilterFSModel",
    "STATUS_COLUMN",
    "VIDEO_EXTS",
]
