"""GroupPickerDialog — multi-select session picker for adding to a group.

PRD Part M / Performance G4: when the user clicks "+ Add Sessions" inside a
group, this dialog lists every session in the workspace with a search box
and checkbox per row. Returns the list of selected session_ids on accept.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from cctv_yolo.theme import (
    BORDER,
    INDIGO,
    OFFWHITE,
    PANEL,
    PINK,
    PURPLE,
    RADIUS,
    TEXT_MUTED,
)


_DIALOG_STYLE = f"""
QDialog {{
    background-color: {INDIGO};
    color: {OFFWHITE};
}}
QLabel {{
    color: {OFFWHITE};
    background: transparent;
}}
QLineEdit {{
    background-color: {PANEL};
    color: {OFFWHITE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS - 2}px;
    padding: 6px 10px;
    font-size: 13px;
}}
QLineEdit:focus {{
    border-color: {PURPLE};
}}
QListWidget {{
    background-color: {PANEL};
    color: {OFFWHITE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    padding: 4px;
    font-size: 13px;
    outline: 0;
}}
QListWidget::item {{
    padding: 6px 8px;
    border-radius: 4px;
}}
QListWidget::item:hover {{
    background-color: rgba(228, 145, 201, 0.15);
}}
QListWidget::item:selected {{
    background-color: rgba(152, 37, 152, 0.30);
    color: {OFFWHITE};
}}
QPushButton {{
    background-color: transparent;
    color: {PURPLE};
    border: 1px solid {PURPLE};
    border-radius: {RADIUS - 2}px;
    padding: 6px 16px;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: {PURPLE};
    color: {OFFWHITE};
}}
QPushButton:default {{
    background-color: {PURPLE};
    color: {OFFWHITE};
}}
"""


class GroupPickerDialog(QDialog):
    """Multi-select picker for sessions.

    Args:
        sessions: iterable of session dicts (must have ``id`` and ``video_name``).
        already_in: iterable of session_ids that are already in the group;
            these are shown disabled with an "(already added)" suffix.
        title: dialog title.
        parent: Qt parent.
    """

    def __init__(
        self,
        sessions,
        already_in=None,
        title: str = "Add Sessions to Group",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setStyleSheet(_DIALOG_STYLE)
        self.setMinimumSize(520, 460)

        self._sessions = list(sessions)
        self._already_in = set(already_in or [])
        self._selected_ids: list[str] = []

        self._build_ui()
        self._populate(self._sessions)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        intro = QLabel("Pick one or more sessions to add. Use the search box to filter.")
        intro.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Search row
        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search sessions by name or ID…")
        self.search_edit.textChanged.connect(self._on_search)
        search_row.addWidget(self.search_edit, stretch=1)

        self.btn_all = QPushButton("Select All")
        self.btn_all.clicked.connect(self._select_all_visible)
        search_row.addWidget(self.btn_all)

        self.btn_none = QPushButton("Clear")
        self.btn_none.clicked.connect(self._clear_selection)
        search_row.addWidget(self.btn_none)

        layout.addLayout(search_row)

        # List
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.NoSelection)
        layout.addWidget(self.list_widget, stretch=1)

        self.summary_label = QLabel("0 selected")
        self.summary_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self.summary_label)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Population + filtering
    # ------------------------------------------------------------------

    def _populate(self, sessions) -> None:
        self.list_widget.clear()
        for s in sessions:
            sid = s.get("id", "")
            label = s.get("video_name") or sid
            track_count = s.get("track_count", 0)
            suffix = f"  ({track_count} tracks)" if track_count else ""
            text = f"{label}{suffix}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, sid)

            if sid in self._already_in:
                item.setFlags(Qt.NoItemFlags)
                item.setText(f"{text}  — already in group")
                item.setCheckState(Qt.Checked)
                # Visually muted
                item.setForeground(Qt.gray)
            else:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

        self.list_widget.itemChanged.connect(self._on_item_changed)
        self._update_summary()

    def _on_search(self, text: str) -> None:
        text_l = (text or "").strip().lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            display = item.text().lower()
            sid = (item.data(Qt.UserRole) or "").lower()
            visible = (text_l in display) or (text_l in sid)
            item.setHidden(not visible)

    def _select_all_visible(self) -> None:
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.isHidden():
                continue
            if not (item.flags() & Qt.ItemIsUserCheckable):
                continue
            item.setCheckState(Qt.Checked)

    def _clear_selection(self) -> None:
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not (item.flags() & Qt.ItemIsUserCheckable):
                continue
            item.setCheckState(Qt.Unchecked)

    def _on_item_changed(self, _item) -> None:  # noqa: ANN001
        self._update_summary()

    def _update_summary(self) -> None:
        n = 0
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not (item.flags() & Qt.ItemIsUserCheckable):
                continue
            if item.checkState() == Qt.Checked:
                n += 1
        self.summary_label.setText(f"{n} selected")

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        self._selected_ids = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not (item.flags() & Qt.ItemIsUserCheckable):
                continue
            if item.checkState() == Qt.Checked:
                sid = item.data(Qt.UserRole)
                if sid:
                    self._selected_ids.append(sid)
        self.accept()

    # ------------------------------------------------------------------
    # Public accessor
    # ------------------------------------------------------------------

    def selected_session_ids(self) -> list[str]:
        return list(self._selected_ids)
