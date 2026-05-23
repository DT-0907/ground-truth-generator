"""CollapsibleSection — minimalistic disclosure panel with PURPLE underline.

Used in the Performance tab to wrap each Analyses sub-panel (Model A/B compare,
Before/After renderer, Confusion matrix) and the "Sessions in this group"
membership list. Click the header to toggle the body in/out.

PRD C11 — section underlines use PURPLE; selection PURPLE@30%; hover PINK@15%.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cctv_yolo.theme import (
    BORDER,
    OFFWHITE,
    PANEL,
    PINK,
    PURPLE,
    RADIUS,
    TEXT_MUTED,
)


_TOGGLE_STYLE = f"""
QPushButton {{
    background-color: transparent;
    color: {OFFWHITE};
    border: none;
    border-bottom: 1px solid {PURPLE};
    border-radius: 0;
    padding: 6px 4px 6px 4px;
    text-align: left;
    font-size: 14px;
    font-weight: 600;
}}
QPushButton:hover {{
    color: {PINK};
    background-color: rgba(228, 145, 201, 0.10);
}}
QPushButton:pressed {{
    color: {PURPLE};
}}
"""

_BODY_STYLE = f"""
QFrame#collapsibleBody {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-top: none;
    border-radius: 0 0 {RADIUS}px {RADIUS}px;
    padding: 12px;
}}
"""


class CollapsibleSection(QWidget):
    """A header button + an optional body widget that hides/shows on click.

    Usage:
        sec = CollapsibleSection("Model A/B Compare")
        sec.set_body(some_widget)
        sec.toggled.connect(on_toggle)
        layout.addWidget(sec)

    The header underline is PURPLE; expand/collapse is indicated by the
    leading triangle in the button text.
    """

    toggled = Signal(bool)  # emitted with the new expanded state

    def __init__(self, title: str, expanded: bool = True,
                 subtitle: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._expanded = expanded
        self._title = title

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Header row -------------------------------------------------
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        self._btn = QPushButton()
        self._btn.setStyleSheet(_TOGGLE_STYLE)
        self._btn.setCursor(Qt.PointingHandCursor)
        self._btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn.clicked.connect(self._toggle)
        header_row.addWidget(self._btn, stretch=1)

        self._subtitle_label: QLabel | None = None
        if subtitle:
            self._subtitle_label = QLabel(subtitle)
            self._subtitle_label.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 11px; padding-right: 4px;"
            )
            header_row.addWidget(self._subtitle_label, 0, Qt.AlignRight)

        outer.addLayout(header_row)

        # --- Body --------------------------------------------------------
        self._body_frame = QFrame()
        self._body_frame.setObjectName("collapsibleBody")
        self._body_frame.setStyleSheet(_BODY_STYLE)
        self._body_layout = QVBoxLayout(self._body_frame)
        self._body_layout.setContentsMargins(12, 12, 12, 12)
        self._body_layout.setSpacing(8)

        outer.addWidget(self._body_frame)

        self._update_header()
        self._body_frame.setVisible(self._expanded)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_body(self, widget: QWidget) -> None:
        """Replace the body with a single widget. Removes any existing children."""
        # Clear existing
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if widget is not None:
            self._body_layout.addWidget(widget)

    def add_body_widget(self, widget: QWidget) -> None:
        """Append a widget into the body layout."""
        self._body_layout.addWidget(widget)

    def body_layout(self) -> QVBoxLayout:
        """Return the body layout so callers can compose their own contents."""
        return self._body_layout

    def set_subtitle(self, text: str | None) -> None:
        if self._subtitle_label is None:
            return
        self._subtitle_label.setText(text or "")

    def set_expanded(self, expanded: bool) -> None:
        if self._expanded == expanded:
            return
        self._expanded = expanded
        self._body_frame.setVisible(expanded)
        self._update_header()
        self.toggled.emit(expanded)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_title(self, title: str) -> None:
        self._title = title
        self._update_header()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def _update_header(self) -> None:
        arrow = "▾" if self._expanded else "▸"
        self._btn.setText(f"{arrow}  {self._title}")
