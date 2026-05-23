"""PresetPicker — small combo widget for saving / loading processing presets.

PRD D2-8: lets the user name a (model, confidence, frame_skip) bundle and
recall it later. Presets live in ``config/presets.json`` under the data root.

Usage::

    picker = PresetPicker(presets_path)
    picker.preset_loaded.connect(self._on_preset_loaded)
    # When the user clicks "Save current…" the picker calls ``current_provider()``
    # to get the dict to store:
    picker.set_current_provider(lambda: {
        "model": self.model_combo.currentText(),
        "conf": self.conf_slider.value() / 100.0,
        "frame_skip": self.skip_spin.value(),
    })
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)

from cctv_yolo.theme import BORDER, OFFWHITE, PANEL, PURPLE

logger = logging.getLogger(__name__)


_COMBO_STYLE = f"""
QComboBox {{
    background-color: {PANEL};
    color: {OFFWHITE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    min-width: 140px;
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    color: {OFFWHITE};
    border: 1px solid {BORDER};
    selection-background-color: {PURPLE};
    selection-color: {OFFWHITE};
}}
"""

_BTN_STYLE = f"""
QPushButton {{
    background-color: transparent;
    color: {PURPLE};
    border: 1px solid {PURPLE};
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11px;
}}
QPushButton:hover {{
    background-color: {PURPLE};
    color: {OFFWHITE};
}}
"""


_SAVE_ENTRY = "+ Save current…"
_DEFAULT_NAME = "Default"


class PresetPicker(QWidget):
    """Combo + save/delete buttons for processing presets."""

    preset_loaded = Signal(dict)  # emitted with the loaded preset payload

    def __init__(self, presets_path: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._presets_path = Path(presets_path)
        self._current_provider: Optional[Callable[[], dict]] = None
        self._presets: dict[str, dict] = {}
        self._setup_ui()
        self._load_from_disk()
        self._populate_combo()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Preset:"))

        self.combo = QComboBox()
        self.combo.setStyleSheet(_COMBO_STYLE)
        self.combo.activated.connect(self._on_combo_activated)
        layout.addWidget(self.combo)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet(_BTN_STYLE)
        self.btn_delete.setToolTip("Delete the selected preset")
        self.btn_delete.clicked.connect(self._on_delete)
        layout.addWidget(self.btn_delete)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_current_provider(self, provider: Callable[[], dict]) -> None:
        """Provide a callable that returns the current settings as a dict."""
        self._current_provider = provider

    def preset_names(self) -> list[str]:
        return sorted(self._presets.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        if not self._presets_path.exists():
            self._presets = {}
            return
        try:
            with open(self._presets_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._presets = {
                    str(k): v for k, v in data.items() if isinstance(v, dict)
                }
            else:
                self._presets = {}
        except Exception as e:
            logger.warning("Failed to read presets %s: %s", self._presets_path, e)
            self._presets = {}

    def _save_to_disk(self) -> None:
        try:
            self._presets_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._presets_path, "w", encoding="utf-8") as f:
                json.dump(self._presets, f, indent=2)
        except Exception as e:
            logger.error("Failed to write presets %s: %s", self._presets_path, e)

    # ------------------------------------------------------------------
    # Combo population
    # ------------------------------------------------------------------

    def _populate_combo(self) -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        # Always start with Default
        self.combo.addItem(_DEFAULT_NAME)
        for name in sorted(self._presets.keys()):
            if name == _DEFAULT_NAME:
                continue
            self.combo.addItem(name)
        self.combo.insertSeparator(self.combo.count())
        self.combo.addItem(_SAVE_ENTRY)
        self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_combo_activated(self, idx: int) -> None:
        text = self.combo.itemText(idx)
        if text == _SAVE_ENTRY:
            self._on_save_current()
            return
        if text in self._presets:
            self.preset_loaded.emit(dict(self._presets[text]))

    def _on_save_current(self) -> None:
        if self._current_provider is None:
            QMessageBox.warning(self, "No data", "No current settings to save.")
            self._populate_combo()
            return
        name, ok = QInputDialog.getText(
            self, "Save preset", "Preset name:"
        )
        if not ok or not name.strip():
            self._populate_combo()
            return
        name = name.strip()
        if name == _SAVE_ENTRY or name == "":
            self._populate_combo()
            return
        if name in self._presets:
            reply = QMessageBox.question(
                self,
                "Overwrite preset",
                f"A preset named '{name}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self._populate_combo()
                return
        try:
            payload = self._current_provider() or {}
        except Exception as e:
            logger.error("Preset provider failed: %s", e)
            payload = {}
        self._presets[name] = payload
        self._save_to_disk()
        self._populate_combo()
        # Select the just-saved entry
        idx = self.combo.findText(name)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def _on_delete(self) -> None:
        name = self.combo.currentText()
        if name in (_DEFAULT_NAME, _SAVE_ENTRY, ""):
            QMessageBox.information(self, "Delete preset", "Select a saved preset first.")
            return
        if name not in self._presets:
            return
        reply = QMessageBox.question(
            self,
            "Delete preset",
            f"Delete preset '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._presets.pop(name, None)
        self._save_to_disk()
        self._populate_combo()
