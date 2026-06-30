"""Manage Class Sets dialog.

Lets the user create / duplicate / edit / delete custom class sets (e.g. the
FHWA 13-class scheme) on top of the built-in presets. Built-in presets are
read-only here — "Duplicate" makes an editable copy. Backed entirely by
``cctv_yolo.classes`` (config/class_sets.json).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog, QDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from cctv_yolo import classes as class_registry
from cctv_yolo.theme import BORDER, ERROR, OFFWHITE, PANEL, PURPLE, TEXT_MUTED


class ClassSetDialog(QDialog):
    """Create / edit class sets. Returns the active set id via ``selected_id``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Class Sets")
        self.resize(720, 480)
        self.setStyleSheet(f"QDialog {{ background: {PANEL}; color: {OFFWHITE}; }}")
        self._current_id: str | None = None

        root = QHBoxLayout(self)

        # --- Left: list of sets ---
        left = QVBoxLayout()
        left.addWidget(self._muted("Class sets"))
        self.list = QListWidget()
        self.list.setStyleSheet(
            f"QListWidget {{ background:{PANEL}; color:{OFFWHITE}; border:1px solid {BORDER};"
            f" border-radius:6px; }} QListWidget::item:selected {{ background:{PURPLE}; }}")
        self.list.currentItemChanged.connect(lambda *_: self._load_selected())
        left.addWidget(self.list, 1)

        btn_row = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_dup = QPushButton("Duplicate")
        self.btn_del = QPushButton("Delete")
        for b in (self.btn_new, self.btn_dup, self.btn_del):
            b.setStyleSheet(self._btn_css())
            btn_row.addWidget(b)
        self.btn_new.clicked.connect(self._new_set)
        self.btn_dup.clicked.connect(self._duplicate_set)
        self.btn_del.clicked.connect(self._delete_set)
        left.addLayout(btn_row)
        root.addLayout(left, 1)

        # --- Right: editor ---
        right = QVBoxLayout()
        name_row = QHBoxLayout()
        name_row.addWidget(self._muted("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setStyleSheet(
            f"QLineEdit {{ background:{PANEL}; color:{OFFWHITE}; border:1px solid {BORDER};"
            f" border-radius:6px; padding:6px; }}")
        name_row.addWidget(self.name_edit, 1)
        right.addLayout(name_row)

        self.builtin_note = self._muted("")
        right.addWidget(self.builtin_note)

        right.addWidget(self._muted(
            "Classes (COCO id seeds detection from a stock model; leave blank "
            "for classes only a custom-trained model detects):"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Class name", "COCO id", "Color"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(
            f"QTableWidget {{ background:{PANEL}; color:{OFFWHITE}; border:1px solid {BORDER}; }}"
            f"QHeaderView::section {{ background:{PANEL}; color:{TEXT_MUTED}; border:none; padding:4px; }}")
        self.table.cellClicked.connect(self._maybe_pick_color)
        right.addWidget(self.table, 1)

        row_btns = QHBoxLayout()
        self.btn_add_row = QPushButton("Add class")
        self.btn_del_row = QPushButton("Remove class")
        for b in (self.btn_add_row, self.btn_del_row):
            b.setStyleSheet(self._btn_css())
            row_btns.addWidget(b)
        self.btn_add_row.clicked.connect(self._add_row)
        self.btn_del_row.clicked.connect(self._remove_row)
        row_btns.addStretch()
        right.addLayout(row_btns)

        action_row = QHBoxLayout()
        self.btn_activate = QPushButton("Set as Active")
        self.btn_save = QPushButton("Save")
        self.btn_close = QPushButton("Close")
        for b in (self.btn_activate, self.btn_save, self.btn_close):
            b.setStyleSheet(self._btn_css(primary=(b is self.btn_save)))
        self.btn_activate.clicked.connect(self._set_active)
        self.btn_save.clicked.connect(self._save)
        self.btn_close.clicked.connect(self.accept)
        action_row.addStretch()
        action_row.addWidget(self.btn_activate)
        action_row.addWidget(self.btn_save)
        action_row.addWidget(self.btn_close)
        right.addLayout(action_row)
        root.addLayout(right, 2)

        self._reload_list(select=class_registry.active_id())

    # -- helpers -----------------------------------------------------------
    def _muted(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:12px;")
        lbl.setWordWrap(True)
        return lbl

    def _btn_css(self, primary: bool = False) -> str:
        bg = PURPLE if primary else "transparent"
        return (f"QPushButton {{ background:{bg}; color:{OFFWHITE}; border:1px solid {BORDER};"
                f" border-radius:6px; padding:6px 12px; }}"
                f"QPushButton:hover {{ border-color:{PURPLE}; }}")

    @property
    def selected_id(self) -> str:
        return class_registry.active_id()

    def _reload_list(self, select: str | None = None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        active = class_registry.active_id()
        target_row = 0
        for i, s in enumerate(class_registry.list_sets()):
            label = s["name"] + ("  (built-in)" if s["builtin"] else "")
            if s["id"] == active:
                label = "● " + label
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, s["id"])
            self.list.addItem(item)
            if select and s["id"] == select:
                target_row = i
        self.list.blockSignals(False)
        self.list.setCurrentRow(target_row)
        self._load_selected()

    def _selected_set_id(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _load_selected(self) -> None:
        sid = self._selected_set_id()
        if not sid:
            return
        self._current_id = sid
        sset = next((s for s in class_registry.list_sets() if s["id"] == sid), None)
        if not sset:
            return
        builtin = sset["builtin"]
        self.name_edit.setText(sset["name"])
        self.name_edit.setEnabled(not builtin)
        self.builtin_note.setText(
            "Built-in preset — read-only. Use Duplicate to make an editable copy."
            if builtin else "")
        self.btn_del.setEnabled(not builtin)
        self.btn_add_row.setEnabled(not builtin)
        self.btn_del_row.setEnabled(not builtin)
        self.btn_save.setEnabled(not builtin)
        self.table.setRowCount(0)
        for c in sset["classes"]:
            self._add_row(c.get("name", ""), c.get("coco_id"), c.get("color"), editable=not builtin)

    def _add_row(self, name: str = "", coco_id=None, color: str | None = None,
                 editable: bool = True) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        name_item = QTableWidgetItem(name)
        id_item = QTableWidgetItem("" if coco_id in (None, "") else str(coco_id))
        color_item = QTableWidgetItem(color or "")
        if color:
            color_item.setBackground(QColor(color))
        for it in (name_item, id_item, color_item):
            if not editable:
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)  # color via picker
        self.table.setItem(r, 0, name_item)
        self.table.setItem(r, 1, id_item)
        self.table.setItem(r, 2, color_item)

    def _remove_row(self) -> None:
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)

    def _maybe_pick_color(self, row: int, col: int) -> None:
        if col != 2 or not self.btn_save.isEnabled():
            return
        item = self.table.item(row, 2)
        initial = QColor(item.text()) if item and item.text() else QColor(PURPLE)
        chosen = QColorDialog.getColor(initial, self, "Pick class color")
        if chosen.isValid():
            hexc = chosen.name()
            item.setText(hexc)
            item.setBackground(chosen)

    def _collect_classes(self) -> list[dict]:
        out = []
        for r in range(self.table.rowCount()):
            name = (self.table.item(r, 0).text() if self.table.item(r, 0) else "").strip()
            if not name:
                continue
            raw_id = (self.table.item(r, 1).text() if self.table.item(r, 1) else "").strip()
            color = (self.table.item(r, 2).text() if self.table.item(r, 2) else "").strip()
            cid = None
            if raw_id:
                try:
                    cid = int(raw_id)
                except ValueError:
                    cid = None
            out.append({"name": name, "coco_id": cid, "color": color or None})
        return out

    # -- actions -----------------------------------------------------------
    def _new_set(self) -> None:
        name, ok = QInputDialog.getText(self, "New Class Set", "Name:")
        if not ok or not name.strip():
            return
        sid = class_registry.create_set(name.strip(), [{"name": "object"}], activate=False)
        self._reload_list(select=sid)

    def _duplicate_set(self) -> None:
        sid = self._selected_set_id()
        if not sid:
            return
        src = next((s for s in class_registry.list_sets() if s["id"] == sid), None)
        if not src:
            return
        new_id = class_registry.create_set(
            src["name"] + " (copy)",
            [dict(c) for c in src["classes"]],
            activate=False,
        )
        self._reload_list(select=new_id)

    def _delete_set(self) -> None:
        sid = self._selected_set_id()
        if not sid:
            return
        try:
            if QMessageBox.question(
                self, "Delete Class Set",
                "Delete this class set? This can't be undone.",
            ) != QMessageBox.Yes:
                return
            class_registry.delete_set(sid)
        except ValueError as e:
            QMessageBox.warning(self, "Can't delete", str(e))
            return
        self._reload_list(select=class_registry.active_id())

    def _save(self) -> None:
        sid = self._current_id
        if not sid:
            return
        classes = self._collect_classes()
        if not classes:
            QMessageBox.warning(self, "No classes", "Add at least one class.")
            return
        try:
            class_registry.update_set(sid, name=self.name_edit.text(), classes=classes)
        except Exception as e:
            QMessageBox.warning(self, "Couldn't save", self._write_err(e))
            return
        self._reload_list(select=sid)
        QMessageBox.information(self, "Saved", "Class set saved.")

    def _set_active(self) -> None:
        sid = self._selected_set_id()
        if not sid:
            return
        try:
            class_registry.set_active(sid)
        except Exception as e:
            QMessageBox.warning(self, "Couldn't set active", self._write_err(e))
            return
        self._reload_list(select=sid)

    @staticmethod
    def _write_err(e: Exception) -> str:
        return (f"Couldn't write the class set file:\n{e}\n\n"
                "If this persists, pause OneDrive sync on the data folder or "
                "exclude it from antivirus scanning.")
