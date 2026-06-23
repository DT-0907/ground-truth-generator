"""
Dialog windows for track operations — class change, merge, new track, ROI name.
"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                QComboBox, QPushButton, QLineEdit, QFormLayout)
from PySide6.QtCore import Qt
from cctv_yolo.theme import BORDER, INDIGO, OFFWHITE, PANEL, PANEL_HI, PINK, PURPLE, YELLOW
from cctv_yolo import classes as class_registry


def _class_names(current: str | None = None) -> list[str]:
    """Class names from the active class set. If ``current`` is supplied and not
    in the set, it's prepended so a track labeled under a different/older set
    isn't silently changed when the dialog opens."""
    names = list(class_registry.class_names())
    if current and current not in names:
        names = [current] + names
    return names or ["car"]

# Finer-grained subclasses keyed by primary class. Used to refine
# annotations beyond the COCO supercategory. ``""`` means "unspecified".
VEHICLE_SUBCLASSES = {
    "car": ["", "sedan", "suv", "hatchback", "coupe", "minivan", "wagon"],
    "truck": ["", "pickup", "box_truck", "flatbed", "semi", "tow_truck", "garbage_truck", "delivery_van"],
    "bus": ["", "city_bus", "school_bus", "coach", "shuttle"],
    "motorcycle": ["", "sport", "cruiser", "scooter", "moped"],
    "bicycle": ["", "road", "mountain", "ebike", "cargo"],
}

# Dark theme stylesheet for dialogs
DIALOG_STYLE = """
QDialog {
    background: ;
    color: #eee;
}
QLabel {
    color: #eee;
}
QComboBox, QLineEdit {
    background: ;
    color: ;
    border: 1px solid ;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
}
QComboBox:focus, QLineEdit:focus {
    border-color: ;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QComboBox QAbstractItemView {
    background: ;
    color: ;
    selection-background-color: ;
}
QPushButton {
    padding: 10px 20px;
    border: none;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
}
QPushButton#cancel {
    background: ;
    color: ;
}
QPushButton#cancel:hover {
    background: ;
}
QPushButton#confirm {
    background: ;
    color: ;
}
QPushButton#confirm:hover {
    background: ;
}
"""


class ClassChangeDialog(QDialog):
    """Dialog to change a track's vehicle class and optional subclass."""

    def __init__(self, current_class: str = "car", current_subclass: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Class")
        self.setFixedWidth(320)
        self.setStyleSheet(DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addWidget(QLabel("Vehicle class:"))
        self.combo = QComboBox()
        names = _class_names(current_class)
        self.combo.addItems(names)
        idx = names.index(current_class) if current_class in names else 0
        self.combo.setCurrentIndex(idx)
        self.combo.currentTextChanged.connect(self._reload_subclasses)
        layout.addWidget(self.combo)

        layout.addWidget(QLabel("Subclass (optional):"))
        self.sub_combo = QComboBox()
        layout.addWidget(self.sub_combo)
        self._reload_subclasses(self.combo.currentText())
        if current_subclass:
            ix = self.sub_combo.findText(current_subclass)
            if ix >= 0:
                self.sub_combo.setCurrentIndex(ix)

        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancel")
        cancel_btn.clicked.connect(self.reject)
        confirm_btn = QPushButton("Change")
        confirm_btn.setObjectName("confirm")
        confirm_btn.clicked.connect(self.accept)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(confirm_btn)
        layout.addLayout(btn_layout)

    def _reload_subclasses(self, cls: str):
        self.sub_combo.clear()
        self.sub_combo.addItems(VEHICLE_SUBCLASSES.get(cls, [""]))

    def selected_class(self) -> str:
        return self.combo.currentText()

    def selected_subclass(self) -> str:
        return self.sub_combo.currentText()


class MergeDialog(QDialog):
    """Dialog to select a target track for merging.

    Shows a dropdown of all other tracks with their class, frame range,
    and a gap indicator showing how many frames will be interpolated.
    """

    def __init__(self, source_track_id: int, tracks: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Merge Tracks")
        self.setFixedWidth(400)
        self.setStyleSheet(DIALOG_STYLE)
        self.source_track_id = source_track_id
        self.tracks = tracks

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        layout.addWidget(QLabel(f"Merge track #{source_track_id} into:"))

        self.combo = QComboBox()
        other_tracks = [t for t in tracks if t["track_id"] != source_track_id]
        other_tracks.sort(key=lambda t: t["track_id"])
        for t in other_tracks:
            sf = t.get('start_frame', t['frames'][0]['frame'] if t.get('frames') else 0)
            ef = t.get('end_frame', t['frames'][-1]['frame'] if t.get('frames') else 0)
            label = f"#{t['track_id']} {t['class']} (F{sf}-{ef})"
            self.combo.addItem(label, t["track_id"])
        layout.addWidget(self.combo)

        self.gap_label = QLabel("")
        self.gap_label.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(self.gap_label)

        self.combo.currentIndexChanged.connect(self._update_gap_info)
        if self.combo.count() > 0:
            self._update_gap_info()

        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancel")
        cancel_btn.clicked.connect(self.reject)
        confirm_btn = QPushButton("Merge")
        confirm_btn.setObjectName("confirm")
        confirm_btn.clicked.connect(self.accept)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(confirm_btn)
        layout.addLayout(btn_layout)

    def _update_gap_info(self):
        """Display gap information between the source and selected target track."""
        target_id = self.combo.currentData()
        if target_id is None:
            return
        source = next((t for t in self.tracks if t["track_id"] == self.source_track_id), None)
        target = next((t for t in self.tracks if t["track_id"] == target_id), None)
        if source and target:
            s_start = source.get('start_frame', source['frames'][0]['frame'] if source.get('frames') else 0)
            s_end = source.get('end_frame', source['frames'][-1]['frame'] if source.get('frames') else 0)
            t_start = target.get('start_frame', target['frames'][0]['frame'] if target.get('frames') else 0)
            t_end = target.get('end_frame', target['frames'][-1]['frame'] if target.get('frames') else 0)
            gap = max(0, max(t_start - s_end, s_start - t_end) - 1)
            if gap > 0:
                self.gap_label.setText(f"Gap: {gap} frames will be interpolated")
                self.gap_label.setStyleSheet(f"color: {YELLOW}; font-size: 12px;")
            else:
                self.gap_label.setText("No gap — tracks overlap or are adjacent")
                self.gap_label.setStyleSheet(f"color: {PURPLE}; font-size: 12px;")

    def selected_target_id(self) -> int | None:
        return self.combo.currentData()


class NewTrackDialog(QDialog):
    """Dialog to create a new track from a drawn bounding box."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Track")
        self.setFixedWidth(300)
        self.setStyleSheet(DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        layout.addWidget(QLabel("Select vehicle class:"))

        self.combo = QComboBox()
        self.combo.addItems(_class_names())
        layout.addWidget(self.combo)

        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancel")
        cancel_btn.clicked.connect(self.reject)
        confirm_btn = QPushButton("Create")
        confirm_btn.setObjectName("confirm")
        confirm_btn.clicked.connect(self.accept)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(confirm_btn)
        layout.addLayout(btn_layout)

    def selected_class(self) -> str:
        return self.combo.currentText()


class RoiNameDialog(QDialog):
    """Dialog to name a new ROI region."""

    def __init__(self, default_name: str = "ROI 1", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Name ROI")
        self.setFixedWidth(350)
        self.setStyleSheet(DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        layout.addWidget(QLabel("Name this region of interest:"))

        self.name_input = QLineEdit(default_name)
        self.name_input.selectAll()
        layout.addWidget(self.name_input)

        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancel")
        cancel_btn.clicked.connect(self.reject)
        confirm_btn = QPushButton("Create")
        confirm_btn.setObjectName("confirm")
        confirm_btn.clicked.connect(self.accept)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(confirm_btn)
        layout.addLayout(btn_layout)

    def roi_name(self) -> str:
        return self.name_input.text().strip() or "ROI"


class RenameRoiDialog(QDialog):
    """Dialog to rename an existing ROI region."""

    def __init__(self, current_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rename ROI")
        self.setFixedWidth(350)
        self.setStyleSheet(DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        layout.addWidget(QLabel("Rename ROI:"))

        self.name_input = QLineEdit(current_name)
        self.name_input.selectAll()
        layout.addWidget(self.name_input)

        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancel")
        cancel_btn.clicked.connect(self.reject)
        confirm_btn = QPushButton("Rename")
        confirm_btn.setObjectName("confirm")
        confirm_btn.clicked.connect(self.accept)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(confirm_btn)
        layout.addLayout(btn_layout)

    def new_name(self) -> str:
        return self.name_input.text().strip()
