"""
Dialog windows for track operations — class change, merge, new track, ROI name.
"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                QComboBox, QPushButton, QLineEdit, QFormLayout)
from PySide6.QtCore import Qt

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]

# Dark theme stylesheet for dialogs
DIALOG_STYLE = """
QDialog {
    background: #16213e;
    color: #eee;
}
QLabel {
    color: #eee;
}
QComboBox, QLineEdit {
    background: #1a1a2e;
    color: #fff;
    border: 1px solid #2d3a5a;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
}
QComboBox:focus, QLineEdit:focus {
    border-color: #4ecca3;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QComboBox QAbstractItemView {
    background: #1a1a2e;
    color: #fff;
    selection-background-color: #2d3a5a;
}
QPushButton {
    padding: 10px 20px;
    border: none;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
}
QPushButton#cancel {
    background: #2d3a5a;
    color: #fff;
}
QPushButton#cancel:hover {
    background: #3d4a6a;
}
QPushButton#confirm {
    background: #4ecca3;
    color: #000;
}
QPushButton#confirm:hover {
    background: #3db892;
}
"""


class ClassChangeDialog(QDialog):
    """Dialog to change a track's vehicle class."""

    def __init__(self, current_class: str = "car", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Class")
        self.setFixedWidth(300)
        self.setStyleSheet(DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        layout.addWidget(QLabel("Select vehicle class:"))

        self.combo = QComboBox()
        self.combo.addItems(VEHICLE_CLASSES)
        idx = VEHICLE_CLASSES.index(current_class) if current_class in VEHICLE_CLASSES else 0
        self.combo.setCurrentIndex(idx)
        layout.addWidget(self.combo)

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

    def selected_class(self) -> str:
        return self.combo.currentText()


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
            label = f"#{t['track_id']} {t['class']} (F{t['start_frame']}-{t['end_frame']})"
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
            gap = max(0, max(target["start_frame"] - source["end_frame"],
                             source["start_frame"] - target["end_frame"]) - 1)
            if gap > 0:
                self.gap_label.setText(f"Gap: {gap} frames will be interpolated")
                self.gap_label.setStyleSheet("color: #f39c12; font-size: 12px;")
            else:
                self.gap_label.setText("No gap — tracks overlap or are adjacent")
                self.gap_label.setStyleSheet("color: #4ecca3; font-size: 12px;")

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
        self.combo.addItems(VEHICLE_CLASSES)
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
