"""
Settings dialog -- local folder shortcuts and NAS connection management.

Converted from settings_tab.py to a QDialog with OK/Cancel buttons.
Accessed via File > Settings (Ctrl+,).

Sections:
1. Local Folders: data path display + "Open" buttons for each folder
2. NAS Connection: status indicator, config inputs, Connect/Disconnect buttons
"""
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QFrame,
    QGroupBox,
    QMessageBox,
    QDialogButtonBox,
)

from cctv_yolo.nas_manager import NasManager

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
from cctv_yolo.theme import (
    INDIGO as BG, PANEL, BORDER, PURPLE as ACCENT, OFFWHITE as TEXT,
    ERROR,
    PINK,
    TEXT_MUTED,
    INDIGO,
    OFFWHITE,
    PURPLE,
)

DIALOG_STYLE = f"""
QDialog {{
    background-color: {BG};
    color: {TEXT};
}}
QLabel {{
    color: {TEXT};
}}
"""

GROUP_STYLE = f"""
QGroupBox {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-top: 2px solid {ACCENT};
    border-radius: 8px;
    margin-top: 14px;
    padding: 16px;
    padding-top: 30px;
    font-size: 14px;
    font-weight: bold;
    color: {TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 14px;
    color: {ACCENT};
    font-size: 14px;
}}
"""

FOLDER_BTN_STYLE = f"""
QPushButton {{
    background-color: {ACCENT};
    color: ;
    border: none;
    border-radius: 4px;
    padding: 4px 12px;
    font-weight: bold;
    font-size: 12px;
    min-width: 60px;
}}
QPushButton:hover {{
    background-color: ;
}}
"""

INPUT_STYLE = f"""
QLineEdit {{
    background-color: ;
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: {ACCENT};
    selection-color: ;
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
    background-color: ;
}}
QLineEdit:hover {{
    border: 1px solid ;
}}
"""

CONNECT_BTN_STYLE = f"""
QPushButton {{
    background-color: {ACCENT};
    color: ;
    border: none;
    border-radius: 4px;
    padding: 8px 24px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: ;
}}
QPushButton:disabled {{
    background-color: {BORDER};
    color: ;
}}
"""

DISCONNECT_BTN_STYLE = f"""
QPushButton {{
    background-color: ;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 24px;
    font-weight: bold;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: ;
}}
"""

PATH_LABEL_STYLE = f"""
QLabel {{
    color: ;
    font-size: 12px;
    font-family: "SF Mono", "Menlo", "Courier New", monospace;
    padding: 6px 8px;
    background-color: rgba(13, 21, 37, 0.5);
    border-radius: 4px;
}}
"""

STATUS_CONNECTED = f"""
QLabel {{
    background-color: {ACCENT};
    color: ;
    border-radius: 12px;
    padding: 6px 16px;
    font-weight: bold;
    font-size: 12px;
    letter-spacing: 0.5px;
}}
"""

STATUS_DISCONNECTED = f"""
QLabel {{
    background-color: ;
    color: white;
    border-radius: 12px;
    padding: 6px 16px;
    font-weight: bold;
    font-size: 12px;
    letter-spacing: 0.5px;
}}
"""

BUTTON_BOX_STYLE = f"""
QDialogButtonBox QPushButton {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 20px;
    font-size: 13px;
    min-width: 80px;
}}
QDialogButtonBox QPushButton:hover {{
    background-color: {BORDER};
}}
QDialogButtonBox QPushButton:default {{
    background-color: {ACCENT};
    color: ;
    border: none;
    font-weight: bold;
}}
QDialogButtonBox QPushButton:default:hover {{
    background-color: ;
}}
"""


class _GpuStatusProbe(QThread):
    """Detect GPU status OFF the GUI thread.

    Querying the NVIDIA driver (nvcuda.dll / nvidia-smi) can take a moment when
    a laptop's Optimus dGPU is parked; doing it inline while building the
    Settings dialog froze the whole page (a colleague reported the Settings
    buttons "do not respond"). Running it here keeps the dialog instantly
    responsive and just updates the label when the answer arrives.
    """

    done = Signal(str)  # status text for the label

    def run(self):
        from cctv_yolo import gpu_runtime
        try:
            if gpu_runtime.is_installed():
                info = gpu_runtime.installed_info() or {}
                txt = (f"Active: PyTorch {info.get('torch', '?')} "
                       f"({info.get('variant', '?')}) — GPU enabled.")
            else:
                name = gpu_runtime.gpu_name()
                if name:
                    txt = (f"{name} detected, currently running on CPU. "
                           "Set up GPU acceleration for a big speedup.")
                else:
                    txt = "No NVIDIA GPU detected — running on CPU."
        except Exception:
            txt = "GPU status unavailable (the setup button still works)."
        self.done.emit(txt)


class SettingsDialog(QDialog):
    """Settings dialog -- local folders and NAS connection."""

    mode_changed = Signal(str)  # "local" or "nas"

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.nas_manager = NasManager(data_manager.config_dir / "nas.json")
        self.setWindowTitle("Settings")
        self.setMinimumWidth(650)
        self.setMinimumHeight(500)
        self.setStyleSheet(DIALOG_STYLE)
        self._setup_ui()
        self._load_nas_config()
        self._sync_nas_status()
        # Make sure the GPU probe thread is joined before this dialog is torn
        # down (OK/Cancel emit finished; the window X goes through closeEvent).
        self.finished.connect(self._stop_gpu_probe)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # --- Header ---
        title = QLabel("Settings")
        title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {TEXT};")
        layout.addWidget(title)

        # --- Local Folders section ---
        folders_group = QGroupBox("Local Folders")
        folders_group.setStyleSheet(GROUP_STYLE)
        folders_layout = QVBoxLayout(folders_group)
        folders_layout.setSpacing(8)

        # Data root display
        data_path_row = QHBoxLayout()
        data_path_row.addWidget(QLabel("Data Directory:"))
        self.lbl_data_path = QLabel(str(self.data_manager.data_root))
        self.lbl_data_path.setStyleSheet(PATH_LABEL_STYLE)
        self.lbl_data_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        data_path_row.addWidget(self.lbl_data_path, stretch=1)
        btn_open_data = QPushButton("Open")
        btn_open_data.setStyleSheet(FOLDER_BTN_STYLE)
        btn_open_data.clicked.connect(lambda: self.data_manager.open_folder("data"))
        data_path_row.addWidget(btn_open_data)
        folders_layout.addLayout(data_path_row)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {BORDER};")
        folders_layout.addWidget(sep)

        # Individual folder buttons
        folder_types = [
            ("Videos", "videos"),
            ("Tracks", "tracks"),
            ("Corrections", "corrections"),
            ("Exports", "exports"),
            ("Models", "models"),
        ]
        for display_name, folder_type in folder_types:
            row = QHBoxLayout()
            lbl = QLabel(display_name)
            lbl.setMinimumWidth(100)
            row.addWidget(lbl)

            if folder_type == "models":
                path_lbl = QLabel(str(self.data_manager.models_dir))
            else:
                path_lbl = QLabel(str(getattr(self.data_manager, f"{folder_type}_dir", "")))
            path_lbl.setStyleSheet(PATH_LABEL_STYLE)
            path_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            row.addWidget(path_lbl, stretch=1)

            btn = QPushButton("Open")
            btn.setStyleSheet(FOLDER_BTN_STYLE)
            if folder_type == "models":
                btn.clicked.connect(
                    lambda checked=False: self.data_manager.open_folder("models")
                )
            else:
                btn.clicked.connect(
                    lambda checked=False, ft=folder_type: self.data_manager.open_folder(ft)
                )
            row.addWidget(btn)
            folders_layout.addLayout(row)

        layout.addWidget(folders_group)

        # --- GPU Acceleration section (Windows only) ---
        import sys as _sys
        if _sys.platform == "win32":
            gpu_group = QGroupBox("GPU Acceleration")
            gpu_group.setStyleSheet(GROUP_STYLE)
            gpu_layout = QVBoxLayout(gpu_group)
            gpu_layout.setSpacing(8)
            # Detect OFF the GUI thread (see _GpuStatusProbe) so a slow NVIDIA
            # driver query can never freeze the Settings page. Show a neutral
            # placeholder until the probe answers.
            self.lbl_gpu_status = QLabel("Checking for an NVIDIA GPU…")
            self.lbl_gpu_status.setWordWrap(True)
            gpu_layout.addWidget(self.lbl_gpu_status)
            self.btn_setup_gpu = QPushButton("Set up / repair GPU acceleration")
            self.btn_setup_gpu.clicked.connect(self._on_setup_gpu)
            gpu_layout.addWidget(self.btn_setup_gpu)
            layout.addWidget(gpu_group)

            self._gpu_probe = _GpuStatusProbe(self)
            self._gpu_probe.done.connect(self._on_gpu_status)
            self._gpu_probe.start()

        # --- NAS Connection section ---
        nas_group = QGroupBox("NAS Connection (Tailscale SMB)")
        nas_group.setStyleSheet(GROUP_STYLE)
        nas_layout = QVBoxLayout(nas_group)
        nas_layout.setSpacing(10)

        # Status indicator
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        self.lbl_nas_status = QLabel("Disconnected")
        self.lbl_nas_status.setStyleSheet(STATUS_DISCONNECTED)
        status_row.addWidget(self.lbl_nas_status)
        status_row.addStretch()

        self.lbl_mode = QLabel(f"Mode: {self.data_manager.active_mode.upper()}")
        self.lbl_mode.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 13px;")
        status_row.addWidget(self.lbl_mode)
        nas_layout.addLayout(status_row)

        # Config form
        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight)

        self.input_ip = QLineEdit()
        self.input_ip.setPlaceholderText("e.g. 100.64.0.1")
        self.input_ip.setStyleSheet(INPUT_STYLE)
        form.addRow("IP Address:", self.input_ip)

        self.input_share = QLineEdit()
        self.input_share.setPlaceholderText("e.g. cctv_footage")
        self.input_share.setStyleSheet(INPUT_STYLE)
        form.addRow("Share Name:", self.input_share)

        self.input_username = QLineEdit()
        self.input_username.setPlaceholderText("e.g. admin")
        self.input_username.setStyleSheet(INPUT_STYLE)
        form.addRow("Username:", self.input_username)

        self.input_password = QLineEdit()
        self.input_password.setPlaceholderText("Password")
        self.input_password.setEchoMode(QLineEdit.Password)
        self.input_password.setStyleSheet(INPUT_STYLE)
        form.addRow("Password:", self.input_password)

        self.input_mount_point = QLineEdit()
        self.input_mount_point.setPlaceholderText("/tmp/cctv_nas_mount  or  Z:")
        self.input_mount_point.setStyleSheet(INPUT_STYLE)
        form.addRow("Mount Point:", self.input_mount_point)

        nas_layout.addLayout(form)

        # NAS buttons
        nas_btn_row = QHBoxLayout()
        nas_btn_row.setSpacing(12)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setStyleSheet(CONNECT_BTN_STYLE)
        self.btn_connect.clicked.connect(self._connect_nas)
        nas_btn_row.addWidget(self.btn_connect)

        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setStyleSheet(DISCONNECT_BTN_STYLE)
        self.btn_disconnect.setEnabled(False)
        self.btn_disconnect.clicked.connect(self._disconnect_nas)
        nas_btn_row.addWidget(self.btn_disconnect)

        nas_btn_row.addStretch()
        nas_layout.addLayout(nas_btn_row)

        # NAS status message
        self.lbl_nas_message = QLabel("")
        self.lbl_nas_message.setStyleSheet("color: #999; font-size: 12px;")
        self.lbl_nas_message.setWordWrap(True)
        nas_layout.addWidget(self.lbl_nas_message)

        layout.addWidget(nas_group)

        # --- OK / Cancel buttons ---
        layout.addStretch()

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setStyleSheet(BUTTON_BOX_STYLE)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    # ------------------------------------------------------------------
    # NAS config persistence
    # ------------------------------------------------------------------

    def _on_setup_gpu(self):
        """Open the GPU acceleration setup dialog (Settings -> GPU)."""
        from cctv_yolo import gpu_runtime
        from cctv_yolo.gpu_setup_dialog import GpuSetupDialog
        # variant_for_setup() (not desired_variant()) so we don't falsely claim
        # "no GPU" when nvidia-smi is off-PATH / the dGPU was asleep but a CUDA
        # torch is already installed and running.
        variant = gpu_runtime.variant_for_setup()
        if not variant:
            QMessageBox.information(
                self, "GPU Acceleration",
                "No NVIDIA GPU was detected on this machine, so there's nothing "
                "to install. CCTV-YOLO runs on CPU here.",
            )
            return
        # Already installed and running on the GPU: don't push a redundant
        # multi-GB re-download at them — confirm it's active and let them
        # reinstall only if they actually want to repair it.
        if gpu_runtime.is_installed():
            info = gpu_runtime.installed_info() or {}
            resp = QMessageBox.question(
                self, "GPU Acceleration",
                f"GPU acceleration is already installed and active "
                f"(PyTorch {info.get('torch', '?')}, {info.get('variant', '?')}).\n\n"
                "Reinstall it anyway? (Only needed if it's behaving oddly — this "
                "re-downloads several GB.)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                return
        gpu_runtime.clear_declined()
        GpuSetupDialog(variant, parent=self).exec()
        if gpu_runtime.is_installed() and hasattr(self, "lbl_gpu_status"):
            info = gpu_runtime.installed_info() or {}
            self.lbl_gpu_status.setText(
                f"Active: PyTorch {info.get('torch', '?')} "
                f"({info.get('variant', '?')}). Restart CCTV-YOLO to use it."
            )

    def _on_gpu_status(self, text: str):
        """Update the GPU label once the off-thread probe finishes."""
        if hasattr(self, "lbl_gpu_status"):
            self.lbl_gpu_status.setText(text)

    def _stop_gpu_probe(self, *_):
        # Never let the dialog be destroyed while the GPU probe thread is still
        # running — that aborts the process. The probe is quick; wait briefly.
        probe = getattr(self, "_gpu_probe", None)
        if probe is not None and probe.isRunning():
            probe.wait(3000)

    def closeEvent(self, event):
        self._stop_gpu_probe()
        super().closeEvent(event)

    def _load_nas_config(self):
        """Load saved NAS config into the form inputs."""
        config = self.nas_manager.load_config()
        if not config:
            return
        self.input_ip.setText(config.get("ip", ""))
        self.input_share.setText(config.get("share", ""))
        self.input_username.setText(config.get("username", ""))
        self.input_password.setText(config.get("password", ""))
        self.input_mount_point.setText(config.get("mount_point", ""))

    def _get_config_from_form(self):
        """Read NAS config from form inputs."""
        return {
            "ip": self.input_ip.text().strip(),
            "share": self.input_share.text().strip(),
            "username": self.input_username.text().strip(),
            "password": self.input_password.text().strip(),
            "mount_point": self.input_mount_point.text().strip(),
        }

    def _sync_nas_status(self):
        """Sync the dialog UI with the current NAS connection state."""
        if self.data_manager.active_mode == "nas":
            mount = str(self.data_manager.nas_mount_point or "")
            self._update_status_connected(mount)
        else:
            self._update_status_disconnected()

    # ------------------------------------------------------------------
    # Connect / Disconnect
    # ------------------------------------------------------------------

    def _connect_nas(self):
        """Attempt to mount the NAS share and switch to NAS mode."""
        config = self._get_config_from_form()

        valid, error = self.nas_manager.validate_config(config)
        if not valid:
            QMessageBox.warning(self, "Invalid Config", error)
            return

        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("Connecting...")

        success, message, mount_point = self.nas_manager.mount(config)

        if success and mount_point:
            self.nas_manager.save_config(config)
            self.data_manager.switch_to_nas(mount_point)
            self._update_status_connected(mount_point)
            self.lbl_nas_message.setText(f"Connected to //{config['ip']}/{config['share']}")
            self.mode_changed.emit("nas")
        else:
            self.lbl_nas_message.setText(f"Connection failed: {message}")
            self.lbl_nas_message.setStyleSheet(f"color: {ERROR}; font-size: 12px;")
            self.btn_connect.setEnabled(True)
            self.btn_connect.setText("Connect")

    def _disconnect_nas(self):
        """Unmount NAS and switch back to local mode."""
        mount_point = self.data_manager.nas_mount_point
        self.nas_manager.unmount(mount_point)
        self.data_manager.switch_to_local()
        self._update_status_disconnected()
        self.lbl_nas_message.setText("Disconnected from NAS")
        self.lbl_nas_message.setStyleSheet("color: #999; font-size: 12px;")
        self.mode_changed.emit("local")

    def _update_status_connected(self, mount_point):
        """Update UI to reflect connected state."""
        self.lbl_nas_status.setText("Connected")
        self.lbl_nas_status.setStyleSheet(STATUS_CONNECTED)
        self.lbl_mode.setText(f"Mode: NAS ({mount_point})")
        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("Connected")
        self.btn_disconnect.setEnabled(True)

    def _update_status_disconnected(self):
        """Update UI to reflect disconnected state."""
        self.lbl_nas_status.setText("Disconnected")
        self.lbl_nas_status.setStyleSheet(STATUS_DISCONNECTED)
        self.lbl_mode.setText("Mode: LOCAL")
        self.btn_connect.setEnabled(True)
        self.btn_connect.setText("Connect")
        self.btn_disconnect.setEnabled(False)

    # ------------------------------------------------------------------
    # Auto-reconnect (called from main window on startup)
    # ------------------------------------------------------------------

    def check_auto_reconnect(self):
        """Check if a previously connected NAS is still mounted and switch if so."""
        mount_point = self.nas_manager.check_auto_reconnect()
        if mount_point:
            self.data_manager.switch_to_nas(mount_point)
            self._update_status_connected(mount_point)
            config = self.nas_manager.load_config()
            if config:
                self.lbl_nas_message.setText(
                    f"Auto-reconnected to //{config.get('ip', '?')}/{config.get('share', '?')}"
                )
            self.mode_changed.emit("nas")
