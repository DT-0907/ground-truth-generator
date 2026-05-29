"""
NAS manager — mount/unmount/config for Tailscale SMB shares.
"""
import json
import os
import sys
import base64
import subprocess
from pathlib import Path

# In a windowed (console=False) build, spawning a console command like
# `net use` flashes a black cmd window. CREATE_NO_WINDOW suppresses it.
# Only defined on Windows; 0 is a no-op everywhere else.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class NasManager:
    """Manages NAS connection via Tailscale SMB."""

    def __init__(self, config_file: Path):
        self.config_file = config_file

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def load_config(self) -> dict | None:
        """Load saved NAS config. Returns None if no config exists.

        The password is stored base64-encoded on disk and decoded here.
        """
        if not self.config_file.exists():
            return None
        try:
            with open(self.config_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        try:
            data["password"] = base64.b64decode(data["password"]).decode()
        except Exception:
            # If decoding fails the password may already be plain text
            pass
        return data

    def save_config(self, config: dict):
        """Save NAS config (password is base64-encoded on disk)."""
        from cctv_yolo.data_manager import _atomic_write_json
        safe = dict(config)
        safe["password"] = base64.b64encode(config["password"].encode()).decode()
        _atomic_write_json(self.config_file, safe)

    def delete_config(self):
        """Remove saved NAS config from disk."""
        if self.config_file.exists():
            self.config_file.unlink()

    # ------------------------------------------------------------------
    # Mount / unmount
    # ------------------------------------------------------------------

    def mount(self, config: dict) -> tuple[bool, str, Path | None]:
        """Mount the NAS share.

        Returns ``(success, message, mount_point)``.
        """
        if sys.platform == "win32":
            drive = config.get("mount_point") or "Z:"
            mount_point = Path(drive + "\\")
        else:
            mount_point = Path(config.get("mount_point") or "/tmp/cctv_nas_mount")
            mount_point.mkdir(parents=True, exist_ok=True)

        # Already mounted — nothing to do
        if sys.platform != "win32" and os.path.ismount(str(mount_point)):
            return True, "Already mounted", mount_point
        if sys.platform == "win32" and mount_point.exists():
            # On Windows, check if the drive letter is accessible
            try:
                list(mount_point.iterdir())
                return True, "Already mounted", mount_point
            except OSError:
                pass

        # Build platform-specific mount command
        if sys.platform == "darwin":
            url = (
                f"//{config['username']}:{config['password']}"
                f"@{config['ip']}/{config['share']}"
            )
            cmd = ["mount_smbfs", url, str(mount_point)]
        elif sys.platform.startswith("linux"):
            cmd = [
                "mount",
                "-t",
                "cifs",
                f"//{config['ip']}/{config['share']}",
                str(mount_point),
                "-o",
                f"username={config['username']},password={config['password']}",
            ]
        elif sys.platform == "win32":
            cmd = [
                "net",
                "use",
                drive,
                f"\\\\{config['ip']}\\{config['share']}",
                f"/user:{config['username']}",
                config["password"],
            ]
        else:
            return False, f"Unsupported platform: {sys.platform}", None

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                creationflags=_NO_WINDOW,
            )
            if result.returncode == 0:
                return True, "Connected", mount_point
            return False, result.stderr.strip() or "Mount failed", None
        except subprocess.TimeoutExpired:
            return False, "Mount timed out", None
        except Exception as e:
            return False, str(e), None

    def unmount(self, mount_point: Path | None):
        """Unmount a NAS share. Silently ignores errors."""
        if not mount_point:
            return
        try:
            if sys.platform == "win32":
                # `net use /delete` wants a bare device name ("Z:"), NOT the
                # Path form with a trailing backslash ("Z:\") that mount()
                # builds — the latter is rejected and the unmount silently
                # fails, leaking the mapped drive.
                drive = str(mount_point).rstrip("\\/")
                subprocess.run(
                    ["net", "use", drive, "/delete", "/y"],
                    capture_output=True,
                    timeout=10,
                    creationflags=_NO_WINDOW,
                )
            else:
                subprocess.run(
                    ["umount", str(mount_point)],
                    capture_output=True,
                    timeout=10,
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Auto-reconnect
    # ------------------------------------------------------------------

    def check_auto_reconnect(self) -> Path | None:
        """Check if NAS was previously connected and auto-reconnect.

        Returns the mount point ``Path`` if the share is currently
        mounted, or ``None`` otherwise.  Does **not** attempt to
        re-mount — it only checks whether the existing mount is still
        live.
        """
        config = self.load_config()
        if not config:
            return None

        if sys.platform == "win32":
            drive = config.get("mount_point") or "Z:"
            mount_point = Path(drive + "\\")
            try:
                list(mount_point.iterdir())
                return mount_point
            except OSError:
                return None
        else:
            mount_point = Path(config.get("mount_point") or "/tmp/cctv_nas_mount")
            if mount_point.exists() and os.path.ismount(str(mount_point)):
                return mount_point

        return None

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def validate_config(self, config: dict) -> tuple[bool, str]:
        """Basic validation of NAS config fields.

        Returns ``(valid, error_message)``.
        """
        required = ["ip", "share", "username", "password"]
        for field in required:
            if not config.get(field, "").strip():
                return False, f"Missing required field: {field}"
        return True, ""
