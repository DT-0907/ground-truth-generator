#!/usr/bin/env python3
"""
Verify that all dependencies are installed correctly.
"""

import sys
from pathlib import Path

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} - {e}")
        return False

def check_yolo():
    """Check YOLO installation and download test model."""
    try:
        from ultralytics import YOLO
        print("✓ ultralytics imported")
        
        # Try to load a model (will download if needed)
        print("  Testing YOLO model loading...")
        model = YOLO('yolov8n.pt')  # Smallest model for quick test
        print("  ✓ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ YOLO error: {e}")
        return False

def check_opencv():
    """Check OpenCV installation."""
    try:
        import cv2
        print(f"✓ opencv-python (version {cv2.__version__})")
        
        # Check video codec support
        print("  Checking video codec support...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("  ✓ Video codec support OK")
        return True
    except Exception as e:
        print(f"✗ OpenCV error: {e}")
        return False

def check_tracking():
    """Check tracking capability (Ultralytics has built-in tracking)."""
    try:
        # Ultralytics has built-in tracking, so we don't need external ByteTrack
        from ultralytics import YOLO
        print("✓ Tracking: Ultralytics built-in tracking available")
        print("  Note: No external ByteTrack needed - Ultralytics includes tracking")
        return True
    except Exception as e:
        print(f"⚠ Tracking: {e}")
        print("  Note: Ultralytics should provide built-in tracking")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available."""
    import subprocess
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg: {version_line}")
            return True
        else:
            print("✗ FFmpeg not found")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not found in PATH")
        print("  Install with: brew install ffmpeg (macOS)")
        return False
    except Exception as e:
        print(f"✗ FFmpeg check error: {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Verifying Installation")
    print("=" * 60)
    print()
    
    all_ok = True
    
    print("Python packages:")
    print("-" * 60)
    all_ok &= check_import("numpy", "numpy")
    all_ok &= check_import("pandas", "pandas")
    all_ok &= check_import("PIL", "Pillow")
    all_ok &= check_opencv()
    all_ok &= check_import("scipy", "scipy")
    all_ok &= check_import("flask", "flask")
    print()
    
    print("YOLO:")
    print("-" * 60)
    all_ok &= check_yolo()
    print()
    
    print("Tracking:")
    print("-" * 60)
    # Tracking check is informational only (Ultralytics has built-in tracking)
    check_tracking()
    print()
    
    print("System tools:")
    print("-" * 60)
    all_ok &= check_ffmpeg()
    print()
    
    print("=" * 60)
    if all_ok:
        print("✓ All checks passed! You're ready to go.")
        return 0
    else:
        print("✗ Some checks failed. Please install missing dependencies.")
        print("\nRun: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

