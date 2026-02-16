#!/bin/bash
# ============================================================
#  Build CCTV-YOLO v2 (Native) for macOS
#  Output: dist/CCTV-YOLO.dmg
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Building CCTV-YOLO v2 (Native) for macOS"
echo "=========================================="

# 1. Create / activate a virtual environment
if [ ! -d "build_venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv build_venv
fi
source build_venv/bin/activate

# 2. Install dependencies
echo "[2/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 3. Run PyInstaller
echo "[3/5] Running PyInstaller..."
pyinstaller cctv_yolo.spec --clean --noconfirm

# 4. Create DMG
echo "[4/5] Creating DMG..."
DMG_DIR="dist/dmg_staging"
rm -rf "$DMG_DIR"
mkdir -p "$DMG_DIR"
cp -r "dist/CCTV-YOLO.app" "$DMG_DIR/"
ln -s /Applications "$DMG_DIR/Applications"

rm -f "dist/CCTV-YOLO.dmg"
hdiutil create \
    -volname "CCTV-YOLO" \
    -srcfolder "$DMG_DIR" \
    -ov -format UDZO \
    "dist/CCTV-YOLO.dmg"

rm -rf "$DMG_DIR"

# 5. Done
echo "[5/5] Cleaning up..."
deactivate 2>/dev/null || true

echo ""
echo "=========================================="
echo "  Build complete!"
echo "  DMG: dist/CCTV-YOLO.dmg"
echo "  APP: dist/CCTV-YOLO.app"
echo "=========================================="
echo ""
echo "To install:"
echo "  1. Open dist/CCTV-YOLO.dmg"
echo "  2. Drag CCTV-YOLO to Applications"
echo "  3. Launch from Applications"
echo "  4. Data will be created at ~/Documents/CCTV-YOLO/"
