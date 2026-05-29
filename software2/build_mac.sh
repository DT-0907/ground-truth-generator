#!/bin/bash
# ============================================================
#  Build CCTV-YOLO v2 (Native) for macOS
#  Output: dist/CCTV-YOLO.dmg + dist/CCTV-YOLO.app
# ============================================================
#
# Notes:
#
#   - First build takes ~5-15 minutes (depends on whether dependencies are
#     already cached). Subsequent builds reuse build_venv/ and finish in
#     about 2-3 minutes.
#
#   - This script reuses the existing build_venv/ if present. To force a
#     clean rebuild, `rm -rf build_venv/ dist/` before running.
#
#   - On Apple Silicon (M1/M2/M3), the python3 you invoke must be the native
#     arm64 build, NOT x86_64-via-Rosetta. Otherwise the bundle will run on
#     Intel only and crash on first launch under Apple Silicon. The script
#     warns if it detects this mismatch.
#
#   - Apple MPS (Metal) GPU acceleration is enabled automatically at runtime
#     via torch.backends.mps. No special install needed.
#

set -e
set -o pipefail

# Always print a final summary, even on errors, so users running this in a
# terminal see what stage failed.
BUILD_STATUS="unknown"
trap 'on_exit' EXIT

on_exit() {
    local code=$?
    echo ""
    echo "=========================================="
    if [ "$BUILD_STATUS" = "success" ]; then
        echo "  Build complete!"
        echo "=========================================="
        echo ""
        echo "  DMG : $(pwd)/dist/CCTV-YOLO.dmg"
        echo "  APP : $(pwd)/dist/CCTV-YOLO.app"
        echo ""
        echo "Opening the dist/ folder in Finder now..."
        open "$(pwd)/dist" 2>/dev/null || true
        echo ""
        echo "To install:"
        echo "  1. Double-click dist/CCTV-YOLO.dmg"
        echo "  2. Drag CCTV-YOLO to Applications (or anywhere)"
        echo "  3. Launch CCTV-YOLO"
        echo ""
        echo "Data location:"
        echo "  All videos / tracks / corrections / models / logs live in a"
        echo "  separate, portable folder named cctv-yolo/. On first launch"
        echo "  the app searches ~/Documents/cctv-yolo, ~/Desktop/cctv-yolo,"
        echo "  and ~/cctv-yolo. If none exists, it creates ~/Documents/cctv-yolo/."
        echo "  You can move that folder anywhere later — the app will find"
        echo "  it again automatically (the chosen path is remembered in"
        echo "  ~/Library/Application Support/CCTV-YOLO/data_root.txt)."
        echo "  Override with the CCTV_YOLO_DATA_DIR env var anytime."
    else
        echo "  Build FAILED (stage: $BUILD_STATUS, exit code: $code)"
        echo "=========================================="
        echo ""
        echo "Scroll up to see the error message. Common fixes:"
        echo "  - Make sure python3 is the arm64 native version (Apple Silicon):"
        echo "      python3 -c 'import platform; print(platform.machine())'"
        echo "    arm64 = native, x86_64 = Intel/Rosetta (wrong arch)."
        echo "  - Delete build_venv/ and re-run for a fresh install:"
        echo "      rm -rf build_venv/ dist/"
        echo "  - Confirm you have ~5 GB free disk space for the build_venv + dist."
    fi
    echo ""
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "  Building CCTV-YOLO v2 (Native) for macOS"
echo "=========================================="
echo ""
echo "This takes 2-15 minutes depending on whether deps are cached."
echo ""

# ---- arch sanity check -----------------------------------------------
HOST_ARCH="$(uname -m)"
PY_ARCH="$(python3 -c 'import platform; print(platform.machine())' 2>/dev/null || echo unknown)"
echo "Host architecture       : $HOST_ARCH"
echo "python3 architecture    : $PY_ARCH"
if [ "$HOST_ARCH" = "arm64" ] && [ "$PY_ARCH" = "x86_64" ]; then
    echo ""
    echo "  WARNING: you're on Apple Silicon but python3 is x86_64 (Rosetta)."
    echo "           The resulting .app will only run under Rosetta and"
    echo "           torch MPS acceleration will not work."
    echo "           Recommended: install python3 via the official"
    echo "           https://www.python.org/downloads/ installer or via"
    echo "           Homebrew (brew install python@3.12) and re-run."
    echo ""
    echo "           Continuing anyway in 5 seconds. Ctrl-C to abort."
    sleep 5
fi
echo ""

# ---- 1. venv ----------------------------------------------------------
BUILD_STATUS="venv"
if [ ! -d "build_venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv build_venv
else
    echo "[1/5] Reusing existing build_venv/"
fi
# shellcheck disable=SC1091
source build_venv/bin/activate

# ---- 2. dependencies --------------------------------------------------
BUILD_STATUS="dependencies"
echo ""
echo "[2/5] Installing dependencies..."
echo "  - Upgrading pip..."
pip install --upgrade pip
echo ""
echo "  - Installing requirements (torch, PySide6, ultralytics, opencv, etc.)..."
echo "    This is the long step. Progress bars will appear below."
pip install -r requirements.txt

# Verify pyinstaller is reachable (it's in requirements.txt; this catches
# venv-activation snafus).
if ! command -v pyinstaller >/dev/null 2>&1; then
    echo ""
    echo "ERROR: pyinstaller is not on PATH after dependency install."
    echo "  Delete build_venv/ and re-run this script."
    exit 1
fi

# ---- 3. PyInstaller ---------------------------------------------------
BUILD_STATUS="pyinstaller"
echo ""
echo "[3/5] Running PyInstaller (5-10 minutes)..."
pyinstaller cctv_yolo.spec --clean --noconfirm

if [ ! -d "dist/CCTV-YOLO.app" ]; then
    echo ""
    echo "ERROR: PyInstaller finished but dist/CCTV-YOLO.app is missing."
    echo "  Check cctv_yolo.spec and the warn-*.txt file in build/."
    exit 1
fi

# Strip macOS extended attributes from the built .app. PyInstaller's
# Python interpreter / dylibs sometimes inherit quarantine xattrs from
# the source pip cache, which makes Gatekeeper refuse to launch the .app
# on other machines with "damaged and can't be opened".
echo ""
echo "[3.5/5] Stripping extended attributes from the .app..."
xattr -cr "dist/CCTV-YOLO.app" 2>/dev/null || true

# Ad-hoc code-sign the whole bundle. Stripping xattrs alone does NOT stop the
# "CCTV-YOLO is damaged and can't be opened" error on OTHER Macs — that comes
# from an unsigned/invalid signature plus the quarantine flag the recipient's
# browser adds on download. A valid (even ad-hoc, "-") deep signature
# downgrades that hard block to the normal "unidentified developer →
# right-click Open" prompt, and is REQUIRED on Apple Silicon (every binary
# must carry at least an ad-hoc signature). This is NOT notarization: proper
# distribution still needs a Developer ID + `xcrun notarytool` + stapling
# (see the README), but ad-hoc signing is the best we can do without a paid
# Apple Developer account and makes the app launchable via right-click → Open.
echo "[3.6/5] Ad-hoc code-signing the .app..."
if command -v codesign >/dev/null 2>&1; then
    codesign --force --deep --sign - "dist/CCTV-YOLO.app" 2>/dev/null \
        && echo "  Signed (ad-hoc)." \
        || echo "  WARNING: ad-hoc codesign failed; first launch may need right-click > Open."
else
    echo "  (codesign not found — skipping; first launch may need right-click > Open.)"
fi

# ---- 4. DMG packaging -------------------------------------------------
BUILD_STATUS="dmg"
echo ""
echo "[4/5] Creating DMG..."

# Free disk check — hdiutil silently fails when /tmp or dist/ can't hold
# the staged .app + the compressed image.
# `|| true` keeps a failed du/df pipe from aborting the build under
# `set -e -o pipefail` right before the DMG step — the preflight is advisory.
APP_SIZE_KB=$(du -sk "dist/CCTV-YOLO.app" 2>/dev/null | awk '{print $1}' || true)
FREE_KB=$(df -k . 2>/dev/null | awk 'NR==2 {print $4}' || true)
if [ -n "$APP_SIZE_KB" ] && [ -n "$FREE_KB" ]; then
    NEEDED_KB=$(( APP_SIZE_KB * 3 ))  # staging + dmg + slack
    if [ "$FREE_KB" -lt "$NEEDED_KB" ]; then
        echo "ERROR: only ${FREE_KB} KB free, need ~${NEEDED_KB} KB for DMG."
        echo "  Free some disk space and re-run, or skip the DMG step."
        exit 1
    fi
else
    echo "  (skipping disk-space preflight — du/df unavailable)"
fi

DMG_DIR="dist/dmg_staging"
rm -rf "$DMG_DIR"
mkdir -p "$DMG_DIR"
cp -R "dist/CCTV-YOLO.app" "$DMG_DIR/"
ln -s /Applications "$DMG_DIR/Applications"

rm -f "dist/CCTV-YOLO.dmg"
if ! hdiutil create \
    -volname "CCTV-YOLO" \
    -srcfolder "$DMG_DIR" \
    -ov -format UDZO \
    "dist/CCTV-YOLO.dmg"; then
    echo ""
    echo "ERROR: hdiutil failed to create the DMG."
    echo "  Common causes: stale mount with the same volname, low disk,"
    echo "  or filesystem doesn't support APFS sparse images."
    echo "  Try: hdiutil detach /Volumes/CCTV-YOLO 2>/dev/null; re-run."
    rm -rf "$DMG_DIR"
    exit 1
fi

rm -rf "$DMG_DIR"

# ---- 5. done ----------------------------------------------------------
BUILD_STATUS="cleanup"
echo ""
echo "[5/5] Cleaning up..."
deactivate 2>/dev/null || true

BUILD_STATUS="success"
