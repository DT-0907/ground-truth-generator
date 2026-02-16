@echo off
REM ============================================================
REM  Build CCTV-YOLO v2 (Native) for Windows
REM  Output: dist\CCTV-YOLO\CCTV-YOLO.exe
REM ============================================================

echo ==========================================
echo   Building CCTV-YOLO v2 (Native) for Windows
echo ==========================================

cd /d "%~dp0"

REM 1. Create virtual environment if needed
if not exist "build_venv" (
    echo [1/4] Creating virtual environment...
    python -m venv build_venv
)

REM Activate
call build_venv\Scripts\activate.bat

REM 2. Install dependencies
echo [2/4] Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q

REM 3. Run PyInstaller
echo [3/4] Running PyInstaller...
pyinstaller cctv_yolo.spec --clean --noconfirm

REM 4. Done
echo [4/4] Done!
call deactivate 2>nul

echo.
echo ==========================================
echo   Build complete!
echo   Executable: dist\CCTV-YOLO\CCTV-YOLO.exe
echo ==========================================
echo.
echo To run:
echo   1. Copy dist\CCTV-YOLO\ folder to desired location
echo   2. Run CCTV-YOLO.exe
echo   3. Data will be created at %%USERPROFILE%%\Documents\CCTV-YOLO\
echo.
pause
