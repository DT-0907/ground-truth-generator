@echo off
REM ============================================================
REM  Build CCTV-YOLO v2 (Native) for Windows
REM  Output: dist\CCTV-YOLO\CCTV-YOLO.exe
REM ============================================================
setlocal enabledelayedexpansion

echo ==========================================
echo   Building CCTV-YOLO v2 (Native) for Windows
echo ==========================================

cd /d "%~dp0"

REM ---------- 1. Virtual environment ----------
if not exist "build_venv" (
    echo [1/4] Creating virtual environment...
    python -m venv build_venv
    if errorlevel 1 (
        echo ERROR: failed to create virtual environment.
        echo Make sure Python 3.10+ is installed and on PATH.
        pause
        exit /b 1
    )
)

call build_venv\Scripts\activate.bat

REM ---------- 2. Dependencies ----------
echo [2/4] Installing dependencies...

REM Pip needs to be current — using `pip install --upgrade pip` can fail mid-replace
REM on Windows. `python -m pip` avoids the file-lock issue.
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip upgrade failed.
    pause
    exit /b 1
)

REM Default `pip install torch` on Windows pulls the CUDA wheels (~2.5 GB).
REM CPU-only wheels are ~200 MB, install in seconds, and run fine on any
REM Windows machine. Install these FIRST so the resolver doesn't pick CUDA
REM when ultralytics later asks for torch.
echo Installing CPU-only torch (smaller + faster than CUDA wheels)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: torch install failed.
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: requirements install failed.
    pause
    exit /b 1
)

REM ---------- 3. PyInstaller ----------
echo [3/4] Running PyInstaller (this can take 5-15 minutes)...

REM --log-level=INFO makes progress visible so the build doesn't look frozen
REM while it churns through torch/ultralytics analysis.
pyinstaller cctv_yolo.spec --clean --noconfirm --log-level=INFO
if errorlevel 1 (
    echo ERROR: PyInstaller build failed. Check warn-cctv_yolo.txt in build\
    pause
    exit /b 1
)

REM ---------- 4. Done ----------
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
echo To package as installer:
echo   Run: iscc installer_windows.iss   (requires Inno Setup 6+)
echo.
pause
