@echo off
REM ============================================================
REM  Build CCTV-YOLO v2 (Native) for Windows
REM  Output: dist\CCTV-YOLO\CCTV-YOLO.exe
REM ============================================================
REM
REM Notes:
REM  - First build takes 15-25 minutes (downloads ~2.5 GB of torch +
REM    runs PyInstaller). Subsequent runs reuse build_venv\ and finish
REM    in about 5 minutes.
REM  - To force a clean rebuild: delete the build_venv\ folder first.
REM  - To build with CPU-only torch (smaller, no NVIDIA GPU support):
REM       set CCTV_YOLO_CPU_TORCH=1
REM    before running this script.
REM

echo.
echo ==========================================
echo   Building CCTV-YOLO v2 (Native) for Windows
echo ==========================================
echo.
echo This will take 5-25 minutes. Do NOT close this window.
echo.

cd /d "%~dp0"

REM ---------- 1. Virtual environment ----------
if not exist "build_venv" (
    echo [1/4] Creating virtual environment...
    python -m venv build_venv
    if errorlevel 1 (
        echo.
        echo ERROR: failed to create virtual environment.
        echo Make sure Python 3.10, 3.11, or 3.12 is installed and on PATH.
        echo Check with: python --version
        goto :fail
    )
) else (
    echo [1/4] Reusing existing build_venv\
)

call build_venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ERROR: could not activate build_venv. The venv may be corrupted.
    echo Delete the build_venv\ folder and re-run this script.
    goto :fail
)

REM ---------- 2. Dependencies ----------
echo.
echo [2/4] Installing dependencies (long download on first run)...

python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip upgrade failed.
    goto :fail
)

REM Try CUDA torch first for NVIDIA GPU support; fall back to CPU on
REM failure (e.g. no wheel for this Python version, network blocked).
if defined CCTV_YOLO_CPU_TORCH goto :install_cpu_torch

echo Installing CUDA torch (NVIDIA GPU support, ~2.5 GB)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo.
    echo NOTE: CUDA torch install failed. Falling back to CPU-only torch.
    echo.
    goto :install_cpu_torch
)
goto :torch_ok

:install_cpu_torch
echo Installing CPU-only torch (~250 MB)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo.
    echo ERROR: torch install failed.
    echo Try one of:
    echo   - Use Python 3.10, 3.11, or 3.12 (delete build_venv, re-run)
    echo   - Check your internet connection
    echo   - Run "python -m pip install torch torchvision" manually
    goto :fail
)

:torch_ok
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: requirements install failed.
    echo See the pip output above for the actual reason.
    goto :fail
)

REM ---------- 3. PyInstaller ----------
echo.
echo [3/4] Running PyInstaller (5-15 minutes)...

REM Send PyInstaller workpath / distpath to short %TEMP% subfolders.
REM Deep project paths (especially under OneDrive) combined with
REM PyInstaller's internal xcopy steps exceed cmd.exe's 8191-char
REM line limit and abort the build with "input line too long".
REM Short %TEMP%\cyb and %TEMP%\cyd keep the line lengths safe.
set "PYI_WORK=%TEMP%\cyb"
set "PYI_DIST=%TEMP%\cyd"
if exist "%PYI_WORK%" rmdir /S /Q "%PYI_WORK%"
if exist "%PYI_DIST%" rmdir /S /Q "%PYI_DIST%"

pyinstaller cctv_yolo.spec --clean --noconfirm --workpath "%PYI_WORK%" --distpath "%PYI_DIST%"
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller failed.
    echo Check warn-cctv_yolo.txt in %PYI_WORK% for missing imports.
    goto :fail
)

REM Move the build output from %TEMP% into the project's dist\ folder.
if exist "dist\CCTV-YOLO" rmdir /S /Q "dist\CCTV-YOLO"
if not exist "dist" mkdir "dist"
xcopy /E /I /Y /Q "%PYI_DIST%\CCTV-YOLO" "dist\CCTV-YOLO\" >nul
if errorlevel 1 (
    echo ERROR: failed to copy build output to dist\CCTV-YOLO\
    goto :fail
)

REM Ship the diagnostic launcher next to the exe (best-effort).
if exist "CCTV-YOLO-debug.bat" copy /Y "CCTV-YOLO-debug.bat" "dist\CCTV-YOLO\" >nul

REM ---------- 4. Done ----------
echo.
echo [4/4] Done!
call deactivate 2>nul

echo.
echo ==========================================
echo   Build complete!
echo ==========================================
echo.
echo   Executable: %CD%\dist\CCTV-YOLO\CCTV-YOLO.exe
echo   Folder    : %CD%\dist\CCTV-YOLO\
echo.
echo To run:
echo   1. Open the dist\CCTV-YOLO\ folder
echo   2. Double-click CCTV-YOLO.exe
echo.
echo Data folder:
echo   The app stores videos, tracks, corrections, models, and logs in a
echo   folder named cctv-yolo\ in your Documents folder by default. You
echo   can move that folder anywhere; the app will find it again on the
echo   next launch.
echo.
echo If CCTV-YOLO.exe will not open, run CCTV-YOLO-debug.bat in the same
echo folder. It captures the real startup error to startup-output.log.
echo.

start "" "%CD%\dist\CCTV-YOLO"

echo Press any key to close this window.
pause >nul
exit /b 0


:fail
echo.
echo ==========================================
echo   Build FAILED
echo ==========================================
echo.
echo Scroll up to see the actual error message.
echo Common fixes:
echo   - Use Python 3.10, 3.11, or 3.12 (check: python --version)
echo   - Delete the build_venv\ folder and re-run for a fresh install
echo   - Move the project out of OneDrive if it is syncing
echo   - Disable AV on this folder and on %%TEMP%%
echo.
echo Press any key to close this window.
pause >nul
exit /b 1
