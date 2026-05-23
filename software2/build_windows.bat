@echo off
REM ============================================================
REM  Build CCTV-YOLO v2 (Native) for Windows
REM  Output: dist\CCTV-YOLO\CCTV-YOLO.exe
REM ============================================================
REM
REM Notes:
REM  - First build takes 15-25 minutes (downloads ~2.5 GB of torch +
REM    runs PyInstaller). Subsequent runs reuse the venv and finish in
REM    about 5 minutes.
REM
REM  - The venv is created at a SHORT path under your user profile,
REM    not inside the project folder. This is so deep pip-install paths
REM    (PySide6, torch/cuda) don't exceed Windows' 260-char MAX_PATH
REM    limit when the project itself is under a long path like
REM    C:\Users\<name>\Downloads\ground-truth-generator-main\...
REM
REM    Default venv location: %USERPROFILE%\.cctv_yolo_build_venv
REM    Override with the CCTV_YOLO_BUILD_VENV env var if you want to
REM    put it somewhere else.
REM
REM  - To force a clean rebuild: delete the venv folder above (NOT the
REM    project folder), then re-run this script.
REM
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

REM Resolve the venv location. Default to a short path under %USERPROFILE%
REM so deeply-nested site-packages paths stay under Windows MAX_PATH.
if defined CCTV_YOLO_BUILD_VENV (
    set "VENV_DIR=%CCTV_YOLO_BUILD_VENV%"
) else (
    set "VENV_DIR=%USERPROFILE%\.cctv_yolo_build_venv"
)
echo Using venv at: %VENV_DIR%
echo.

REM ---------- 1. Virtual environment ----------
if not exist "%VENV_DIR%" (
    echo [1/4] Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo ERROR: failed to create virtual environment.
        echo Make sure Python 3.10, 3.11, or 3.12 is installed and on PATH.
        echo Check with: python --version
        goto :fail
    )
) else (
    echo [1/4] Reusing existing venv at %VENV_DIR%
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo ERROR: could not activate venv. The venv may be corrupted.
    echo Delete the folder "%VENV_DIR%" and re-run this script.
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
    echo   - Use Python 3.10, 3.11, or 3.12 (delete the venv folder, re-run)
    echo   - Check your internet connection
    echo   - Run "python -m pip install torch torchvision" manually
    goto :fail
)

:torch_ok
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: requirements install failed.
    echo.
    echo If the error mentions "Windows Long Path support" or "[Errno 2] No
    echo such file or directory" with a very long path, your project folder
    echo is too deep. Move it to a shorter path like C:\cctv-yolo\ and
    echo re-run. The venv itself is already at a short path, so this should
    echo be rare.
    goto :fail
)

REM ---------- 3. PyInstaller ----------
echo.
echo [3/4] Running PyInstaller (5-15 minutes)...

REM Send PyInstaller workpath / distpath to short %TEMP% subfolders.
REM Deep project paths combined with PyInstaller's internal xcopy steps
REM exceed cmd.exe's 8191-char line limit and abort with "input line
REM too long". Short %TEMP%\cyb and %TEMP%\cyd keep that under control.
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
echo   Build venv: %VENV_DIR%
echo.
echo To run:
echo   1. Open the dist\CCTV-YOLO\ folder (opening it now)
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
echo.
echo Common fixes:
echo   - Use Python 3.10, 3.11, or 3.12 (check: python --version)
echo   - Delete the venv folder "%VENV_DIR%" and re-run for a fresh install
echo   - Move the project to a SHORT path like C:\cctv-yolo\ if the error
echo     mentions long paths or "[Errno 2] No such file or directory"
echo   - Disable AV on the project folder and on %%TEMP%%
echo.
echo Press any key to close this window.
pause >nul
exit /b 1
