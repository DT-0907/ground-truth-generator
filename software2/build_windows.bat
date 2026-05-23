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

REM Install CUDA-enabled torch so the bundled exe automatically uses an
REM NVIDIA GPU when one is present (dramatically faster than CPU for YOLO).
REM On machines WITHOUT a CUDA GPU the same torch falls back to CPU at
REM runtime via `torch.cuda.is_available()`, so this is strictly better
REM than the old CPU-only build -- it just costs ~2 GB more on disk.
REM
REM To opt out and ship a smaller (~200 MB) CPU-only build, set the env
REM var CCTV_YOLO_CPU_TORCH=1 before running this script.
if defined CCTV_YOLO_CPU_TORCH (
    echo Installing CPU-only torch (CCTV_YOLO_CPU_TORCH=1 set)...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    echo Installing CUDA torch (auto-uses NVIDIA GPU when available, falls back to CPU)...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)
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
REM The spec uses collect_submodules('cctv_yolo') so any new module added
REM under cctv_yolo/ (theme.py, __version__.py, logging_config.py, widgets/,
REM etc.) is bundled automatically -- no spec edits required when adding
REM more pure-Python modules.
echo [3/4] Running PyInstaller (this can take 5-15 minutes)...

REM PyInstaller's COLLECT phase shells out internal xcopy/file-list commands
REM that include the full workpath + every binary path. With a deep project
REM path like C:\Users\<name>\Documents\cctv-yolo\software2\build\... those
REM internal commands routinely exceed cmd.exe's 8191-char limit and the
REM build aborts with "The command line is too long" / "input line too long".
REM
REM Fix: point --workpath and --distpath at a SHORT absolute path under %TEMP%.
REM Internal commands then stay well under the limit. After the build we copy
REM the result back into dist\CCTV-YOLO\ so the installer step and the
REM diagnostic-launcher copy still work the same as before.
set "PYI_WORK=%TEMP%\cyb"
set "PYI_DIST=%TEMP%\cyd"
if exist "%PYI_WORK%" rmdir /S /Q "%PYI_WORK%"
if exist "%PYI_DIST%" rmdir /S /Q "%PYI_DIST%"

REM --log-level=INFO makes progress visible so the build doesn't look frozen
REM while it churns through torch/ultralytics analysis.
pyinstaller cctv_yolo.spec --clean --noconfirm --log-level=INFO ^
    --workpath "%PYI_WORK%" --distpath "%PYI_DIST%"
if errorlevel 1 (
    echo ERROR: PyInstaller build failed. Check warn-cctv_yolo.txt in %PYI_WORK%\
    pause
    exit /b 1
)

REM Move the built app back into the project's dist\ so installer_windows.iss
REM and the rest of this script see it at the expected path.
if exist "dist\CCTV-YOLO" rmdir /S /Q "dist\CCTV-YOLO"
if not exist "dist" mkdir "dist"
xcopy /E /I /Y /Q "%PYI_DIST%\CCTV-YOLO" "dist\CCTV-YOLO\" >nul
if errorlevel 1 (
    echo ERROR: failed to copy build output from %PYI_DIST% to dist\CCTV-YOLO\
    pause
    exit /b 1
)

REM Ship the diagnostic launcher next to the exe. If CCTV-YOLO.exe won't
REM open, running CCTV-YOLO-debug.bat captures the real error to a log.
copy /Y "CCTV-YOLO-debug.bat" "dist\CCTV-YOLO\" >nul

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
echo If CCTV-YOLO.exe will not open:
echo   Run CCTV-YOLO-debug.bat instead. It captures the real error
echo   to startup-output.log next to the exe.
echo.
echo To package as installer:
echo   Run: iscc installer_windows.iss   (requires Inno Setup 6+)
echo.
pause
