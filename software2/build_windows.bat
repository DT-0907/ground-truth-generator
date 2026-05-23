@echo off
REM ============================================================
REM  Build CCTV-YOLO v2 (Native) for Windows
REM  Output: dist\CCTV-YOLO\CCTV-YOLO.exe
REM ============================================================
REM
REM Notes for first-time builders:
REM
REM   - First build takes ~15-25 minutes (downloads ~2.5 GB of CUDA torch
REM     wheels + runs PyInstaller). Subsequent builds reuse build_venv and
REM     finish in ~5 minutes.
REM
REM   - If %TEMP% is inside OneDrive or actively scanned by AV (Windows
REM     Defender real-time protection in particular), PyInstaller's COLLECT
REM     phase will get cryptic permission errors. Exclude %TEMP%\cyb and
REM     %TEMP%\cyd from AV scanning or move them outside any OneDrive-synced
REM     location if that happens.
REM
REM   - To force a clean rebuild, delete the build_venv\ folder before running.
REM     Otherwise the script reuses the existing venv (much faster).
REM
REM   - To skip the CUDA torch download (smaller build, no GPU), set the env
REM     var CCTV_YOLO_CPU_TORCH=1 before running.
REM
setlocal enabledelayedexpansion

REM Make em-dashes / other utf-8 chars render in the console.
chcp 65001 >nul 2>&1

REM Make sure we always end at :end so the terminal stays open even on
REM unexpected exits (syntax errors, etc.).
set "BUILD_STATUS=unknown"
set "BUILD_EXIT_CODE=0"

echo.
echo ==========================================
echo   Building CCTV-YOLO v2 (Native) for Windows
echo ==========================================
echo.
echo This will take 5-25 minutes depending on whether dependencies are
echo cached. The terminal will stay open at the end. Do not close it.
echo.

cd /d "%~dp0"

REM ---------- 1. Virtual environment ----------
if not exist "build_venv" (
    echo [1/5] Creating virtual environment...
    python -m venv build_venv
    if errorlevel 1 (
        echo.
        echo ERROR: failed to create virtual environment.
        echo   Make sure Python 3.10, 3.11, or 3.12 is installed and on PATH.
        echo   Verify with: python --version
        set "BUILD_STATUS=failed-venv"
        set "BUILD_EXIT_CODE=1"
        goto :end
    )
) else (
    echo [1/5] Reusing existing build_venv\
)

call build_venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ERROR: couldn't activate build_venv. The venv may be corrupted.
    echo   Delete the build_venv\ folder and re-run this script.
    set "BUILD_STATUS=failed-activate"
    set "BUILD_EXIT_CODE=1"
    goto :end
)

REM ---------- 2. Dependencies ----------
echo.
echo [2/5] Installing dependencies (this is where the long download happens)...

REM Python version sanity check. PyTorch wheels exist for 3.10..3.12 on
REM Windows. 3.13+ wheels are slow to arrive, especially for CUDA builds.
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
echo   Detected Python: !PY_VER!
echo !PY_VER! | findstr /R "^3\.1[0-2]\." >nul
if errorlevel 1 (
    echo.
    echo   WARNING: Python !PY_VER! may not have prebuilt torch wheels.
    echo            Recommended: install Python 3.10, 3.11, or 3.12 from
    echo            https://www.python.org/downloads/ and re-run this script.
    echo            Continuing anyway -- pip will fall back to CPU torch.
    echo.
)

REM Pip upgrade. `python -m pip` avoids the file-lock issue that hits when
REM `pip install --upgrade pip` rewrites its own .exe on Windows.
echo   - Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo ERROR: pip upgrade failed.
    echo   Check your internet connection and that python.exe is on PATH.
    set "BUILD_STATUS=failed-pip"
    set "BUILD_EXIT_CODE=1"
    goto :end
)

REM Install torch. Default = CUDA wheels (cu121) so the bundled exe uses
REM an NVIDIA GPU when present. On machines without CUDA the same wheels
REM transparently fall back to CPU via torch.cuda.is_available().
REM
REM If the CUDA wheel install fails (no Python wheel available for this
REM Python version, blocked network, etc.) we AUTOMATICALLY retry with
REM CPU-only wheels so the build can still complete.
echo.
echo   - Installing PyTorch (this is the long one -- 2.5 GB for CUDA, ~250 MB for CPU)...
echo     Be patient; pip prints a progress bar but it can pause briefly.
set "TORCH_INSTALLED=0"
if defined CCTV_YOLO_CPU_TORCH (
    echo     Mode: CPU-only ^(CCTV_YOLO_CPU_TORCH=1 set^)
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if !ERRORLEVEL! EQU 0 set "TORCH_INSTALLED=1"
) else (
    echo     Mode: CUDA ^(auto-uses NVIDIA GPU; falls back to CPU at runtime^)
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if !ERRORLEVEL! EQU 0 (
        set "TORCH_INSTALLED=1"
    ) else (
        echo.
        echo     WARNING: CUDA torch install failed for Python !PY_VER!.
        echo              Falling back to CPU-only torch so the build can continue.
        echo.
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        if !ERRORLEVEL! EQU 0 set "TORCH_INSTALLED=1"
    )
)
if "!TORCH_INSTALLED!"=="0" (
    echo.
    echo ERROR: both CUDA and CPU torch installs failed.
    echo   Try one of:
    echo     - Use Python 3.10, 3.11, or 3.12 ^(delete build_venv, re-run^).
    echo     - Check your network connection and any corporate proxy.
    echo     - Run "python -m pip install torch torchvision" manually to see
    echo       the full error.
    set "BUILD_STATUS=failed-torch"
    set "BUILD_EXIT_CODE=1"
    goto :end
)

echo.
echo   - Installing remaining requirements ^(PySide6, ultralytics, opencv, etc.^)...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: requirements install failed.
    echo   See full pip output above. Often this is ultralytics or
    echo   opencv-python -- try "python -m pip install ultralytics
    echo   opencv-python" manually to see the underlying error.
    set "BUILD_STATUS=failed-requirements"
    set "BUILD_EXIT_CODE=1"
    goto :end
)

REM Verify pyinstaller actually landed.
where pyinstaller >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: pyinstaller is not on PATH after dependency install.
    echo   This usually means the venv didn't activate properly.
    echo   Delete build_venv\ and try again.
    set "BUILD_STATUS=failed-pyinstaller-missing"
    set "BUILD_EXIT_CODE=1"
    goto :end
)

REM ---------- 3. PyInstaller ----------
REM The spec uses collect_submodules('cctv_yolo') so any new module added
REM under cctv_yolo\ (theme.py, __version__.py, logging_config.py, widgets\,
REM exports\, etc.) is bundled automatically -- no spec edits required.
echo.
echo [3/5] Running PyInstaller ^(this takes another 5-15 minutes^)...

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
    echo.
    echo ERROR: PyInstaller build failed.
    echo   Check warn-cctv_yolo.txt in %PYI_WORK%\ for missing imports.
    echo   Common causes: AV blocking writes, missing C++ redistributable,
    echo   or the project path is on a OneDrive-synced folder.
    set "BUILD_STATUS=failed-pyinstaller"
    set "BUILD_EXIT_CODE=1"
    goto :end
)

REM ---------- 4. Move result ----------
echo.
echo [4/5] Copying build output to dist\CCTV-YOLO\ ...

if exist "dist\CCTV-YOLO" rmdir /S /Q "dist\CCTV-YOLO"
if not exist "dist" mkdir "dist"
xcopy /E /I /Y /Q "%PYI_DIST%\CCTV-YOLO" "dist\CCTV-YOLO\" >nul
if errorlevel 1 (
    echo.
    echo ERROR: failed to copy build output from %PYI_DIST% to dist\CCTV-YOLO\
    echo   Check that you have write permission to "%CD%\dist\"
    set "BUILD_STATUS=failed-copy"
    set "BUILD_EXIT_CODE=1"
    goto :end
)

REM Ship the diagnostic launcher next to the exe. If CCTV-YOLO.exe won't
REM open, running CCTV-YOLO-debug.bat captures the real error to a log.
if exist "CCTV-YOLO-debug.bat" (
    copy /Y "CCTV-YOLO-debug.bat" "dist\CCTV-YOLO\" >nul
)

REM ---------- 5. Done ----------
echo.
echo [5/5] Done!
call deactivate 2>nul

set "BUILD_STATUS=success"

REM ---------- :end ----------
:end
echo.
echo ==========================================
if "%BUILD_STATUS%"=="success" (
    echo   Build complete!
    echo ==========================================
    echo.
    echo   Executable : %CD%\dist\CCTV-YOLO\CCTV-YOLO.exe
    echo   Folder     : %CD%\dist\CCTV-YOLO\
    echo.
    echo   Opening the folder in Explorer now...
    start "" "%CD%\dist\CCTV-YOLO"
    echo.
    echo To run on this machine: double-click CCTV-YOLO.exe inside that folder.
    echo To share: copy the entire dist\CCTV-YOLO\ folder to another machine.
    echo.
    echo Data location:
    echo   All videos / tracks / corrections / models / logs are stored IN
    echo   THE SAME FOLDER as the .exe -- so the install is portable. Just
    echo   copy the dist\CCTV-YOLO\ folder anywhere to move everything together.
    echo   Override with the CCTV_YOLO_DATA_DIR env var if you want a different
    echo   location.
    echo.
    echo If CCTV-YOLO.exe will not open, run CCTV-YOLO-debug.bat instead --
    echo it captures the real startup error to startup-output.log.
    echo.
    echo To package as installer:
    echo   iscc installer_windows.iss      ^(requires Inno Setup 6+^)
) else (
    echo   Build FAILED ^(stage: %BUILD_STATUS%^)
    echo ==========================================
    echo.
    echo Scroll up to see the error message. Common fixes:
    echo   - Use Python 3.10, 3.11, or 3.12 ^(check: python --version^)
    echo   - Delete build_venv\ and re-run for a fresh install
    echo   - Disable AV on this folder and on %%TEMP%%\
    echo   - Move the project out of OneDrive if it's syncing
)
echo.
echo Press any key to close this window...
pause >nul
exit /b %BUILD_EXIT_CODE%
