@echo off
REM ============================================================
REM  Build CCTV-YOLO v2 (Native) for Windows
REM  Output: dist\CCTV-YOLO\CCTV-YOLO.exe
REM ============================================================
REM
REM NOTE: %TEMP% is sometimes inside OneDrive or actively scanned by AV
REM (Windows Defender real-time protection in particular). If the build
REM fails with cryptic permission errors during PyInstaller's COLLECT
REM phase, exclude %TEMP%\cyb and %TEMP%\cyd from AV scanning or move
REM the working dirs outside any OneDrive-synced location.
REM
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

REM Python version sanity check. PyTorch wheels exist for 3.10..3.12 on
REM Windows. 3.13+ wheels are slow to arrive, especially for CUDA builds.
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo Detected Python: %PY_VER%
echo %PY_VER% | findstr /R "^3\.1[0-2]\." >nul
if errorlevel 1 (
    echo.
    echo WARNING: Python %PY_VER% may not have prebuilt torch wheels.
    echo          Recommended: install Python 3.10, 3.11, or 3.12 from
    echo          https://www.python.org/downloads/ and re-run this script.
    echo.
    echo Continuing anyway — pip will fail loudly if wheels are missing.
    echo.
)

REM Pip needs to be current. `python -m pip` avoids the file-lock issue
REM that hits when `pip install --upgrade pip` rewrites its own .exe.
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip upgrade failed.
    echo   Check your internet connection and that python.exe is on PATH.
    pause
    exit /b 1
)

REM Install torch. Default = CUDA wheels (cu121) so the bundled exe uses
REM an NVIDIA GPU when present. On machines without CUDA the same wheels
REM transparently fall back to CPU via torch.cuda.is_available().
REM
REM If the CUDA wheel install fails (no Python wheel available for this
REM Python version, blocked network, etc.) we AUTOMATICALLY retry with
REM CPU-only wheels so the build can still complete.
REM
REM To skip the CUDA attempt entirely, set CCTV_YOLO_CPU_TORCH=1 first.
REM Use !ERRORLEVEL! (delayed expansion, requires `setlocal enabledelayedexpansion`
REM at the top of this file) — `if errorlevel N` nested inside an outer if/else
REM block evaluates inconsistently in some cmd.exe versions.
set TORCH_INSTALLED=0
if defined CCTV_YOLO_CPU_TORCH (
    echo Installing CPU-only torch (CCTV_YOLO_CPU_TORCH=1 set)...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if !ERRORLEVEL! EQU 0 set TORCH_INSTALLED=1
) else (
    echo Installing CUDA torch (auto-uses NVIDIA GPU; falls back to CPU at runtime)...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if !ERRORLEVEL! EQU 0 (
        set TORCH_INSTALLED=1
    ) else (
        echo.
        echo WARNING: CUDA torch install failed for Python !PY_VER!.
        echo          Falling back to CPU-only torch so the build can continue.
        echo.
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        if !ERRORLEVEL! EQU 0 set TORCH_INSTALLED=1
    )
)
if "!TORCH_INSTALLED!"=="0" (
    echo ERROR: both CUDA and CPU torch installs failed.
    echo   Try one of:
    echo     - Use Python 3.10, 3.11, or 3.12 (delete build_venv, re-run).
    echo     - Check your network connection and any corporate proxy.
    echo     - Run "python -m pip install torch torchvision" manually to see
    echo       the full error.
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: requirements install failed.
    echo   See full pip output above. Often this is ultralytics or
    echo   opencv-python — try "python -m pip install ultralytics
    echo   opencv-python" manually to see the underlying error.
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
REM (Best-effort — don't fail the whole build if the debug launcher is
REM missing for some reason.)
if exist "CCTV-YOLO-debug.bat" (
    copy /Y "CCTV-YOLO-debug.bat" "dist\CCTV-YOLO\" >nul
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
echo If CCTV-YOLO.exe will not open:
echo   Run CCTV-YOLO-debug.bat instead. It captures the real error
echo   to startup-output.log next to the exe.
echo.
echo To package as installer:
echo   Run: iscc installer_windows.iss   (requires Inno Setup 6+)
echo.
pause
