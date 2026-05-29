@echo off
REM ============================================================
REM  Build CCTV-YOLO v2 (Native) for Windows
REM  Output: dist\CCTV-YOLO\CCTV-YOLO.exe + dist\CCTV-YOLO-Setup.exe
REM ============================================================
REM
REM Notes:
REM  - First build takes 15-25 minutes (downloads ~2.5 GB of torch +
REM    runs PyInstaller). Subsequent runs reuse the venv and finish in
REM    about 5 minutes.
REM
REM  - PYTHON VERSION: PyTorch publishes wheels for Python 3.10, 3.11, and
REM    3.12 ONLY. Python 3.13 and 3.14 have NO torch wheels yet, so a build
REM    on them HANGS forever in pip's resolver (it looks frozen). This
REM    script now refuses to build on an unsupported Python and prefers the
REM    `py -3.12` launcher automatically. If you only have 3.13/3.14
REM    installed, install Python 3.12 from python.org and re-run.
REM
REM  - The venv is created at a SHORT path under your user profile, not
REM    inside the project folder, so deep pip-install paths (PySide6,
REM    torch/cuda) don't exceed Windows' 260-char MAX_PATH limit.
REM    Default: %USERPROFILE%\.cctv_yolo_build_venv
REM    Override with the CCTV_YOLO_BUILD_VENV env var.
REM    A reused venv built with an unsupported Python is auto-recreated.
REM
REM  - GPU / CUDA: this script AUTO-DETECTS your GPU via nvidia-smi and
REM    picks the matching torch wheels:
REM       cu128  RTX 50-series (Blackwell) / driver CUDA >= 12.8
REM       cu126  driver CUDA >= 12.6
REM       cu124  driver CUDA >= 12.4
REM       cu121  driver CUDA >= 12.1
REM       cu118  older NVIDIA GPUs / drivers
REM       cpu    no NVIDIA GPU detected (~250 MB instead of ~2.5 GB)
REM    Override the auto-pick:  set CCTV_YOLO_TORCH_VARIANT=cu128
REM    Legacy alias:            set CCTV_YOLO_CPU_TORCH=1   (forces cpu)
REM
REM    IMPORTANT: cu118/cu121/cu124/cu126 have NO kernels for Blackwell
REM    (RTX 50-series, compute sm_120). Those GPUs MUST use cu128, which is
REM    why auto-detection exists — the old fixed cu118 default ran such
REM    cards on CPU or crashed with "no kernel image is available".
REM

echo.
echo ==========================================
echo   Building CCTV-YOLO v2 (Native) for Windows
echo ==========================================
echo.
echo This will take 5-25 minutes. Do NOT close this window.
echo.

cd /d "%~dp0"

REM ---------- 0. Find a supported Python (3.10-3.12) ----------
REM Prefer the py launcher with an explicit version so we never grab a
REM too-new 3.13/3.14 that torch has no wheels for.
set "PYCMD="
call :try_py "py -3.12"
call :try_py "py -3.11"
call :try_py "py -3.10"
call :try_py "python"
call :try_py "py"
if not defined PYCMD (
    echo.
    echo ERROR: No supported Python found on this machine.
    echo.
    echo PyTorch ships wheels for Python 3.10, 3.11, and 3.12 ONLY.
    echo Your "python" is most likely 3.13 or 3.14 ^(too new^) or missing.
    echo This is the exact cause of the "pip install torch ... hangs forever"
    echo problem: with no matching wheel, pip's resolver never finishes.
    echo.
    echo Fix: install Python 3.12 from
    echo   https://www.python.org/downloads/
    echo tick "Add python.exe to PATH" during setup, then re-run this script.
    echo.
    echo Detected on PATH:
    python --version 2>nul
    py --version 2>nul
    goto :fail
)
echo Using Python interpreter: %PYCMD%
%PYCMD% --version
echo.

REM Resolve the venv location (short path -> stays under MAX_PATH).
if defined CCTV_YOLO_BUILD_VENV (
    set "VENV_DIR=%CCTV_YOLO_BUILD_VENV%"
) else (
    set "VENV_DIR=%USERPROFILE%\.cctv_yolo_build_venv"
)
echo Using venv at: %VENV_DIR%
echo.

REM ---------- 1. Virtual environment ----------
REM Validate a REUSED venv's interpreter. A venv built with 3.13/3.14
REM (the bug that bit a tester) is recreated instead of silently reused.
if exist "%VENV_DIR%\Scripts\python.exe" (
    "%VENV_DIR%\Scripts\python.exe" -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo [1/5] Existing venv uses an unsupported Python version - recreating it...
        rmdir /S /Q "%VENV_DIR%"
    ) else (
        echo [1/5] Reusing existing venv at %VENV_DIR%
    )
)
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/5] Creating virtual environment with %PYCMD% ...
    %PYCMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo ERROR: failed to create virtual environment.
        goto :fail
    )
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
echo [2/5] Installing dependencies (long download on first run)...

python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip upgrade failed.
    goto :fail
)

REM Resolve which torch wheel index to use.
REM   - CCTV_YOLO_CPU_TORCH=1 (legacy)  -> cpu
REM   - CCTV_YOLO_TORCH_VARIANT set      -> honored as-is (manual override)
REM   - otherwise                        -> auto-detected from nvidia-smi
if defined CCTV_YOLO_CPU_TORCH set "CCTV_YOLO_TORCH_VARIANT=cpu"
if not defined CCTV_YOLO_TORCH_VARIANT (
    echo Detecting GPU / CUDA to choose the right torch wheels...
    for /f "usebackq delims=" %%v in (`python detect_torch_variant.py`) do set "CCTV_YOLO_TORCH_VARIANT=%%v"
)
if not defined CCTV_YOLO_TORCH_VARIANT set "CCTV_YOLO_TORCH_VARIANT=cu118"
echo Selected torch variant: %CCTV_YOLO_TORCH_VARIANT%

set "TORCH_INDEX=https://download.pytorch.org/whl/%CCTV_YOLO_TORCH_VARIANT%"
echo Installing torch (%CCTV_YOLO_TORCH_VARIANT%) from %TORCH_INDEX%
python -m pip install torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo.
    echo NOTE: torch (%CCTV_YOLO_TORCH_VARIANT%) install failed. Falling back to CPU-only torch.
    echo.
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo.
        echo ERROR: torch install failed.
        echo Try one of:
        echo   - Confirm Python 3.10, 3.11, or 3.12 (delete the venv folder, re-run)
        echo   - Check your internet connection
        echo   - Run "python -m pip install torch torchvision" manually
        goto :fail
    )
    set "CCTV_YOLO_TORCH_VARIANT=cpu"
)

REM Verify the installed torch and catch a SILENT CPU-wheel fallback (pip
REM sometimes resolves a CPU wheel even from a CUDA index). Exit code 2 =
REM we asked for CUDA but got CPU-only torch.
echo.
echo Verifying torch install...
python -c "import torch,sys; cuda=torch.version.cuda; print('  torch version :', torch.__version__); print('  CUDA build    :', cuda or '(CPU-only)'); print('  CUDA available:', torch.cuda.is_available()); sys.exit(2 if ('%CCTV_YOLO_TORCH_VARIANT%'!='cpu' and not cuda) else 0)"
if errorlevel 2 (
    echo.
    echo WARNING: a CUDA torch was requested (%CCTV_YOLO_TORCH_VARIANT%) but a
    echo CPU-only torch got installed. The app will run on CPU. This usually
    echo means no matching CUDA wheel exists for your Python version. Use
    echo Python 3.12 and re-run, or set CCTV_YOLO_TORCH_VARIANT explicitly.
    echo.
) else (
    REM exit code 1 = the verification python itself failed (torch did not
    REM even import). Surface it instead of silently continuing.
    if errorlevel 1 (
        echo.
        echo WARNING: could not import torch for verification - it may have
        echo failed to install correctly. Continuing, but the build may not run.
        echo.
    )
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: requirements install failed.
    echo.
    echo If the error mentions "Windows Long Path support" or "[Errno 2] No
    echo such file or directory" with a very long path, move the project to a
    echo shorter path like C:\cctv-yolo\ and re-run.
    goto :fail
)

REM ---------- 3. PyInstaller ----------
echo.
echo [3/5] Running PyInstaller (5-15 minutes)...

REM Send PyInstaller workpath / distpath to short %TEMP% subfolders so deep
REM project paths don't blow cmd.exe's 8191-char line limit.
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

REM Ship the end-user MOTW-strip helper alongside the exe.
if exist "Unblock.bat" copy /Y "Unblock.bat" "dist\CCTV-YOLO\" >nul

REM Strip Mark-of-the-Web (Zone.Identifier ADS) from the bundle on this
REM build machine. End users still re-Unblock after extracting the ZIP.
echo.
echo Stripping Mark-of-the-Web from build output...
powershell -NoProfile -Command "Get-ChildItem -LiteralPath 'dist\CCTV-YOLO' -Recurse -File | Unblock-File" 2>nul

REM ---------- 4. Installer (Inno Setup) ----------
echo.
echo [4/5] Building single-file installer (Inno Setup)...

REM Read the app version from the single source of truth (__version__.py).
set "APP_VERSION="
for /f "usebackq delims=" %%v in (`powershell -NoProfile -Command "(Select-String -Path 'cctv_yolo\__version__.py' -Pattern '__version__\s*=\s*\"([^\"]+)\"').Matches.Groups[1].Value"`) do set "APP_VERSION=%%v"
if not defined APP_VERSION set "APP_VERSION=0.0.0"
echo   App version: %APP_VERSION%

REM Locate the Inno Setup compiler (ISCC.exe).
set "ISCC="
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC for %%I in (iscc.exe ISCC.exe) do if not defined ISCC if exist "%%~$PATH:I" set "ISCC=%%~$PATH:I"

if defined ISCC (
    echo   Using Inno Setup: %ISCC%
    "%ISCC%" /DAppVersion=%APP_VERSION% installer_windows.iss
    if errorlevel 1 (
        echo.
        echo WARNING: Inno Setup compile failed. The folder build in
        echo dist\CCTV-YOLO\ is still usable; only the single-file
        echo installer was not produced.
    ) else (
        set "INSTALLER_BUILT=1"
    )
) else (
    echo.
    echo NOTE: Inno Setup not found - skipping single-file installer.
    echo The folder build in dist\CCTV-YOLO\ still works.
    echo To get CCTV-YOLO-Setup.exe, install Inno Setup 6 from
    echo   https://jrsoftware.org/isdl.php
    echo and re-run this script.
)

REM ---------- 5. Done ----------
echo.
echo [5/5] Done!
call deactivate 2>nul

echo.
echo ==========================================
echo   Build complete!
echo ==========================================
echo.
if defined INSTALLER_BUILT (
    echo   SHARE THIS:  %CD%\dist\CCTV-YOLO-Setup.exe
    echo                ^(single-file installer - this is what you send^)
    echo.
)
echo   Folder build: %CD%\dist\CCTV-YOLO\  ^(CCTV-YOLO.exe inside^)
echo   Build venv  : %VENV_DIR%
echo   torch wheel : %CCTV_YOLO_TORCH_VARIANT%
echo.
echo To run locally now:
echo   1. Open the dist\CCTV-YOLO\ folder (opening it now)
echo   2. Double-click CCTV-YOLO.exe  (no console window - production mode)
echo.
echo Data folder:
echo   The app stores videos, tracks, corrections, models, and logs in a
echo   folder named cctv-yolo\ in your Documents folder by default.
echo.
echo If CCTV-YOLO.exe will not open, run CCTV-YOLO-debug.bat in the same
echo folder. It launches CCTV-YOLO-debug.exe (console build) and captures
echo the real startup error to startup-output.log.
echo.

start "" "%CD%\dist\CCTV-YOLO"

echo Press any key to close this window.
pause >nul
exit /b 0


:try_py
REM %~1 = a candidate launcher command (e.g. "py -3.12" or "python").
REM Sets PYCMD to the first one that reports Python 3.10-3.12. Skips the
REM rest once PYCMD is set. Missing launchers fail the probe harmlessly.
if defined PYCMD goto :eof
%~1 -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
if not errorlevel 1 set "PYCMD=%~1"
goto :eof


:fail
echo.
echo ==========================================
echo   Build FAILED
echo ==========================================
echo.
echo Scroll up to see the actual error message.
echo.
echo Common fixes:
echo   - Use Python 3.10, 3.11, or 3.12 (NOT 3.13/3.14). Install 3.12 from
echo     https://www.python.org/downloads/ and re-run.
echo   - Delete the venv folder "%VENV_DIR%" and re-run for a fresh install
echo   - Move the project to a SHORT path like C:\cctv-yolo\ if the error
echo     mentions long paths or "[Errno 2] No such file or directory"
echo   - Disable AV on the project folder and on %%TEMP%%
echo.
echo Press any key to close this window.
pause >nul
exit /b 1
