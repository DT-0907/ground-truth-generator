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
REM    script prefers the `py -3.12` launcher automatically; if NO supported
REM    Python is found it OFFERS TO AUTO-INSTALL Python 3.12 for you
REM    (per-user, no admin needed) from python.org, then builds with it.
REM
REM  - The venv is created at a SHORT path under your user profile, not
REM    inside the project folder, so deep pip-install paths (PySide6,
REM    torch/cuda) don't exceed Windows' 260-char MAX_PATH limit.
REM    Default: %USERPROFILE%\.cctv_yolo_build_venv
REM    Override with the CCTV_YOLO_BUILD_VENV env var.
REM    A reused venv built with an unsupported Python is auto-recreated.
REM
REM  - GPU / CUDA (HYBRID): this build bakes a UNIVERSAL CPU PyTorch into the
REM    installer, so the app runs on ANY PC out of the box. GPU acceleration is
REM    NOT baked in. On first launch, if an NVIDIA GPU is detected, the app
REM    offers to download the matching CUDA build of PyTorch (cu128 for
REM    RTX 50-series / Blackwell, cu118 for older drivers) into a per-user
REM    folder, then uses it after a restart. See cctv_yolo/gpu_runtime.py.
REM    Result: ONE installer works everywhere — CPU and any NVIDIA GPU
REM    (including Blackwell, which needs cu128 and is handled at runtime).
REM

echo.
echo ==========================================
echo   Building CCTV-YOLO v2 (Native) for Windows
echo ==========================================
echo.
echo This will take 5-25 minutes. Do NOT close this window.
echo.

cd /d "%~dp0"

REM Python version to auto-install if no supported interpreter is found.
set "PY_VER=3.12.8"

REM ---------- 0. Find a supported Python (3.10-3.12) ----------
REM Prefer the py launcher with an explicit version so we never grab a
REM too-new 3.13/3.14 that torch has no wheels for.
set "PYCMD="
call :try_py "py -3.12"
call :try_py "py -3.11"
call :try_py "py -3.10"
call :try_py "python"
call :try_py "py"
REM Nothing supported on the machine? Offer to auto-install Python 3.12.
if not defined PYCMD call :install_python
if not defined PYCMD (
    echo.
    echo ERROR: No supported Python ^(3.10-3.12^) is available, and the
    echo automatic install did not complete.
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

REM PyTorch CPU baseline. HYBRID packaging: the shipped app bakes a UNIVERSAL
REM CPU PyTorch (staged as a data tree by cctv_yolo.spec). GPU acceleration is
REM a per-machine, first-run DOWNLOAD handled at runtime by
REM cctv_yolo.gpu_runtime, so the build only needs CPU torch here. Pinned to
REM match the runtime GPU pin (torch 2.8.0 / torchvision 0.23.0) so the baked
REM CPU build and a downloaded CUDA build line up.
echo.
echo Installing CPU PyTorch baseline (torch 2.8.0 + torchvision 0.23.0)...
echo (GPU acceleration is offered automatically on first run if an NVIDIA card is present.)
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo.
    echo ERROR: CPU torch 2.8.0 install failed. The baked baseline must match
    echo the GPU runtime pin ^(2.8.0^), so this does NOT fall back to a different
    echo version. Try one of:
    echo   - Confirm Python 3.10, 3.11, or 3.12 ^(delete the venv folder, re-run^)
    echo   - Check your internet connection
    goto :fail
)

echo.
echo Verifying torch install...
python -c "import torch; print('  torch version :', torch.__version__); print('  build         : CPU baseline (GPU is a first-run download)')"
if errorlevel 1 (
    echo WARNING: could not import torch for verification - continuing anyway.
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
REM Parse the version with findstr (no PowerShell, no parens) so cmd's
REM for /f paren-matcher can't choke on parentheses inside an embedded
REM command. tokens=2 delims== grabs the ` "2.2.0"` half; then strip the
REM surrounding spaces and quotes.
for /f "usebackq tokens=2 delims==" %%v in (`findstr /b /c:"__version__" "cctv_yolo\__version__.py"`) do set "APP_VERSION=%%v"
set "APP_VERSION=%APP_VERSION: =%"
set APP_VERSION=%APP_VERSION:"=%
if not defined APP_VERSION set "APP_VERSION=0.0.0"
echo   App version: %APP_VERSION%

REM Locate the Inno Setup compiler (ISCC.exe).
set "ISCC="
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
REM Per-user install (winget `JRSoftware.InnoSetup` lands here, no admin).
if not defined ISCC if exist "%LocalAppData%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC for %%I in (iscc.exe ISCC.exe) do if not defined ISCC if exist "%%~$PATH:I" set "ISCC=%%~$PATH:I"

if defined ISCC (
    echo   Using Inno Setup: "%ISCC%"
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
echo   torch       : CPU baseline (GPU auto-offered on first run for NVIDIA cards)
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


:try_py_path
REM %~1 = a full path to a python.exe. Like :try_py, but stores PYCMD
REM QUOTED so later "%PYCMD% -m venv ..." works even when the path has
REM spaces (e.g. C:\Program Files\...). Used to find a freshly-installed
REM Python whose new PATH entry isn't live in this cmd session yet.
if defined PYCMD goto :eof
if not exist "%~1" goto :eof
"%~1" -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
if not errorlevel 1 set PYCMD="%~1"
goto :eof


:install_python
REM No supported Python found -> offer to download + install Python %PY_VER%
REM (per-user, no admin) from python.org, then re-probe. Declining or any
REM failure just returns with PYCMD unset (caller then prints manual help).
echo.
echo ==========================================
echo   No supported Python (3.10-3.12) found
echo ==========================================
echo.
echo PyTorch has no wheels for Python 3.13/3.14, so the build cannot proceed
echo with what is installed. This script can download and install Python
echo %PY_VER% for you now (per-user, no admin required, alongside any other
echo Python you have).
echo.
choice /C YN /T 20 /D Y /M "Install Python %PY_VER% automatically now"
if errorlevel 2 (
    echo Skipping auto-install.
    goto :eof
)
echo.
echo Downloading Python %PY_VER% installer from python.org...
set "PY_SETUP=%TEMP%\python-%PY_VER%-amd64.exe"
set "PY_URL=https://www.python.org/ftp/python/%PY_VER%/python-%PY_VER%-amd64.exe"
powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -UseBasicParsing -Uri '%PY_URL%' -OutFile '%PY_SETUP%'; Unblock-File '%PY_SETUP%' } catch { exit 1 }"
if errorlevel 1 (
    echo.
    echo Download failed ^(no internet, or the URL changed^). Falling back to
    echo the manual install instructions below.
    goto :eof
)
echo Installing Python %PY_VER% silently (about a minute)...
"%PY_SETUP%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1
REM The installer just updated PATH, but that is NOT live in this session.
REM Probe the deterministic per-user install path directly (and the py
REM launcher, in case it landed somewhere already on PATH).
call :try_py_path "%LocalAppData%\Programs\Python\Python312\python.exe"
call :try_py_path "%ProgramFiles%\Python312\python.exe"
call :try_py "py -3.12"
del /Q "%PY_SETUP%" >nul 2>&1
if defined PYCMD (
    echo.
    echo Installed Python %PY_VER%: %PYCMD%
) else (
    echo.
    echo Python installer ran but a working 3.12 was not found afterward.
)
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
