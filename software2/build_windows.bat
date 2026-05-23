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

REM ---------- 0. Long-path-support check + workarounds ----------
REM
REM Modern PyTorch + PyInstaller produce paths like
REM   build_venv\Lib\site-packages\torch\lib\cudnn_cnn_train64_8.dll
REM which combined with a deep project path (especially under OneDrive)
REM regularly exceeds Windows' MAX_PATH (260 chars) limit and crashes pip
REM with: "does not have Windows long path support enabled".
REM
REM We work around this two ways:
REM   1. Check whether LongPathsEnabled is set in the registry. If not,
REM      print one-line, copy-paste-able instructions to enable it.
REM   2. Force pip + PyInstaller caches into SHORT %TEMP% subfolders so
REM      relative paths under them stay well under 260 chars.

reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled 2>nul | findstr /R /C:"0x1" >nul
if errorlevel 1 (
    echo.
    echo WARNING: Windows Long Path support is NOT enabled on this system.
    echo          pip and PyInstaller can fail when extracting deep paths
    echo          like torch/cuda DLLs.
    echo.
    echo Recommended fix ^(one-time, requires Administrator + reboot^):
    echo.
    echo   Open PowerShell ^(Admin^) and run:
    echo     New-ItemProperty -Path ^"HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem^" `
    echo         -Name LongPathsEnabled -PropertyType DWord -Value 1 -Force
    echo   then reboot.
    echo.
    echo Alternatively in an Admin Command Prompt:
    echo     reg add ^"HKLM\SYSTEM\CurrentControlSet\Control\FileSystem^" /v LongPathsEnabled /t REG_DWORD /d 1 /f
    echo     then reboot.
    echo.
    echo Continuing with short-path workarounds. If pip still fails with a
    echo path-length error, enable long paths first and try again.
    echo.
)

REM Sanity check: warn if the project path itself is long. Anything past
REM ~120 chars leaves little headroom for nested torch/cuda paths.
call :strlen PROJECT_PATH_LEN "%CD%"
if !PROJECT_PATH_LEN! GTR 120 (
    echo.
    echo WARNING: Project path is !PROJECT_PATH_LEN! characters long:
    echo            %CD%
    echo          PyInstaller can hit MAX_PATH limits with deep wheels.
    echo          If the build fails, move this folder to a shorter path
    echo          ^(e.g. C:\cctv-yolo\^) and re-run.
    echo.
)

REM Short pip cache dir keeps wheel extract paths short.
set "PIP_CACHE_DIR=%TEMP%\pc"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%" 2>nul

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
    echo To share: copy the entire dist\CCTV-YOLO\ folder to another machine
    echo   ^(or use the Inno Setup installer: iscc installer_windows.iss^).
    echo.
    echo Data location:
    echo   All videos / tracks / corrections / models / logs live in a
    echo   separate, portable folder named cctv-yolo\. On first launch the
    echo   app searches %%USERPROFILE%%\Documents\cctv-yolo,
    echo   %%USERPROFILE%%\Desktop\cctv-yolo, and %%USERPROFILE%%\cctv-yolo.
    echo   If none exists, it creates %%USERPROFILE%%\Documents\cctv-yolo\.
    echo   You can move that folder anywhere later -- the app will find it
    echo   again automatically ^(remembered in %%APPDATA%%\CCTV-YOLO\data_root.txt^).
    echo   Override with the CCTV_YOLO_DATA_DIR env var anytime.
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


REM ---------- helper subroutine: string length ----------
REM Usage: call :strlen <result_var_name> <string>
:strlen
setlocal enabledelayedexpansion
set "_s=%~2"
set "_len=0"
:_strlen_loop
if defined _s (
    set "_s=!_s:~1!"
    set /a "_len+=1"
    goto :_strlen_loop
)
endlocal & set "%~1=%_len%"
goto :eof
