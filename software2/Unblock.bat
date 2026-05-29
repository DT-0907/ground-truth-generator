@echo off
REM ============================================================
REM  CCTV-YOLO - Unblock downloaded files (one-time speedup)
REM
REM  When you extract a ZIP downloaded from a browser, Windows
REM  attaches a "Mark-of-the-Web" (MOTW) flag to every file. The
REM  first launch of CCTV-YOLO.exe then triggers SmartScreen +
REM  Defender full-scan, which can take 30-60 seconds for the
REM  2.5 GB PyTorch/CUDA bundle.
REM
REM  Stripping MOTW is safe: it does not change file content,
REM  permissions, or executability. It just tells Windows "this
REM  file is local, no need to re-scan it from scratch every
REM  time".
REM
REM  Run this batch file ONCE after extracting the ZIP and the
REM  app will launch as fast as a normal local program.
REM ============================================================
cd /d "%~dp0"

echo Unblocking all files in:
echo   %CD%
echo.
echo (Strips the Mark-of-the-Web flag so Windows doesn't re-scan
echo  the entire bundle on every launch. Safe and reversible.)
echo.

powershell -NoProfile -Command "Get-ChildItem -LiteralPath '%CD%' -Recurse -File | Unblock-File"
if errorlevel 1 (
    echo.
    echo ERROR: PowerShell could not run Unblock-File.
    echo You can run this manually:
    echo   1. Open PowerShell in this folder
    echo   2. Run: Get-ChildItem -Recurse ^| Unblock-File
    echo.
    pause
    exit /b 1
)

echo.
echo Done. Double-click CCTV-YOLO.exe to launch.
echo.
pause
