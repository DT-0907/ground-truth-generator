@echo off
REM ============================================================
REM  CCTV-YOLO diagnostic launcher
REM
REM  Use this when CCTV-YOLO.exe will not open.
REM  It runs the exe and captures ALL output -- including native
REM  crash messages (DLL load failed, OMP errors, Qt platform
REM  plugin failures) that never reach Python's crash handler.
REM
REM  The window stays open so you can read the error, and the
REM  full output is saved to startup-output.log next to the exe.
REM ============================================================
cd /d "%~dp0"

echo Launching CCTV-YOLO with full output capture...
echo This window will stay open so you can read any errors.
echo.

CCTV-YOLO.exe > startup-output.log 2>&1
set EXITCODE=%errorlevel%

echo.
echo ============================================================
echo  CCTV-YOLO exited with code %EXITCODE%
echo  Full output saved to:
echo    %~dp0startup-output.log
echo ============================================================
echo.
echo ----- startup-output.log -----------------------------------
type startup-output.log
echo ------------------------------------------------------------
echo.
pause
