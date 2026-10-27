@echo off
setlocal
title Phone as GPU - PC Bridge & Web UI

echo ====================================================
echo       PHONE AS GPU - PC USB BRIDGE & LAUNCHER
echo ====================================================
echo.

where adb >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    if exist "C:\platform-tools\adb.exe" (
        set "ADB_PATH=C:\platform-tools\adb.exe"
    ) else if exist "C:\scrcpy-win64-v4.0\adb.exe" (
        set "ADB_PATH=C:\scrcpy-win64-v4.0\adb.exe"
    ) else (
        echo [ERROR] ADB not found.
        pause
        exit /b 1
    )
) else (
    set "ADB_PATH=adb"
)

echo [*] Using ADB from: %ADB_PATH%

echo [*] Checking connected devices...
"%ADB_PATH%" devices
echo.

echo [*] Forwarding Web UI & OpenAI API port (8080) over USB...
"%ADB_PATH%" forward tcp:8080 tcp:8080

echo [*] Forwarding llama.cpp engine port (50053) over USB...
"%ADB_PATH%" forward tcp:50053 tcp:50053

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ====================================================
    echo   [SUCCESS] USB BRIDGE IS LIVE!
    echo ====================================================
    echo.
    echo  1. Phone Web UI on PC : http://localhost:8080
    echo  2. OpenAI API endpoint: http://localhost:8080/v1
    echo.
    echo Opening Web UI in your default browser...
    start http://localhost:8080
) else (
    echo [ERROR] Failed to forward ports. Check USB cable and USB Debugging.
)

echo.
echo Press any key to view active ADB forwarded ports or Ctrl+C to close.
pause
"%ADB_PATH%" forward --list
pause
