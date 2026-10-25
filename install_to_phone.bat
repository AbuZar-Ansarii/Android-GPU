@echo off
setlocal
title Push Local AI Hub to Phone

echo ====================================================
echo    PUSH PHONE AS GPU (HUB + WEB UI) TO PHONE
echo ====================================================
echo.

where adb >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    if exist "C:\platform-tools\adb.exe" (
        set "ADB_PATH=C:\platform-tools\adb.exe"
    ) else if exist "C:\scrcpy-win64-v4.0\adb.exe" (
        set "ADB_PATH=C:\scrcpy-win64-v4.0\adb.exe"
    ) else (
        echo [ERROR] ADB not found. Please install Android Platform Tools.
        pause
        exit /b 1
    )
) else (
    set "ADB_PATH=adb"
)

echo [*] Checking connected phone...
"%ADB_PATH%" devices
echo.

echo [*] Creating target folder on phone: /sdcard/Download/PhoneAsGPU ...
"%ADB_PATH%" shell mkdir -p /sdcard/Download/PhoneAsGPU

echo [*] Transferring setup_termux.sh, app.py, and web interface...
"%ADB_PATH%" push setup_termux.sh /sdcard/Download/PhoneAsGPU/setup_termux.sh
"%ADB_PATH%" push app.py /sdcard/Download/PhoneAsGPU/app.py
"%ADB_PATH%" push web /sdcard/Download/PhoneAsGPU/

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ====================================================
    echo [SUCCESS] All files transferred to your phone!
    echo ====================================================
    echo.
    echo Next Steps on Your Phone:
    echo  1. Open the Termux app.
    echo  2. Copy and paste these 3 commands:
    echo.
    echo     termux-setup-storage
    echo     cp -r ~/storage/downloads/PhoneAsGPU ~/
    echo     cd ~/PhoneAsGPU && chmod +x setup_termux.sh && ./setup_termux.sh
    echo.
) else (
    echo [ERROR] Could not transfer files. Make sure USB Debugging is ON and phone is unlocked.
)

pause
