@echo off
cd /D %~dp0
setlocal

echo ============================================
echo Optional FFmpeg Installer
echo ============================================
echo This step is optional and needed for video pipeline scripts.
echo.

where winget >nul 2>nul
if errorlevel 1 (
    echo ERROR: winget not found. Install FFmpeg manually and update .env
    pause
    exit /b 1
)

winget install -e --id Gyan.FFmpeg
if errorlevel 1 (
    echo ERROR: winget failed to install FFmpeg.
    pause
    exit /b 1
)

for /f "delims=" %%F in ('where ffmpeg 2^>nul') do set "FFMPEG_EXE=%%F"
for /f "delims=" %%P in ('where ffprobe 2^>nul') do set "FFPROBE_EXE=%%P"

if not defined FFMPEG_EXE (
    echo WARNING: ffmpeg.exe not found in PATH. Update .env manually.
    pause
    exit /b 0
)
if not defined FFPROBE_EXE (
    echo WARNING: ffprobe.exe not found in PATH. Update .env manually.
    pause
    exit /b 0
)

powershell -NoProfile -Command ^
  "$envFile='.\.env'; if(!(Test-Path $envFile)){ New-Item -Path $envFile -ItemType File | Out-Null }; " ^
  "$lines=Get-Content $envFile; " ^
  "$lines=$lines | Where-Object { $_ -notmatch '^FFMPEG_PATH=' -and $_ -notmatch '^FFPROBE_PATH=' }; " ^
  "$lines += 'FFMPEG_PATH=%FFMPEG_EXE%'; " ^
  "$lines += 'FFPROBE_PATH=%FFPROBE_EXE%'; " ^
  "Set-Content -Path $envFile -Value $lines"

echo.
echo FFmpeg installed.
echo .env updated:
echo   FFMPEG_PATH=%FFMPEG_EXE%
echo   FFPROBE_PATH=%FFPROBE_EXE%
pause
exit /b 0
