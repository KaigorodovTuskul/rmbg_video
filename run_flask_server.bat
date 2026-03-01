@echo off
cd /D %~dp0
setlocal

set "PYTHON_EXE=.\python_embeded\python.exe"
set "APP=.\flask_batch_ui.py"

if not exist "%PYTHON_EXE%" (
    echo ERROR: python_embeded not found. Run setup.bat first.
    pause
    exit /b 1
)

echo ============================================
echo RMBG Flask Launcher UI
echo ============================================
echo Open in browser: http://127.0.0.1:7860
echo.

%PYTHON_EXE% %APP%
