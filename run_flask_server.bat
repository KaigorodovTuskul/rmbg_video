@echo off
cd /D %~dp0
setlocal

set "PYTHON_EXE=.\python_embeded\python.exe"
set "APP=.\flask_batch_ui.py"
set "PIP_ARGS=--no-cache-dir --no-warn-script-location --timeout=1000 --retries 10"

if not exist "%PYTHON_EXE%" (
    echo ERROR: python_embeded not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Checking Flask...
%PYTHON_EXE% -c "import flask" >nul 2>nul
if errorlevel 1 (
    echo Flask is missing. Installing...
    %PYTHON_EXE% -I -m pip install Flask==3.1.2 %PIP_ARGS%
    if errorlevel 1 (
        echo ERROR: Failed to install Flask.
        pause
        exit /b 1
    )
)

echo ============================================
echo RMBG Flask Launcher UI
echo ============================================
echo Open in browser: http://127.0.0.1:7860
echo.

%PYTHON_EXE% %APP%
