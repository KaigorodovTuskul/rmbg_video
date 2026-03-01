@echo off
cd /D %~dp0
setlocal

set "SCRIPT_TITLE=RMBG TRT Setup"
title %SCRIPT_TITLE%

set "PIP_ARGS=--no-cache-dir --no-warn-script-location --timeout=1000 --retries 10"
set "PYTHON_EXE=python_embeded\python.exe"
set "PY_EMBED_URL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130"
set "NVIDIA_INDEX_URL=https://pypi.nvidia.com"

echo.
echo ================================
echo   RMBG TRT Setup
echo ================================
echo.

if exist "%PYTHON_EXE%" (
    echo python_embeded already exists.
    set /p REINSTALL="Reinstall embedded Python? (y/N): "
    if /I "%REINSTALL%"=="y" (
        echo Removing existing python_embeded...
        rd /s /q "python_embeded"
    ) else (
        goto :skip_python_install
    )
)

echo Creating python_embeded...
mkdir python_embeded
cd python_embeded

echo Downloading Python 3.12.10 Embedded...
powershell -NoProfile -Command "Invoke-WebRequest '%PY_EMBED_URL%' -OutFile 'python.zip' -UseBasicParsing"
if errorlevel 1 goto :fail

echo Extracting Python...
tar.exe -xf python.zip
if errorlevel 1 goto :fail
del /f /q python.zip

echo Downloading get-pip.py...
powershell -NoProfile -Command "Invoke-WebRequest 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py' -UseBasicParsing"
if errorlevel 1 goto :fail

echo Configuring python312._pth...
(
  echo .
  echo python312.zip
  echo .
  echo Lib/site-packages
  echo Lib
  echo Scripts
) > python312._pth

echo Installing pip...
.\python.exe -I get-pip.py %PIP_ARGS%
if errorlevel 1 goto :fail

echo Upgrading base packaging tools...
.\python.exe -I -m pip install --upgrade pip setuptools wheel %PIP_ARGS%
if errorlevel 1 goto :fail

cd ..

:skip_python_install

echo.
echo ================================
echo Installing PyTorch and TensorRT
echo ================================
echo.

%PYTHON_EXE% -I -m pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130 --index-url %TORCH_INDEX_URL% %PIP_ARGS%
if errorlevel 1 goto :fail

%PYTHON_EXE% -I -m pip install wheel-stub==0.4.2 --extra-index-url %NVIDIA_INDEX_URL% %PIP_ARGS%
if errorlevel 1 goto :fail

%PYTHON_EXE% -I -m pip install tensorrt-cu13-bindings==10.13.3.9.post1 tensorrt-cu13-libs==10.13.3.9.post1 tensorrt-cu13==10.13.3.9.post1 --extra-index-url %NVIDIA_INDEX_URL% %PIP_ARGS%
if errorlevel 1 goto :fail

echo.
echo ================================
echo Installing requirements.txt
echo ================================
echo.

if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found.
    goto :fail
)

%PYTHON_EXE% -I -m pip install -r requirements.txt --extra-index-url %TORCH_INDEX_URL% %PIP_ARGS%
if errorlevel 1 goto :fail

echo.
echo ================================
echo Installation Complete
echo ================================
echo.
echo Run:
echo   birefnet_trt_launcher.bat
echo.
pause
exit /b 0

:fail
echo.
echo Setup failed with error code %errorlevel%.
echo.
pause
exit /b 1
