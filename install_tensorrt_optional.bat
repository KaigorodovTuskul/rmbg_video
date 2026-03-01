@echo off
cd /D %~dp0
setlocal

set "PYTHON_EXE=.\python_embeded\python.exe"
set "PIP_ARGS=--no-cache-dir --no-warn-script-location --timeout=1000 --retries 10"
set "NVIDIA_INDEX_URL=https://pypi.nvidia.com"

if not exist "%PYTHON_EXE%" (
    echo ERROR: python_embeded not found. Run setup.bat first.
    pause
    exit /b 1
)

echo ============================================
echo Optional TensorRT Installer
echo ============================================
echo This step is optional and needed only for TRT launcher.
echo.

%PYTHON_EXE% -I -m pip install wheel-stub==0.4.2 --extra-index-url %NVIDIA_INDEX_URL% %PIP_ARGS%
if errorlevel 1 goto :fail

%PYTHON_EXE% -I -m pip install tensorrt-cu13-bindings==10.13.3.9.post1 tensorrt-cu13-libs==10.13.3.9.post1 tensorrt-cu13==10.13.3.9.post1 --extra-index-url %NVIDIA_INDEX_URL% %PIP_ARGS%
if errorlevel 1 goto :fail

echo.
echo TensorRT optional install complete.
pause
exit /b 0

:fail
echo.
echo TensorRT optional install failed with error code %errorlevel%.
pause
exit /b 1
