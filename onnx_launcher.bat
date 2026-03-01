@echo off
cd /D %~dp0
setlocal EnableDelayedExpansion

set "PYTHON_EXE=.\python_embeded\python.exe"
set "SCRIPT=.\onnx_launcher.py"

echo ============================================
echo ONNX Model Launcher
echo ============================================
echo.

for /f %%C in ('powershell -NoProfile -Command "(Get-ChildItem 'models\\*.onnx' -File -ErrorAction SilentlyContinue | Measure-Object).Count"') do set "COUNT=%%C"
if "%COUNT%"=="0" (
    echo No ONNX files found in models\
    pause
    exit /b 1
)

echo Available ONNX models:
for /f "tokens=1,2 delims=|" %%A in ('powershell -NoProfile -Command "$i=1; Get-ChildItem 'models\\*.onnx' -File | Sort-Object Name | ForEach-Object { Write-Output ($i.ToString() + '|' + $_.Name); $i++ }"') do (
    echo   %%A^) %%B
)

set /p MODEL_CHOICE="Select ONNX [1]: "
if "%MODEL_CHOICE%"=="" set "MODEL_CHOICE=1"
set "ONNX_NAME="
for /f %%N in ('powershell -NoProfile -Command "$i=%MODEL_CHOICE%; $e=Get-ChildItem 'models\\*.onnx' -File | Sort-Object Name; if($i -ge 1 -and $i -le $e.Count){$e[$i-1].Name}"') do set "ONNX_NAME=%%N"
if not defined ONNX_NAME (
    echo Invalid ONNX selection.
    pause
    exit /b 1
)
set "ONNX_PATH=models\%ONNX_NAME%"

set "MASK_ARGS=--mask-threshold 0.65"
set /p MASK_CHOICE="Mask mode binary or soft? (B/s): "
if /I "%MASK_CHOICE%"=="s" set "MASK_ARGS=--soft-mask"
if /I not "%MASK_CHOICE%"=="s" (
    set "THRESH="
    set /p THRESH="Binary threshold [0.65]: "
    if defined THRESH set "MASK_ARGS=--mask-threshold !THRESH!"
)

set "PROVIDERS=auto"
set /p PROVIDERS_IN="Providers auto/cuda/cpu [auto]: "
if defined PROVIDERS_IN set "PROVIDERS=%PROVIDERS_IN%"

echo.
echo Selected ONNX: %ONNX_NAME%
echo.

for %%f in (workfolder\*) do (
    echo Processing: %%~nxf
    %PYTHON_EXE% %SCRIPT% --input-source "%%~nxf" --onnx-path "%ONNX_PATH%" --providers %PROVIDERS% !MASK_ARGS!
)

echo ============================================
echo ONNX processing complete!
echo ============================================
pause
