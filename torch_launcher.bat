@echo off
cd /D %~dp0
setlocal EnableDelayedExpansion

set "PYTHON_EXE=.\python_embeded\python.exe"
set "SCRIPT=.\torch_launcher.py"

echo ============================================
echo Torch Model Launcher
echo ============================================
echo.

for /f %%C in ('powershell -NoProfile -Command "(Get-ChildItem 'models\\hf' -Directory -ErrorAction SilentlyContinue | Measure-Object).Count"') do set "COUNT=%%C"
if "%COUNT%"=="0" (
    echo No torch model directories found in models\hf\
    echo Use download_convert_hf_to_trt.bat first.
    pause
    exit /b 1
)

echo Available torch models:
for /f "tokens=1,2 delims=|" %%A in ('powershell -NoProfile -Command "$i=1; Get-ChildItem 'models\\hf' -Directory | Sort-Object Name | ForEach-Object { Write-Output ($i.ToString() + '|' + $_.Name); $i++ }"') do (
    echo   %%A^) %%B
)

set /p MODEL_CHOICE="Select model [1]: "
if "%MODEL_CHOICE%"=="" set "MODEL_CHOICE=1"
set "MODEL_DIR="
for /f %%N in ('powershell -NoProfile -Command "$i=%MODEL_CHOICE%; $e=Get-ChildItem 'models\\hf' -Directory | Sort-Object Name; if($i -ge 1 -and $i -le $e.Count){$e[$i-1].FullName}"') do set "MODEL_DIR=%%N"
if not defined MODEL_DIR (
    echo Invalid model selection.
    pause
    exit /b 1
)

set /p WIDTH="Inference width [1024]: "
if "%WIDTH%"=="" set "WIDTH=1024"
set /p HEIGHT="Inference height [1024]: "
if "%HEIGHT%"=="" set "HEIGHT=1024"

set "MASK_ARGS=--mask-threshold 0.65"
set /p MASK_CHOICE="Mask mode binary or soft? (B/s): "
if /I "%MASK_CHOICE%"=="s" set "MASK_ARGS=--soft-mask"
if /I not "%MASK_CHOICE%"=="s" (
    set "THRESH="
    set /p THRESH="Binary threshold [0.65]: "
    if defined THRESH set "MASK_ARGS=--mask-threshold !THRESH!"
)

set "DEVICE=auto"
set /p DEVICE_IN="Device auto/cuda/cpu [auto]: "
if defined DEVICE_IN set "DEVICE=%DEVICE_IN%"

echo.
echo Selected model dir: %MODEL_DIR%
echo.

for %%f in (workfolder\*) do (
    echo Processing: %%~nxf
    %PYTHON_EXE% %SCRIPT% --input-source "%%~nxf" --model-dir "%MODEL_DIR%" --width %WIDTH% --height %HEIGHT% --device %DEVICE% !MASK_ARGS!
)

echo ============================================
echo Torch processing complete!
echo ============================================
pause
