@echo off
cd /D %~dp0
setlocal EnableDelayedExpansion

echo ============================================
echo BiRefNet TRT Launcher
echo ============================================

set "PYTHON_EXE=.\python_embeded\python.exe"
set "SCRIPT=.\birefnet_trt\birefnet_trt.py"
set "TRT_LIBS=.\python_embeded\Lib\site-packages\tensorrt_libs"
if exist "%TRT_LIBS%\nvinfer_10.dll" set "PATH=%CD%\python_embeded\Lib\site-packages\tensorrt_libs;%PATH%"

for /f %%C in ('powershell -NoProfile -Command "(Get-ChildItem 'models\\birefnet*.engine' | Measure-Object).Count"') do set "COUNT=%%C"

if "%COUNT%"=="0" (
    echo No BiRefNet TRT engines found in models\
    pause
    exit /b 1
)

echo Available models:
for /f "tokens=1,2 delims=|" %%A in ('powershell -NoProfile -Command "$i=1; Get-ChildItem 'models\\birefnet*.engine' | Sort-Object Name | ForEach-Object { Write-Output ($i.ToString() + '|' + $_.Name); $i++ }"') do (
    echo   %%A^) %%B
)

set /p MODEL_CHOICE="Select model [1]: "
if "%MODEL_CHOICE%"=="" set "MODEL_CHOICE=1"
set "ENGINE_NAME="
for /f %%N in ('powershell -NoProfile -Command "$i=%MODEL_CHOICE%; $e=Get-ChildItem 'models\\birefnet*.engine' | Sort-Object Name; if($i -ge 1 -and $i -le $e.Count){$e[$i-1].Name}"') do set "ENGINE_NAME=%%N"
if not defined ENGINE_NAME (
    echo Invalid model selection.
    pause
    exit /b 1
)
set "ENGINE_PATH=models\%ENGINE_NAME%"

for /f "tokens=1,2" %%A in ('
  powershell -NoProfile -Command "$n='%ENGINE_NAME%'; if($n -match '_(\d+)x(\d+)_b\d+\.engine$'){ Write-Output ($matches[1]+' '+$matches[2]); } elseif($n -match '_(\d+)_b\d+\.engine$'){ Write-Output ($matches[1]+' '+$matches[1]); } else { Write-Output '1024 1024'; }"
') do (
  set "ENGINE_W=%%A"
  set "ENGINE_H=%%B"
)

set "MASK_ARGS=--mask-threshold 0.65"
set /p MASK_CHOICE="Mask mode binary or soft? (B/s): "
if /I "%MASK_CHOICE%"=="s" set "MASK_ARGS=--soft-mask"
if /I not "%MASK_CHOICE%"=="s" (
    set "THRESH="
    set /p THRESH="Binary threshold [0.65]: "
    if defined THRESH set "MASK_ARGS=--mask-threshold !THRESH!"
)

echo.
echo Selected:
echo   Engine: %ENGINE_NAME%
echo   Engine size: %ENGINE_W%x%ENGINE_H%
echo   Mask args: %MASK_ARGS%
echo.

for %%f in (workfolder\*) do (
    echo Processing: %%~nxf
    %PYTHON_EXE% %SCRIPT% --input-source "%%~nxf" --engine-path "%ENGINE_PATH%" --num_max_workers 32 --base-edge %ENGINE_H% --precision fp16 --gpu-limit 1 --auto-workers --batch-size 1 !MASK_ARGS!
)

echo ============================================
echo Processing complete!
echo ============================================
pause
