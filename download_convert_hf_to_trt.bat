@echo off
cd /D %~dp0
setlocal EnableDelayedExpansion

set "PYTHON_EXE=.\python_embeded\python.exe"
set "SCRIPT=.\download_convert_hf_to_trt.py"

echo ============================================
echo HF Model Download + ONNX + TRT Mixed Builder
echo ============================================
echo.

echo Select model:
echo   1^) ZhengPeng7/BiRefNet
echo   2^) ZhengPeng7/BiRefNet_lite
echo   3^) briaai/RMBG-2.0
echo   4^) Custom repo id
set /p MODEL_CHOICE="Choice [1]: "
if "%MODEL_CHOICE%"=="" set "MODEL_CHOICE=1"

set "REPO_ID=ZhengPeng7/BiRefNet"
if "%MODEL_CHOICE%"=="2" set "REPO_ID=ZhengPeng7/BiRefNet_lite"
if "%MODEL_CHOICE%"=="3" set "REPO_ID=briaai/RMBG-2.0"
if "%MODEL_CHOICE%"=="4" (
    set /p REPO_ID="Enter repo id (owner/name): "
)

if "%REPO_ID%"=="" (
    echo ERROR: repo id is empty.
    pause
    exit /b 1
)

set /p WIDTH="Width [1024]: "
if "%WIDTH%"=="" set "WIDTH=1024"
set /p HEIGHT="Height [1024]: "
if "%HEIGHT%"=="" set "HEIGHT=1024"
set /p BATCH="Batch [1]: "
if "%BATCH%"=="" set "BATCH=1"
set /p WORKSPACE="TRT workspace GB [6]: "
if "%WORKSPACE%"=="" set "WORKSPACE=6"
set /p DL_WORKERS="Download workers [16]: "
if "%DL_WORKERS%"=="" set "DL_WORKERS=16"

echo.
echo Repo: %REPO_ID%
echo Size: %WIDTH%x%HEIGHT%
echo Batch: %BATCH%
echo Workspace: %WORKSPACE% GB
echo Download workers: %DL_WORKERS%
echo.

%PYTHON_EXE% %SCRIPT% --repo-id "%REPO_ID%" --width %WIDTH% --height %HEIGHT% --batch %BATCH% --workspace-gb %WORKSPACE% --download-workers %DL_WORKERS%
if errorlevel 1 (
    echo.
    echo FAILED.
    pause
    exit /b 1
)

echo.
echo Complete.
pause
