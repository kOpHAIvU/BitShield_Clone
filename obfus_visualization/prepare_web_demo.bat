@echo off
REM Prepare web demo models for Windows

echo ========================================
echo Preparing Web Demo Models
echo ========================================

REM Default parameters
set MODEL_NAME=ResNetSEBlockIoT
set DATASET_NAME=IoTID20
set ATTACK_MODE=pbs
set ATTACK_ITERS=25
set DEVICE=cuda

REM Parse arguments
if "%1" neq "" set MODEL_NAME=%1
if "%2" neq "" set DATASET_NAME=%2
if "%3" neq "" set ATTACK_MODE=%3
if "%4" neq "" set ATTACK_ITERS=%4
if "%5" neq "" set DEVICE=%5

echo Model: %MODEL_NAME%
echo Dataset: %DATASET_NAME%
echo Attack Mode: %ATTACK_MODE%
echo Attack Iterations: %ATTACK_ITERS%
echo Device: %DEVICE%
echo.

python prepare_web_demo_models.py %MODEL_NAME% %DATASET_NAME% --attack-mode %ATTACK_MODE% --attack-iters %ATTACK_ITERS% --device %DEVICE%

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================
    echo SUCCESS! Models created successfully
    echo ========================================
    echo.
    echo Models saved to: models/web_demo/%DATASET_NAME%_%MODEL_NAME%/
    echo   - original.pt
    echo   - attacked.pt
    echo   - protected.pt
    echo   - obfus_config.json
) else (
    echo.
    echo ========================================
    echo ERROR! Failed to create models
    echo ========================================
    exit /b 1
)

pause

