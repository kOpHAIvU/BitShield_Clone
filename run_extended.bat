@echo off
echo BitShield Extended Pipeline Runner
echo ==================================

REM Set Python path
set PYTHON_PATH=C:\Users\ADMIN\AppData\Local\Programs\Python\Python312\python.exe

REM Check if arguments provided
if "%1"=="" (
    echo Usage: run_extended.bat [command] [args...]
    echo.
    echo Examples:
    echo   run_extended.bat train ResNetSEBlockIoT WUSTL --epochs 5
    echo   run_extended.bat attack dig ResNetSEBlockIoT WUSTL
    echo   run_extended.bat demo --mode specific --dataset WUSTL --model ResNetSEBlockIoT
    goto :end
)

REM Parse command
if "%1"=="train" (
    shift
    echo Running training...
    %PYTHON_PATH% support/models/train_extended.py %*
) else if "%1"=="attack" (
    shift
    echo Running attack simulation...
    %PYTHON_PATH% attack_with_defense_extended.py %*
) else if "%1"=="demo" (
    shift
    echo Running demo...
    %PYTHON_PATH% demo_extended_pipeline.py %*
) else (
    echo Unknown command: %1
    echo Available commands: train, attack, demo
)

:end
pause
