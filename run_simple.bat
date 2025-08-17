@echo off
REM Simple Windows runner for BitShield
REM This script runs basic commands without Docker

setlocal enabledelayedexpansion

echo ========================================
echo BitShield - Simple Windows Runner
echo ========================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo Activating virtual environment...
    call env.bat
    if errorlevel 1 (
        echo Error: Failed to activate virtual environment.
        pause
        exit /b 1
    )
)

REM Get command from arguments
set "CMD=%*"
if "%CMD%"=="" (
    echo Usage: %0 ^<command^>
    echo.
    echo Available commands:
    echo   train ^<model^> ^<dataset^>     - Train a model
    echo   test ^<model^> ^<dataset^>      - Test a model
    echo   download-datasets               - Download datasets
    echo   list-models                     - List available models
    echo   list-datasets                   - List available datasets
    echo.
    echo Examples:
    echo   %0 train resnet50 CIFAR10
    echo   %0 test resnet50 CIFAR10
    echo   %0 download-datasets
    echo.
    pause
    exit /b 1
)

echo Running command: %CMD%

REM Parse command
for /f "tokens=1,2,3" %%a in ("%CMD%") do (
    set "action=%%a"
    set "model=%%b"
    set "dataset=%%c"
)

if "%action%"=="train" (
    if "%model%"=="" (
        echo Error: Model name required for training.
        echo Example: %0 train resnet50 CIFAR10
        pause
        exit /b 1
    )
    if "%dataset%"=="" (
        echo Error: Dataset name required for training.
        echo Example: %0 train resnet50 CIFAR10
        pause
        exit /b 1
    )
    echo Training %model% on %dataset%...
    python support\models\train.py %model% %dataset%
    goto :end
)

if "%action%"=="test" (
    if "%model%"=="" (
        echo Error: Model name required for testing.
        echo Example: %0 test resnet50 CIFAR10
        pause
        exit /b 1
    )
    if "%dataset%"=="" (
        echo Error: Dataset name required for testing.
        echo Example: %0 test resnet50 CIFAR10
        pause
        exit /b 1
    )
    echo Testing %model% on %dataset%...
    python support\models\train.py %model% %dataset% --test-only
    goto :end
)

if "%action%"=="download-datasets" (
    echo Downloading datasets...
    python tools\ensure_datasets.py
    goto :end
)

if "%action%"=="list-models" (
    echo Available models:
    echo   resnet50
    echo   densenet121
    echo   googlenet
    echo   lenet1
    echo   dcgan_g
    goto :end
)

if "%action%"=="list-datasets" (
    echo Available datasets:
    echo   CIFAR10
    echo   CIFAR100
    echo   MNIST
    echo   MNISTC
    echo   FashionC
    echo   ImageNet (if available)
    goto :end
)

echo Unknown command: %action%
echo Use %0 without arguments to see available commands.
pause
exit /b 1

:end
echo.
echo Command completed successfully!
pause
