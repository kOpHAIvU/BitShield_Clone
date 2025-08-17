@echo off
REM Docker runner for BitShield
REM This script runs all commands inside Docker container

setlocal enabledelayedexpansion

echo ========================================
echo BitShield - Docker Runner
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running or not installed.
    echo Please install Docker Desktop and start it.
    pause
    exit /b 1
)

REM Check if Docker image exists
docker image inspect debfd-runner:latest >nul 2>&1
if errorlevel 1 (
    echo Docker image not found. Building...
    docker\setup.bat
    if errorlevel 1 (
        echo Error: Failed to build Docker image.
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
    echo   setup                    - Setup Docker environment
    echo   download-datasets        - Download datasets
    echo   train ^<model^> ^<dataset^>     - Train a model
    echo   test ^<model^> ^<dataset^>      - Test a model
    echo   build                    - Build all models
    echo   sweep ^<model^> ^<dataset^>     - Run bit-flip sweep
    echo   attack ^<model^> ^<dataset^>    - Run attack simulation
    echo   shell                    - Open shell in container
    echo   list-models              - List available models
    echo   list-datasets            - List available datasets
    echo   test-tvm                 - Test TVM installation
    echo.
    echo Examples:
    echo   %0 train resnet50 CIFAR10
    echo   %0 sweep resnet50 CIFAR10
    echo   %0 attack resnet50 CIFAR10
    echo   %0 shell
    echo.
    pause
    exit /b 1
)

echo Running in Docker: %CMD%

REM Parse command
for /f "tokens=1,2,3" %%a in ("%CMD%") do (
    set "action=%%a"
    set "model=%%b"
    set "dataset=%%c"
)

if "%action%"=="setup" (
    echo Setting up Docker environment...
    docker\setup.bat
    goto :end
)

if "%action%"=="download-datasets" (
    echo Downloading datasets...
    docker\run-in-docker.bat python tools\ensure_datasets.py
    goto :end
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
    docker\run-in-docker.bat python simple_train_test.py train %model% %dataset%
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
    docker\run-in-docker.bat python simple_train_test.py test %model% %dataset%
    goto :end
)

if "%action%"=="build" (
    echo Building all models...
    docker\run-in-docker.bat python buildmodels.py
    goto :end
)

if "%action%"=="sweep" (
    if "%model%"=="" (
        echo Error: Model name required for sweep.
        echo Example: %0 sweep resnet50 CIFAR10
        pause
        exit /b 1
    )
    if "%dataset%"=="" (
        echo Error: Dataset name required for sweep.
        echo Example: %0 sweep resnet50 CIFAR10
        pause
        exit /b 1
    )
    echo Running simple bit-flip sweep for %model% on %dataset%...
    docker\run-in-docker.bat python simple_sweep_attack.py sweep %model% %dataset%
    goto :end
)

if "%action%"=="attack" (
    if "%model%"=="" (
        echo Error: Model name required for attack simulation.
        echo Example: %0 attack resnet50 CIFAR10
        pause
        exit /b 1
    )
    if "%dataset%"=="" (
        echo Error: Dataset name required for attack simulation.
        echo Example: %0 attack resnet50 CIFAR10
        pause
        exit /b 1
    )
    echo Running simple attack simulation for %model% on %dataset%...
    docker\run-in-docker.bat python simple_sweep_attack.py attack %model% %dataset%
    goto :end
)

if "%action%"=="shell" (
    echo Opening shell in Docker container...
    docker\run-in-docker.bat /bin/bash
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

if "%action%"=="test-tvm" (
    echo Testing TVM installation...
    docker\run-in-docker.bat python test_tvm.py
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
