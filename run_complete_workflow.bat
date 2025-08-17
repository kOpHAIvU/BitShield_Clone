@echo off
REM Complete workflow runner for BitShield
REM This script runs the complete workflow from training to analysis

setlocal enabledelayedexpansion

echo ========================================
echo BitShield - Complete Workflow Runner
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

REM Get parameters
set "model=%1"
set "dataset=%2"

if "%model%"=="" (
    echo Usage: %0 ^<model^> ^<dataset^>
    echo.
    echo Available models: resnet50, densenet121, googlenet, lenet1
    echo Available datasets: CIFAR10, MNISTC, FashionC, MNIST
    echo.
    echo Example: %0 resnet50 CIFAR10
    pause
    exit /b 1
)

if "%dataset%"=="" (
    echo Error: Dataset is required.
    echo Example: %0 resnet50 CIFAR10
    pause
    exit /b 1
)

echo Starting complete workflow for %model% on %dataset%
echo.

REM Step 1: Setup Docker environment
echo [1/6] Setting up Docker environment...
call run_docker.bat setup
if errorlevel 1 (
    echo Error: Failed to setup Docker environment.
    pause
    exit /b 1
)

REM Step 2: Download datasets
echo.
echo [2/6] Downloading datasets...
call run_docker.bat download-datasets
if errorlevel 1 (
    echo Error: Failed to download datasets.
    pause
    exit /b 1
)

REM Step 3: Train model
echo.
echo [3/6] Training %model% on %dataset%...
call run_docker.bat train %model% %dataset%
if errorlevel 1 (
    echo Error: Failed to train model.
    pause
    exit /b 1
)

REM Step 4: Test model
echo.
echo [4/6] Testing %model% on %dataset%...
call run_docker.bat test %model% %dataset%
if errorlevel 1 (
    echo Warning: Model testing failed, but continuing...
)

REM Step 5: Build binary
echo.
echo [5/6] Building binary for %model% on %dataset%...
call run_docker.bat build
if errorlevel 1 (
    echo Warning: Binary build failed, but continuing...
)

REM Step 6: Run bit-flip sweep
echo.
echo [6/6] Running bit-flip sweep for %model% on %dataset%...
call run_docker.bat sweep %model% %dataset%
if errorlevel 1 (
    echo Warning: Bit-flip sweep failed.
)

echo.
echo ========================================
echo Workflow completed!
echo ========================================
echo.
echo Results are available in:
echo   - models/     (trained models)
echo   - built/      (binary files)
echo   - results/    (analysis results)
echo.
echo To run attack simulation:
echo   run_docker.bat attack %model% %dataset%
echo.
pause
