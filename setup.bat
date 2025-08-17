@echo off
setlocal enabledelayedexpansion

REM Setup script for BitShield project on Windows
REM This script sets up the BitShield project on Windows

echo ========================================
echo BitShield - Windows Setup
echo ========================================
echo.

REM Check if Python 3.8+ is available
python --version 2>nul | findstr "3\.[8-9]\|3\.1[0-2]" >nul
if errorlevel 1 (
    echo ERROR: Python 3.8+ is required. Please install Python 3.8+ first.
    echo You can download it from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Python version check passed.

REM Initialize git submodules
echo.
echo Initializing git submodules...
git submodule update --init --recursive
if errorlevel 1 (
    echo WARNING: Git submodule initialization failed. Continuing anyway...
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment and install dependencies
echo.
echo Installing dependencies...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
echo Installing PyTorch...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo Installing other dependencies...
python -m pip install -r requirements_simple.txt

REM Create necessary directories
echo.
echo Creating necessary directories...
if not exist "datasets" mkdir datasets
if not exist "models" mkdir models
if not exist "built" mkdir built
if not exist "results" mkdir results
if not exist ".cache" mkdir .cache
if not exist "built-aux" mkdir built-aux
if not exist "ghidra\db" mkdir ghidra\db
if not exist "ghidra\analysis" mkdir ghidra\analysis

REM Set environment variable for ImageNet (user needs to set this)
if not defined IMAGENET_ROOT (
    echo.
    echo NOTE: IMAGENET_ROOT environment variable is not set.
    echo If you have ImageNet dataset, please set it:
    echo   set IMAGENET_ROOT=C:\path\to\imagenet
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Activate environment: venv\Scripts\activate.bat
echo 2. Download datasets: python tools\ensure_datasets.py
echo 3. Install Docker Desktop if you want to build models
echo 4. Run: docker\setup.bat (if you have Docker)
echo.
echo For detailed instructions, see README.md
echo.
pause 