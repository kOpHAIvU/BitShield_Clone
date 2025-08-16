@echo off
echo ========================================
echo BitShield - Quick Start for Next Time
echo ========================================
echo.

echo Step 1: Activating environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment.
    echo Please run setup_fix.ps1 first.
    pause
    exit /b 1
)

echo.
echo Step 2: Checking environment...
python --version
python -c "import torch; print('PyTorch:', torch.__version__)"

echo.
echo ========================================
echo Environment ready!
echo ========================================
echo.
echo Available commands:
echo 1. Training: python support\models\train.py resnet50 CIFAR10 --epochs 3
echo 2. Docker setup: docker\setup.bat
echo 3. Build models: docker\run-in-docker.bat python buildmodels.py
echo 4. Experiments: docker\run-in-docker.bat python flipsweep.py -m resnet50 -d CIFAR10
echo.
echo For detailed instructions, see README_Windows_Complete.md
echo.
