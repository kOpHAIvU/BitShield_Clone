@echo off
echo ========================================
echo BitShield - Quick Start for Windows
echo ========================================
echo.

echo Step 1: Running initial setup...
call setup.bat
if errorlevel 1 (
    echo Setup failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo Step 2: Activating environment...
call env.bat

echo.
echo Step 3: Downloading datasets...
python tools\ensure_datasets.py
if errorlevel 1 (
    echo Dataset download failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. If you have ImageNet dataset, set IMAGENET_ROOT environment variable
echo 2. Install Docker Desktop if you want to build models
echo 3. Run: docker\setup.bat (if you have Docker)
echo 4. Run: python support\models\train.py resnet50 CIFAR10 (to train a model)
echo.
echo For detailed instructions, see README_Windows.md
echo.
pause 