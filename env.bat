@echo off
REM Environment activation script for BitShield
REM This script activates the virtual environment and sets up the project environment

echo Activating BitShield environment...

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo Please run setup.bat first to create the environment.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Set project environment variables
set PYTHONPATH=%CD%;%PYTHONPATH%

echo Environment activated successfully!
echo.
echo You can now run BitShield commands.
echo Type 'deactivate' to exit the environment.
echo. 