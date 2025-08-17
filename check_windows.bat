@echo off
REM Windows compatibility check for BitShield
REM This script checks if the system is ready to run BitShield

setlocal enabledelayedexpansion

echo ========================================
echo BitShield - Windows Compatibility Check
echo ========================================
echo.

set "all_good=true"

echo Checking system requirements...
echo.

REM Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo   ❌ Python not found. Please install Python 3.8+ from python.org
    set "all_good=false"
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do (
        echo   ✅ Python %%i found
    )
)

REM Check Git
echo.
echo [2/5] Checking Git...
git --version >nul 2>&1
if errorlevel 1 (
    echo   ❌ Git not found. Please install Git from git-scm.com
    set "all_good=false"
) else (
    for /f "tokens=3" %%i in ('git --version 2^>^&1') do (
        echo   ✅ Git %%i found
    )
)

REM Check Docker (optional)
echo.
echo [3/5] Checking Docker (optional)...
docker --version >nul 2>&1
if errorlevel 1 (
    echo   ⚠️  Docker not found. Advanced features will not be available.
    echo   💡 Install Docker Desktop for full functionality.
) else (
    for /f "tokens=3" %%i in ('docker --version 2^>^&1') do (
        echo   ✅ Docker %%i found
    )
)

REM Check virtual environment
echo.
echo [4/5] Checking virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo   ✅ Virtual environment found
) else (
    echo   ❌ Virtual environment not found. Run setup.bat first.
    set "all_good=false"
)

REM Check project structure
echo.
echo [5/5] Checking project structure...
if exist "support\models" (
    echo   ✅ Project structure looks good
) else (
    echo   ❌ Project structure incomplete. Run setup.bat first.
    set "all_good=false"
)

echo.
echo ========================================
echo Summary
echo ========================================

if "%all_good%"=="true" (
    echo ✅ System is ready to run BitShield!
    echo.
    echo Next steps:
    echo 1. Run: env.bat
    echo 2. Run: run_simple.bat download-datasets
    echo 3. Run: run_simple.bat train lenet1 MNIST
    echo.
    echo For advanced features, install Docker Desktop and run:
    echo   docker\setup.bat
) else (
    echo ❌ System needs configuration.
    echo.
    echo Please fix the issues above and run setup.bat again.
)

echo.
echo For detailed instructions, see README.md
echo.
pause
