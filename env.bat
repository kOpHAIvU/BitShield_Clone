@echo off
REM Environment setup script for Windows
REM Source this script to set up the environment

setlocal enabledelayedexpansion

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Set project paths
set "TVM_DIR=%SCRIPT_DIR%\compilers\tvm-main"
set "NNFUSION_DIR=%SCRIPT_DIR%\compilers\nnfusion-main"
set "TOOLS_DIR=%SCRIPT_DIR%\tools"

REM Set virtual environment path
set "VENV_DIR=%SCRIPT_DIR%\venv"

REM Change to project directory
cd /d "%SCRIPT_DIR%"

REM Check if virtual environment exists
if not exist "%VENV_DIR%" (
    echo Virtual environment not found. Please run setup.bat first.
    exit /b 1
)

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

REM Add tools directory to PATH
set "PATH=%TOOLS_DIR%;%PATH%"

REM Set Python path for TVM and NNFusion
set "PYTHONPATH=%TVM_DIR%\python;%NNFUSION_DIR%\src\python;%PYTHONPATH%"

REM Set torch home
set "TORCH_HOME=%SCRIPT_DIR%\.cache\torch"

REM Set ImageNet root if not already set
if not defined IMAGENET_ROOT (
    echo Warning: IMAGENET_ROOT not set. Some functionality may not work.
    echo Please set it to your ImageNet dataset path.
)

echo Environment activated successfully!
echo Python: %PYTHONPATH%
echo Tools: %TOOLS_DIR%
echo. 