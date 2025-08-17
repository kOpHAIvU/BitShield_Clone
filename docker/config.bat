@echo off
REM Windows version of config.sh
REM Configuration variables for Docker setup

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Set project paths
set "PROJECT_DIR=%SCRIPT_DIR%\.."
REM Convert to absolute path
for %%i in ("%PROJECT_DIR%") do set "PROJECT_DIR=%%~fi"

REM Convert Windows path to Docker path format (use forward slashes)
set "PROJECT_DIR_DOCKER=%PROJECT_DIR%"
set "PROJECT_DIR_DOCKER=%PROJECT_DIR_DOCKER:\=/%"

REM Set compiler paths with forward slashes for Docker
set "TVM_DIR=%PROJECT_DIR%/compilers/tvm-main"
set "GLOW_DIR=%PROJECT_DIR%/compilers/glow-main"
set "NNFUSION_DIR=%PROJECT_DIR%/compilers/nnfusion-main"
set "RESOURCES_DIR=%PROJECT_DIR%/resources"

REM Convert all paths to forward slashes for Docker
set "TVM_DIR=%TVM_DIR:\=/%"
set "GLOW_DIR=%GLOW_DIR:\=/%"
set "NNFUSION_DIR=%NNFUSION_DIR:\=/%"
set "RESOURCES_DIR=%RESOURCES_DIR:\=/%"

REM Set Docker image name
set "BUILT_IMAGE=debfd-runner"

REM Check if IMAGENET_ROOT is set
if not defined IMAGENET_ROOT (
    echo Warning: IMAGENET_ROOT environment variable is not set.
    echo Some functionality may not work without ImageNet dataset.
    set "IMAGENET_ROOT=D:/temp/imagenet"
)

REM Debug: Print paths to verify
echo Debug: PROJECT_DIR=%PROJECT_DIR%
echo Debug: PROJECT_DIR_DOCKER=%PROJECT_DIR_DOCKER%
echo Debug: TVM_DIR=%TVM_DIR% 