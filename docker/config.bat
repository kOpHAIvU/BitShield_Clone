@echo off
REM Windows version of config.sh
REM Configuration variables for Docker setup

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Set project paths
set "PROJECT_DIR=%SCRIPT_DIR%\.."
set "TVM_DIR=%PROJECT_DIR%\compilers\tvm-main"
set "GLOW_DIR=%PROJECT_DIR%\compilers\glow-main"
set "NNFUSION_DIR=%PROJECT_DIR%\compilers\nnfusion-main"
set "RESOURCES_DIR=%PROJECT_DIR%\resources"

REM Set Docker image name
set "BUILT_IMAGE=debfd-runner"

REM Check if IMAGENET_ROOT is set
if not defined IMAGENET_ROOT (
    echo Warning: IMAGENET_ROOT environment variable is not set.
    echo Some functionality may not work without ImageNet dataset.
    set "IMAGENET_ROOT=C:\temp\imagenet"
) 