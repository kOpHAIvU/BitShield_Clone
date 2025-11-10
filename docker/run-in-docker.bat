@echo off
REM Windows version of run-in-docker.sh
REM This script runs commands inside the Docker container

setlocal enabledelayedexpansion

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Source config
call "%SCRIPT_DIR%\config.bat"

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running or not installed.
    echo Please install Docker Desktop and start it.
    pause
    exit /b 1
)

REM Check if the Docker image exists
docker image inspect "%BUILT_IMAGE%:latest" >nul 2>&1
if errorlevel 1 (
    echo Error: Docker image %BUILT_IMAGE% not found.
    echo Please run docker\setup.bat first to build the image.
    pause
    exit /b 1
)

REM Build the command
set "CMD=%*"
if "%CMD%"=="" (
    echo Usage: %0 ^<command^>
    echo Example: %0 python tools\ensure_datasets.py
    pause
    exit /b 1
)

echo Running in Docker: %CMD%

REM Convert Windows path separators to Unix path separators for Docker
set "DOCKER_CMD=%CMD%"
set "DOCKER_CMD=%DOCKER_CMD:\=/%"

REM Run the command in Docker with TVM environment setup
docker run --rm -it ^
    -v "%cd%:/workspace" ^
    -w "/workspace" ^
    -e TVM_LIBRARY_PATH=/workspace/compilers/tvm-main/build.docker ^
    --ulimit core=0 ^
    --shm-size 1G ^
    "%BUILT_IMAGE%" ^
    /bin/bash -c "if [ -d /workspace/compilers/tvm-main ]; then export TVM_HOME=/workspace/compilers/tvm-main && export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} && export LD_LIBRARY_PATH=$TVM_HOME/build.docker:${LD_LIBRARY_PATH} && cd $TVM_HOME && pip install -e python -q 2>/dev/null || true; fi && %DOCKER_CMD%"

set "ret=%errorlevel%"

echo Command completed with exit code: %ret%
exit /b %ret% 