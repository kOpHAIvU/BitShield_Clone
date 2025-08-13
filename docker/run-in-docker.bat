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
    exit /b 1
)

REM Check if the Docker image exists
docker image inspect "%BUILT_IMAGE%:latest" >nul 2>&1
if errorlevel 1 (
    echo Error: Docker image %BUILT_IMAGE% not found.
    echo Please run docker\setup.bat first to build the image.
    exit /b 1
)

REM Build the command
set "CMD=%*"
if "%CMD%"=="" (
    echo Usage: %0 ^<command^>
    echo Example: %0 python tools\ensure_datasets.py
    exit /b 1
)

echo Running in Docker: %CMD%

REM Run the command in Docker
docker run --rm -it ^
    -v "%IMAGENET_ROOT%:%IMAGENET_ROOT%:ro" ^
    -v "%PROJECT_DIR%:%PROJECT_DIR%" ^
    -w "%PROJECT_DIR%" ^
    -e TVM_LIBRARY_PATH=%TVM_DIR%\build.docker ^
    --ulimit core=0 ^
    --shm-size 1G ^
    "%BUILT_IMAGE%" ^
    /bin/bash -ic "source \"%PROJECT_DIR%\"/env.sh && ( %CMD% )"

set "ret=%errorlevel%"

echo Fixing potential permissions issues...
docker run --rm -it ^
    -v "%PROJECT_DIR%:%PROJECT_DIR%" ^
    -w "%PROJECT_DIR%" ^
    "%BUILT_IMAGE%" ^
    /bin/bash -c "for x in ghidra/{db,analysis} models built results built-aux .cache; do chown -R $(id -u):$(id -g) \"%PROJECT_DIR%/\$x\" 2>/dev/null; done" 2>nul

exit /b %ret% 