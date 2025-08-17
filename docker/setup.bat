@echo off
REM Windows version of docker/setup.sh
REM This script builds the Docker image for the project

setlocal enabledelayedexpansion

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Source config
call "%SCRIPT_DIR%\config.bat"

echo ========================================
echo Building Docker image for BitShield
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running or not installed.
    echo Please install Docker Desktop and start it.
    pause
    exit /b 1
)

echo Docker is running. Checking image...

REM Check if the image already exists
docker image inspect "%BUILT_IMAGE%:latest" >nul 2>&1
if not errorlevel 1 (
    echo Docker image %BUILT_IMAGE% already exists.
    echo Skipping build...
    goto :build_compilers
)

echo Building Docker image...
docker build -t "%BUILT_IMAGE%" -f "%SCRIPT_DIR%\Dockerfile" .
if errorlevel 1 (
    echo Error: Failed to build Docker image.
    pause
    exit /b 1
)

:build_compilers
echo.
echo Building compilers...

REM Build TVM for Docker container
if not exist "%PROJECT_DIR%\compilers\tvm-main\build.docker" (
    echo Building TVM...
    docker run --rm -i ^
        -v "%PROJECT_DIR%:%PROJECT_DIR_DOCKER%" ^
        -w "%TVM_DIR%" ^
        "%BUILT_IMAGE%" ^
        /bin/bash -c "mkdir -p build.docker && cd build.docker && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4"
    if errorlevel 1 (
        echo Warning: TVM build failed. Some functionality may not work.
    )
)

REM Build Glow for Docker container
if not exist "%PROJECT_DIR%\compilers\glow-main\build.docker" (
    echo Building Glow...
    docker run --rm -i ^
        -v "%PROJECT_DIR%:%PROJECT_DIR_DOCKER%" ^
        -w "%GLOW_DIR%" ^
        "%BUILT_IMAGE%" ^
        /bin/bash -c "mkdir -p build.docker && cd build.docker && cmake -DGLOW_BUILD_TESTS=0 -DCMAKE_BUILD_TYPE=Release .. && make -j4"
    if errorlevel 1 (
        echo Warning: Glow build failed. Some functionality may not work.
    )
)

echo.
echo ========================================
echo Docker setup completed successfully!
echo ========================================
echo.
echo Use docker\run-in-docker.bat to run commands inside the container.
echo Example: docker\run-in-docker.bat python buildmodels.py
echo.
pause 