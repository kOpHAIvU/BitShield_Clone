@echo off
REM Reset Docker environment for BitShield
REM This script removes old Docker images and rebuilds them

echo ========================================
echo Resetting Docker Environment
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

echo Removing old Docker image...
docker rmi debfd-runner:latest 2>nul
if errorlevel 1 (
    echo No old image to remove.
)

echo.
echo Removing old build directories...
if exist "compilers\tvm-main\build.docker" (
    rmdir /s /q "compilers\tvm-main\build.docker"
    echo Removed TVM build directory.
)

if exist "compilers\glow-main\build.docker" (
    rmdir /s /q "compilers\glow-main\build.docker"
    echo Removed Glow build directory.
)

echo.
echo Building new Docker image...
docker\setup.bat

echo.
echo ========================================
echo Docker environment reset completed!
echo ========================================
echo.
echo You can now run:
echo   run_docker.bat download-datasets
echo   run_docker.bat train resnet50 CIFAR10
echo.
pause
