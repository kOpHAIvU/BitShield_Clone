@echo off
REM Test Docker setup for BitShield

echo ========================================
echo Testing Docker Setup
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running or not installed.
    pause
    exit /b 1
)

echo Docker is running.

REM Check if image exists
docker image inspect debfd-runner:latest >nul 2>&1
if errorlevel 1 (
    echo Error: Docker image debfd-runner not found.
    echo Please run docker\setup.bat first.
    pause
    exit /b 1
)

echo Docker image found.

REM Test simple command
echo Testing simple command...
docker run --rm debfd-runner:latest echo "Hello from Docker"

if errorlevel 1 (
    echo Error: Failed to run simple command.
    pause
    exit /b 1
)

echo Simple command successful.

REM Test with volume mount
echo Testing volume mount...
docker run --rm -v "%cd%:/workspace" -w "/workspace" debfd-runner:latest echo "Hello from mounted volume"

if errorlevel 1 (
    echo Error: Failed to run with volume mount.
    pause
    exit /b 1
)

echo Volume mount successful.

echo.
echo ========================================
echo Docker test completed successfully!
echo ========================================
echo.
pause
