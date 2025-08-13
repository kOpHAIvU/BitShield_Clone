@echo off
REM Windows version of docker/setup.sh
REM This script builds the Docker image for the project

setlocal enabledelayedexpansion

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Source config
call "%SCRIPT_DIR%\config.bat"

echo Building Docker image for BitShield project...

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running or not installed.
    echo Please install Docker Desktop and start it.
    exit /b 1
)

REM Check if the image already exists
docker image inspect "%BUILT_IMAGE%:latest" >nul 2>&1
if not errorlevel 1 (
    echo Docker image %BUILT_IMAGE% already exists.
    echo Skipping build...
    goto :build_venv
)

echo Building Docker image...
docker build -t "%BUILT_IMAGE%" -f - . <<EOF
FROM cnly/dotfiles-full:bullseye-20230109-c20df35

# TVM deps
RUN apt update && apt install -y \
    ninja-build zlib1g zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev llvm-13 libopenblas-dev

# Glow deps
RUN apt install -y graphviz libpng-dev \
    libprotobuf-dev ninja-build protobuf-compiler wget \
    opencl-headers libgoogle-glog-dev libboost-all-dev \
    libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev \
    libjemalloc-dev libpthread-stubs0-dev liblz4-dev libzstd-dev libbz2-dev \
    libsodium-dev libfmt-dev clang-13
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-13 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-13 100 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 100

# NNFusion deps
RUN apt install -y build-essential cmake git curl zlib1g zlib1g-dev libtinfo-dev unzip \
    autoconf automake libtool ca-certificates gdb sqlite3 libsqlite3-dev libcurl4-openssl-dev \
    libprotobuf-dev protobuf-compiler libgflags-dev libgtest-dev \
    libhwloc-dev libgmock-dev

# Ghidra deps
RUN apt install -y openjdk-17-jdk-headless unzip

# Python
RUN apt install -y libreadline-dev
RUN /home/linuxbrew/.linuxbrew/bin/pyenv install 3.8.12
EOF

:build_venv
REM Initialize venv in Docker if it doesn't exist
if not exist "%PROJECT_DIR%\venv.docker" (
    echo Initializing virtual environment in Docker...
    docker run --rm -i ^
        -v "%PROJECT_DIR%:%PROJECT_DIR%" ^
        -w "%PROJECT_DIR%" ^
        "%BUILT_IMAGE%" ^
        /bin/zsh -ic "source env.sh"
)

REM Install Ghidra if not already installed
if not exist "%PROJECT_DIR%\ghidra\ghidra-app" (
    echo Installing Ghidra...
    call "%SCRIPT_DIR%\run-in-docker.bat" ghidra\install-ghidra.sh
)

REM Build TVM for Docker container
if not exist "%TVM_DIR%\build.docker" (
    echo Building TVM for Docker container...
    docker run --rm -i ^
        -v "%PROJECT_DIR%:%PROJECT_DIR%" ^
        -w "%TVM_DIR%" ^
        "%BUILT_IMAGE%" ^
        /bin/bash -c "set -e && mkdir -p build.docker && cd build.docker && cp \"%RESOURCES_DIR%\tvm-main.config.cmake\" ./config.cmake && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Release .. -G Ninja && ninja"
)

REM Build Glow for Docker container
if not exist "%GLOW_DIR%\build.docker" (
    echo Building Glow for Docker container...
    REM Fix folly version issue
    git -C "%GLOW_DIR%\thirdparty\folly" checkout v2020.10.05.00
    docker run --rm -i ^
        -v "%PROJECT_DIR%:%PROJECT_DIR%" ^
        -w "%GLOW_DIR%" ^
        "%BUILT_IMAGE%" ^
        /bin/bash -c "set -e && mkdir -p build.docker && cd build.docker && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DGLOW_BUILD_TESTS=0 -DCMAKE_BUILD_TYPE=Release .. -G Ninja && ninja"
)

echo Docker setup completed successfully!
echo Use %SCRIPT_DIR%\run-in-docker.bat to run commands inside the container. 