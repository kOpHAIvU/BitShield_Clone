@echo off
REM Complete TVM installation script with proper Python 3.11 and tvm.ffi build

setlocal enabledelayedexpansion

call "%~dp0config.bat"

echo ========================================
echo Complete TVM Installation
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running or not installed.
    pause
    exit /b 1
)

REM Step 1: Install Python 3.11 pip and dependencies
echo Step 1: Installing Python 3.11 pip and build dependencies...
docker run --rm -i ^
    -v "%cd%:/workspace" ^
    "%BUILT_IMAGE%" ^
    /bin/bash -c "apt-get update -qq && apt-get install -y python3.11-venv python3.11-dev python3-pip python3-distutils curl 2>&1 | tail -3 && pip3 install --break-system-packages --upgrade pip setuptools wheel cython scikit-build-core setuptools-scm ninja 2>&1 | tail -5"

REM Step 2: Check/Clone TVM
if not exist "%PROJECT_DIR%\compilers\tvm-main" (
    echo Step 2: Cloning TVM...
    mkdir "%PROJECT_DIR%\compilers" 2>nul
    docker run --rm -i ^
        -v "%cd%:/workspace" ^
        -w "/workspace/compilers" ^
        "%BUILT_IMAGE%" ^
        /bin/bash -c "git clone --recursive https://github.com/apache/tvm.git tvm-main"
) else (
    echo Step 2: TVM directory exists. Skipping clone.
)

REM Step 3: Build TVM C++ library
if not exist "%PROJECT_DIR%\compilers\tvm-main\build.docker" (
    echo Step 3: Building TVM C++ library...
    docker run --rm -i ^
        -v "%cd%:/workspace" ^
        -w "/workspace/compilers/tvm-main" ^
        "%BUILT_IMAGE%" ^
        /bin/bash -c "apt-get update -qq && apt-get install -y llvm-13-dev llvm-13 2>&1 || apt-get install -y llvm-14-dev llvm-14 2>&1 || true; mkdir -p build.docker; cd build.docker; cp ../cmake/config.cmake .; if which llvm-config-14 2>&1; then sed -i s/USE_LLVM\ OFF/USE_LLVM\ llvm-config-14/ config.cmake; elif which llvm-config-13 2>&1; then sed -i s/USE_LLVM\ OFF/USE_LLVM\ llvm-config-13/ config.cmake; fi; cmake -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON=ON -DPython3_EXECUTABLE=/usr/bin/python3.11 ..; make -j4 2>&1 | tail -10"
) else (
    echo Step 3: TVM C++ library already built. Skipping.
)

REM Step 4: Build and install tvm-ffi
echo Step 4: Building and installing tvm-ffi package...
docker run --rm -i ^
    -v "%cd%:/workspace" ^
    -w "/workspace/compilers/tvm-main/3rdparty/tvm-ffi" ^
    -e TVM_LIBRARY_PATH=/workspace/compilers/tvm-main/build.docker ^
    "%BUILT_IMAGE%" ^
    /bin/bash -c "export TVM_LIBRARY_PATH=/workspace/compilers/tvm-main/build.docker && export LD_LIBRARY_PATH=/workspace/compilers/tvm-main/build.docker/lib && pip3 install --break-system-packages -e . --no-build-isolation 2>&1 | tail -30"

REM Step 5: Setup tvm.ffi module symlink
echo Step 5: Setting up tvm.ffi module...
docker run --rm -i ^
    -v "%cd%:/workspace" ^
    -w "/workspace/compilers/tvm-main" ^
    "%BUILT_IMAGE%" ^
    /bin/bash -c "rm -f python/tvm_ffi; ln -sf ../3rdparty/tvm-ffi/python/tvm_ffi python/tvm_ffi; mkdir -p python/tvm/ffi; ls -la python/tvm_ffi python/tvm/ffi/__init__.py 2>&1 | head -3"

REM Step 6: Install TVM Python package
echo Step 6: Installing TVM Python package...
docker run --rm -i ^
    -v "%cd%:/workspace" ^
    -w "/workspace/compilers/tvm-main" ^
    -e TVM_LIBRARY_PATH=/workspace/compilers/tvm-main/build.docker ^
    "%BUILT_IMAGE%" ^
    /bin/bash -c "export TVM_LIBRARY_PATH=/workspace/compilers/tvm-main/build.docker && export LD_LIBRARY_PATH=/workspace/compilers/tvm-main/build.docker/lib && pip3 install --break-system-packages -e python 2>&1 | tail -15"

REM Step 7: Test TVM import
echo Step 7: Testing TVM import...
docker run --rm -i ^
    -v "%cd%:/workspace" ^
    -w "/workspace" ^
    -e TVM_HOME=/workspace/compilers/tvm-main ^
    -e PYTHONPATH=/workspace/compilers/tvm-main/python ^
    -e LD_LIBRARY_PATH=/workspace/compilers/tvm-main/build.docker/lib:/workspace/compilers/tvm-main/build.docker ^
    -e TVM_LIBRARY_PATH=/workspace/compilers/tvm-main/build.docker ^
    "%BUILT_IMAGE%" ^
    /bin/bash -c "python3.11 -c 'import tvm; print(\"TVM version:\", tvm.__version__)'"

if errorlevel 1 (
    echo.
    echo ERROR: TVM import failed!
    echo.
    echo Troubleshooting:
    echo 1. Check if tvm.ffi is installed: python3.11 -c \"from tvm import ffi\"
    echo 2. Check TVM library: ls -la /workspace/compilers/tvm-main/build.docker/lib/libtvm*.so
    echo 3. Check environment variables are set correctly
) else (
    echo.
    echo ========================================
    echo TVM installation completed successfully!
    echo ========================================
    echo.
    echo You can now run:
    echo   docker\run-in-docker.bat python3.11 buildmodels.py --compiler tvm ...
    echo.
)

pause

