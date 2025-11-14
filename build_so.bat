@echo off
REM Script để build TVM .so files từ models (Windows version)
REM Usage: build_so.bat [options]

setlocal enabledelayedexpansion

REM Default values
set COMPILER=tvm
set COMPILER_VER=main
set MODEL=resnet50
set DATASET=CIFAR10
set CIG=ncnp
set DIG=nd
set AVX=true
set OPT_LEVEL=3
set CHECK_ACC=false
set FORCE=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="-c" set COMPILER=%~2 & shift & shift & goto parse_args
if /i "%~1"=="--compiler" set COMPILER=%~2 & shift & shift & goto parse_args
if /i "%~1"=="-v" set COMPILER_VER=%~2 & shift & shift & goto parse_args
if /i "%~1"=="--compiler-ver" set COMPILER_VER=%~2 & shift & shift & goto parse_args
if /i "%~1"=="-m" set MODEL=%~2 & shift & shift & goto parse_args
if /i "%~1"=="--model" set MODEL=%~2 & shift & shift & goto parse_args
if /i "%~1"=="-d" set DATASET=%~2 & shift & shift & goto parse_args
if /i "%~1"=="--dataset" set DATASET=%~2 & shift & shift & goto parse_args
if /i "%~1"=="-i" set CIG=%~2 & shift & shift & goto parse_args
if /i "%~1"=="--cig" set CIG=%~2 & shift & shift & goto parse_args
if /i "%~1"=="-I" set DIG=%~2 & shift & shift & goto parse_args
if /i "%~1"=="--dig" set DIG=%~2 & shift & shift & goto parse_args
if /i "%~1"=="-X" set AVX=false & shift & goto parse_args
if /i "%~1"=="--no-avx" set AVX=false & shift & goto parse_args
if /i "%~1"=="-O" set OPT_LEVEL=%~2 & shift & shift & goto parse_args
if /i "%~1"=="--opt-level" set OPT_LEVEL=%~2 & shift & shift & goto parse_args
if /i "%~1"=="-A" set CHECK_ACC=true & shift & goto parse_args
if /i "%~1"=="--check-acc" set CHECK_ACC=true & shift & goto parse_args
if /i "%~1"=="-f" set FORCE=true & shift & goto parse_args
if /i "%~1"=="--force" set FORCE=true & shift & goto parse_args
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="--help" goto show_help
shift
goto parse_args

:end_parse

REM Build command
set BUILD_CMD=python buildmodels.py
set BUILD_CMD=!BUILD_CMD! --compiler %COMPILER%
set BUILD_CMD=!BUILD_CMD! --compiler_ver %COMPILER_VER%
set BUILD_CMD=!BUILD_CMD! --model %MODEL%
set BUILD_CMD=!BUILD_CMD! --dataset %DATASET%
set BUILD_CMD=!BUILD_CMD! --cig %CIG%
set BUILD_CMD=!BUILD_CMD! --dig %DIG%
set BUILD_CMD=!BUILD_CMD! --opt-level %OPT_LEVEL%

if "%AVX%"=="false" set BUILD_CMD=!BUILD_CMD! --no-avx

if "%CHECK_ACC%"=="true" (
    set BUILD_CMD=!BUILD_CMD! --check-acc
) else (
    set BUILD_CMD=!BUILD_CMD! --no-check-acc
)

if "%FORCE%"=="true" set BUILD_CMD=!BUILD_CMD! --force

REM Print build information
echo ========================================
echo Building .so File
echo ========================================
echo Compiler:     %COMPILER% (%COMPILER_VER%)
echo Model:        %MODEL%
echo Dataset:      %DATASET%
echo CIG Mode:     %CIG%
echo DIG Mode:     %DIG%
echo AVX:          %AVX%
echo Opt Level:    %OPT_LEVEL%
echo Check Acc:    %CHECK_ACC%
echo Force:        %FORCE%
echo ========================================
echo.

REM Check if model file exists
set MODEL_FILE=models\%DATASET%\%MODEL%\%MODEL%.pt
if not exist "%MODEL_FILE%" (
    echo Error: Model file not found: %MODEL_FILE%
    echo Please train the model first.
    exit /b 1
)
echo Model file found: %MODEL_FILE%
echo.

REM Run build
echo Running build command...
echo %BUILD_CMD%
echo.

call %BUILD_CMD%
if errorlevel 1 (
    echo.
    echo ========================================
    echo Build failed!
    echo ========================================
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo You can now use this .so file for:
echo   - Attack simulation
echo   - Ghidra analysis
echo   - Performance benchmarking
echo.
echo See README_BUILD.md for more information.
goto end

:show_help
echo Usage: build_so.bat [OPTIONS]
echo.
echo Build TVM/Glow/NNFusion .so files from trained models
echo.
echo Options:
echo   -c, --compiler COMPILER      Compiler to use (tvm, glow, nnfusion) [default: tvm]
echo   -v, --compiler-ver VERSION    Compiler version [default: main]
echo   -m, --model MODEL             Model name (resnet50, googlenet, densenet121) [default: resnet50]
echo   -d, --dataset DATASET         Dataset name (CIFAR10, MNISTC, FashionC) [default: CIFAR10]
echo   -i, --cig CIG                 CIG mode (nc, ncnp, cc1, cc2) [default: ncnp]
echo   -I, --dig DIG                 DIG mode (nd, gn1, gn2, gninf) [default: nd]
echo   -X, --no-avx                  Disable AVX optimization
echo   -O, --opt-level LEVEL         Optimization level (0-3) [default: 3]
echo   -A, --check-acc               Check accuracy after build
echo   -f, --force                   Force rebuild even if file exists
echo   -h, --help                    Show this help message
echo.
echo Examples:
echo   build_so.bat -m resnet50 -d CIFAR10 -I nd
echo   build_so.bat -m resnet50 -d CIFAR10 -I gn1 -f
goto end

:end
endlocal

