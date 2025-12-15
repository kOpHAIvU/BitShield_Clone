@echo off
REM ========================================================================
REM OBFUS Defense Visualization Pipeline for Windows
REM ========================================================================

echo.
echo ========================================================================
echo   OBFUS Defense Experiment and Visualization
echo ========================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Default parameters
set MODEL=ResNetSEBlockIoT
set DATASET=CICIoT2023
set DEVICE=cuda
set ATTACK_ITERS=25
set ATTACK_MODES=pbs,random,pbs2random,random2pbs
set SIG_PERIOD=20
set SIG_K=3.0
set OBFUS_TARGETS=linear,conv1d
set OBFUS_AUTO_RESEED=10

REM Parse arguments (optional: can be customized)
if not "%1"=="" set MODEL=%1
if not "%2"=="" set DATASET=%2

echo Configuration:
echo   Model: %MODEL%
echo   Dataset: %DATASET%
echo   Device: %DEVICE%
echo   Attack Iterations: %ATTACK_ITERS%
echo   Attack Modes: %ATTACK_MODES%
echo.

REM Activate virtual environment if exists
if exist "bitshield\Scripts\activate.bat" (
    echo Activating virtual environment: bitshield
    call bitshield\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment: venv
    call venv\Scripts\activate.bat
) else (
    echo Warning: No virtual environment found, using system Python
)

echo.
echo ========================================================================
echo   Running OBFUS Experiment Pipeline
echo ========================================================================
echo.

REM Change to project root
cd ..

REM Run the full pipeline
python obfus_visualization\run_full_obfus_pipeline.py %MODEL% %DATASET% ^
  --device %DEVICE% ^
  --attack-iters %ATTACK_ITERS% ^
  --attack-modes %ATTACK_MODES% ^
  --sig-period %SIG_PERIOD% ^
  --sig-k %SIG_K% ^
  --obfus-targets %OBFUS_TARGETS% ^
  --obfus-auto-reseed %OBFUS_AUTO_RESEED%

if errorlevel 1 (
    echo.
    echo Error: Pipeline failed!
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo   Pipeline Complete!
echo ========================================================================
echo.
echo Results saved to: results\obfus_experiments\%DATASET%_%MODEL%_obfus_experiment.json
echo Visualizations saved to: results\obfus_visualizations\%DATASET%_%MODEL%\
echo.
echo To view visualizations, open the PNG files in:
echo   results\obfus_visualizations\%DATASET%_%MODEL%\
echo.
pause

