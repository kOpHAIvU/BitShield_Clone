@echo off
REM Windows version of runattacksim.sh
REM This script runs attack simulations

setlocal enabledelayedexpansion

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "attacksim=%SCRIPT_DIR%\..\attacksim.py"

REM Function to run attack simulation
:run_attacksim
set "args=%* --skip-existing"
echo Running with %args%
python "%attacksim%" %args%
goto :eof

REM Function to run with DRAM configurations
:_run_with_drams
call :run_attacksim %* --vuln-pct 0.16e-4 --zero-one-pct 51.15 --nexps 50
call :run_attacksim %* --vuln-pct 0.39e-4 --zero-one-pct 48.89 --nexps 50
call :run_attacksim %* --vuln-pct 3.04e-4 --zero-one-pct 50.59 --nexps 50
call :run_attacksim %* --vuln-pct 26.40e-4 --zero-one-pct 50.75 --nexps 50
call :run_attacksim %* --vuln-pct 64.54e-4 --zero-one-pct 51.16 --nexps 50
goto :eof

REM Function to run with protections and DRAM configurations
:_run_with_protections_and_drams
call :_run_with_drams %* --cig ncnp --dig nd
call :_run_with_drams %* --cig cc2 --dig gn1
goto :eof

REM Main logic
if "%1"=="" (
    REM No arguments - run all combinations
    for %%m in (resnet50 googlenet densenet121) do (
        for %%d in (CIFAR10 MNISTC FashionC) do (
            for %%a in (w a s) do (
                call :_run_with_protections_and_drams --model-name %%m --dataset %%d --attacker-type %%a
            )
        )
    )
    goto :eof
)

if "%1"=="--check-sig-bypass" (
    call :_run_with_drams %*
    goto :eof
)

if "%4"=="" (
    echo Usage: %0 [ -m model-name -d dataset | --check-sig-bypass ]
    exit /b 1
)

REM 4 arguments provided - run with all attacker types
for %%a in (w a s) do (
    call :_run_with_protections_and_drams %* --attacker-type %%a
) 