@echo off
echo ========================================
echo TRAINING ALL MODELS ON ALL DATASETS
echo ========================================
echo.

echo Choose training mode:
echo 1. Full training (10 epochs each)
echo 2. Train specific dataset only
echo 3. Exit
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting full training...
    python train_all_models.py
)  else if "%choice%"=="2" (
    echo.
    echo Available datasets:
    echo 1. IoTID20
    echo 2. WUSTL
    echo 3. CICIoT2023
    echo.
    set /p dataset_choice="Choose dataset (1-3): "
    
    if "%dataset_choice%"=="1" (
        python train_dataset_models.py IoTID20
    ) else if "%dataset_choice%"=="2" (
        python train_dataset_models.py WUSTL
    ) else if "%dataset_choice%"=="3" (
        python train_dataset_models.py CICIoT2023
    ) else (
        echo Invalid choice!
        pause
        exit /b 1
    )
) else if "%choice%"=="3" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice!
    pause
    exit /b 1
)

echo.
echo Training completed!
pause

