@echo off
REM Batch script to run dataset generation with Isaac Sim
REM Place this file in your isaac-sim directory

echo ============================================================
echo Multimodal World Model Grounding Test - Dataset Generation
echo ============================================================
echo.

REM Change to the script directory if needed
REM cd /d %~dp0

echo [1/2] Testing Isaac Sim setup...
python.bat test_isaac_sim.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Test failed. Please check your Isaac Sim installation.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting dataset generation...
echo This may take 5-10 minutes for 100 episodes...
echo.

python.bat generate_dataset.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Dataset generation failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! Dataset generated in ./physics_dataset/
echo ============================================================
echo.
echo Next steps:
echo 1. Check ./physics_dataset/metadata.json for dataset info
echo 2. Adjust config.py to generate more episodes
echo 3. Use data_loader.py for model training
echo.
pause
