@echo off
REM Stepik Retention - Setup and Run
REM Run from: models_chat\ML contest\stepik_retention

cd /d "%~dp0"
cd ..

echo === Step 1: Training model ===
python stepik_retention/train_model.py
if errorlevel 1 (
    echo Failed. Make sure pandas, numpy, scikit-learn, xgboost are installed.
    pause
    exit /b 1
)

echo.
echo === Step 2: Precomputing user features ===
python stepik_retention/precompute_features.py
if errorlevel 1 (
    echo Failed to precompute features.
    pause
    exit /b 1
)

echo.
echo === Step 3: Starting Docker containers ===
cd stepik_retention
docker-compose up -d --build

echo.
echo Done! Open http://localhost:3003
pause
