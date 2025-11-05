@echo off
echo ==========================================
echo Starting PredictWell Backend with Miniconda
echo ==========================================
echo.

REM Check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda is not installed or not in PATH
    echo Please install Miniconda from https://docs.conda.io/en/latest/miniconda.html
    echo Or activate your conda base environment first
    pause
    exit /b 1
)

REM Navigate to backend directory
cd /d "%~dp0"

REM Check if environment exists
conda env list | findstr "predictwell" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Creating conda environment 'predictwell'...
    echo This may take a few minutes...
    conda env create -f environment.yml
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create conda environment
        pause
        exit /b 1
    )
    echo.
    echo Environment created successfully!
    echo.
)

REM Activate environment
echo Activating conda environment 'predictwell'...
call conda activate predictwell

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate conda environment
    pause
    exit /b 1
)

echo.
echo Environment activated!
echo.

REM Check if dependencies are installed
python -c "import flask" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo.
    echo Dependencies installed!
    echo.
)

REM Start the Flask server
echo ==========================================
echo Starting Flask Backend Server...
echo ==========================================
echo.
echo Backend will be available at: http://localhost:10000
echo Press Ctrl+C to stop the server
echo.

python run.py

pause

