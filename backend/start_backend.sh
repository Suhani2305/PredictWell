#!/bin/bash

echo "=========================================="
echo "Starting PredictWell Backend with Miniconda"
echo "=========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda is not installed or not in PATH"
    echo "Please install Miniconda from https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Navigate to backend directory
cd "$(dirname "$0")"

# Check if environment exists
if ! conda env list | grep -q "predictwell"; then
    echo "Creating conda environment 'predictwell'..."
    echo "This may take a few minutes..."
    conda env create -f environment.yml
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create conda environment"
        exit 1
    fi
    echo ""
    echo "Environment created successfully!"
    echo ""
fi

# Activate environment
echo "Activating conda environment 'predictwell'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate predictwell

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

echo ""
echo "Environment activated!"
echo ""

# Check if dependencies are installed
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
    echo ""
    echo "Dependencies installed!"
    echo ""
fi

# Start the Flask server
echo "=========================================="
echo "Starting Flask Backend Server..."
echo "=========================================="
echo ""
echo "Backend will be available at: http://localhost:10000"
echo "Press Ctrl+C to stop the server"
echo ""

python run.py


