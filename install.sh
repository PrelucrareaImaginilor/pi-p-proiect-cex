#!/bin/bash
set -e

VENV_NAME=".venv"

echo "--- Sign Language Detector Setup (Python 3.11 Fix) ---"

# 1. Look for Python 3.11 specifically
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "Found Python 3.11: $($PYTHON_CMD --version)"
elif command -v python3 &> /dev/null; then
    # Check if python3 is actually 3.11
    VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$VERSION" == "3.11" ]; then
        PYTHON_CMD="python3"
        echo "Found Python 3.11 (via python3)"
    else
        echo "ERROR: Python 3.11 is required. You have version $VERSION."
        echo "Please run: brew install python@3.11"
        exit 1
    fi
else
    echo "ERROR: Python 3.11 not found."
    echo "Please install it: brew install python@3.11"
    exit 1
fi

# 2. Delete old environment to avoid conflicts
if [ -d "$VENV_NAME" ]; then
    echo "Removing old virtual environment..."
    rm -rf "$VENV_NAME"
fi

# 3. Create new environment
echo "Creating virtual environment using $PYTHON_CMD..."
$PYTHON_CMD -m venv $VENV_NAME

# 4. Activate and Install
source "$VENV_NAME/bin/activate"
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
# Install in specific order to prevent dependency resolution errors
pip install "protobuf==3.20.3"
pip install "tensorflow==2.15.0"
pip install -r requirements.txt

echo ""
echo "--- Setup Complete! ---"
echo "Run: source .venv/bin/activate"