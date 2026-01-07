#!/bin/bash

echo "================================================="
echo "   Setting up Optimized Environment for M4 Pro   "
echo "================================================="

# 1. Create a virtual environment specifically for this project
#    We use 'venv' to keep it isolated from your system Python.
if [ ! -d "venv_m4" ]; then
    echo "[INFO] Creating virtual environment 'venv_m4'..."
    python3 -m venv .venv_m4
else
    echo "[INFO] Virtual environment 'venv_m4' already exists."
fi

# 2. Activate the environment
source venv_m4/bin/activate

# 3. Upgrade pip to handle the latest wheels
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# 4. Install Dependencies
echo "[INFO] Installing Optimized Packages..."
pip install -r requirements.txt

# 5. Verify GPU Detection
echo "================================================="
echo "   Verifying GPU Access...                       "
echo "================================================="
python3 -c "import tensorflow as tf; print(f'TensorFlow Version: {tf.__version__}'); print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')));"

echo "================================================="
echo "   Setup Complete!                               "
echo "   To run your tools, use:                       "
echo "   source venv_m4/bin/activate                   "
echo "   python sequence_collector.py                  "
echo "================================================="