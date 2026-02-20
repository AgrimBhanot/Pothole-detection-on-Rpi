#!/bin/bash

# Quick start script for RPi5 Object Detection System
# This script helps with initial setup and testing

set -e

echo "=============================================="
echo "RPi5 Object Detection System - Quick Start"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for linux sys
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${YELLOW}Warning: This script is optimized for Linux/RPi5${NC}"
fi


echo "[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python $python_version found"
echo ""

# Check if virtual environment exists
echo "[2/6] Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv --system-site-packages
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi
echo ""

# Activate venv
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated"
echo ""

# Install dependencies
echo "[4/6] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed"
echo ""

# Check for model files
echo "[5/6] Checking for model files..."
model_found=false

if [ -f "new_model/Model1_Pothole.onnx" ]; then
    echo "Pothole model found: Model1_Pothole.onnx"
    model_found=true
else
    echo -e "${YELLOW} Pothole model not found${NC}"
fi

if [ -f "new_model/Model2_General.onnx" ]; then
    echo " General model found"
    model_found=true
else
    echo -e "${YELLOW} General model not found: Model2_General.onnx${NC}"
    echo "  Note: You can run in single-model mode with --single-model flag"
fi

if [ "$model_found" = false ]; then
    echo -e "${RED}No model files found!${NC}"
    echo "  Please place your ONNX model new_model in this directory."
    exit 1
fi
echo ""

echo "[6/6] Final Checkup"
source venv/bin/activate



echo "=============================================="
echo "Setup complete! Here are some usage examples:"
echo "=============================================="
echo ""
echo "Run with camera:"
echo "  python Source/main.py --source camera"
echo ""
echo "Run with video file:"
echo "  python Source/main.py --source video.mp4"
echo ""
echo "Single model mode:"
echo "  python Source/main.py --source camera --single-model"
echo ""
echo "Disable saving:"
echo "  python Source/main.py --source camera --no-save"
echo ""
echo "Custom thresholds:"
echo "  python Source/main.py --source camera --conf-threshold 0.6 --save-threshold 0.8"
echo ""
echo "For more options, run:"
echo "  python Source/main.py --help"
echo ""
echo "Please verify the activation of VIRTUAL ENVIRONMENT first"
echo "=============================================="
echo -e "${GREEN}Ready to detect! ${NC}"
echo "=============================================="
