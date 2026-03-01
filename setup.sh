#!/bin/bash
# --- Setup script for the project --- #
# --- Create virtual environment --- #
echo "Setting up the project..."
# Check if python-venv is available
if ! python3 -m venv --help > /dev/null 2>&1; then
    echo "python3-venv is not installed. Installing it now..."
    sudo apt update
    sudo apt install python3-venv -y
fi
# Create virtual environment
python3 -m venv venv
echo "Virtual environment created successfully."
# Activate virtual environment
source venv/bin/activate
# --- Install dependencies --- #
echo "Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "Setup completed"

