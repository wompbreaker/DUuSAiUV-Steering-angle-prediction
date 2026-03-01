# --- Setup script for the project --- #
Write-Host "Setting up the project..."

# --- Create virtual environment --- #
python -m venv venv
Write-Host "Virtual environment created successfully."
# Activate virtual environment
.\venv\Scripts\Activate.ps1
# --- Install dependencies --- #
Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Write-Host "Setup completed"