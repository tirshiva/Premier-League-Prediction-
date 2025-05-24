import subprocess
import os
import sys
from pathlib import Path

# Set MLFLOW_TRACKING_URI to local mlruns
project_root = Path(__file__).resolve().parent
mlruns_path = project_root / "mlruns"
os.environ['MLFLOW_TRACKING_URI'] = mlruns_path.as_uri()
print(f"MLFLOW_TRACKING_URI set to: {os.environ['MLFLOW_TRACKING_URI']}")

# Check if the virtual environment is activated
if 'VIRTUAL_ENV' not in os.environ:
    print("Virtual environment is not activated. Please activate it manually or run this script from within the virtual environment.")
    sys.exit(1)

# Install dependencies
subprocess.run(['pip', 'install', '--upgrade', '--no-cache-dir', '-r', 'requirements.txt'], check=True)

# Start the Streamlit app
print("Starting frontend application...")
try:
    subprocess.run(['streamlit', 'run', 'src/frontend/app.py', '--server.port', '8502'], check=True)
except KeyboardInterrupt:
    print("Frontend application stopped by user.")