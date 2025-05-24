import subprocess
import os
import sys
from pathlib import Path # Add Path import

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

# Start the backend API (FastAPI with Uvicorn) in the background
print("Starting backend API...")
backend_process = subprocess.Popen(['uvicorn', 'src.api.main:app', '--host', '0.0.0.0', '--port', '8000'])
print(f"Backend API started with PID: {backend_process.pid}")

# Give the backend a moment to start
import time
time.sleep(5) # Adjust sleep time if necessary

# Start the Streamlit app
print("Starting frontend application...")
try:
    subprocess.run(['streamlit', 'run', 'src/frontend/app.py'], check=True)
except KeyboardInterrupt:
    print("Frontend application stopped by user.")
finally:
    print("Stopping backend API...")
    backend_process.terminate()
    backend_process.wait()
    print("Backend API stopped.")
