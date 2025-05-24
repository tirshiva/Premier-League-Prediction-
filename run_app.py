import subprocess
import os
import sys

# Check if the virtual environment is activated
if 'VIRTUAL_ENV' not in os.environ:
    print("Virtual environment is not activated. Please activate it manually or run this script from within the virtual environment.")
    sys.exit(1)

# Install dependencies
subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)

# Start the API
subprocess.run(['python', '-m', 'src.api.main'], check=True)
