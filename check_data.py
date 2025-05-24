import kagglehub
import pandas as pd
import os

# Download dataset
path = kagglehub.dataset_download("panaaaaa/english-premier-league-and-championship-full-dataset")
print("Dataset path:", path)

# List all files
print("\nAll files in directory:")
for root, dirs, files in os.walk(path):
    for file in files:
        full_path = os.path.join(root, file)
        print(f"Found file: {full_path}")
        if file.endswith('.csv'):
            df = pd.read_csv(full_path)
            print(f"\nFile: {file}")
            print("Shape:", df.shape)
            print("Columns:", df.columns.tolist())
            print("\nFirst few rows:")
            print(df.head())
            print("\n-------------------\n")
