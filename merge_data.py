import pandas as pd
import os
from pathlib import Path

# Set the data directory
data_dir = Path('Data')

# Get all training and testing files
training_files = list(data_dir.glob('*-training.parquet'))
testing_files = list(data_dir.glob('*-testing.parquet'))

print(f"Found {len(training_files)} training files and {len(testing_files)} testing files")

# Function to merge files
def merge_files(file_list, output_name):
    if not file_list:
        print(f"No {output_name} files found")
        return

    # Read and concatenate all files
    dfs = []
    for file in file_list:
        print(f"Reading {file}")
        df = pd.read_parquet(file)
        dfs.append(df)

    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged dataframe
    output_path = data_dir / f'merged_{output_name}.parquet'
    merged_df.to_parquet(output_path, index=False)

    print(f"Merged {len(file_list)} files into {output_path}")
    print(f"Shape: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}")

# Merge training files
merge_files(training_files, 'training')

# Merge testing files
merge_files(testing_files, 'testing')

print("Data merging completed!")