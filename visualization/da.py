import os
import pandas as pd

def process_csv_files(folder_path):
    target_columns = ['env_step', 'pqn_Breakout-MinAtar - returned_episode_returns']

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Keep only the target columns
                df = df[target_columns]

                # Save the updated CSV file back to disk
                df.to_csv(file_path, index=False)
                print(f"Processed file: {file_name}")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

# Example usage
folder_path = "/home/catalin/University/Thesis/purejaxql/visualization/Datasets2/Breakout-MinAtar/Breakout-Baseline"
process_csv_files(folder_path)
