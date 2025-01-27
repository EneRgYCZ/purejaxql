import os
import pandas as pd

def filter_csv_in_place_recursive(folder_path, column_index=0, threshold=20_000_000):
    """
    Recursively filters CSV files in a folder and its subfolders in place,
    removing rows where the value in the specified column exceeds the threshold.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        column_index (int): Index of the column to check against the threshold (default: 0 for the first column).
        threshold (int): Threshold value for filtering rows (default: 20,000,000).
    """
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return

    # Walk through the directory structure
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)
                try:
                    # Load the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Get the column name for the specified index
                    column_name = df.columns[column_index]
                    
                    # Filter rows where the column value is less than or equal to the threshold
                    filtered_df = df[df[column_name] <= threshold]
                    
                    # Overwrite the original file with the filtered data
                    filtered_df.to_csv(file_path, index=False)
                    
                    print(f"Filtered and overwritten file: {file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

# Example usage
folder_path = "/home/catalin/University/Thesis/purejaxql/visualization/Dataset"
filter_csv_in_place_recursive(folder_path)
