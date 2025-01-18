import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_data(folder_path):
     """
     Load and process data from the selected folder, distinguishing between target and behavior policies.
     """
     subfolder_data = {}
     
     # Iterate over subfolders
     for subdir, dirs, _ in os.walk(folder_path):
          for subfolder in dirs:
               subfolder_path = os.path.join(subdir, subfolder)
               target_policy_files = []
               behavior_policy_files = []
               
               # Distinguish between target and behavior policy files
               for file in os.listdir(subfolder_path):
                    if "NN" in file:
                         target_policy_files.append(os.path.join(subfolder_path, file))
                    else:
                         behavior_policy_files.append(os.path.join(subfolder_path, file))
               
               # Save files by subfolder
               subfolder_data[subfolder] = {
                    "Target": target_policy_files,
                    "Behavior": behavior_policy_files
               }
     return subfolder_data

def process_files(file_list, rolling_window=10, policy_type="", subfolder_name=""):
     """
     Process files into a single DataFrame for the specified policy type,
     applying a rolling mean for smoothing.
     """
     data_frames = []
     for file in file_list:
          df = pd.read_csv(file, usecols=[0, 4], header=0)  # Column A (env_step) and E (return)
          df.columns = ['env_step', 'return']
          df['return'] = df['return'].rolling(window=rolling_window, min_periods=1).mean()  # Apply rolling mean
          df['Policy'] = policy_type
          df['Subfolder'] = subfolder_name
          data_frames.append(df)
     return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def plot_data(df, policy_type):
     """
     Plot data with returns against environment steps for each policy type,
     using hue, style, estimator, and errorbar.
     """
     plt.figure(figsize=(10, 10))
     sns.lineplot(
          data=df,
          x='env_step',
          y='return',
          hue='Subfolder',
          style='Subfolder',
          estimator='mean',
          errorbar=('se', 1)
     )
     plt.title(f'{policy_type} in the Breakout Environment', fontsize=16)
     plt.xlabel('Environment Steps (Millions)', fontsize=14)
     plt.ylabel('Return', fontsize=14)
     plt.xticks([0, 5_000_000, 10_000_000, 15_000_000, 20_000_000], ['0', '5', '10', '15', '20'], fontsize=12)
     plt.yticks(fontsize=12)
     plt.legend(title='Configurations', title_fontsize=13, fontsize=12, loc='lower right')
     plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
     plt.grid(True)
     plt.show()

def main():
     """
     Main function to select a folder, process data, and generate separate plots for target and behavior policies.
     """
     # Set the folder path for visualization (e.g., "Datasets/Asterix")
     folder = input("Enter the folder name (e.g., 'Datasets/Asterix'): ")
     
     if not os.path.exists(folder):
          print(f"Folder '{folder}' does not exist. Please check the path.")
          return
     
     rolling_window = int(input("Enter the rolling window size for smoothing (e.g., 10): "))
     print(f"Processing data from: {folder}")
     subfolder_data = load_data(folder)
     
     all_target_data = []
     all_behavior_data = []
     
     # Process files for each subfolder
     for subfolder, files in subfolder_data.items():
          target_data = process_files(files['Target'], rolling_window=rolling_window, policy_type="Target", subfolder_name=subfolder)
          behavior_data = process_files(files['Behavior'], rolling_window=rolling_window, policy_type="Behavior", subfolder_name=subfolder)
          all_target_data.append(target_data)
          all_behavior_data.append(behavior_data)
     
     # Combine data across subfolders for each policy type
     target_policy_df = pd.concat(all_target_data, ignore_index=True) if all_target_data else pd.DataFrame()
     behavior_policy_df = pd.concat(all_behavior_data, ignore_index=True) if all_behavior_data else pd.DataFrame()
     
     # Plot target policy data
     if not target_policy_df.empty:
          plot_data(target_policy_df, policy_type='Target Policy (Greedy)')
     else:
          print("No target policy data found.")
     
     # Plot behavior policy data
     if not behavior_policy_df.empty:
          plot_data(behavior_policy_df, policy_type='Behavior Policy (E-Greedy)')
     else:
          print("No behavior policy data found.")

if __name__ == "__main__":
     main()
