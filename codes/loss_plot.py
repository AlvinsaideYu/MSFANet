import pandas as pd
import re
import matplotlib.pyplot as plt

# Initialize lists to store the epoch number and L1 averages
epoch_list = []
l1_average_list = []

# Pattern to identify epochs and their respective L1 values
pattern_epoch = re.compile(r'\[Epoch (\d+)\]')
pattern_l1 = re.compile(r'\[L1: ([\d.]+)\]')

# Load log data from file
with open(r"D:\Wz_Project_Learning\Super_Resolution_Reconstruction\HAUNet_RSISR\Ablation_experiment\FSMamba6x4_WHU-RS19\log.txt", 'r') as file:
    lines = file.readlines()

# Variables to store the current epoch and cumulative L1 values
current_epoch = None
l1_values = []

# Loop through each line in the log data
for line in lines:
    # Check for epoch line
    match_epoch = pattern_epoch.search(line)
    if match_epoch:
        # If we are moving to a new epoch, calculate and store the previous epoch's average L1
        if current_epoch is not None and l1_values:
            l1_average = sum(l1_values) / len(l1_values)
            epoch_list.append(current_epoch)
            l1_average_list.append(l1_average)
        
        # Update current epoch and reset L1 values list
        current_epoch = int(match_epoch.group(1))
        l1_values = []
    
    # Check for L1 values in the line
    match_l1 = pattern_l1.search(line)
    if match_l1:
        l1_value = float(match_l1.group(1))
        l1_values.append(l1_value)

# Calculate and store the last epoch's average L1 if it exists
if current_epoch is not None and l1_values:
    l1_average = sum(l1_values) / len(l1_values)
    epoch_list.append(current_epoch)
    l1_average_list.append(l1_average)

# Create DataFrame
df_l1 = pd.DataFrame({'Epoch': epoch_list, 'L6': l1_average_list})

# Save DataFrame to Excel file
output_path = "l1_average_loss_per_epoch.xlsx"
df_l1.to_excel(output_path, index=False)

output_path


