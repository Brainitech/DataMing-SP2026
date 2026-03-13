import os
import pandas as pd
import numpy as np

def process_csv_and_window(file_path, window_size=50, overlap=0.5):
    df = pd.read_csv(file_path)
    df = df.iloc[:, 2:] # Drop first two columns
    
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    
    windowed_features = []
    windowed_labels = []
    step_size = int(window_size * (1 - overlap))
    
    for start_i in range(0, len(features) - window_size + 1, step_size):
        end_i = start_i + window_size
        window_f = features[start_i:end_i]
        window_l = labels[start_i:end_i]
        
        # Label '1' if > 40% are '1', else '0'
        final_label = 1 if np.mean(window_l == 1) > 0.4 else 0
        
        windowed_features.append(window_f)
        windowed_labels.append(final_label)
        
    return windowed_features, windowed_labels

def process_and_save_dataset(folder_paths, save_dir, save_prefix):
    # Ensure the output directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_features = []
    all_labels = []
    
    for folder_path in folder_paths:
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)
                    w_features, w_labels = process_csv_and_window(file_path)
                    all_features.extend(w_features)
                    all_labels.extend(w_labels)
                    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Define full paths for saving
    save_path_X = os.path.join(save_dir, f'{save_prefix}_X.npy')
    save_path_y = os.path.join(save_dir, f'{save_prefix}_y.npy')

    # Save the processed arrays to disk
    np.save(save_path_X, X)
    np.save(save_path_y, y)
    print(f"Saved {save_path_X} with shape {X.shape}")
    print(f"Saved {save_path_y} with shape {y.shape}")

# --- Execution ---
data_dir = '../data'
train_base_dir = os.path.join(data_dir, 'Sample_Training')
test_base_dir = os.path.join(data_dir, 'Sample_Test') # Removed leading slash for correct joining

# Define output directories
train_out_dir = os.path.join(data_dir, 'train')
test_out_dir = os.path.join(data_dir, 'test')

# Automatically generate the list of folder paths
train_folders = [os.path.join(train_base_dir, folder) for folder in os.listdir(train_base_dir) 
                 if os.path.isdir(os.path.join(train_base_dir, folder))]

test_folders = [os.path.join(test_base_dir, folder) for folder in os.listdir(test_base_dir) 
                if os.path.isdir(os.path.join(test_base_dir, folder))]

# Process and save training data to ../data/train
process_and_save_dataset(train_folders, train_out_dir, 'train_data')

# Process and save testing data to ../data/test
process_and_save_dataset(test_folders, test_out_dir, 'test_data')
