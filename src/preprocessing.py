import os
import pandas as pd
import numpy as np

def process_csv_and_window(file_path, window_size=50, overlap=0.5):
    # read the file and drop the first two columns since they aren't features
    df = pd.read_csv(file_path)
    df = df.iloc[:, 2:] 
    
    # split features and the true labels
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    
    windowed_features = []
    windowed_labels = []
    
    # step size for 50% overlap
    step_size = int(window_size * (1 - overlap))
    
    # sliding window loop
    for start_i in range(0, len(features) - window_size + 1, step_size):
        end_i = start_i + window_size
        window_f = features[start_i:end_i]
        window_l = labels[start_i:end_i]
        
        # rule: label 1 if > 40% of the window is 1
        final_label = 1 if np.mean(window_l == 1) > 0.4 else 0
        
        windowed_features.append(window_f)
        windowed_labels.append(final_label)
        
    return windowed_features, windowed_labels

def process_and_save_dataset(folder_paths, save_dir, save_prefix):
    # make sure the folder exists first
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_features = []
    all_labels = []
    
    # loop through all folders and csvs
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
    
    save_path_X = os.path.join(save_dir, f'{save_prefix}_X.npy')
    save_path_y = os.path.join(save_dir, f'{save_prefix}_y.npy')

    # save as npy arrays to load faster later
    np.save(save_path_X, X)
    np.save(save_path_y, y)
    print(f"Saved {save_path_X} shape: {X.shape}")
    print(f"Saved {save_path_y} shape: {y.shape}")