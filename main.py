import os
import numpy as np
from src.preprocessing import process_and_save_dataset
from src.evaluation import calculate_metrics

# import the models we wrote from scratch
from src.models.logistic_regression import CustomLogisticRegression
from src.models.knn import CustomKNN
from src.models.naive_bayes import CustomNaiveBayes
from src.models.perceptron import CustomPerceptron
from src.models.xgboost import CustomXGBoost

def ensure_data_processed():
    # checking if we already made the npy files to save time
    if not os.path.exists('data/train/train_data_X.npy'):
        print("Preprocessed data not found. Running the preprocessing script...")
        
        # Paths to the raw extracted folders
        train_base_dir = 'data/Sample_Training'
        test_base_dir = 'data/Sample_Test'
        
        if not os.path.exists(train_base_dir) or not os.path.exists(test_base_dir):
            raise FileNotFoundError("Raw folders 'Sample_Training' or 'Sample_Test' not found.")

        # grab all subfolders
        train_folders = [os.path.join(train_base_dir, f) for f in os.listdir(train_base_dir) 
                         if os.path.isdir(os.path.join(train_base_dir, f))]
        test_folders = [os.path.join(test_base_dir, f) for f in os.listdir(test_base_dir) 
                        if os.path.isdir(os.path.join(test_base_dir, f))]
        
        # save them to the proper target directories
        process_and_save_dataset(train_folders, 'data/train', 'train_data')
        process_and_save_dataset(test_folders, 'data/test', 'test_data')
    else:
        print("Found preprocessed .npy files. Skipping the slow stuff.")

def flatten_windows(X):
    # flatten the 3D windows into 2D so the basic models don't crash
    return X.reshape(X.shape[0], -1)

def balance_data_downsample(X, y):
    # handling imbalance by down-sampling class 0 to match class 1 count
    indices_0 = np.where(y == 0)[0]
    indices_1 = np.where(y == 1)[0]
    
    min_class_count = len(indices_1)
    np.random.seed(42) # fixed seed for reproducibility
    
    # randomly pick the same amount of 0s
    sampled_indices_0 = np.random.choice(indices_0, size=min_class_count, replace=False)
    balanced_indices = np.concatenate([sampled_indices_0, indices_1])
    np.random.shuffle(balanced_indices)
    
    return X[balanced_indices], y[balanced_indices]

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # train and get predictions
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return calculate_metrics(y_test, predictions)

def main():
    ensure_data_processed()

    print("Loading data...")
    X_train_raw = np.load('data/train/train_data_X.npy')
    y_train = np.load('data/train/train_data_y.npy')
    X_test_raw = np.load('data/test/test_data_X.npy')
    y_test = np.load('data/test/test_data_y.npy')

    # flatten for the shallow models
    X_train = flatten_windows(X_train_raw)
    X_test = flatten_windows(X_test_raw)

    # create a balanced version of the training set
    X_train_bal, y_train_bal = balance_data_downsample(X_train, y_train)

    # setup models with some basic hyperparameters
    models = {
        "Logistic Regression": CustomLogisticRegression(learning_rate=0.01, epochs=500),
        "K-Nearest Neighbors": CustomKNN(k=3),
        "Naive Bayes": CustomNaiveBayes(),
        "Perceptron": CustomPerceptron(learning_rate=0.01, epochs=500),
        "XGBoost": CustomXGBoost()
    }

    # test on imbalanced data
    print("\n" + "="*40)
    print(" RESULTS (WITHOUT BALANCING)")
    print("="*40)
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # test on balanced data
    print("\n" + "="*40)
    print(" RESULTS (WITH BALANCING)")
    print("="*40)
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(model, X_train_bal, y_train_bal, X_test, y_test)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

if __name__ == "__main__":
    main()