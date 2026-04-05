import os
import numpy as np
from src.preprocessing import process_and_save_dataset
from src.evaluation import calculate_metrics, stratified_k_fold
from src.balancing import compute_class_weights, get_sample_weights

from src.models.logistic_regression import CustomLogisticRegression
from src.models.knn import CustomKNN
from src.models.naive_bayes import CustomNaiveBayes
from src.models.perceptron import CustomPerceptron
from src.models.xgboost import CustomXGBoost


def ensure_data_processed():
    if not os.path.exists('data/train/train_data_X.npy'):
        print("Preprocessed data not found. Running preprocessing...")
        train_base_dir = 'data/Sample_Training'
        test_base_dir = 'data/Sample_Test'
        if not os.path.exists(train_base_dir) or not os.path.exists(test_base_dir):
            raise FileNotFoundError("Raw data folders not found.")
        train_folders = [os.path.join(train_base_dir, f) for f in os.listdir(train_base_dir)
                         if os.path.isdir(os.path.join(train_base_dir, f))]
        test_folders = [os.path.join(test_base_dir, f) for f in os.listdir(test_base_dir)
                        if os.path.isdir(os.path.join(test_base_dir, f))]
        process_and_save_dataset(train_folders, 'data/train', 'train_data')
        process_and_save_dataset(test_folders, 'data/test', 'test_data')
    else:
        print("Found preprocessed .npy files. Skipping preprocessing.")


def flatten_windows(X):
    return X.reshape(X.shape[0], -1)


def make_model(name, params, scale_pos_weight=1.0):
    """Instantiates a model by name with a given hyperparameter dict."""
    if name == "Logistic Regression":
        return CustomLogisticRegression(**params)
    elif name == "K-Nearest Neighbors":
        return CustomKNN(**params)
    elif name == "Naive Bayes":
        return CustomNaiveBayes()
    elif name == "Perceptron":
        return CustomPerceptron(**params)
    elif name == "XGBoost":
        return CustomXGBoost(**params, scale_pos_weight=scale_pos_weight)


def tune_and_evaluate(name, param_grid, X_train, y_train, X_test, y_test,
                       use_weights=False, k_folds=5):
    """
    Stratified K-Fold cross-validation over param_grid.
    Selects the best hyperparameters by mean balanced accuracy across folds,
    then trains a final model on the full training set and evaluates on test set.
    """
    folds = stratified_k_fold(X_train, y_train, k=k_folds)

    # Compute class weights once from the full training set if balancing
    if use_weights:
        class_weights = compute_class_weights(y_train)
    else:
        class_weights = None

    best_params = None
    best_score = -1.0

    print(f"\n  Tuning {name} over {len(param_grid)} configs x {k_folds} folds...")

    for params in param_grid:
        fold_scores = []

        for train_idx, val_idx in folds:
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            # Compute per-sample weights for this fold's training subset
            if use_weights:
                fold_class_weights = compute_class_weights(y_fold_train)
                sw = get_sample_weights(y_fold_train, fold_class_weights)
            else:
                sw = None

            # Compute scale_pos_weight for XGBoost (negatives / positives)
            spw = 1.0
            if use_weights and name == "XGBoost":
                n0 = np.sum(y_fold_train == 0)
                n1 = np.sum(y_fold_train == 1)
                spw = n0 / n1 if n1 > 0 else 1.0

            model = make_model(name, params, scale_pos_weight=spw)
            model.fit(X_fold_train, y_fold_train, sample_weights=sw)
            preds = model.predict(X_fold_val)
            metrics = calculate_metrics(y_fold_val, preds)
            fold_scores.append(metrics['Balanced Accuracy'])

        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    print(f"  Best params: {best_params}  |  CV Balanced Accuracy: {best_score:.4f}")

    # Train final model on full training set with best hyperparameters
    if use_weights:
        sw_full = get_sample_weights(y_train, class_weights)
        n0 = np.sum(y_train == 0)
        n1 = np.sum(y_train == 1)
        spw_full = n0 / n1 if n1 > 0 else 1.0
    else:
        sw_full = None
        spw_full = 1.0

    final_model = make_model(name, best_params, scale_pos_weight=spw_full)
    final_model.fit(X_train, y_train, sample_weights=sw_full)
    final_preds = final_model.predict(X_test)
    return calculate_metrics(y_test, final_preds), best_params


# ---------------------------------------------------------------------------
# Hyperparameter grids (kept small for speed; expand as needed)
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    "Logistic Regression": [
        {"learning_rate": lr, "epochs": ep}
        for lr in [0.001, 0.01]
        for ep in [300, 500]
    ],
    "K-Nearest Neighbors": [
        {"k": k} for k in [3, 5, 7]
    ],
    "Naive Bayes": [
        {}  # No tunable hyperparameters in our implementation
    ],
    "Perceptron": [
        {"learning_rate": lr, "epochs": ep}
        for lr in [0.001, 0.01]
        for ep in [300, 500]
    ],
    "XGBoost": [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        for n in [100, 200]
        for d in [4, 6]
        for lr in [0.05, 0.1]
    ]
}


def print_metrics(metrics, best_params):
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"    Best Params: {best_params}")


def main():
    ensure_data_processed()

    print("\nLoading data...")
    X_train_raw = np.load('data/train/train_data_X.npy')
    y_train = np.load('data/train/train_data_y.npy')
    X_test_raw = np.load('data/test/test_data_X.npy')
    y_test = np.load('data/test/test_data_y.npy')

    X_train = flatten_windows(X_train_raw)
    X_test = flatten_windows(X_test_raw)

    model_names = list(PARAM_GRIDS.keys())

    # -----------------------------------------------------------------------
    # WITHOUT BALANCING
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" RESULTS — WITHOUT CLASS WEIGHTING")
    print("=" * 50)
    for name in model_names:
        print(f"\n[{name}]")
        metrics, best_params = tune_and_evaluate(
            name, PARAM_GRIDS[name],
            X_train, y_train, X_test, y_test,
            use_weights=False
        )
        print_metrics(metrics, best_params)

    # -----------------------------------------------------------------------
    # WITH BALANCING (class-weight based — no data removed, no upsampling)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" RESULTS — WITH CLASS WEIGHTING (BALANCED)")
    print("=" * 50)
    for name in model_names:
        print(f"\n[{name}]")
        metrics, best_params = tune_and_evaluate(
            name, PARAM_GRIDS[name],
            X_train, y_train, X_test, y_test,
            use_weights=True
        )
        print_metrics(metrics, best_params)


if __name__ == "__main__":
    main()
