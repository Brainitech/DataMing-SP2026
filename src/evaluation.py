import numpy as np

def calculate_metrics(y_true, y_pred):
    # get the basic confusion matrix values
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # safe division to avoid errors if denominator is 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # final assignment metrics
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'F1 Score': f1_score,
        'Balanced Accuracy': balanced_accuracy
    }

def stratified_k_fold(X, y, k=5):
    # built this just in case we need cross validation later
    indices_0 = np.where(y == 0)[0]
    indices_1 = np.where(y == 1)[0]

    np.random.seed(42)
    np.random.shuffle(indices_0)
    np.random.shuffle(indices_1)

    folds_0 = np.array_split(indices_0, k)
    folds_1 = np.array_split(indices_1, k)

    folds = []
    for i in range(k):
        test_indices = np.concatenate([folds_0[i], folds_1[i]])
        np.random.shuffle(test_indices)
        
        train_indices_0 = np.concatenate([folds_0[j] for j in range(k) if j != i])
        train_indices_1 = np.concatenate([folds_1[j] for j in range(k) if j != i])
        train_indices = np.concatenate([train_indices_0, train_indices_1])
        np.random.shuffle(train_indices)

        folds.append((train_indices, test_indices))
        
    return folds