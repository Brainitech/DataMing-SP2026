import numpy as np

class CustomKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # knn doesn't really train, just stores the data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # FAST WAY: numpy calculates distance to all training samples at once
        distances = np.linalg.norm(self.X_train - x, axis=1)
        
        # get indices of the k nearest ones
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # very fast way to find the most frequent label (0 or 1)
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common