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
        # find distance between input and all training samples
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        
        # get indices of the k nearest ones
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # return the most common label among the k nearest
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common