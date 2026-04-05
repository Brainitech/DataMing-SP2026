import numpy as np

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y, sample_weights=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # If no weights given, treat all samples equally
        if sample_weights is None:
            sample_weights = np.ones(n_samples)

        # Normalize so weights sum to n_samples (keeps LR scale stable)
        sample_weights = sample_weights / sample_weights.sum() * n_samples

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            error = y_predicted - y

            # Scale each sample's error by its weight before averaging the gradient
            # This makes minority class mistakes matter more during gradient descent
            weighted_error = error * sample_weights

            dw = (1 / n_samples) * np.dot(X.T, weighted_error)
            db = (1 / n_samples) * np.sum(weighted_error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return np.array([1 if i > 0.5 else 0 for i in y_predicted])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
