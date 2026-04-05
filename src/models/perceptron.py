import numpy as np

class CustomPerceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.train_losses = []

    def fit(self, X, y, sample_weights=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Perceptron needs labels in {-1, +1}
        y_ = np.where(y <= 0, -1, 1)

        if sample_weights is None:
            sample_weights = np.ones(n_samples)

        sample_weights = sample_weights / sample_weights.sum() * n_samples

        self.train_losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                if y_[idx] * linear_output <= 0:
                    # Scale the update step by this sample's weight
                    # Minority class samples get a larger correction step
                    w = sample_weights[idx]
                    self.weights += self.lr * w * y_[idx] * x_i
                    self.bias += self.lr * w * y_[idx]
                    epoch_loss += 1

            self.train_losses.append(epoch_loss / n_samples)

            if epoch_loss == 0:
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
