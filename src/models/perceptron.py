import numpy as np

class CustomPerceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.train_losses = []  # Store training losses for visualization

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # perceptron math needs -1 and 1 instead of 0 and 1
        y_ = np.where(y <= 0, -1, 1) 
        
        self.train_losses = []  # Reset losses
        
        # update weights based on errors
        for epoch in range(self.epochs):
            epoch_loss = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                if y_[idx] * linear_output <= 0:
                    self.weights += self.lr * y_[idx] * x_i
                    self.bias += self.lr * y_[idx]
                    epoch_loss += 1  # Count misclassifications as loss
            
            # Store average loss for this epoch
            self.train_losses.append(epoch_loss / n_samples)
            
            # Early stopping if perfect classification
            if epoch_loss == 0:
                break

    def predict(self, X):
        # dot product and thresholding
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)