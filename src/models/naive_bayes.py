import numpy as np

class CustomNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y, sample_weights=None):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        if sample_weights is None:
            sample_weights = np.ones(n_samples)

        total_weight = sample_weights.sum()

        for idx, c in enumerate(self.classes):
            mask = (y == c)
            X_c = X[mask]
            w_c = sample_weights[mask]

            # Weighted mean: sum(w_i * x_i) / sum(w_i)
            w_sum = w_c.sum()
            self.mean[idx, :] = np.average(X_c, axis=0, weights=w_c)

            # Weighted variance
            diff = X_c - self.mean[idx, :]
            self.var[idx, :] = np.average(diff ** 2, axis=0, weights=w_c) + 1e-9

            # Prior is now the weighted fraction of this class
            # With class weights, minority class prior is inflated — model pays more attention to it
            self.priors[idx] = w_sum / total_weight

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posteriors.append(prior + class_conditional)
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
