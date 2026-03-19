import xgboost as xgb
import numpy as np

class CustomXGBoost:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, objective='binary:logistic'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.model = None

    def fit(self, X, y):
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y)

        # Set parameters
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': self.objective,
            'eval_metric': 'logloss'
        }

        # Train the model
        self.model = xgb.train(params, dtrain, num_boost_round=self.n_estimators)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X)

        # Get probability predictions
        y_pred_proba = self.model.predict(dtest)

        # Convert to binary predictions (threshold at 0.5)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        return y_pred