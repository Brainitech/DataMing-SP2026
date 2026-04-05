import xgboost as xgb
import numpy as np

# XGBoost has built-in CUDA support — no extra library needed.
# Setting device='cuda' and tree_method='hist' hands all tree-building
# math to the GPU. On a 4050 this typically gives 5-10x faster training.
try:
    # Quick probe: try to allocate a tiny DMatrix on GPU
    _probe = xgb.DMatrix(np.zeros((1, 1)))
    xgb.train({'device': 'cuda', 'tree_method': 'hist'}, _probe,
               num_boost_round=1, verbose_eval=False)
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


class CustomXGBoost:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 objective='binary:logistic', scale_pos_weight=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        # scale_pos_weight: negatives / positives ratio
        # XGBoost upweights the positive class by this factor at each split
        self.scale_pos_weight = scale_pos_weight
        self.model = None

    def fit(self, X, y, sample_weights=None):
        dtrain = xgb.DMatrix(X, label=y, weight=sample_weights)

        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': self.objective,
            'eval_metric': 'logloss',
            'scale_pos_weight': self.scale_pos_weight,
        }

        if GPU_AVAILABLE:
            # tree_method='hist' is the only method that supports GPU in XGBoost 2.x
            # device='cuda' moves both data and computation onto the GPU
            params['device'] = 'cuda'
            params['tree_method'] = 'hist'

        self.model = xgb.train(params, dtrain, num_boost_round=self.n_estimators)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        dtest = xgb.DMatrix(X)
        y_pred_proba = self.model.predict(dtest)
        return (y_pred_proba >= 0.5).astype(int)
