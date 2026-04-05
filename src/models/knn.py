import numpy as np

# -----------------------------------------------------------------------
# GPU availability probe
# -----------------------------------------------------------------------
# The previous probe only checked `import cupy`, which succeeds even when
# cuBLAS is missing. cuBLAS is needed for matmul (@), but NOT for
# elementwise ops like subtraction, squaring, and summing.
#
# Strategy:
#   1. Try to import cupy — if this fails, GPU_AVAILABLE = False.
#   2. Try a real matmul probe — if cuBLAS is missing this will raise
#      ImportError for libcublas. If it works, use the fast matmul path.
#   3. Try a pure elementwise op — no cuBLAS needed at all. If this works,
#      use the elementwise path (slightly slower but still much faster
#      than the CPU loop for large datasets).
# -----------------------------------------------------------------------
GPU_AVAILABLE = False
GPU_USE_MATMUL = False

try:
    import cupy as cp

    # Test 1: can we do matmul? (requires cuBLAS)
    try:
        _a = cp.ones((4, 4), dtype=cp.float32)
        _b = _a @ _a
        del _a, _b
        GPU_AVAILABLE = True
        GPU_USE_MATMUL = True
    except Exception:
        # cuBLAS missing — try elementwise path instead (no cuBLAS needed)
        try:
            _a = cp.ones((4, 4), dtype=cp.float32)
            _b = (_a ** 2).sum(axis=1)
            del _a, _b
            GPU_AVAILABLE = True
            GPU_USE_MATMUL = False
        except Exception:
            GPU_AVAILABLE = False

except ImportError:
    GPU_AVAILABLE = False


class CustomKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.sample_weights = None

    def fit(self, X, y, sample_weights=None):
        self.X_train = X
        self.y_train = y
        if sample_weights is not None:
            self.sample_weights = sample_weights / sample_weights.sum() * len(y)
        else:
            self.sample_weights = np.ones(len(y))

    def predict(self, X):
        if GPU_AVAILABLE:
            return self._predict_gpu(X)
        return self._predict_cpu(X)

    # ------------------------------------------------------------------
    # CPU PATH — per-sample loop with weighted voting
    # ------------------------------------------------------------------
    def _predict_cpu(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        k_weights = self.sample_weights[k_indices]
        vote_0 = k_weights[k_labels == 0].sum()
        vote_1 = k_weights[k_labels == 1].sum()
        return 1 if vote_1 >= vote_0 else 0

    # ------------------------------------------------------------------
    # GPU PATH A — matmul-based (fast, requires cuBLAS)
    # ------------------------------------------------------------------
    # Uses: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*(a . b)
    # The dot product is one large (n_test, n_train) matmul.
    # Batched to stay within VRAM limits.
    # ------------------------------------------------------------------
    def _predict_gpu_matmul(self, X, batch_size=1024):
        X_train_gpu = cp.array(self.X_train, dtype=cp.float32)
        y_train_gpu = cp.array(self.y_train)
        sw_gpu = cp.array(self.sample_weights, dtype=cp.float32)

        X_train_norm_sq = (X_train_gpu ** 2).sum(axis=1)

        all_preds = []
        for start in range(0, len(X), batch_size):
            X_batch = cp.array(X[start:start + batch_size], dtype=cp.float32)
            X_batch_norm_sq = (X_batch ** 2).sum(axis=1)

            dot = X_batch @ X_train_gpu.T
            dist_sq = X_batch_norm_sq[:, None] + X_train_norm_sq[None, :] - 2.0 * dot
            dist_sq = cp.maximum(dist_sq, 0.0)

            k_indices = cp.argsort(dist_sq, axis=1)[:, :self.k]
            k_labels  = y_train_gpu[k_indices]
            k_weights = sw_gpu[k_indices]

            vote_1 = (k_weights * (k_labels == 1)).sum(axis=1)
            vote_0 = (k_weights * (k_labels == 0)).sum(axis=1)
            all_preds.append(cp.asnumpy((vote_1 >= vote_0).astype(cp.int32)))

        return np.concatenate(all_preds)

    # ------------------------------------------------------------------
    # GPU PATH B — elementwise-only (no cuBLAS needed)
    # ------------------------------------------------------------------
    # Computes distances without matmul:
    #   diff = X_batch[:, None, :] - X_train[None, :, :]   shape (batch, n_train, features)
    #   dist_sq = (diff ** 2).sum(axis=2)
    #
    # This builds the full 3D diff tensor, so batch_size must be kept
    # small to stay within VRAM. At batch=64, n_train=46k, features=450:
    #   64 * 46000 * 450 * 4 bytes = ~5.3 GB — too large.
    #
    # So we use an intermediate trick: compute one feature dimension at a
    # time and accumulate into the (batch, n_train) distance matrix.
    # Memory = batch * n_train * 4 bytes = 64 * 46000 * 4 = ~11 MB. Safe.
    # ------------------------------------------------------------------
    def _predict_gpu_elementwise(self, X, batch_size=64):
        X_train_gpu = cp.array(self.X_train, dtype=cp.float32)
        y_train_gpu = cp.array(self.y_train)
        sw_gpu = cp.array(self.sample_weights, dtype=cp.float32)

        all_preds = []
        for start in range(0, len(X), batch_size):
            X_batch = cp.array(X[start:start + batch_size], dtype=cp.float32)
            b = len(X_batch)
            n_train = len(X_train_gpu)

            # Accumulate squared distances feature-by-feature
            dist_sq = cp.zeros((b, n_train), dtype=cp.float32)
            for f in range(X_batch.shape[1]):
                diff_f = X_batch[:, f:f+1] - X_train_gpu[:, f:f+1].T
                dist_sq += diff_f ** 2

            k_indices = cp.argsort(dist_sq, axis=1)[:, :self.k]
            k_labels  = y_train_gpu[k_indices]
            k_weights = sw_gpu[k_indices]

            vote_1 = (k_weights * (k_labels == 1)).sum(axis=1)
            vote_0 = (k_weights * (k_labels == 0)).sum(axis=1)
            all_preds.append(cp.asnumpy((vote_1 >= vote_0).astype(cp.int32)))

        return np.concatenate(all_preds)

    def _predict_gpu(self, X):
        if GPU_USE_MATMUL:
            return self._predict_gpu_matmul(X)
        return self._predict_gpu_elementwise(X)
