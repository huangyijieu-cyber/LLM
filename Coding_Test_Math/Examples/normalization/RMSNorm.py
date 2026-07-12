import math


class RMSNorm:
    """RMSNorm over the last dimension using plain Python lists."""

    def __init__(self, model_dim, eps=1e-8, gamma=None):
        self.model_dim = model_dim
        self.eps = eps
        self.gamma = gamma if gamma is not None else [1.0] * model_dim

    def _norm_vector(self, x):
        mean_square = sum(value * value for value in x) / len(x)
        inv_rms = 1.0 / math.sqrt(mean_square + self.eps)
        return [value * inv_rms * self.gamma[i] for i, value in enumerate(x)]

    def _norm(self, x):
        if x and isinstance(x[0], list):
            return [self._norm(item) for item in x]
        return self._norm_vector(x)

    def forward(self, x):
        return self._norm(x)

    __call__ = forward
