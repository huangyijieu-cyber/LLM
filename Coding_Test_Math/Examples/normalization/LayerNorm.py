import math


class LayerNorm:
    """LayerNorm over the last dimension using plain Python lists."""

    def __init__(self, model_dim, eps=1e-5, gamma=None, beta=None):
        self.model_dim = model_dim
        self.eps = eps
        self.gamma = gamma if gamma is not None else [1.0] * model_dim
        self.beta = beta if beta is not None else [0.0] * model_dim

    def _norm_vector(self, x):
        mean = sum(x) / len(x)
        var = sum((value - mean) ** 2 for value in x) / len(x)
        inv_std = 1.0 / math.sqrt(var + self.eps)
        return [
            (value - mean) * inv_std * self.gamma[i] + self.beta[i]
            for i, value in enumerate(x)
        ]

    def forward(self, x):
        if x and isinstance(x[0], list):
            return [self.forward(item) for item in x]
        return self._norm_vector(x)

    __call__ = forward
