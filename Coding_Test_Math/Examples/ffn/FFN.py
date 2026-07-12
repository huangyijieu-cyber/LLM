from ...common import linear, relu


class FFN:
    """Two-layer feed-forward network: Linear -> ReLU -> Linear."""

    def __init__(self, model_dim, intermediate_dim, w_up=None, b_up=None, w_down=None, b_down=None):
        self.model_dim = model_dim
        self.intermediate_dim = intermediate_dim
        self.w_up = w_up if w_up is not None else [
            [0.0] * model_dim for _ in range(intermediate_dim)
        ]
        self.b_up = b_up if b_up is not None else [0.0] * intermediate_dim
        self.w_down = w_down if w_down is not None else [
            [0.0] * intermediate_dim for _ in range(model_dim)
        ]
        self.b_down = b_down if b_down is not None else [0.0] * model_dim

    def forward(self, x):
        hidden = relu(linear(x, self.w_up, self.b_up))
        return linear(hidden, self.w_down, self.b_down)

    __call__ = forward
