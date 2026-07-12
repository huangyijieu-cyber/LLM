from ...common import linear, silu, zip_map


class SwiGLUFFN:
    """SwiGLU feed-forward block: Down(SiLU(Gate(x)) * Up(x))."""

    def __init__(self, model_dim, intermediate_dim, w_gate=None, w_up=None, w_down=None):
        self.model_dim = model_dim
        self.intermediate_dim = intermediate_dim
        self.w_gate = w_gate if w_gate is not None else [
            [0.0] * model_dim for _ in range(intermediate_dim)
        ]
        self.w_up = w_up if w_up is not None else [
            [0.0] * model_dim for _ in range(intermediate_dim)
        ]
        self.w_down = w_down if w_down is not None else [
            [0.0] * intermediate_dim for _ in range(model_dim)
        ]

    def forward(self, x):
        gate = silu(linear(x, self.w_gate))
        up = linear(x, self.w_up)
        activated = zip_map(lambda a, b: a * b, gate, up)
        return linear(activated, self.w_down)

    __call__ = forward
